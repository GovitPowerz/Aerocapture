"""Calibrate NN input normalization scales from observed raw distributions.

Runs the deployed NN over a reserved seed pool, collects the normalized 35-wide
candidate trace, inverts each input's CURRENT transform to recover the raw
distribution, then proposes new normalization entries so each input's
[p_lo, p_hi] (default [5, 95], tunable via --target-percentiles) maps exactly to
[-1, 1].

The transform is chosen FROM THE DATA: a tail-heaviness statistic
(tail_ratio = (p99.9 - p0.1) / (p_hi - p_lo)) decides between affine ("none",
unbounded linear -- good for near-bounded inputs) and asinh (log-compressed
tails -- good for heavy-tailed inputs). Both use a two-parameter endpoint fit
(center + scale) so the chosen percentiles land on +-1 exactly, even for
one-sided / skewed inputs. `_FORCE_ASINH` is an OPTIONAL override (respected
unless --no-force-asinh): inputs known a priori heavy-tailed (DV, energy,
accelerations) stay asinh even if a thin MC sample under-estimates their tail.
`tanh` is never auto-selected (asinh dominates it for heavy tails, affine for
bounded); it survives only on `_SKIP` inputs that already carry it.

The trace is inverted with the normalization the SIM ACTUALLY used, resolved by
`_resolve_normalization` (`[network.normalization]` override > embedded model >
Rust DEFAULT). Inverting with a transform that differs from the forward pass
distorts the recovered raw by `s_forward/s_invert` -- inverting with the fixed
DEFAULT against a moving deployed scale makes the proposed scale oscillate
across retrain+recalibrate cycles instead of converging.
The DV inputs (32/33/34) are ordinary asinh inputs (no sentinel); they are
calibrated over their full distribution like any other heavy-tailed input.

The proposed normalization is emitted in the model-JSON form (a list of 35
`{transform, scale, center}` entries) and can be written into a model JSON's
top-level `"normalization"` field via `--write-model PATH`, or as a paste-ready
`[network.normalization]` TOML snippet via `--emit-toml PATH` (drop under the
`[network]` table of a training config without losing hand-written comments).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import cast

import numpy as np

from aerocapture.training.evaluate import CALIBRATION_SEED_OFFSET

# Inputs forced to asinh regardless of the observed tail ratio (known heavy-tailed:
# velocities, energy, accelerations, DV). Optional override -- disable with
# --no-force-asinh to let the data-driven selector decide every non-skip input.
_FORCE_ASINH: set[int] = set()
# Bounded inputs to skip entirely (binary / tanh / sin-cos already in [-1,1]).
_SKIP: set[int] = {15, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30}


def _current_transforms() -> list[dict]:
    """Return the Rust DEFAULT_NORMALIZATION table.

    Each entry is `{"transform": "none"|"asinh"|"tanh", "scale": float, "center": float}`.
    """
    import aerocapture_rs  # type: ignore[import-not-found]

    return cast("list[dict]", aerocapture_rs.default_normalization())


def _resolve_normalization(toml_path: str) -> list[dict]:
    """Resolve the normalization the SIM ACTUALLY applies for this config.

    Mirrors Rust precedence (config.rs / data/mod.rs): `[network.normalization]`
    override REPLACES the model's table; absent that, the embedded model
    `normalization` block; absent that, the Rust DEFAULT.

    This MUST match what `collect_nn_inputs` used to produce the trace -- the
    inversion `raw = invert(norm)` only recovers the true raw when it uses the
    same constants as the forward pass. Inverting with DEFAULT against a trace
    normalized with a deployed override gives `recovered = (s_default/s_deployed)
    * raw_true`, a scale-dependent distortion that makes the proposed scale
    oscillate across retrain+recalibrate cycles.
    """
    from aerocapture.training.toml_utils import load_toml_with_bases

    cfg = load_toml_with_bases(Path(toml_path))
    override = cfg.get("network", {}).get("normalization")
    if override is not None:
        return [dict(e) for e in override]
    model_path = cfg.get("data", {}).get("neural_network")
    if model_path and Path(model_path).exists():
        emb = json.loads(Path(model_path).read_text()).get("normalization")
        if emb:
            return [dict(e) for e in emb]
    return _current_transforms()


def invert_transform(norm: np.ndarray, spec: dict) -> np.ndarray:
    """Invert a `{transform, scale, center}` normalization back to raw values.

    Forward is `norm = transform((raw - center) / scale)`, so:
      none/affine -> raw = norm * scale + center
      asinh       -> raw = scale * sinh(norm) + center
      tanh        -> raw = scale * atanh(norm) + center  (guarded |norm| < 1)
    """
    kind = spec["transform"]
    scale = float(spec["scale"])
    center = float(spec["center"])
    if kind == "none":
        return np.asarray(norm * scale + center)
    if kind == "asinh":
        return np.asarray(scale * np.sinh(norm) + center)
    if kind == "tanh":
        clamped = np.clip(norm, -1.0 + 1e-12, 1.0 - 1e-12)
        return np.asarray(scale * np.arctanh(clamped) + center)
    raise ValueError(f"unknown transform {kind!r}")


def derive_asinh_scale(p1: float, p99: float) -> float:
    span = max(abs(p1), abs(p99))
    span = max(span, 1e-12)
    return span / math.sinh(1.0)


def derive_affine(p1: float, p99: float) -> tuple[float, float]:
    center = (p1 + p99) / 2.0
    half = max((p99 - p1) / 2.0, 1e-6)
    return center, half


def derive_asinh_endpoints(p_lo: float, p_hi: float) -> tuple[float, float]:
    """Two-parameter asinh fit mapping (p_lo, p_hi) -> (-1, +1) exactly.

    Solving asinh((p_lo - c)/s) = -1 and asinh((p_hi - c)/s) = +1 gives
    c = (p_lo + p_hi)/2, s = (p_hi - p_lo)/(2 sinh 1). Unlike the legacy
    `derive_asinh_scale` (center pinned at 0), this hits BOTH endpoints even for
    one-sided inputs (e.g. always-positive DV: p_lo would otherwise map to 0).
    """
    center = (p_lo + p_hi) / 2.0
    scale = max((p_hi - p_lo) / (2.0 * math.sinh(1.0)), 1e-12)
    return center, scale


def tail_ratio(vals: np.ndarray, lo_pct: float, hi_pct: float) -> float:
    """Heavy-tailedness: outer (p0.1..p99.9) span over core (p_lo..p_hi) span.

    ~1.05 for uniform, ~1.5 for Gaussian, >3 for lognormal / spiky inputs.
    """
    p_lo, p_hi = np.percentile(vals, [lo_pct, hi_pct])
    core = max(float(p_hi - p_lo), 1e-12)
    o_lo, o_hi = np.percentile(vals, [0.1, 99.9])
    return float(o_hi - o_lo) / core


def choose_transform(
    vals: np.ndarray,
    *,
    lo_pct: float,
    hi_pct: float,
    tail_threshold: float,
    force_asinh: bool = False,
) -> tuple[dict, float]:
    """Pick {none, asinh} from the data and fit it so (p_lo, p_hi) -> (-1, +1).

    asinh when the input is heavy-tailed (tail_ratio >= threshold) or forced;
    affine ("none") otherwise. Returns (spec, observed_tail_ratio).
    """
    p_lo, p_hi = (float(v) for v in np.percentile(vals, [lo_pct, hi_pct]))
    tr = tail_ratio(vals, lo_pct, hi_pct)
    if force_asinh or tr >= tail_threshold:
        center, scale = derive_asinh_endpoints(p_lo, p_hi)
        return {"transform": "asinh", "scale": scale, "center": center}, tr
    center, half = derive_affine(p_lo, p_hi)
    return {"transform": "none", "scale": half, "center": center}, tr


def _collect_raw(toml_path: str, n_sims: int, transforms: list[dict] | None = None) -> dict[int, np.ndarray]:
    import aerocapture_rs  # type: ignore[import-not-found]

    from aerocapture.training.evaluate import make_reserved_seeds

    # Invert with the normalization the SIM actually used, not the Rust DEFAULT
    # (see _resolve_normalization); otherwise the recovered raw is distorted by
    # s_default/s_deployed and the proposed scale oscillates between retrains.
    if transforms is None:
        transforms = _resolve_normalization(toml_path)
    seeds = make_reserved_seeds(0, CALIBRATION_SEED_OFFSET, n_sims)
    recs = aerocapture_rs.collect_nn_inputs(toml_path, seeds, overrides=None)
    cols: dict[int, list[np.ndarray]] = {}
    for r in recs:
        x = np.asarray(r["X"])  # (T, 35) normalized
        for idx in range(x.shape[1]):
            cols.setdefault(idx, []).append(x[:, idx])
    raw: dict[int, np.ndarray] = {}
    for idx, parts in cols.items():
        norm = np.concatenate(parts)
        norm = norm[np.isfinite(norm)]
        if idx < len(transforms):
            raw[idx] = invert_transform(norm, transforms[idx])
    return raw


def _propose_normalization(
    toml_path: str,
    n_sims: int,
    *,
    lo_pct: float = 5.0,
    hi_pct: float = 95.0,
    tail_threshold: float = 1.6,
    respect_force_asinh: bool = True,
) -> tuple[list[dict], list[str]]:
    """Return (proposed normalization entries, human-readable table lines).

    The proposed list has 35 entries (model-JSON form). Inputs in `_SKIP` and any
    input not observed keep their CURRENT transform unchanged. Every other input
    is fit so its (p_lo, p_hi) maps to (-1, +1), with the transform chosen by
    `choose_transform` (data-driven, with the `_FORCE_ASINH` override).
    """
    from aerocapture.training.ablation import NN_INPUT_NAMES

    current = _resolve_normalization(toml_path)
    raw = _collect_raw(toml_path, n_sims, transforms=current)
    proposed = [dict(spec) for spec in current]
    table: list[str] = []
    for idx in range(len(current)):
        name = NN_INPUT_NAMES[idx] if idx < len(NN_INPUT_NAMES) else f"idx{idx}"
        if idx in _SKIP:
            continue
        vals = raw.get(idx)
        if vals is None or len(vals) == 0:
            table.append(f"  [{idx:2d}] {name:22s} SKIPPED (no finite values)")
            continue
        force = respect_force_asinh and idx in _FORCE_ASINH
        spec, tr = choose_transform(vals, lo_pct=lo_pct, hi_pct=hi_pct, tail_threshold=tail_threshold, force_asinh=force)
        proposed[idx] = spec
        p_lo, p50, p_hi = np.percentile(vals, [lo_pct, 50, hi_pct])
        forced = " (forced)" if force and spec["transform"] == "asinh" and tr < tail_threshold else ""
        table.append(
            f"  [{idx:2d}] {name:22s} {spec['transform']:5s} "
            f"p{lo_pct:g}={p_lo:11.4g} p50={p50:11.4g} p{hi_pct:g}={p_hi:11.4g} "
            f"tail={tr:5.2f} -> scale={spec['scale']:.6e} center={spec['center']:.6e}{forced}"
        )
    return proposed, table


def calibrate(
    toml_path: str,
    n_sims: int,
    *,
    lo_pct: float = 5.0,
    hi_pct: float = 95.0,
    tail_threshold: float = 1.6,
    respect_force_asinh: bool = True,
) -> tuple[list[dict], str]:
    proposed, table = _propose_normalization(
        toml_path,
        n_sims,
        lo_pct=lo_pct,
        hi_pct=hi_pct,
        tail_threshold=tail_threshold,
        respect_force_asinh=respect_force_asinh,
    )
    header = f"RAW DISTRIBUTION TABLE (target p{lo_pct:g}/p{hi_pct:g} -> -1/+1, asinh when tail_ratio >= {tail_threshold:g}):"
    report = "\n".join([header, *table])
    return proposed, report


def format_toml_normalization(entries: list[dict], names: list[str]) -> str:
    """Render a calibrated normalization list as a paste-ready TOML
    `normalization = [...]` assignment (inline tables, one per line, each
    annotated with its candidate-input index + name). Paste under [network]."""
    lines = ["normalization = ["]
    for i, e in enumerate(entries):
        name = names[i] if i < len(names) else f"idx{i}"
        # repr() on floats is round-trip-exact and valid TOML float syntax.
        lines.append(f'  {{ transform = "{e["transform"]}", scale = {float(e["scale"])!r}, center = {float(e["center"])!r} }}, # {i} {name}')
    lines.append("]")
    return "\n".join(lines) + "\n"


def _write_model_normalization(model_path: str, normalization: list[dict]) -> None:
    path = Path(model_path)
    doc = json.loads(path.read_text())
    doc["normalization"] = normalization
    path.write_text(json.dumps(doc, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate NN input normalization scales")
    ap.add_argument("--toml", required=True)
    ap.add_argument("--n-sims", type=int, default=300)
    ap.add_argument(
        "--target-percentiles",
        type=float,
        nargs=2,
        metavar=("LO", "HI"),
        default=(5.0, 95.0),
        help="percentiles mapped to (-1, +1). Default 5 95.",
    )
    ap.add_argument(
        "--tail-threshold",
        type=float,
        default=1.6,
        help="tail_ratio (p99.9-p0.1)/(p_hi-p_lo) above which asinh is chosen over affine. Default 1.6.",
    )
    ap.add_argument(
        "--no-force-asinh",
        action="store_true",
        help="ignore the _FORCE_ASINH override list and let tail_ratio decide every non-skip input.",
    )
    ap.add_argument("--output", default=None, help="optional path to write the readable report")
    ap.add_argument("--write-model", default=None, help="optional model JSON path to write the proposed normalization array into")
    ap.add_argument(
        "--emit-toml",
        default=None,
        help="optional path to write a paste-ready [network.normalization] TOML snippet",
    )
    args = ap.parse_args()
    lo_pct, hi_pct = args.target_percentiles
    if not 0.0 <= lo_pct < hi_pct <= 100.0:
        ap.error(f"--target-percentiles must satisfy 0 <= LO < HI <= 100 (got {lo_pct} {hi_pct})")
    proposed, report = calibrate(
        args.toml,
        args.n_sims,
        lo_pct=lo_pct,
        hi_pct=hi_pct,
        tail_threshold=args.tail_threshold,
        respect_force_asinh=not args.no_force_asinh,
    )
    print(report)
    if args.output:
        Path(args.output).write_text(report)
    if args.write_model:
        _write_model_normalization(args.write_model, proposed)
        print(f"\nWrote {len(proposed)} normalization entries to {args.write_model}")
    if args.emit_toml:
        from aerocapture.training.ablation import NN_INPUT_NAMES

        snippet = "# Generated by calibrate_inputs.py -- paste under the [network] table of your training config.\n"
        snippet += format_toml_normalization(proposed, NN_INPUT_NAMES)
        Path(args.emit_toml).write_text(snippet)
        print(f"normalization TOML snippet written to {args.emit_toml} (paste under [network])")


if __name__ == "__main__":
    main()
