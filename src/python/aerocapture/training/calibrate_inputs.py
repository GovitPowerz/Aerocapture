"""Calibrate NN input normalization scales from observed raw distributions.

Runs the deployed NN over a reserved seed pool, collects the normalized 35-wide
candidate trace, inverts each input's CURRENT transform to recover the raw
distribution, then proposes new normalization entries so each input's [p1, p99]
fills ~[-1, 1]. Heavy-tailed / acceleration / DV inputs -> asinh; bounded -> affine.

Single source of truth for the current transforms is the Rust
`aerocapture_rs.default_normalization()` table (a list of 35
`{transform, scale, center}` dicts) -- there is no hand-mirrored Python copy.
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

# Reserved seed pool, disjoint from train/val/final-eval/report streams.
CALIBRATION_SEED_OFFSET = 6_000_000

# Inputs that always use asinh (heavy-tailed / spiky), regardless of tail ratio.
_FORCE_ASINH = {
    2,
    3,
    5,
    11,
    12,
    13,
    14,
    18,
    19,
    31,
    32,
    33,
    34,
}
# Bounded inputs to skip entirely (binary / tanh / sin-cos already in [-1,1]).
_SKIP = {15, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30}


def _current_transforms() -> list[dict]:
    """Return the Rust DEFAULT_NORMALIZATION table (single source of truth).

    Each entry is `{"transform": "none"|"asinh"|"tanh", "scale": float, "center": float}`.
    """
    import aerocapture_rs  # type: ignore[import-not-found]

    return cast("list[dict]", aerocapture_rs.default_normalization())


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


def _collect_raw(toml_path: str, n_sims: int) -> dict[int, np.ndarray]:
    import aerocapture_rs  # type: ignore[import-not-found]

    from aerocapture.training.evaluate import make_reserved_seeds

    transforms = _current_transforms()
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


def _propose_normalization(toml_path: str, n_sims: int) -> tuple[list[dict], list[str]]:
    """Return (proposed normalization entries, human-readable table lines).

    The proposed list has 35 entries (model-JSON form). Inputs in `_SKIP` and any
    input not observed keep their CURRENT transform unchanged.
    """
    from aerocapture.training.ablation import NN_INPUT_NAMES

    current = _current_transforms()
    raw = _collect_raw(toml_path, n_sims)
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
        p1, p50, p99 = np.percentile(vals, [1, 50, 99])
        if idx in _FORCE_ASINH:
            s = derive_asinh_scale(p1, p99)
            proposed[idx] = {"transform": "asinh", "scale": s, "center": 0.0}
            table.append(f"  [{idx:2d}] {name:22s} asinh  p1={p1:11.4g} p50={p50:11.4g} p99={p99:11.4g} -> scale={s:.6e}")
        else:
            center, half = derive_affine(p1, p99)
            proposed[idx] = {"transform": "none", "scale": half, "center": center}
            table.append(f"  [{idx:2d}] {name:22s} none   p1={p1:11.4g} p50={p50:11.4g} p99={p99:11.4g} -> scale={half:.6e} center={center:.6e}")
    return proposed, table


def calibrate(toml_path: str, n_sims: int) -> tuple[list[dict], str]:
    proposed, table = _propose_normalization(toml_path, n_sims)
    report = "\n".join(["RAW DISTRIBUTION TABLE:", *table])
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
    ap.add_argument("--output", default=None, help="optional path to write the readable report")
    ap.add_argument("--write-model", default=None, help="optional model JSON path to write the proposed normalization array into")
    ap.add_argument(
        "--emit-toml",
        default=None,
        help="optional path to write a paste-ready [network.normalization] TOML snippet",
    )
    args = ap.parse_args()
    proposed, report = calibrate(args.toml, args.n_sims)
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
