"""Calibrate NN input normalization scales from observed raw distributions.

Runs the deployed NN over a reserved seed pool, collects the normalized 35-wide
candidate trace, inverts each input's KNOWN current transform to recover the raw
distribution, then emits new scale constants so each input's [p1, p99] fills
~[-1, 1]. Heavy-tailed / acceleration / DV inputs -> asinh; bounded -> affine.
The 3 live correction-DV inputs (32/33/34) get per-component asinh scales with
pre-capture sentinel ticks (normalized == 1.5) excluded before percentiles.

One-time tool: paste the emitted Rust const block into neural.rs (Task 5), then
re-run nn_input_report to verify ~1% saturation.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

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

# DV inputs (live correction-DV components) -- per-component scales, sentinel-excluded.
_DV_INDICES = {32, 33, 34}
# Pre-capture sentinel saturates the asinh output to exactly 1.5 (asinh(sinh(1.5))).
_SENTINEL_NORM = 1.5

# Current transform per index (must mirror build_nn_input at calibration time).
# Forms: ("asinh", s) | ("affine", a, b) meaning norm = a*raw + b | ("raw",).
CURRENT_TRANSFORMS: dict[int, tuple] = {
    0: ("affine", 1.0, -1.0),  # ecc - 1
    1: ("affine", 3.0 / 5.0, 0.0),  # deg * 3/5
    2: ("asinh", 8.802043e02),
    3: ("asinh", 5.554906e06),
    4: ("affine", 2.0 / 3e3, -3.0),  # (raw/3e3 - 1.5)*2
    5: ("affine", 1.0 / 20.0, -1.0),  # raw/20 - 1
    6: ("affine", 2.0, -1.0),  # frac*2 - 1
    7: ("affine", 2.0, -1.0),
    8: ("affine", 1.0 / 65.0, -1.0),  # (raw-65)/65
    9: ("affine", 1.0 / 0.3, 0.0),
    10: ("affine", 2.0 / math.pi, 0.0),
    11: ("affine", 1.0 / 50.0, -1.0),  # raw/50 - 1
    12: ("affine", 1.0 / 10.0, 0.0),  # raw/10
    13: ("asinh", 3.259362e07),
    14: ("asinh", 6.626041e07),
    16: ("raw",),
    17: ("affine", 1.0 / 2e3, -1.0),  # raw/2e3 - 1
    18: ("asinh", 7.333648e02),
    19: ("affine", 1.0 / 2e3, 0.0),  # raw/2e3
    31: ("asinh", 9.158960e05),
    32: ("asinh", 1.28e02),  # provisional S_DV
    33: ("asinh", 1.28e02),
    34: ("asinh", 1.28e02),
}

# Rust const name per asinh index (for the emitted block).
_ASINH_CONST_NAME = {
    2: "S_RADIAL_VELOCITY",
    3: "S_ORBITAL_ENERGY",
    5: "S_ACCEL_MAGNITUDE",
    11: "S_DRAG_ACCEL",
    12: "S_LIFT_ACCEL",
    13: "S_SMA_ERROR",
    14: "S_APOAPSIS_ALT",
    18: "S_HDOT_NOMINAL",
    19: "S_PDYN_ERROR",
    31: "S_PERIAPSIS_ALT",
    32: "S_DV1",
    33: "S_DV2",
    34: "S_DV3",
}


def drop_sentinel(norm: np.ndarray, idx: int) -> np.ndarray:
    """Drop pre-capture sentinel ticks (norm == 1.5) for DV inputs; pass others through."""
    if idx in _DV_INDICES:
        return norm[np.abs(norm - _SENTINEL_NORM) > 1e-6]
    return norm


def invert_transform(norm: np.ndarray, transform: tuple) -> np.ndarray:
    kind = transform[0]
    if kind == "asinh":
        (_, s) = transform
        return np.asarray(s * np.sinh(norm))
    if kind == "affine":
        (_, a, b) = transform
        return np.asarray((norm - b) / a)
    if kind == "raw":
        return norm
    raise ValueError(f"unknown transform {transform!r}")


def derive_asinh_scale(p1: float, p99: float) -> float:
    span = max(abs(p1), abs(p99))
    span = max(span, 1e-12)
    return span / math.sinh(1.0)


def derive_affine(p1: float, p99: float) -> tuple[float, float]:
    center = (p1 + p99) / 2.0
    half = max((p99 - p1) / 2.0, 1e-6)
    return center, half


def _collect_raw(toml_path: str, n_sims: int) -> dict[int, np.ndarray]:
    import aerocapture_rs

    from aerocapture.training.evaluate import make_reserved_seeds

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
        norm = drop_sentinel(norm, idx)
        if idx in CURRENT_TRANSFORMS:
            raw[idx] = invert_transform(norm, CURRENT_TRANSFORMS[idx])
    return raw


def calibrate(toml_path: str, n_sims: int) -> str:
    from aerocapture.training.ablation import NN_INPUT_NAMES

    raw = _collect_raw(toml_path, n_sims)
    lines: list[str] = []
    lines.append("// === calibrated input scales (calibrate_inputs.py) ===")
    seen_const: set[str] = set()
    table: list[str] = []
    for idx in sorted(raw):
        if idx in _SKIP:
            continue
        vals = raw[idx]
        name = NN_INPUT_NAMES[idx] if idx < len(NN_INPUT_NAMES) else f"idx{idx}"
        if len(vals) == 0:
            table.append(f"  [{idx:2d}] {name:22s} SKIPPED (no finite values)")
            continue
        p1, p50, p99 = np.percentile(vals, [1, 50, 99])
        if idx in _FORCE_ASINH:
            s = derive_asinh_scale(p1, p99)
            const = _ASINH_CONST_NAME.get(idx, f"S_IDX{idx}")
            if const not in seen_const:
                lines.append(f"const {const}: f64 = {s:.6e}; // {name}: p1={p1:.3g} p99={p99:.3g}")
                seen_const.add(const)
            table.append(f"  [{idx:2d}] {name:22s} asinh  p1={p1:11.4g} p50={p50:11.4g} p99={p99:11.4g} -> s={s:.4e}")
        else:
            center, half = derive_affine(p1, p99)
            table.append(f"  [{idx:2d}] {name:22s} affine p1={p1:11.4g} p50={p50:11.4g} p99={p99:11.4g} -> center={center:.6e} half={half:.6e}")
    report = "\n".join(["RAW DISTRIBUTION TABLE:", *table, "", *lines])
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate NN input normalization scales")
    ap.add_argument("--toml", required=True)
    ap.add_argument("--n-sims", type=int, default=300)
    ap.add_argument("--output", default=None, help="optional path to write the report")
    args = ap.parse_args()
    report = calibrate(args.toml, args.n_sims)
    print(report)
    if args.output:
        Path(args.output).write_text(report)


if __name__ == "__main__":
    main()
