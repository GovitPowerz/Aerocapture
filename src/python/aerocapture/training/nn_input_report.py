"""Standalone NN input behavior report. See
docs/superpowers/specs/2026-05-29-nn-input-report-design.md."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import aerocapture_rs
import numpy as np
import numpy.typing as npt

from aerocapture.training.ablation import NN_INPUT_NAMES, _load_cost_kwargs
from aerocapture.training.charts_nn_inputs import chart_nn_input_panel
from aerocapture.training.toml_utils import load_toml_with_bases

# class codes
BLUE_LOW_DV = 0
RED_HIGH_DV = 1


def classify_by_dv(dv: npt.NDArray[np.float64], threshold: float) -> npt.NDArray[np.int8]:
    """Blue (0) if final DV < threshold, red (1) otherwise."""
    return np.where(np.asarray(dv) < threshold, BLUE_LOW_DV, RED_HIGH_DV).astype(np.int8)


def input_summary(
    X_list: list[npt.NDArray[np.float64]],
    traj_class: npt.NDArray[np.int8],
    names: list[str],
    in_mask: set[int],
) -> list[dict[str, object]]:
    """Per-input stats over all (trajectory x timestep) samples.

    Returns one dict per input with index, name, p1/p50/p99, frac_out_of_range
    (fraction of samples with |value| > 1), separation
    (|mean_red - mean_blue| / pooled_std), and in_mask. Sorted by
    frac_out_of_range desc, then separation desc.
    """
    n_inputs = len(names)
    blue_parts = [X_list[i] for i in range(len(X_list)) if traj_class[i] == BLUE_LOW_DV]
    red_parts = [X_list[i] for i in range(len(X_list)) if traj_class[i] == RED_HIGH_DV]
    blue = np.concatenate(blue_parts, axis=0) if blue_parts else np.empty((0, n_inputs))
    red = np.concatenate(red_parts, axis=0) if red_parts else np.empty((0, n_inputs))
    alls = np.concatenate(list(X_list), axis=0)
    rows: list[dict[str, object]] = []
    for j in range(n_inputs):
        col = alls[:, j]
        p1, p50, p99 = (float(v) for v in np.percentile(col, [1, 50, 99]))
        frac_oor = float(np.mean(np.abs(col) > 1.0))
        if blue.shape[0] and red.shape[0]:
            mb, mr = float(blue[:, j].mean()), float(red[:, j].mean())
            pooled = float(np.sqrt(0.5 * (blue[:, j].var() + red[:, j].var()))) + 1e-12
            sep = abs(mr - mb) / pooled
        else:
            sep = 0.0
        rows.append(
            {
                "index": j,
                "name": names[j],
                "p1": p1,
                "p50": p50,
                "p99": p99,
                "frac_out_of_range": frac_oor,
                "separation": sep,
                "in_mask": j in in_mask,
            }
        )
    rows.sort(key=lambda r: (r["frac_out_of_range"], r["separation"]), reverse=True)
    return rows


NN_INPUT_REPORT_SEED_OFFSET = 5_000_000


def _resolve_mask(toml_path: str) -> set[int]:
    cfg = load_toml_with_bases(Path(toml_path))
    mask = cfg.get("network", {}).get("input_mask")
    return set(mask) if mask is not None else set(range(16))


def _default_dv_threshold(toml_path: str) -> float:
    return float(_load_cost_kwargs(toml_path).get("dv_threshold", 1000.0))


def run_report(
    toml_path: str,
    n_sims: int = 500,
    output_dir: Path | None = None,
    dv_threshold: float | None = None,
    overrides: dict[str, object] | None = None,
) -> Path:
    """Run the deployed NN over n_sims seeds, classify by final DV, render
    per-input panels (time + energy) + a summary table."""
    out_dir = Path(output_dir) if output_dir else Path("nn_input_report")
    out_dir.mkdir(parents=True, exist_ok=True)
    thr = dv_threshold if dv_threshold is not None else _default_dv_threshold(toml_path)
    in_mask = _resolve_mask(toml_path)

    seeds = [NN_INPUT_REPORT_SEED_OFFSET + i for i in range(n_sims)]
    recs = aerocapture_rs.collect_nn_inputs(toml_path, seeds, overrides=overrides)

    X_list = [r["X"] for r in recs]
    time_list = [r["time"] for r in recs]
    energy_list = [r["energy"] for r in recs]
    dv = np.array([float(r["dv"]) for r in recs], dtype=np.float64)
    klass = classify_by_dv(dv, thr)

    rows = input_summary(X_list, klass, NN_INPUT_NAMES, in_mask)
    summary = {
        "dv_threshold": thr,
        "n_sims": n_sims,
        "n_blue": int(np.sum(klass == BLUE_LOW_DV)),
        "n_red": int(np.sum(klass == RED_HIGH_DV)),
        "inputs": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    for j, nm in enumerate(NN_INPUT_NAMES):
        chart_nn_input_panel(
            X_list, time_list, klass, j, nm, j in in_mask,
            out_dir / f"nn_input_{j:02d}_{nm}_time.svg", x_label="time (s)",
        )
        chart_nn_input_panel(
            X_list, energy_list, klass, j, nm, j in in_mask,
            out_dir / f"nn_input_{j:02d}_{nm}_energy.svg", x_label="energy (MJ/kg)",
        )
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="NN input behavior report")
    ap.add_argument("training_dir")
    ap.add_argument("--toml", required=True)
    ap.add_argument("--n-sims", type=int, default=500)
    ap.add_argument("--dv-threshold", type=float, default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()
    out = args.output_dir or str(Path(args.training_dir) / "nn_input_report")
    run_report(args.toml, n_sims=args.n_sims, output_dir=Path(out), dv_threshold=args.dv_threshold)
    print(f"NN input report written to {out}")


if __name__ == "__main__":
    main()
