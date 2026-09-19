"""Standalone NN input behavior report. See
docs/design/2026-05-29-nn-input-report-design.md."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt

from aerocapture.training.ablation import NN_INPUT_NAMES, _load_cost_kwargs
from aerocapture.training.charts_nn_inputs import chart_nn_input_panel
from aerocapture.training.report_render import render_pdf
from aerocapture.training.seeds import NN_INPUT_REPORT_SEED_OFFSET, make_reserved_seeds
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


def _resolve_mask(toml_path: str, model_path: str | None = None) -> set[int]:
    """Resolve the deployed model's input_mask (which the Rust runtime actually
    uses) for panel "(unused)" greying. Prefers the JSON model's embedded mask
    over the TOML config -- the two can differ when --toml is not the training
    config. Falls back to the TOML mask, then the [0..16) backward-compat default."""
    cfg = load_toml_with_bases(Path(toml_path))
    nn_path = model_path or cfg.get("data", {}).get("neural_network")
    if nn_path:
        try:
            model_mask = json.loads(Path(nn_path).read_text()).get("input_mask")
            if model_mask is not None:
                return set(model_mask)
        except (OSError, json.JSONDecodeError) as e:  # fmt: skip  # ruff fmt strips the parens -> invalid syntax
            print(f"Warning: could not read input_mask from model file {nn_path} ({type(e).__name__}: {e}); falling back to TOML", file=sys.stderr)
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
    # Honor an overridden model path so panel greying reflects the model actually run.
    override_model = overrides.get("data.neural_network") if overrides else None
    in_mask = _resolve_mask(toml_path, str(override_model) if override_model is not None else None)

    # Derive the pool from the config's monte_carlo.seed (like report.py /
    # param_sweep) -- a hardcoded base 0 loses disjointness from training
    # draws for configs with a non-zero base seed.
    from aerocapture.training.toml_utils import load_toml_with_bases

    base_mc_seed = int(load_toml_with_bases(Path(toml_path)).get("monte_carlo", {}).get("seed", 42))
    seeds = make_reserved_seeds(base_mc_seed, NN_INPUT_REPORT_SEED_OFFSET, n_sims)
    import aerocapture_rs

    recs = aerocapture_rs.collect_nn_inputs(toml_path, seeds, overrides=overrides)

    X_list = [r["X"] for r in recs]
    time_list = [r["time"] for r in recs]
    energy_list = [r["energy"] for r in recs]
    dv = np.array([float(r["dv"]) for r in recs], dtype=np.float64)
    klass = classify_by_dv(dv, thr)

    rows = input_summary(X_list, klass, NN_INPUT_NAMES, in_mask)
    # Attach the per-input SVG basenames; the Typst template prefixes out_dir.
    for r in rows:
        j, nm = r["index"], r["name"]
        r["time_svg"] = f"nn_input_{j:02d}_{nm}_time.svg"
        r["energy_svg"] = f"nn_input_{j:02d}_{nm}_energy.svg"
    summary = {
        "scheme": Path(toml_path).stem,
        "dv_threshold": thr,
        "n_sims": n_sims,
        "n_blue": int(np.sum(klass == BLUE_LOW_DV)),
        "n_red": int(np.sum(klass == RED_HIGH_DV)),
        "inputs": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    for j, nm in enumerate(NN_INPUT_NAMES):
        chart_nn_input_panel(
            X_list,
            time_list,
            klass,
            j,
            nm,
            j in in_mask,
            out_dir / f"nn_input_{j:02d}_{nm}_time.svg",
            x_label="time (s)",
        )
        chart_nn_input_panel(
            X_list,
            energy_list,
            klass,
            j,
            nm,
            j in in_mask,
            out_dir / f"nn_input_{j:02d}_{nm}_energy.svg",
            x_label="energy estimated (MJ/kg)",
        )

    # PDF is best-effort: the SVGs + summary.json stay usable without typst.
    render_pdf("nn_input_report", out_dir, out_dir / "nn_input_report.pdf")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="NN input behavior report")
    ap.add_argument("training_dir")
    ap.add_argument("--toml", required=True)
    ap.add_argument("--n-sims", type=int, default=500)
    ap.add_argument("--dv-threshold", type=float, default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument(
        "--model",
        default=None,
        help="NN model JSON to report on (default: <training_dir>/best_model.json when present; the TOML's "
        "data.neural_network deploy path is shared across --output-dir siblings and may hold a foreign cell's weights)",
    )
    args = ap.parse_args()

    # Pin the run-local model + apply its co-trained scaffolding so the recorded
    # inputs reflect the deployed operating point (mirrors the ablation CLI).
    from aerocapture.training.deploy_overrides import load_scaffolding_overrides  # noqa: PLC0415

    training_dir = Path(args.training_dir)
    model = args.model
    if model is None and (training_dir / "best_model.json").exists():
        model = str(training_dir / "best_model.json")
    overrides: dict[str, object] = dict(load_scaffolding_overrides(training_dir))
    if model:
        overrides["data.neural_network"] = model
        print(f"Model: {model}")

    out = args.output_dir or str(Path(args.training_dir) / "nn_input_report")
    run_report(args.toml, n_sims=args.n_sims, output_dir=Path(out), dv_threshold=args.dv_threshold, overrides=overrides or None)
    print(f"NN input report written to {out}")


if __name__ == "__main__":
    main()
