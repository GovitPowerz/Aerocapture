"""Ablation analysis for NN input importance ranking.

Zeroes out each input one at a time on a trained network and measures
cost degradation. Ranks inputs by importance (high delta = important).

Uses the same cost function as the training pipeline (softplus-quadratic
DV + constraint penalties) for consistency.
"""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

NN_INPUT_NAMES: list[str] = [
    "eccentricity_excess",  # 0
    "inclination_error",  # 1
    "radial_velocity",  # 2
    "orbital_energy",  # 3
    "velocity",  # 4
    "accel_magnitude",  # 5
    "heat_flux_fraction",  # 6
    "heat_load_fraction",  # 7
    "altitude",  # 8
    "fpa",  # 9
    "latitude",  # 10
    "drag_accel",  # 11
    "lift_accel",  # 12
    "sma_error",  # 13
    "apoapsis_alt",  # 14
    "bounce_flag",  # 15
    "cos_bank_nominal",  # 16
    "pdyn_nominal",  # 17
    "hdot_nominal",  # 18
    "pdyn_error",  # 19
    "exit_bank_angle",  # 20
    "density_exit",  # 21
    "ref_velocity_latched",  # 22
]

# Index of dv_total_m_s in the 52-column final_record array (0-based, includes sim_number).
# Verified against FINAL_CSV_COLUMNS in output.rs and results.rs comment (final_record[41]).
_DV_TOTAL_COL = 41


def _resolve_nn_path(toml_path: str) -> Path:
    """Return the absolute path to the neural network JSON model file."""
    from aerocapture.training.toml_utils import load_toml_with_bases

    config = load_toml_with_bases(Path(toml_path))
    nn_path_str: str | None = config.get("data", {}).get("neural_network")
    if nn_path_str is None:
        raise ValueError(f"No data.neural_network path found in {toml_path}")
    # Resolve relative to CWD (matches Rust simulator behavior)
    return Path(nn_path_str).resolve()


def _load_cost_kwargs(toml_path: str) -> dict[str, Any]:
    """Extract cost function kwargs from TOML config (mirrors training pipeline)."""
    from aerocapture.training.toml_utils import load_toml_with_bases

    config = load_toml_with_bases(Path(toml_path))
    kwargs: dict[str, Any] = {}
    cost_cfg = config.get("cost_function", {})
    if "dv_threshold" in cost_cfg:
        kwargs["dv_threshold"] = float(cost_cfg["dv_threshold"])
    if "g_load_weight" in cost_cfg:
        kwargs["g_load_weight"] = float(cost_cfg["g_load_weight"])
    if "heat_flux_weight" in cost_cfg:
        kwargs["heat_flux_weight"] = float(cost_cfg["heat_flux_weight"])
    if "heat_load_weight" in cost_cfg:
        kwargs["heat_load_weight"] = float(cost_cfg["heat_load_weight"])
    if "cost_transform" in cost_cfg:
        kwargs["cost_transform"] = str(cost_cfg["cost_transform"])
    constraints = config.get("flight", {}).get("constraints", {})
    if "max_load_factor" in constraints:
        kwargs["g_load_limit"] = float(constraints["max_load_factor"])
    if "max_heat_flux" in constraints:
        kwargs["heat_flux_limit"] = float(constraints["max_heat_flux"])
    if "max_heat_load" in constraints:
        kwargs["heat_load_limit"] = float(constraints["max_heat_load"])
    return kwargs


def _mean_per_sim_cost(final_records: np.ndarray, cost_kwargs: dict[str, Any]) -> float:
    """Compute mean per-sim cost using the training cost function."""
    from aerocapture.training.evaluate import compute_cost

    costs = np.array([compute_cost(fr.reshape(1, 52), **cost_kwargs) for fr in final_records])
    return float(np.mean(costs))


def run_ablation(
    toml_path: str,
    n_sims: int = 1000,
    sim_timeout_secs: float | None = None,
) -> dict:
    """Run ablation analysis on a trained NN model.

    For each of the 23 inputs, writes a temp model JSON with ablated_input set,
    overrides data.neural_network to point at it, and measures cost degradation
    vs baseline using the same cost function as the training pipeline.

    Returns dict with keys: baseline_cost, n_sims, results, ranked.
    """
    import aerocapture_rs

    nn_path = _resolve_nn_path(toml_path)
    model_json = json.loads(nn_path.read_text())
    cost_kwargs = _load_cost_kwargs(toml_path)

    common_overrides: dict = {"simulation.n_sims": n_sims}

    # Baseline run (no ablation)
    baseline = aerocapture_rs.run_mc(toml_path, overrides=common_overrides, sim_timeout_secs=sim_timeout_secs)
    baseline_mean = _mean_per_sim_cost(baseline.final_records, cost_kwargs)

    # Only ablate inputs that the model actually reads (in the mask).
    active_mask: set[int] | None = None
    if "input_mask" in model_json and model_json["input_mask"] is not None:
        active_mask = set(model_json["input_mask"])

    results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_model_path = Path(tmpdir) / "ablated_model.json"

        for idx in range(len(NN_INPUT_NAMES)):
            if active_mask is not None and idx not in active_mask:
                results.append(
                    {
                        "index": idx,
                        "name": NN_INPUT_NAMES[idx],
                        "baseline_cost": baseline_mean,
                        "ablated_cost": baseline_mean,
                        "delta": 0.0,
                        "abs_delta": 0.0,
                        "masked_out": True,
                    }
                )
                continue

            ablated_json = copy.deepcopy(model_json)
            ablated_json["ablated_input"] = idx
            tmp_model_path.write_text(json.dumps(ablated_json))

            overrides = {**common_overrides, "data.neural_network": str(tmp_model_path)}
            ablated = aerocapture_rs.run_mc(toml_path, overrides=overrides, sim_timeout_secs=sim_timeout_secs)
            ablated_mean = _mean_per_sim_cost(ablated.final_records, cost_kwargs)
            delta = ablated_mean - baseline_mean

            results.append(
                {
                    "index": idx,
                    "name": NN_INPUT_NAMES[idx],
                    "baseline_cost": baseline_mean,
                    "ablated_cost": ablated_mean,
                    "delta": delta,
                    "abs_delta": abs(delta),
                }
            )

    ranked = sorted(results, key=lambda r: float(r["abs_delta"]), reverse=True)  # type: ignore[arg-type]
    for rank, r in enumerate(ranked):
        r["rank"] = rank + 1

    return {
        "baseline_cost": baseline_mean,
        "n_sims": n_sims,
        "results": results,
        "ranked": ranked,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="NN input ablation analysis")
    parser.add_argument("training_dir", help="Path to training output directory")
    parser.add_argument("--toml", required=True, help="TOML config path")
    parser.add_argument("--n-sims", type=int, default=1000, help="MC sims per ablation run")
    parser.add_argument("--sim-timeout", type=float, default=None, help="Per-sim timeout (seconds)")
    args = parser.parse_args()

    print(f"Running ablation analysis with {args.n_sims} sims per input...")
    results = run_ablation(args.toml, args.n_sims, args.sim_timeout)

    # Print table
    print(f"\nBaseline mean cost: {results['baseline_cost']:.4f}")
    print(f"{'Rank':<6}{'Index':<8}{'Name':<25}{'Delta':>12}{'Ablated Cost':>15}")
    print("-" * 66)
    for r in results["ranked"]:
        print(f"{r['rank']:<6}{r['index']:<8}{r['name']:<25}{r['delta']:>12.4f}{r['ablated_cost']:>15.4f}")

    # Save JSON
    out_path = Path(args.training_dir) / "ablation_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")

    # Generate chart
    from aerocapture.training.charts_ablation import chart_ablation_bar

    svg_path = Path(args.training_dir) / "ablation_chart.svg"
    chart_ablation_bar(results["ranked"], str(svg_path))
    print(f"Chart saved to {svg_path}")


if __name__ == "__main__":
    main()
