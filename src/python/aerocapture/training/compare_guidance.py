"""Compare guidance schemes on identical Monte Carlo scenarios.

Flies each scheme's deployed cell (`training_output/<scheme>/`: the optimized
TOML of a classical scheme, or the base training TOML + `best_model.json` +
co-trained scaffolding of an NN scheme) through the config's own Monte Carlo
(`cell_eval.fly_mc`), so every scheme sees the same `n_sims` dispersed scenarios
drawn from the shared `[monte_carlo] seed`, then prints a summary table.

Usage:
    uv run python -m aerocapture.training.compare_guidance \
        --n-sims 500 \
        --schemes equilibrium_glide energy_controller pred_guid fnpag ftc neural_network \
            neural_network_gru_pso neural_network_gru_ppo \
            neural_network_lstm_pso neural_network_lstm_ppo \
            neural_network_scaledpi_pso neural_network_delta_pso piecewise_constant
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from aerocapture.training import charts
from aerocapture.training.cell_eval import fly_mc
from aerocapture.training.cost import compute_cost

SCHEMES = [
    "equilibrium_glide",
    "energy_controller",
    "pred_guid",
    "fnpag",
    "ftc",
    "neural_network",
    "neural_network_rl",
    "neural_network_atan2_rl",
    "neural_network_gru_pso",
    "neural_network_gru_pso_magonly",
    "neural_network_gru_ppo",
    "neural_network_lstm_pso",
    "neural_network_lstm_ppo",
    "neural_network_window_pso",
    "neural_network_transformer_pso",
    "neural_network_mamba_pso",
    "neural_network_scaledpi_pso",
    "neural_network_delta_pso",
    "neural_network_joint",
    "piecewise_constant",
]

# Each scheme's training TOML (relative to repo root).
# These inherit from missions/ and common.toml, so they carry the full
# mission config including MC dispersions, cost function, and constraints.
SCHEME_TRAINING_CONFIGS: dict[str, str] = {
    "equilibrium_glide": "configs/training/msr_aller_eqglide_train.toml",
    "energy_controller": "configs/training/msr_aller_energy_controller_train.toml",
    "pred_guid": "configs/training/msr_aller_pred_guid_train.toml",
    "fnpag": "configs/training/msr_aller_fnpag_train.toml",
    "ftc": "configs/training/msr_aller_ftc_train.toml",
    "neural_network": "configs/training/msr_aller_nn_train_consolidated.toml",
    "neural_network_rl": "configs/training/msr_aller_rl_train.toml",
    "neural_network_atan2_rl": "configs/training/msr_aller_nn_atan2_ppo_train.toml",
    "neural_network_gru_pso": "configs/training/msr_aller_gru_pso_train.toml",
    "neural_network_gru_pso_magonly": "configs/training/msr_aller_gru_pso_magonly_train.toml",
    "neural_network_gru_ppo": "configs/training/msr_aller_gru_ppo_train.toml",
    "neural_network_lstm_pso": "configs/training/msr_aller_lstm_pso_train.toml",
    "neural_network_lstm_ppo": "configs/training/msr_aller_lstm_ppo_train.toml",
    "neural_network_window_pso": "configs/training/msr_aller_window_pso_train.toml",
    "neural_network_transformer_pso": "configs/training/msr_aller_transformer_pso_train.toml",
    "neural_network_mamba_pso": "configs/training/msr_aller_mamba_pso_train.toml",
    "neural_network_scaledpi_pso": "configs/training/msr_aller_nn_scaledpi_train.toml",
    "neural_network_delta_pso": "configs/training/msr_aller_nn_delta_train.toml",
    "neural_network_joint": "configs/training/msr_aller_nn_joint_train.toml",
    "piecewise_constant": "configs/training/msr_aller_piecewise_constant_train.toml",
}

# Schemes that deploy via the Rust `neural_network` runtime (they provide a
# best_model.json but the guidance scheme name the Rust sim knows is "neural_network").
_NN_DEPLOY_SCHEMES = {
    "neural_network",
    "neural_network_rl",
    "neural_network_atan2_rl",
    "neural_network_gru_pso",
    "neural_network_gru_pso_magonly",
    "neural_network_gru_ppo",
    "neural_network_lstm_pso",
    "neural_network_lstm_ppo",
    "neural_network_window_pso",
    "neural_network_transformer_pso",
    "neural_network_mamba_pso",
    "neural_network_scaledpi_pso",
    "neural_network_delta_pso",
    "neural_network_joint",
}


def run_scheme(
    scheme: str,
    n_sims: int,
    params_dir: Path | None = None,
    cost_kwargs: dict[str, Any] | None = None,
    base_toml_override: Path | None = None,
    sim_timeout_secs: float | None = None,
) -> dict | None:
    """Fly one scheme's deployed cell on the config's Monte Carlo; return its metrics.

    Uses the scheme's own training TOML as base config (so network architecture,
    navigation params, etc. are preserved). If base_toml_override is provided,
    uses that instead (fallback for schemes without a dedicated config). The cell
    is `params_dir/<scheme>`; a missing cell flies the TOML's defaults.
    """
    scheme_toml = Path(SCHEME_TRAINING_CONFIGS.get(scheme, ""))
    if scheme_toml.exists():
        print(f"  Config: {scheme_toml}")
    elif base_toml_override and base_toml_override.exists():
        scheme_toml = base_toml_override
        print(f"  Config: {base_toml_override} (fallback)")
    else:
        print(f"  ERROR: No training config found for {scheme}")
        return None

    # NN-deploying schemes all route through the Rust `neural_network` guidance runtime.
    guidance_type = "neural_network" if scheme in _NN_DEPLOY_SCHEMES else scheme
    cell_dir = params_dir / scheme if params_dir is not None else None
    if cell_dir is None or not cell_dir.exists():
        print("  Using TOML defaults (no deployed cell)")
    try:  # a classical cell with best_params.json but no optimized TOML is refused by cell_eval
        res = fly_mc(cell_dir, scheme_toml, n_sims=n_sims, extra_overrides={"guidance.type": guidance_type}, sim_timeout_secs=sim_timeout_secs)
    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR: {exc}")
        return None
    if res.toml_path != scheme_toml.resolve():
        print(f"  Using optimized params from {res.toml_path}")
    if "data.neural_network" in res.overrides:
        print(f"  Using optimized NN from {res.overrides['data.neural_network']}")
    scaffolding = sorted(k for k in res.overrides if k not in ("simulation.n_sims", "guidance.type", "data.neural_network"))
    if scaffolding:
        print(f"  Using optimized scaffolding: {', '.join(scaffolding)}")

    final = res.final_records
    if len(final) == 0:
        return None
    captured = res.captured

    metrics: dict = {
        "n_sims": len(final),
        "captured": int(captured.sum()),
        "capture_rate": float(captured.sum()) / len(final) * 100,
        "cost": compute_cost(final, **(cost_kwargs or {})),
    }

    if captured.any():
        metrics["apo_err_mean"] = float(np.abs(final[captured, charts._FR_APO_ERR]).mean())
        metrics["apo_err_std"] = float(np.abs(final[captured, charts._FR_APO_ERR]).std())
        metrics["peri_err_mean"] = float(np.abs(final[captured, charts._FR_PERI_ERR]).mean())
        metrics["peri_err_std"] = float(np.abs(final[captured, charts._FR_PERI_ERR]).std())
        dv = res.dv
        metrics["dv_mean"] = float(np.mean(dv))
        metrics["dv_std"] = float(np.std(dv))
    else:
        metrics["apo_err_mean"] = float("inf")
        metrics["peri_err_mean"] = float("inf")
        metrics["dv_mean"] = float("inf")

    return metrics


def print_comparison_table(results: dict[str, dict]) -> None:
    """Print a formatted comparison table."""
    header = f"{'Scheme':<22} {'Capture':>8} {'Cost':>12} {'Apo err':>10} {'Peri err':>10} {'Delta-V':>10}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for scheme, m in sorted(results.items(), key=lambda x: x[1].get("cost", 1e30)):
        cap = f"{m['captured']}/{m['n_sims']}"
        cost = f"{m['cost']:.2e}"
        apo = f"{m.get('apo_err_mean', float('inf')):.1f}" if m.get("apo_err_mean", float("inf")) < 1e10 else "N/A"
        peri = f"{m.get('peri_err_mean', float('inf')):.1f}" if m.get("peri_err_mean", float("inf")) < 1e10 else "N/A"
        dv = f"{m.get('dv_mean', float('inf')):.1f}" if m.get("dv_mean", float("inf")) < 1e10 else "N/A"
        print(f"{scheme:<22} {cap:>8} {cost:>12} {apo:>10} {peri:>10} {dv:>10}")

    print("=" * len(header))
    print("Apo/Peri err in km, Delta-V in m/s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare guidance schemes on identical MC scenarios")
    parser.add_argument(
        "--base-toml",
        type=str,
        default=None,
        help="Fallback TOML config for schemes without a dedicated training config",
    )
    parser.add_argument("--n-sims", type=int, default=500, help="Number of MC sims per scheme")
    parser.add_argument(
        "--schemes",
        nargs="+",
        default=SCHEMES,
        choices=SCHEMES,
        help="Schemes to compare",
    )
    parser.add_argument("--params-dir", type=str, default="training_output", help="Directory with the deployed cells (one per scheme)")
    parser.add_argument(
        "--sim-timeout", type=float, default=30.0, help="Per-sim wall-clock timeout in seconds (a non-terminating sim would hang the comparison)"
    )
    args = parser.parse_args()

    base_toml = Path(args.base_toml) if args.base_toml else None
    params_dir = Path(args.params_dir)

    # Parse cost function config from the first scheme's TOML (all inherit from
    # same common.toml). read_cost_kwargs is the canonical TOML->kwargs reader
    # (report.py, rl/train.py, param_sweep.py) -- the previous hand-rolled dict
    # omitted heat_load_weight/heat_load_limit, so compute_cost fell back to
    # weight 10000 vs the trained-with 1.0, mis-ranking heat-load violators.
    from aerocapture.training.report import read_cost_kwargs

    first_scheme = args.schemes[0]
    cost_toml_path = Path(SCHEME_TRAINING_CONFIGS.get(first_scheme, ""))
    if cost_toml_path.exists():
        cost_kwargs: dict[str, Any] = read_cost_kwargs(cost_toml_path)
    elif base_toml and base_toml.exists():
        cost_kwargs = read_cost_kwargs(base_toml)
    else:
        print(f"ERROR: No config found for cost function parsing (tried {first_scheme})")
        sys.exit(1)

    results: dict[str, dict] = {}
    for scheme in args.schemes:
        print(f"\nRunning {scheme}...")
        metrics = run_scheme(scheme, args.n_sims, params_dir, cost_kwargs=cost_kwargs, base_toml_override=base_toml, sim_timeout_secs=args.sim_timeout)
        if metrics:
            results[scheme] = metrics
            print(f"  Captured: {metrics['captured']}/{metrics['n_sims']}, cost={metrics['cost']:.2e}")
        else:
            print("  FAILED")

    print_comparison_table(results)

    # Save results to JSON
    output_file = params_dir / "comparison_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
