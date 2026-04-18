"""Compare guidance schemes on identical Monte Carlo scenarios.

Runs each guidance scheme with its own training TOML config (so scheme-specific
settings like network architecture, navigation params, etc. are preserved),
then prints a summary table of performance metrics.

Usage:
    uv run python -m aerocapture.training.compare_guidance \
        --n-sims 500 \
        --schemes equilibrium_glide energy_controller pred_guid fnpag ftc neural_network neural_network_gru_pso piecewise_constant
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from aerocapture.training.evaluate import compute_cost

SCHEMES = [
    "equilibrium_glide",
    "energy_controller",
    "pred_guid",
    "fnpag",
    "ftc",
    "neural_network",
    "neural_network_rl",
    "neural_network_gru_pso",
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
    "neural_network_gru_pso": "configs/training/msr_aller_gru_pso_train.toml",
    "piecewise_constant": "configs/training/msr_aller_piecewise_constant_train.toml",
}

# Schemes that deploy via the Rust `neural_network` runtime (they provide a
# best_model.json but the guidance scheme name the Rust sim knows is "neural_network").
_NN_DEPLOY_SCHEMES = {"neural_network", "neural_network_rl", "neural_network_gru_pso"}


def run_scheme(
    scheme: str,
    n_sims: int,
    executable: str,
    cwd: Path,
    params_dir: Path | None = None,
    cost_kwargs: dict[str, float] | None = None,
    base_toml_override: Path | None = None,
) -> dict | None:
    """Run a single guidance scheme and return metrics.

    Uses the scheme's own training TOML as base config (so network architecture,
    navigation params, etc. are preserved). If base_toml_override is provided,
    uses that instead (fallback for schemes without a dedicated config).

    If params_dir/<scheme>/best_params.json exists, uses optimized params.
    Otherwise uses defaults from the training TOML.
    """
    from aerocapture.training.toml_utils import load_toml_with_bases

    # Use scheme-specific training TOML, or fallback to override
    scheme_toml_path = cwd / SCHEME_TRAINING_CONFIGS.get(scheme, "")
    if scheme_toml_path.exists():
        toml_data = load_toml_with_bases(scheme_toml_path)
        print(f"  Config: {SCHEME_TRAINING_CONFIGS[scheme]}")
    elif base_toml_override and base_toml_override.exists():
        toml_data = load_toml_with_bases(base_toml_override)
        print(f"  Config: {base_toml_override} (fallback)")
    else:
        print(f"  ERROR: No training config found for {scheme}")
        return None

    # Override n_sims and results suffix
    results_suffix = f".compare_{scheme}"
    toml_data.setdefault("simulation", {})["n_sims"] = n_sims
    toml_data.setdefault("data", {})["results_suffix"] = results_suffix

    # Set guidance type. NN-deploying schemes (neural_network, neural_network_rl)
    # both route through the Rust `neural_network` guidance runtime.
    if scheme in _NN_DEPLOY_SCHEMES:
        toml_data.setdefault("guidance", {})["type"] = "neural_network"
    else:
        toml_data.setdefault("guidance", {})["type"] = scheme

    # Handle NN: always prefer best_model.json from training output
    if scheme in _NN_DEPLOY_SCHEMES:
        nn_path = params_dir / scheme / "best_model.json" if params_dir else None
        if nn_path and nn_path.exists():
            toml_data.setdefault("data", {})["neural_network"] = str(nn_path)
            print(f"  Using optimized NN from {nn_path}")
        elif "neural_network" not in toml_data.get("data", {}):
            default_nn = "data/neural_network/nn_model.json"
            toml_data["data"]["neural_network"] = default_nn
            print(f"  Using default NN weights from {default_nn}")
    else:
        toml_data.get("data", {}).pop("neural_network", None)

    # Load optimized params if available
    if params_dir and scheme not in _NN_DEPLOY_SCHEMES:
        params_file = params_dir / scheme / "best_params.json"
        if params_file.exists():
            with open(params_file) as f:
                params = json.load(f)
            from aerocapture.training.param_spaces import GUIDANCE_TOML_SECTIONS

            section = GUIDANCE_TOML_SECTIONS[scheme]
            # Route prefixed params to correct TOML sections (same logic as evaluate.py)
            for k, v in params.items():
                if k.startswith("lateral."):
                    bare = k.removeprefix("lateral.")
                    if bare == "max_reversals":
                        v = int(round(v))
                    toml_data["guidance"].setdefault("lateral", {})[bare] = v
                elif k.startswith("exit."):
                    toml_data["guidance"].setdefault("ftc", {})[k.removeprefix("exit.")] = v
                elif k.startswith("nav."):
                    toml_data.setdefault("navigation", {})[k.removeprefix("nav.")] = v
                elif k.startswith("thermal."):
                    toml_data["guidance"].setdefault("thermal_limiter", {})[k.removeprefix("thermal.")] = v
                elif k.startswith("shaping."):
                    toml_data["guidance"].setdefault("command_shaping", {})[k.removeprefix("shaping.")] = v
                    toml_data["guidance"]["command_shaping"].setdefault("enabled", True)
                else:
                    toml_data["guidance"].setdefault(section, {})[k] = v
            print(f"  Using optimized params from {params_file}")
        else:
            print(f"  Using default params (no {params_file})")

    # Delete stale output files to avoid reading old results (both CSV and text)
    output_dir = toml_data.get("data", {}).get("output_dir", "output")
    suffix = results_suffix.lstrip(".")
    for pattern in [f"final{results_suffix}", f"final.{suffix}.csv"]:
        stale_file = cwd / output_dir / pattern
        stale_file.unlink(missing_ok=True)

    # Write temp TOML
    from aerocapture.training.evaluate import _write_toml

    temp_toml = cwd / f"_compare_{scheme}.toml"
    _write_toml(toml_data, temp_toml)

    # Run simulator
    exe = (cwd / executable).resolve()
    try:
        result = subprocess.run(
            [str(exe), str(temp_toml.resolve())],
            capture_output=True,
            cwd=str(cwd.resolve()),
            timeout=600,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  ERROR: {e}")
        temp_toml.unlink(missing_ok=True)
        return None

    temp_toml.unlink(missing_ok=True)

    # Parse final file -- auto-detect CSV vs legacy text
    output_dir = toml_data.get("data", {}).get("output_dir", "output")
    suffix = results_suffix.lstrip(".")
    final_file = cwd / output_dir / f"final.{suffix}.csv"
    if not final_file.exists():
        final_file = cwd / output_dir / f"final{results_suffix}"
    if not final_file.exists():
        print(f"  ERROR: final file not found in {output_dir}")
        if result.stderr:
            print(f"  stderr: {result.stderr.decode()[:500]}")
        return None

    from aerocapture.training.evaluate import _parse_final_to_legacy_array

    final = _parse_final_to_legacy_array(final_file)
    if final is None or len(final) == 0:
        return None
    ecc = final[:, 9]
    ifinal = final[:, 31]
    captured = (ifinal == 3) & (ecc < 1.0)

    metrics: dict = {
        "n_sims": len(final),
        "captured": int(captured.sum()),
        "capture_rate": float(captured.sum()) / len(final) * 100,
        "cost": compute_cost(final, **(cost_kwargs or {})),
    }

    if captured.any():
        metrics["apo_err_mean"] = float(np.abs(final[captured, 30]).mean())
        metrics["apo_err_std"] = float(np.abs(final[captured, 30]).std())
        metrics["peri_err_mean"] = float(np.abs(final[captured, 29]).mean())
        metrics["peri_err_std"] = float(np.abs(final[captured, 29]).std())
        dv = final[captured, 41]
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
    parser.add_argument("--params-dir", type=str, default="training_output", help="Directory with optimized params")
    parser.add_argument("--executable", type=str, default="src/rust/target/release/aerocapture")
    parser.add_argument("--cwd", type=str, default=".")
    args = parser.parse_args()

    base_toml = Path(args.base_toml) if args.base_toml else None
    cwd = Path(args.cwd)
    params_dir = Path(args.params_dir)

    # Parse cost function config from the first scheme's TOML (all inherit from same common.toml)
    from aerocapture.training.toml_utils import load_toml_with_bases

    first_scheme = args.schemes[0]
    cost_toml_path = cwd / SCHEME_TRAINING_CONFIGS.get(first_scheme, "")
    if cost_toml_path.exists():
        _toml = load_toml_with_bases(cost_toml_path)
    elif base_toml and base_toml.exists():
        _toml = load_toml_with_bases(base_toml)
    else:
        print(f"ERROR: No config found for cost function parsing (tried {first_scheme})")
        sys.exit(1)

    cost_cfg = _toml.get("cost_function", {})
    constraints = _toml.get("flight", {}).get("constraints", {})
    cost_kwargs: dict[str, float] = {
        "dv_threshold": float(cost_cfg.get("dv_threshold", 1000.0)),
        "g_load_limit": float(constraints.get("max_load_factor", 15.0)),
        "heat_flux_limit": float(constraints.get("max_heat_flux", 200.0)),
        "g_load_weight": float(cost_cfg.get("g_load_weight", 1000.0)),
        "heat_flux_weight": float(cost_cfg.get("heat_flux_weight", 1000.0)),
    }

    results: dict[str, dict] = {}
    for scheme in args.schemes:
        print(f"\nRunning {scheme}...")
        metrics = run_scheme(
            scheme,
            args.n_sims,
            args.executable,
            cwd,
            params_dir,
            cost_kwargs=cost_kwargs,
            base_toml_override=base_toml,
        )
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
