"""Compare guidance schemes on identical Monte Carlo scenarios.

Runs each guidance scheme with the same random seed and dispersion config,
then prints a summary table of performance metrics.

Usage:
    uv run python -m aerocapture.training.compare_guidance \
        --base-toml configs/training/msr_aller_eqglide_train.toml \
        --n-sims 100 \
        --schemes equilibrium_glide energy_controller pred_guid fnpag ftc neural_network
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from aerocapture.training.evaluate import compute_cost

SCHEMES = ["equilibrium_glide", "energy_controller", "pred_guid", "fnpag", "ftc", "neural_network"]


def run_scheme(
    scheme: str,
    base_toml: Path,
    n_sims: int,
    executable: str,
    cwd: Path,
    params_dir: Path | None = None,
    cost_kwargs: dict[str, float] | None = None,
) -> dict | None:
    """Run a single guidance scheme and return metrics.

    If params_dir/<scheme>/best_params.json exists, uses optimized params.
    Otherwise uses defaults.
    """
    from aerocapture.training.toml_utils import load_toml_with_bases

    toml_data = load_toml_with_bases(Path(base_toml))

    # Override n_sims and results suffix
    results_suffix = f".compare_{scheme}"
    toml_data.setdefault("simulation", {})["n_sims"] = n_sims
    toml_data.setdefault("data", {})["results_suffix"] = results_suffix

    # Set guidance type
    toml_data.setdefault("guidance", {})["type"] = scheme

    # Handle NN: ensure neural_network data path is set
    if scheme == "neural_network":
        if "neural_network" not in toml_data.get("data", {}):
            # Use best_model.json if available, otherwise default
            nn_path = params_dir / "neural_network" / "best_model.json" if params_dir else None
            if nn_path and nn_path.exists():
                toml_data["data"]["neural_network"] = str(nn_path)
                print(f"  Using optimized NN from {nn_path}")
            else:
                default_nn = "data/neural_network/nn_model.json"
                toml_data["data"]["neural_network"] = default_nn
                print(f"  Using default NN weights from {default_nn}")
    else:
        toml_data.get("data", {}).pop("neural_network", None)

    # Load optimized params if available
    if params_dir and scheme != "neural_network":
        params_file = params_dir / scheme / "best_params.json"
        if params_file.exists():
            with open(params_file) as f:
                params = json.load(f)
            from aerocapture.training.param_spaces import GUIDANCE_TOML_SECTIONS

            section = GUIDANCE_TOML_SECTIONS[scheme]
            # Merge into existing section (FTC has many required fields beyond optimized ones)
            existing = toml_data.get("guidance", {}).get(section, {})
            if existing:
                existing.update(params)
                toml_data["guidance"][section] = existing
            else:
                toml_data["guidance"][section] = params
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

    # Parse final file — auto-detect CSV vs legacy text
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
    energy = final[:, 7]
    ecc = final[:, 9]
    ifinal = final[:, 31]
    captured = (ecc < 1.0) & (energy < 0) & (ifinal != 4.0)

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
    parser.add_argument("--base-toml", type=str, required=True, help="Base TOML config file")
    parser.add_argument("--n-sims", type=int, default=100, help="Number of MC sims per scheme")
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

    base_toml = Path(args.base_toml)
    cwd = Path(args.cwd)
    params_dir = Path(args.params_dir)

    if not base_toml.exists():
        print(f"ERROR: Base TOML not found: {base_toml}")
        sys.exit(1)

    # Parse cost function config from TOML (with defaults)
    from aerocapture.training.toml_utils import load_toml_with_bases

    cost_kwargs: dict[str, float] = {}
    _toml = load_toml_with_bases(Path(base_toml))
    cost_cfg = _toml.get("cost_function", {})
    cost_kwargs = {
        "dv_threshold": float(cost_cfg.get("dv_threshold", 1000.0)),
        "g_load_limit": float(cost_cfg.get("g_load_limit", 15.0)),
        "heat_flux_limit": float(cost_cfg.get("heat_flux_limit", 200.0)),
        "g_load_weight": float(cost_cfg.get("g_load_weight", 1000.0)),
        "heat_flux_weight": float(cost_cfg.get("heat_flux_weight", 1000.0)),
    }

    results: dict[str, dict] = {}
    for scheme in args.schemes:
        print(f"\nRunning {scheme}...")
        metrics = run_scheme(
            scheme,
            base_toml,
            args.n_sims,
            args.executable,
            cwd,
            params_dir,
            cost_kwargs=cost_kwargs,
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
