"""Compare guidance schemes on identical Monte Carlo scenarios.

Runs each guidance scheme with the same random seed and dispersion config,
then prints a summary table of performance metrics.

Usage:
    uv run python -m aerocapture.training.compare_guidance \
        --base-toml configs/msr_aller_eqglide_train.toml \
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
) -> dict | None:
    """Run a single guidance scheme and return metrics.

    If params_dir/<scheme>/best_params.json exists, uses optimized params.
    Otherwise uses defaults.
    """
    import tomllib

    with open(base_toml, "rb") as f:
        toml_data = tomllib.load(f)

    # Override n_sims
    toml_data.setdefault("simulation", {})["n_sims"] = n_sims

    # Set guidance type
    toml_data.setdefault("guidance", {})["type"] = scheme

    # Remove neural_network reference if not NN
    if scheme != "neural_network":
        toml_data.get("data", {}).pop("neural_network", None)

    # Load optimized params if available
    if params_dir and scheme != "neural_network":
        params_file = params_dir / scheme / "best_params.json"
        if params_file.exists():
            with open(params_file) as f:
                params = json.load(f)
            from aerocapture.training.param_spaces import GUIDANCE_TOML_SECTIONS

            section = GUIDANCE_TOML_SECTIONS[scheme]
            toml_data["guidance"][section] = params
            print(f"  Using optimized params from {params_file}")
        else:
            print(f"  Using default params (no {params_file})")

    # Write temp TOML
    from aerocapture.training.evaluate import _write_toml

    temp_toml = cwd / f"_compare_{scheme}.toml"
    _write_toml(toml_data, temp_toml)

    # Set results suffix per scheme to avoid file collisions
    results_suffix = f".compare_{scheme}"
    # Patch the suffix in the temp TOML
    with open(temp_toml) as f:
        content = f.read()
    content = content.replace('.train_nn_temp', results_suffix)
    with open(temp_toml, "w") as f:
        f.write(content)

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

    # Parse final file
    output_dir = toml_data.get("data", {}).get("output_dir", "old_codebase/sorties")
    final_file = cwd / output_dir / f"final{results_suffix}"
    if not final_file.exists():
        print(f"  ERROR: {final_file} not found")
        if result.stderr:
            print(f"  stderr: {result.stderr.decode()[:500]}")
        return None

    from aerocapture.io._fortran import parse_fortran_line

    rows = []
    with open(final_file) as f:
        for line in f:
            values = parse_fortran_line(line)
            if values:
                rows.append(values)

    if not rows:
        return None

    final = np.array(rows)
    energy = final[:, 8]
    ecc = final[:, 10]
    captured = (ecc < 1.0) & (energy < 0)

    metrics: dict = {
        "n_sims": len(final),
        "captured": int(captured.sum()),
        "capture_rate": float(captured.sum()) / len(final) * 100,
        "cost": compute_cost(final),
    }

    if captured.any():
        metrics["apo_err_mean"] = float(np.abs(final[captured, 31]).mean())
        metrics["apo_err_std"] = float(np.abs(final[captured, 31]).std())
        metrics["peri_err_mean"] = float(np.abs(final[captured, 30]).mean())
        metrics["peri_err_std"] = float(np.abs(final[captured, 30]).std())
        dv = final[captured, 42]
        dv_clean = np.where(dv > 1e10, np.nan, dv)
        metrics["dv_mean"] = float(np.nanmean(dv_clean))
        metrics["dv_std"] = float(np.nanstd(dv_clean))
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
    parser.add_argument("--params-dir", type=str, default="save_net", help="Directory with optimized params")
    parser.add_argument("--executable", type=str, default="src/rust/target/release/aerocapture")
    parser.add_argument("--cwd", type=str, default=".")
    args = parser.parse_args()

    base_toml = Path(args.base_toml)
    cwd = Path(args.cwd)
    params_dir = Path(args.params_dir)

    if not base_toml.exists():
        print(f"ERROR: Base TOML not found: {base_toml}")
        sys.exit(1)

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
