"""Cost function evaluation: write NN/guidance params, run simulator, compute cost.

Replaces MATLAB ComputeCost_Aerocap.m.
Supports both NN weight optimization and generic guidance parameter optimization.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from io import TextIOWrapper
from pathlib import Path

import numpy as np
import numpy.typing as npt

from aerocapture.training.config import NetworkConfig, TrainingConfig


def binary_to_decimal(
    xbit: npt.NDArray[np.int8],
    conv_bd: npt.NDArray[np.float64],
    p_min: float,
) -> npt.NDArray[np.float64]:
    """Convert binary chromosome to decimal parameter values.

    Args:
        xbit: Binary chromosome array of shape (n_coef * n_bit,).
        conv_bd: Conversion matrix of shape (n_coef, n_bit).
        p_min: Minimum parameter value.

    Returns:
        Array of shape (n_coef,) with decimal values in [p_min, p_max].
    """
    n_coef, n_bit = conv_bd.shape
    bits = xbit[: n_coef * n_bit].reshape(n_coef, n_bit)
    result: npt.NDArray[np.float64] = np.sum(bits * conv_bd, axis=1) + p_min
    return result


def perturb_network(
    xbit: npt.NDArray[np.int8],
    base_network: npt.NDArray[np.float64],
    config: TrainingConfig,
) -> npt.NDArray[np.float64]:
    """Apply binary-encoded perturbation to base network weights.

    Args:
        xbit: Full binary chromosome (coefficients + sign bits).
        base_network: Base network weight vector.
        config: Training configuration.

    Returns:
        Perturbed network weight vector.
    """
    conv_bd = config.build_conversion_matrix()
    n_coef = config.network.n_coef
    n_bit = config.ga.n_bit

    # Decode decimal values from binary
    params = binary_to_decimal(xbit, conv_bd, config.ga.p_min)

    # Extract sign bits (second half of chromosome)
    sign_bits = xbit[n_bit * n_coef :]
    signs = np.where(sign_bits[:n_coef], 1.0, -1.0)

    # Perturbation: sign * base_weight * (1 + Var * normalized_mid * first_third)
    third = n_coef // 3
    first_third = params[:third]
    mid_section = (params[third : 2 * third] + 1) / 2  # normalize to [0, 1]

    perturbation = np.ones(n_coef)
    perturbation[:third] = 1 + config.ga.variation * mid_section * first_third

    return signs * base_network * perturbation


def write_nn_json(
    weights: npt.NDArray[np.float64],
    network: NetworkConfig,
    filepath: str | Path,
) -> None:
    """Write neural network weights in JSON format readable by Rust.

    Partitions the flat weight vector into layers according to network.layer_sizes.
    """
    filepath = Path(filepath)
    layer_weights: dict[str, dict] = {}
    idx = 0

    for i in range(len(network.layer_sizes) - 1):
        n_in = network.layer_sizes[i]
        n_out = network.layer_sizes[i + 1]

        w = []
        for _ in range(n_out):
            w.append(weights[idx : idx + n_in].tolist())
            idx += n_in
        b = weights[idx : idx + n_out].tolist()
        idx += n_out

        layer_weights[f"layer_{i}"] = {"w": w, "b": b}

    data = {
        "format_version": 1,
        "architecture": {
            "layers": network.layer_sizes,
            "activations": network.activations,
        },
        "weights": layer_weights,
        "output_interpretation": "atan2",
    }

    with open(filepath, "w") as f:
        json.dump(data, f)


def _parse_final_to_legacy_array(filepath: Path) -> npt.NDArray[np.float64] | None:
    """Parse a final conditions CSV file, returning legacy-compatible 53-column array.

    Maps named CSV columns back to the legacy 53-column positions so
    compute_cost() works unchanged.
    """
    import pandas as pd

    from aerocapture.io.parse_final import CSV_TO_LEGACY_INDEX

    df = pd.read_csv(filepath)
    if df.empty:
        return None
    n = len(df)
    result = np.zeros((n, 53))
    result[:, 0] = df["sim_number"].to_numpy()
    for col_name, legacy_idx in CSV_TO_LEGACY_INDEX.items():
        if col_name in df.columns:
            result[:, legacy_idx + 1] = df[col_name].to_numpy()
    return result


def run_simulation(config: TrainingConfig, cwd: str | Path | None = None) -> npt.NDArray[np.float64] | None:
    """Run the Rust simulator and parse final conditions.

    Args:
        config: Training configuration.
        cwd: Working directory (defaults to config.sim.exec_dir).

    Returns:
        Array of final conditions, or None if simulation failed.
    """
    if cwd is None:
        cwd = config.sim.exec_dir
    cwd = Path(cwd)

    executable = (cwd / config.sim.executable).resolve()

    if not config.sim.toml_config:
        return None

    toml_path = (cwd / config.sim.toml_config).resolve()
    try:
        subprocess.run(
            [str(executable), str(toml_path)],
            capture_output=True,
            cwd=str(cwd.resolve()),
            timeout=300,
        )
    except subprocess.TimeoutExpired, FileNotFoundError:
        return None

    # Parse final conditions — auto-detect CSV vs legacy text
    final_file = cwd / config.sim.final_file
    csv_final = Path(str(final_file) + ".csv")
    if csv_final.exists():
        final_file = csv_final
    elif not final_file.exists():
        return None

    try:
        return _parse_final_to_legacy_array(final_file)
    except Exception:
        return None


def compute_cost(final_conditions: npt.NDArray[np.float64]) -> float:
    """Compute RMS cost from simulation final conditions.

    Uses a smooth, continuous cost function that provides gradient signal
    even for non-capturing (hyperbolic) trajectories. This is critical for
    GA convergence: random NNs produce different bank angle profiles that
    dissipate different amounts of orbital energy, so energy-based cost
    differentiates between candidates that a binary crash/no-crash penalty cannot.

    Final file columns (0-indexed):
        8  = orbital energy (MJ/kg), >0 hyperbolic, <0 bound
        10 = eccentricity, >1 hyperbolic
        28 = total simulation time (s)
        30 = periapsis altitude error vs target (km)
        31 = apoapsis altitude error vs target (km)
        42 = total delta-V to reach target orbit (m/s)

    Cost hierarchy (smooth within each level):
        Level 0: Hyperbolic escape → 1e6 + 1e3 * |energy|
        Level 1: Captured, large orbit errors → 1e4 + |apo_err| + |peri_err|
        Level 2: Captured, small errors → |apo_err| + |peri_err| + dv_total

    Returns:
        RMS cost value. Lower is better.
    """
    energy = final_conditions[:, 8]  # MJ/kg
    ecc = final_conditions[:, 10]  # dimensionless
    sim_time = final_conditions[:, 28]  # s
    peri_err = final_conditions[:, 30]  # km
    apo_err = final_conditions[:, 31]  # km
    dv_total = final_conditions[:, 42]  # m/s

    hyperbolic = (ecc > 1.0) | (energy > 0)

    costs = np.zeros(len(final_conditions))

    # Level 0: Non-capturing (hyperbolic) — smooth energy-based cost
    # Energy varies between NNs: more atmospheric drag = lower energy = better
    # Also reward longer flight time (more atmospheric interaction)
    mask = hyperbolic
    costs[mask] = 1e6 + 1e3 * np.abs(energy[mask]) - 0.1 * sim_time[mask]

    # Captured trajectories
    mask = ~hyperbolic
    abs_apo = np.abs(apo_err[mask])
    abs_peri = np.abs(peri_err[mask])
    orbit_err = abs_apo + abs_peri

    # Sanitize dv_total: Fortran writes 1e30 when maneuver computation fails
    dv_clean = np.clip(dv_total[mask], 0, 1e4)
    dv_clean = np.where(dv_total[mask] > 1e10, 0.0, dv_clean)  # ignore bogus values

    # Smooth continuous cost: orbit error + small delta-V contribution
    costs[mask] = orbit_err + 0.01 * dv_clean

    return float(np.sqrt(np.mean(costs**2)))


def decode_direct(
    xbit: npt.NDArray[np.int8],
    config: TrainingConfig,
) -> npt.NDArray[np.float64]:
    """Decode chromosome directly to weight values (no base network needed).

    Maps n_base_coef groups of n_bit binary digits to values in [p_min, p_max].
    """
    n_base = config.network.n_base_coef
    n_bit = config.ga.n_bit
    p_range = config.ga.p_max - config.ga.p_min

    bit_weights = np.power(2.0, np.arange(n_bit - 1, -1, -1))
    bits = xbit[: n_base * n_bit].reshape(n_base, n_bit)
    result: npt.NDArray[np.float64] = np.sum(bits * bit_weights, axis=1) / (2**n_bit - 1) * p_range + config.ga.p_min
    return result


def decode_params_from_chromosome(
    xbit: npt.NDArray[np.int8],
    config: TrainingConfig,
) -> dict[str, float]:
    """Decode binary chromosome to named guidance parameter values.

    Each parameter has its own [p_min, p_max] bounds. Parameters with
    log_scale=True are decoded in log space then exponentiated.

    Returns:
        Dict mapping parameter name to decoded float value.
    """
    from aerocapture.training.param_spaces import PARAM_SPACES

    specs = PARAM_SPACES[config.guidance_type]
    n_bit = config.ga.n_bit
    bit_weights = np.power(2.0, np.arange(n_bit - 1, -1, -1))
    max_val = 2**n_bit - 1

    result = {}
    for i, spec in enumerate(specs):
        bits = xbit[i * n_bit : (i + 1) * n_bit]
        normalized = float(np.sum(bits * bit_weights)) / max_val  # [0, 1]

        if spec.log_scale:
            log_min = np.log10(spec.p_min)
            log_max = np.log10(spec.p_max)
            result[spec.name] = 10.0 ** (log_min + normalized * (log_max - log_min))
        else:
            result[spec.name] = spec.p_min + normalized * (spec.p_max - spec.p_min)

    return result


def write_guidance_toml(
    base_toml_path: str | Path,
    guidance_type: str,
    params: dict[str, float],
    output_path: str | Path | None = None,
) -> Path:
    """Patch a base TOML config with optimized guidance parameters.

    Reads the base TOML, adds/overwrites the [guidance.<section>] with
    the provided parameter values, and writes to output_path (or a temp file).

    Returns:
        Path to the written TOML file.
    """
    import tomllib

    from aerocapture.training.param_spaces import GUIDANCE_TOML_SECTIONS

    base_toml_path = Path(base_toml_path)
    with open(base_toml_path, "rb") as f:
        toml_data = tomllib.load(f)

    # Set the guidance type
    toml_data.setdefault("guidance", {})["type"] = guidance_type

    # Set the parameter section
    section_name = GUIDANCE_TOML_SECTIONS[guidance_type]
    toml_data["guidance"][section_name] = params

    # Write TOML (minimal writer — machine-consumed only)
    if output_path is None:
        fd, path_str = tempfile.mkstemp(suffix=".toml", prefix="guidance_")
        output_path = Path(path_str)
        import os

        os.close(fd)
    else:
        output_path = Path(output_path)

    _write_toml(toml_data, output_path)
    return output_path


def _write_toml(data: dict, path: Path) -> None:
    """Minimal TOML writer for nested dicts with scalar/list values."""
    with open(path, "w") as f:
        _write_toml_section(f, data, prefix="")


def _write_toml_section(f: TextIOWrapper, data: dict, prefix: str) -> None:
    """Recursively write TOML sections."""
    # First pass: write scalar/list values at this level
    for key, value in data.items():
        if isinstance(value, dict):
            # Check if it's an inline table (contains only scalars) or a section
            if _is_table_array(value):
                # Array of inline tables (like aerodynamics.points)
                continue
            if any(isinstance(v, dict) for v in value.values()):
                continue  # Will be written as subsection
            if not _has_non_dict_values(value):
                continue
        if not isinstance(value, dict):
            f.write(f"{key} = {_toml_value(value)}\n")

    # Second pass: write array-of-tables
    for key, value in data.items():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            full_key = f"{prefix}{key}" if prefix else key
            for item in value:
                f.write(f"\n[[{full_key}]]\n")
                _write_toml_section(f, item, prefix=f"{full_key}.")

    # Third pass: write subsections
    for key, value in data.items():
        if isinstance(value, dict):
            full_key = f"{prefix}{key}" if prefix else key
            # Write section header if this dict has scalar values
            scalars = {k: v for k, v in value.items() if not isinstance(v, dict) and not _is_table_array_entry(v)}
            if scalars:
                f.write(f"\n[{full_key}]\n")
                for sk, sv in scalars.items():
                    if isinstance(sv, list) and sv and isinstance(sv[0], dict):
                        continue  # handled separately
                    f.write(f"{sk} = {_toml_value(sv)}\n")
            # Write array-of-tables within this section
            for sk, sv in value.items():
                if isinstance(sv, list) and sv and isinstance(sv[0], dict):
                    aot_key = f"{full_key}.{sk}"
                    for item in sv:
                        f.write(f"\n[[{aot_key}]]\n")
                        for ik, iv in item.items():
                            f.write(f"{ik} = {_toml_value(iv)}\n")
            # Recurse into nested dicts
            for sk, sv in value.items():
                if isinstance(sv, dict):
                    _write_toml_section(f, {sk: sv}, prefix=f"{full_key}.")


def _is_table_array(value: object) -> bool:
    return isinstance(value, list) and bool(value) and isinstance(value[0], dict)


def _is_table_array_entry(value: object) -> bool:
    return isinstance(value, list) and bool(value) and isinstance(value[0], dict)


def _has_non_dict_values(d: dict) -> bool:
    return any(not isinstance(v, dict) for v in d.values())


def _toml_value(value: object) -> str:
    """Format a Python value as TOML."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    # Coerce numpy scalar floats (np.float64, np.float32, etc.) to plain Python float
    # before formatting; repr(np.float64(...)) produces invalid TOML like "np.float64(1e-07)".
    import numbers

    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return repr(float(value))
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, list):
        if value and isinstance(value[0], dict):
            # Inline table array — should be handled as [[section]]
            items = []
            for item in value:
                fields = ", ".join(f"{k} = {_toml_value(v)}" for k, v in item.items())
                items.append(f"{{ {fields} }}")
            return f"[{', '.join(items)}]"
        return f"[{', '.join(_toml_value(v) for v in value)}]"
    return str(value)


def patch_toml_mc_seed(base_toml_path: str | Path, mc_seed: int) -> Path:
    """Create a temp TOML with [monte_carlo].seed overridden.

    Args:
        base_toml_path: Path to the base TOML config.
        mc_seed: The Monte Carlo seed to set.

    Returns:
        Path to the temp TOML file (caller must clean up).
    """
    import os
    import tomllib

    base_toml_path = Path(base_toml_path)
    with open(base_toml_path, "rb") as f:
        toml_data = tomllib.load(f)

    toml_data.setdefault("monte_carlo", {})["seed"] = mc_seed

    fd, path_str = tempfile.mkstemp(suffix=".toml", prefix="mc_seed_")
    output_path = Path(path_str)
    os.close(fd)
    _write_toml(toml_data, output_path)
    return output_path


def evaluate_chromosome(
    xbit: npt.NDArray[np.int8],
    base_network: npt.NDArray[np.float64],
    config: TrainingConfig,
    cwd: str | Path | None = None,
) -> tuple[float, npt.NDArray[np.float64] | None]:
    """Full evaluation pipeline: decode, simulate, score.

    Dispatches between NN weight optimization and generic guidance
    parameter optimization based on config.guidance_type.

    Args:
        xbit: Binary chromosome.
        base_network: Base network weights (ignored for non-NN schemes).
        config: Training configuration.
        cwd: Working directory.

    Returns:
        (cost, final_conditions) tuple.
    """
    if cwd is None:
        cwd = config.sim.exec_dir

    if config.guidance_type == "neural_network":
        # NN path: decode weights, write JSON, run sim
        weights = decode_direct(xbit, config) if config.ga.direct_encoding else perturb_network(xbit, base_network, config)
        nn_path = Path(cwd) / config.sim.nn_param_file
        write_nn_json(weights, config.network, nn_path)
        final = run_simulation(config, cwd=cwd)
    else:
        # Generic guidance param path: decode params, patch TOML, run sim
        params = decode_params_from_chromosome(xbit, config)
        if config.sim.toml_config is None:
            msg = f"toml_config must be set for guidance_type={config.guidance_type}"
            raise ValueError(msg)
        base_toml = Path(cwd) / config.sim.toml_config
        patched_toml = write_guidance_toml(base_toml, config.guidance_type, params)
        try:
            # Temporarily override TOML config to use patched file
            orig_toml = config.sim.toml_config
            config.sim.toml_config = str(patched_toml)
            final = run_simulation(config, cwd=cwd)
            config.sim.toml_config = orig_toml
        finally:
            # Clean up temp file
            patched_toml.unlink(missing_ok=True)

    if final is None:
        return 1e30, None

    cost = compute_cost(final)
    return cost, final
