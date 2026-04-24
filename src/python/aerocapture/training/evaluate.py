"""Cost function evaluation: write NN/guidance params, run simulator, compute cost.

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

try:
    import aerocapture_rs as _aero_rs  # type: ignore[import-not-found, import-untyped]

    _HAS_PYO3 = True
except ImportError:
    _aero_rs = None  # type: ignore[assignment]
    _HAS_PYO3 = False

# Reserved seed offsets -- guarantees training, validation, and final eval
# never share the same RNG stream.
VALIDATION_SEED_OFFSET = 1_000_000
FINAL_EVAL_SEED_OFFSET = 2_000_000


def make_reserved_seeds(base_mc_seed: int, offset: int, n: int) -> list[int]:
    """Generate a deterministic, reproducible list of MC seeds from a reserved RNG stream.

    Given the same (base_mc_seed, offset, n), always returns the same seeds.
    Different offsets produce independent streams.
    """
    seeds: list[int] = np.random.default_rng(base_mc_seed + offset).integers(0, 2**31, size=n).tolist()
    return seeds


def write_nn_json(
    weights: npt.NDArray[np.float64],
    network: NetworkConfig,
    filepath: str | Path,
    input_mask: list[int] | None = None,
) -> None:
    """Write PSO chromosome weights as v2 NN JSON via the Rust LayerWeights trait.

    Routes through `aerocapture_rs.flat_weights_to_json` so the Rust side is the
    single source of truth for weight serialization (closes Phase 0 review
    carry-over #2). Legacy dense-only `NetworkConfig` is translated into a v2
    architecture list before the call.
    """
    if not _HAS_PYO3 or _aero_rs is None:
        raise RuntimeError(
            "write_nn_json now requires the aerocapture_rs PyO3 module. "
            "Build it with `maturin develop --release --manifest-path src/rust/aerocapture-py/Cargo.toml`."
        )

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if network.architecture is not None:
        arch: list[dict[str, object]] = [dict(entry) for entry in network.architecture]
    else:
        arch = []
        for i in range(len(network.layer_sizes) - 1):
            arch.append(
                {
                    "type": "dense",
                    "input_size": network.layer_sizes[i],
                    "output_size": network.layer_sizes[i + 1],
                    "activation": network.activations[i],
                }
            )
    _aero_rs.flat_weights_to_json(
        flat=weights.astype(np.float64).tolist(),
        architecture_json=json.dumps(arch),
        path=str(filepath),
        input_mask=input_mask,
    )


def _parse_final_to_legacy_array(filepath: Path) -> npt.NDArray[np.float64] | None:
    """Parse a final conditions CSV file, returning 0-based 52-column array.

    Maps named CSV columns to their xsauve indices (0-based, no sim_number prefix).
    """
    import pandas as pd

    from aerocapture.io.parse_final import CSV_TO_LEGACY_INDEX

    df = pd.read_csv(filepath)
    if df.empty:
        return None
    n = len(df)
    result = np.zeros((n, 52))
    for col_name, legacy_idx in CSV_TO_LEGACY_INDEX.items():
        if col_name in df.columns:
            result[:, legacy_idx] = df[col_name].to_numpy()
    return result


def run_simulation(
    config: TrainingConfig,
    cwd: str | Path | None = None,
    overrides: dict[str, object] | None = None,
) -> npt.NDArray[np.float64] | None:
    """Run the Rust simulator and parse final conditions.

    Dispatches to PyO3 direct call when available, falling back to subprocess.

    Args:
        config: Training configuration.
        cwd: Working directory (defaults to config.sim.exec_dir).
        overrides: Optional TOML override dict (PyO3 path only).

    Returns:
        Array of final conditions (n, 52), or None if simulation failed.
    """
    if _HAS_PYO3 and config.sim.toml_config:
        return _run_via_pyo3(config, cwd, overrides)
    return _run_via_subprocess(config, cwd)


def _run_via_pyo3(
    config: TrainingConfig,
    cwd: str | Path | None = None,
    overrides: dict[str, object] | None = None,
) -> npt.NDArray[np.float64] | None:
    """Run simulation via PyO3 direct call (in-process, no subprocess)."""
    assert _aero_rs is not None
    if cwd is None:
        cwd = config.sim.exec_dir
    cwd = Path(cwd)
    if not config.sim.toml_config:
        return None
    toml_path = str((cwd / config.sim.toml_config).resolve())
    try:
        result = _aero_rs.run(toml_path=toml_path, overrides=overrides, sim_timeout_secs=config.sim.sim_timeout_secs)
        arr: npt.NDArray[np.float64] = result.final_record.reshape(1, 52)
        return arr
    except Exception:
        import traceback

        traceback.print_exc()
        return None


def _run_via_subprocess(config: TrainingConfig, cwd: str | Path | None = None) -> npt.NDArray[np.float64] | None:
    """Run simulation via subprocess (legacy path)."""
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
    except (subprocess.TimeoutExpired, FileNotFoundError):  # fmt: skip
        return None

    # Parse final conditions -- auto-detect CSV vs legacy text
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


def log_cap(dv: npt.NDArray[np.float64], threshold: float = 1000.0) -> npt.NDArray[np.float64]:
    """C1-continuous log-capped cost: linear below threshold, log above.

    DEPRECATED: kept for backward compatibility. Use dv_cost() instead.
    log_cap compresses the non-capture range (10000-20000) into 3302-3996,
    creating a near-flat plateau that starves the optimizer of gradient.
    """
    dv = np.maximum(dv, 1e-6)  # safety floor
    below = dv <= threshold
    result = np.empty_like(dv)
    result[below] = dv[below]
    result[~below] = threshold * (1.0 + np.log(dv[~below] / threshold))
    return result


# Scale for the quadratic growth above threshold. Controls how fast the
# cost grows on the non-capture side.
_DV_PENALTY_SCALE = 10000.0

# Sharpness of the softplus knee at the DV threshold. Larger = sharper wall.
# k=0.01 gives ~200 m/s transition width (captures < 500 m/s are untouched).
_DV_KNEE_SHARPNESS = 0.01

# Sharpness of the softplus knee for constraint penalties. Operates on
# normalized fractions (val-limit)/limit, so k=100 means ~1% transition.
_CONSTRAINT_KNEE_SHARPNESS = 100.0


def _softplus(x: npt.NDArray[np.float64], k: float) -> npt.NDArray[np.float64]:
    """Numerically stable softplus: ln(1 + exp(k*x)) / k."""
    kx = k * x
    return np.where(kx > 20.0, x, np.log1p(np.exp(kx)) / k)


def dv_cost(dv: npt.NDArray[np.float64], threshold: float = 1000.0) -> npt.NDArray[np.float64]:
    """C-infinity softplus-quadratic DV cost function.

    Uses softplus to smoothly transition from linear (captures) to
    quadratic penalty (non-captures). The softplus replaces the hard
    max(0, dv-T) knee with a C-infinity smooth version, while the
    quadratic term provides strong, always-increasing gradient on the
    non-capture side.

    cost(dv) = dv + sp(dv-T) + sp(dv-T)^2 / (2*S)

    where sp(x) = ln(1 + exp(k*x)) / k  (softplus with sharpness k).

    Properties:
        - C-infinity everywhere (no kinks or discontinuities)
        - Captures nearly untouched: dv=200 -> cost=200.0, dv=500 -> cost=500.7
        - Wall at threshold: slope rises from 1.0 to 1.5 across ~200 m/s
        - Strong far gradient: slope=2.9 at dv=10000, slope=3.9 at dv=20000
        - Wide non-capture spread: dv=10000 -> 23050, dv=20000 -> 57050
    """
    dv = np.maximum(dv, 1e-6)  # safety floor
    s = _DV_PENALTY_SCALE
    x = _softplus(dv - threshold, _DV_KNEE_SHARPNESS)
    return dv + x + x**2 / (2.0 * s)


def compute_cost(
    final_conditions: npt.NDArray[np.float64],
    *,
    dv_threshold: float = 1000.0,
    g_load_limit: float = 15.0,  # fallback; overridden by [flight.constraints] via cost_kwargs
    heat_flux_limit: float = 200.0,  # fallback; overridden by [flight.constraints] via cost_kwargs
    heat_load_limit: float = 25000.0,  # fallback; overridden by [flight.constraints] via cost_kwargs
    g_load_weight: float = 10000.0,
    heat_flux_weight: float = 10000.0,
    heat_load_weight: float = 10000.0,
    cost_transform: str = "linear",
) -> float:
    """Compute RMS cost from simulation final conditions.

    Uses quadratic-penalty DV cost as the primary objective with normalized
    soft constraint penalties for g-load, heat flux, and heat load exceedances.

    All termination outcomes produce meaningful DV values from Rust:
    - Captured: real orbital correction DV
    - Hyperbolic: 10000 + excess velocity
    - Crash/PendingCrash/Timeout: 20000 * proportional time decay

    Returns:
        RMS cost value. Lower is better.
    """
    dv_total = final_conditions[:, 41]
    g_max = final_conditions[:, 17]
    q_max = final_conditions[:, 16]

    costs = dv_cost(dv_total, threshold=dv_threshold)

    g_penalty = g_load_weight * _softplus((g_max - g_load_limit) / g_load_limit, _CONSTRAINT_KNEE_SHARPNESS)
    q_penalty = heat_flux_weight * _softplus((q_max - heat_flux_limit) / heat_flux_limit, _CONSTRAINT_KNEE_SHARPNESS)
    heat_load = final_conditions[:, 28] * 1e3  # MJ/m2 -> kJ/m2
    hl_penalty = heat_load_weight * _softplus((heat_load - heat_load_limit) / heat_load_limit, _CONSTRAINT_KNEE_SHARPNESS)
    costs = costs + g_penalty + q_penalty + hl_penalty

    if cost_transform == "sqrt":
        costs = np.sqrt(costs)
    elif cost_transform == "squared":
        costs = costs**2
    elif cost_transform == "cubed":
        costs = costs**3
    elif cost_transform != "linear":
        raise ValueError(f"unknown cost_transform={cost_transform!r} (expected 'linear', 'sqrt', 'squared', or 'cubed')")

    return float(np.sqrt(np.mean(costs**2)))


def write_guidance_toml(
    base_toml_path: str | Path,
    guidance_type: str,
    params: dict[str, float],
    output_path: str | Path | None = None,
    mc_seed: int | None = None,
    n_sims_override: int | None = None,
) -> Path:
    """Patch a base TOML config with optimized guidance parameters.

    Reads the base TOML, adds/overwrites the [guidance.<section>] with
    the provided parameter values, and writes to output_path (or a temp file).

    Returns:
        Path to the written TOML file.
    """
    from aerocapture.training.param_spaces import GUIDANCE_TOML_SECTIONS
    from aerocapture.training.toml_utils import load_toml_with_bases

    base_toml_path = Path(base_toml_path)
    toml_data = load_toml_with_bases(base_toml_path)

    # Set the guidance type
    toml_data.setdefault("guidance", {})["type"] = guidance_type

    # Split lateral, exit, nav, and thermal params from scheme-specific params
    lateral_params = {k.removeprefix("lateral."): v for k, v in params.items() if k.startswith("lateral.")}
    exit_params = {k.removeprefix("exit."): v for k, v in params.items() if k.startswith("exit.")}
    nav_params = {k.removeprefix("nav."): v for k, v in params.items() if k.startswith("nav.")}
    thermal_params = {k.removeprefix("thermal."): v for k, v in params.items() if k.startswith("thermal.")}
    shaping_params = {k.removeprefix("shaping."): v for k, v in params.items() if k.startswith("shaping.")}
    scheme_params = {
        k: v
        for k, v in params.items()
        if not k.startswith("lateral.")
        and not k.startswith("exit.")
        and not k.startswith("nav.")
        and not k.startswith("thermal.")
        and not k.startswith("shaping.")
    }

    # Round max_reversals to integer
    if "max_reversals" in lateral_params:
        lateral_params["max_reversals"] = int(round(lateral_params["max_reversals"]))

    # Merge scheme params into [guidance.<scheme>]
    section_name = GUIDANCE_TOML_SECTIONS[guidance_type]
    toml_data["guidance"].setdefault(section_name, {}).update(scheme_params)

    # Merge lateral params into [guidance.lateral]
    if lateral_params:
        toml_data["guidance"].setdefault("lateral", {}).update(lateral_params)

    # Merge exit params into [guidance.ftc] (exit guidance is loaded from ftc section for all schemes)
    if exit_params:
        ftc_section = toml_data["guidance"].setdefault("ftc", {})
        ftc_section.update(exit_params)

    # Merge nav params into [navigation] (density filter config used by all schemes)
    if nav_params:
        toml_data.setdefault("navigation", {}).update(nav_params)

    # Merge thermal limiter params into [guidance.thermal_limiter]
    if thermal_params:
        toml_data["guidance"].setdefault("thermal_limiter", {}).update(thermal_params)

    # Merge command shaping params into [guidance.command_shaping]
    if shaping_params:
        toml_data["guidance"].setdefault("command_shaping", {}).update(shaping_params)
        toml_data["guidance"]["command_shaping"].setdefault("enabled", True)

    if mc_seed is not None:
        toml_data.setdefault("monte_carlo", {})["seed"] = mc_seed

    if n_sims_override is not None:
        toml_data.setdefault("simulation", {})["n_sims"] = n_sims_override

    # Write TOML (minimal writer -- machine-consumed only)
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
            # Inline table array -- should be handled as [[section]]
            items = []
            for item in value:
                fields = ", ".join(f"{k} = {_toml_value(v)}" for k, v in item.items())
                items.append(f"{{ {fields} }}")
            return f"[{', '.join(items)}]"
        return f"[{', '.join(_toml_value(v) for v in value)}]"
    return str(value)


def patch_toml_mc_seed(base_toml_path: str | Path, mc_seed: int, n_sims_override: int | None = None) -> Path:
    """Create a temp TOML with [monte_carlo].seed overridden.

    Args:
        base_toml_path: Path to the base TOML config.
        mc_seed: The Monte Carlo seed to set.

    Returns:
        Path to the temp TOML file (caller must clean up).
    """
    import os

    from aerocapture.training.toml_utils import load_toml_with_bases

    base_toml_path = Path(base_toml_path)
    toml_data = load_toml_with_bases(base_toml_path)

    toml_data.setdefault("monte_carlo", {})["seed"] = mc_seed

    if n_sims_override is not None:
        toml_data.setdefault("simulation", {})["n_sims"] = n_sims_override

    fd, path_str = tempfile.mkstemp(suffix=".toml", prefix="mc_seed_")
    output_path = Path(path_str)
    os.close(fd)
    _write_toml(toml_data, output_path)
    return output_path
