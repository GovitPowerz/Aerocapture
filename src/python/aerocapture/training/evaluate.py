"""Cost function evaluation: write NN weights, run simulator, compute cost.

Replaces MATLAB ComputeCost_Aerocap.m.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import numpy.typing as npt

from aerocapture.training.config import TrainingConfig


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
    return np.sum(bits * conv_bd, axis=1) + p_min


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


def write_nn_params(
    weights: npt.NDArray[np.float64],
    filepath: str | Path,
    n_input: int,
    n_hidden: int,
    n_output: int,
) -> None:
    """Write neural network parameters to Fortran-readable file.

    Args:
        weights: Network weight vector.
        filepath: Output file path.
        n_input: Number of inputs.
        n_hidden: Number of hidden neurons.
        n_output: Number of outputs.
    """
    filepath = Path(filepath)
    with open(filepath, "w") as f:
        f.write(f"{n_input}\n")
        f.write(f"{n_hidden}\n")
        f.write(f"{n_output}\n")
        for w in weights:
            f.write(f"{w:.30f}\n")


def run_simulation(config: TrainingConfig, cwd: str | Path | None = None) -> npt.NDArray[np.float64] | None:
    """Run the Fortran simulator and parse final conditions.

    Args:
        config: Training configuration.
        cwd: Working directory (defaults to config.sim.exec_dir).

    Returns:
        Array of final conditions, or None if simulation failed.
    """
    if cwd is None:
        cwd = config.sim.exec_dir
    cwd = Path(cwd)

    init_file = cwd / config.sim.init_file
    executable = cwd / config.sim.executable

    try:
        with open(init_file) as f:
            result = subprocess.run(
                [str(executable)],
                stdin=f,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(cwd),
                timeout=300,
            )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None

    # Parse final conditions
    final_file = cwd / config.sim.final_file
    if not final_file.exists():
        return None

    try:
        from aerocapture.io._fortran import parse_fortran_line

        rows = []
        with open(final_file) as f:
            for line in f:
                values = parse_fortran_line(line)
                if values:
                    rows.append(values)
        return np.array(rows) if rows else None
    except Exception:
        return None


def compute_cost(
    final_conditions: npt.NDArray[np.float64],
    duration_col: int = 29,
    periapsis_col: int = 39,
    apoapsis_col: int = 40,
    inclination_col: int = 41,
) -> float:
    """Compute RMS cost from simulation final conditions.

    Implements the hierarchical penalty cost function from MATLAB.

    Args:
        final_conditions: Array of final conditions (n_sims, n_cols).
        duration_col: Column index for duration error.
        periapsis_col: Column index for periapsis error.
        apoapsis_col: Column index for apoapsis error.
        inclination_col: Column index for inclination error.

    Returns:
        RMS cost value. Lower is better.
    """
    # Extract absolute errors (1-indexed in MATLAB, 0-indexed here)
    # Adjust for 0-indexed columns
    dur = np.abs(final_conditions[:, duration_col - 1])
    peri = np.abs(final_conditions[:, periapsis_col - 1])
    apo = np.abs(final_conditions[:, apoapsis_col - 1])
    incl = np.abs(final_conditions[:, inclination_col - 1])

    # Constraint hierarchy
    crash = (peri > 1e20) | (apo > 1e20) | (incl > 1e20)
    apo_viol = apo > 40
    peri_viol = (peri - 113) > 40
    incl_viol = incl > 40

    err = np.zeros_like(dur)

    # Level 1: Crash
    mask = crash
    err[mask] = 1e30 / np.maximum(dur[mask], 1e-10)

    # Level 2: Apoapsis violation (not crash)
    mask = ~crash & apo_viol
    err[mask] = 1e18 * apo[mask] + 1e12 * dur[mask] + 1e6 * incl[mask]

    # Level 3: Periapsis violation (not crash, not apo violation)
    mask = ~crash & ~apo_viol & peri_viol
    err[mask] = apo[mask] + 1e12 * dur[mask] + 1e6 * incl[mask]

    # Level 4: Inclination violation (no other violations)
    mask = ~crash & ~apo_viol & ~peri_viol & incl_viol
    err[mask] = apo[mask] + dur[mask] + 1e6 * incl[mask]

    # Level 5: All nominal
    mask = ~crash & ~apo_viol & ~peri_viol & ~incl_viol
    err[mask] = apo[mask] + dur[mask] + incl[mask]

    return float(np.sqrt(np.mean(err**2)))


def evaluate_chromosome(
    xbit: npt.NDArray[np.int8],
    base_network: npt.NDArray[np.float64],
    config: TrainingConfig,
    cwd: str | Path | None = None,
) -> tuple[float, npt.NDArray[np.float64] | None]:
    """Full evaluation pipeline: decode, perturb, simulate, score.

    Args:
        xbit: Binary chromosome.
        base_network: Base network weights.
        config: Training configuration.
        cwd: Working directory.

    Returns:
        (cost, final_conditions) tuple.
    """
    # Perturb network
    weights = perturb_network(xbit, base_network, config)

    # Write to file
    if cwd is None:
        cwd = config.sim.exec_dir
    nn_path = Path(cwd) / config.sim.nn_param_file
    write_nn_params(
        weights, nn_path,
        config.network.n_input,
        config.network.n_hidden,
        config.network.n_output,
    )

    # Run simulation
    final = run_simulation(config, cwd=cwd)
    if final is None:
        return 1e30, None

    # Compute cost
    cost = compute_cost(final)
    return cost, final
