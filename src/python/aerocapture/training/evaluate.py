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
        # 6-line header matching Fortran lecgnn.f (skips 6 reads before weights)
        f.write(" \n")
        f.write("   Caracteristiques neural network\n")
        f.write(" \n")
        f.write(f"           {n_input}   ninput\n")
        f.write(f"           {n_hidden}  {n_hidden}  {n_hidden}   nhid\n")
        f.write(f"           {n_output}   noutput\n")
        for w in weights:
            f.write(f"       {w: .30f}\n")


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

    init_file = (cwd / config.sim.init_file).resolve()
    executable = (cwd / config.sim.executable).resolve()

    try:
        with open(init_file) as f:
            result = subprocess.run(
                [str(executable)],
                stdin=f,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(cwd.resolve()),
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
    energy = final_conditions[:, 8]      # MJ/kg
    ecc = final_conditions[:, 10]        # dimensionless
    sim_time = final_conditions[:, 28]   # s
    peri_err = final_conditions[:, 30]   # km
    apo_err = final_conditions[:, 31]    # km
    dv_total = final_conditions[:, 42]   # m/s

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
    return np.sum(bits * bit_weights, axis=1) / (2**n_bit - 1) * p_range + config.ga.p_min


def evaluate_chromosome(
    xbit: npt.NDArray[np.int8],
    base_network: npt.NDArray[np.float64],
    config: TrainingConfig,
    cwd: str | Path | None = None,
) -> tuple[float, npt.NDArray[np.float64] | None]:
    """Full evaluation pipeline: decode, simulate, score.

    Args:
        xbit: Binary chromosome.
        base_network: Base network weights (ignored in direct encoding mode).
        config: Training configuration.
        cwd: Working directory.

    Returns:
        (cost, final_conditions) tuple.
    """
    if config.ga.direct_encoding:
        weights = decode_direct(xbit, config)
    else:
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
