"""Aerocapture corridor boundary computation.

Computes three boundary trajectories for a given mission config:
- Nominal: run with the configured guidance algorithm, no dispersions
- Undershoot: constant bank angle at the maximum lift-up angle that still captures
- Overshoot: constant bank angle at the maximum lift-down angle that still captures

The boundaries are found by bisecting on bank angle at the nominal entry FPA.

Usage (standalone):
    uv run python -m aerocapture.training.corridor \\
        --toml configs/missions/mars.toml \\
        --output corridor_boundaries.npz

Any mission TOML works — the guidance scheme is irrelevant since all corridor
trajectories use constant bank angle (reference_trajectory mode).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt


def _get_aero() -> object:
    """Import and return the aerocapture_rs module."""
    try:
        import aerocapture_rs as aero  # type: ignore[import-not-found, import-untyped]
    except ImportError as e:
        msg = "PyO3 aerocapture_rs module required for corridor computation"
        raise ImportError(msg) from e
    return aero


_COL_DV_APO_PERI = 40  # |dv1| + |dv2| (apoapsis + periapsis corrections, excludes inclination)


def _run_constant_bank(toml_path: str, bank_angle_deg: float) -> tuple[bool, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Run a single sim with constant bank angle (reference_trajectory mode).

    Returns (captured, trajectory (T, 12), final_record (52,)).
    """
    aero = _get_aero()

    overrides = {
        "guidance.type": "ftc",  # dummy — overridden by reference_trajectory=true
        "guidance.reference_trajectory": True,
        "guidance.reference_bank_angle": float(bank_angle_deg),
        "simulation.n_sims": 1,
    }
    result = aero.run(toml_path, overrides=overrides)  # type: ignore[attr-defined]
    traj: npt.NDArray[np.float64] = result.trajectory
    fr: npt.NDArray[np.float64] = result.final_record
    return bool(result.captured), traj, fr


def _bisect_bank_angle(
    toml_path: str,
    lo: float,
    hi: float,
    captured_side: str,
    tol: float = 0.1,
    max_iter: int = 30,
) -> tuple[float, npt.NDArray[np.float64]]:
    """Bisect on bank angle to find the capture/escape boundary.

    Args:
        toml_path: Path to TOML config.
        lo: Lower bank angle bound (deg).
        hi: Upper bank angle bound (deg).
        captured_side: "lo" if lo is the captured side, "hi" if hi is.
        tol: Bank angle tolerance (deg) for bisection convergence.
        max_iter: Maximum bisection iterations.

    Returns:
        (boundary_bank_angle_deg, boundary_trajectory)
    """
    last_captured_traj: npt.NDArray[np.float64] = np.array([])
    last_captured_bank = lo if captured_side == "lo" else hi

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        captured, traj, _ = _run_constant_bank(toml_path, mid)

        if captured:
            last_captured_traj = traj
            last_captured_bank = mid

        if abs(hi - lo) < tol:
            break

        if captured_side == "lo":
            if captured:
                lo = mid
            else:
                hi = mid
        else:
            if captured:
                hi = mid
            else:
                lo = mid

    return last_captured_bank, last_captured_traj


def _find_optimal_bank(
    toml_path: str,
    lo_bank: float,
    hi_bank: float,
    tol: float = 0.1,
) -> tuple[float, npt.NDArray[np.float64]]:
    """Find the constant bank angle that minimizes |dv1|+|dv2| (apoapsis+periapsis correction).

    Uses scipy.optimize.minimize_scalar with bounded method within the captured corridor.
    """
    from scipy.optimize import minimize_scalar

    best_traj: npt.NDArray[np.float64] = np.array([])
    best_bank = (lo_bank + hi_bank) / 2.0

    def objective(bank_deg: float) -> float:
        nonlocal best_traj, best_bank
        captured, traj, fr = _run_constant_bank(toml_path, bank_deg)
        if not captured:
            return 1e30
        dv = float(fr[_COL_DV_APO_PERI])
        best_traj = traj
        best_bank = bank_deg
        return dv

    result = minimize_scalar(objective, bounds=(lo_bank, hi_bank), method="bounded", options={"xatol": tol, "maxiter": 30})
    # Run one final time at the optimum to ensure we have its trajectory
    captured, traj, fr = _run_constant_bank(toml_path, float(result.x))
    if captured:
        return float(result.x), traj
    return best_bank, best_traj


def compute_corridor(
    toml_path: str,
    bank_tol: float = 0.1,
) -> dict[str, npt.NDArray[np.float64]]:
    """Compute corridor boundaries and optimal nominal trajectory.

    All three trajectories use constant bank angle:
    - **Undershoot boundary**: bank angle closest to 0° (full lift-up) that still captures.
    - **Overshoot boundary**: bank angle closest to 180° (full lift-down) that still captures.
    - **Nominal**: bank angle within the corridor that minimizes |dv1|+|dv2|
      (apoapsis + periapsis correction cost, excluding inclination).

    Args:
        toml_path: Path to TOML config with [guidance] section.
        bank_tol: Bank angle tolerance in degrees for bisection/optimization.

    Returns:
        Dict with keys: "nominal", "undershoot", "overshoot" → (T, 12) trajectory arrays,
        plus "nominal_bank_deg", "undershoot_bank_deg", "overshoot_bank_deg" scalar arrays.
    """
    toml_str = str(Path(toml_path).resolve())

    # 1. Find undershoot boundary: bisect bank angle in [0, 90]
    print("  Bisecting undershoot boundary (bank 0°→90°)...")
    cap_0, _, _ = _run_constant_bank(toml_str, 0.0)
    cap_90, _, _ = _run_constant_bank(toml_str, 90.0)

    if cap_0 and not cap_90:
        udr_bank, udr_traj = _bisect_bank_angle(toml_str, 0.0, 90.0, captured_side="lo", tol=bank_tol)
    elif cap_0 and cap_90:
        print("  Both 0° and 90° capture; using 90° as undershoot boundary")
        udr_bank, udr_traj = 90.0, _run_constant_bank(toml_str, 90.0)[1]
    else:
        print("  WARNING: Bank=0° does not capture; using 0° trajectory")
        udr_bank, udr_traj = 0.0, _run_constant_bank(toml_str, 0.0)[1]

    print(f"    Undershoot boundary: bank={udr_bank:.2f}°")

    # 2. Find overshoot boundary: bisect bank angle in [90, 180]
    print("  Bisecting overshoot boundary (bank 90°→180°)...")
    cap_180, _, _ = _run_constant_bank(toml_str, 180.0)

    if cap_90 and not cap_180:
        ovr_bank, ovr_traj = _bisect_bank_angle(toml_str, 90.0, 180.0, captured_side="lo", tol=bank_tol)
    elif not cap_90 and not cap_180:
        print("  90° and 180° both escape; bisecting from 45°→180°...")
        cap_45, _, _ = _run_constant_bank(toml_str, 45.0)
        if cap_45:
            ovr_bank, ovr_traj = _bisect_bank_angle(toml_str, 45.0, 180.0, captured_side="lo", tol=bank_tol)
        else:
            print("  WARNING: Cannot find overshoot boundary")
            ovr_bank, ovr_traj = 180.0, _run_constant_bank(toml_str, 180.0)[1]
    else:
        print("  Both 90° and 180° capture; using 180° as overshoot boundary")
        ovr_bank, ovr_traj = 180.0, _run_constant_bank(toml_str, 180.0)[1]

    print(f"    Overshoot boundary: bank={ovr_bank:.2f}°")

    # 3. Find optimal nominal: minimize |dv1|+|dv2| within the captured corridor
    # Search between the two boundaries (with a small inward margin to stay captured)
    margin = bank_tol * 2
    nom_lo = min(udr_bank, ovr_bank) + margin
    nom_hi = max(udr_bank, ovr_bank) - margin
    print(f"  Optimizing nominal bank angle for min DV (apo+peri) in [{nom_lo:.1f}°, {nom_hi:.1f}°]...")
    nom_bank, nom_traj = _find_optimal_bank(toml_str, nom_lo, nom_hi, tol=bank_tol)
    _, _, nom_fr = _run_constant_bank(toml_str, nom_bank)
    nom_dv = float(nom_fr[_COL_DV_APO_PERI])
    print(f"    Nominal: bank={nom_bank:.2f}°, DV(apo+peri)={nom_dv:.1f} m/s")

    return {
        "nominal": nom_traj,
        "undershoot": udr_traj,
        "overshoot": ovr_traj,
        "nominal_bank_deg": np.array([nom_bank]),
        "undershoot_bank_deg": np.array([udr_bank]),
        "overshoot_bank_deg": np.array([ovr_bank]),
    }


def save_corridor(data: dict[str, npt.NDArray[np.float64]], output_path: Path) -> None:
    """Save corridor boundaries to a compressed .npz file."""
    np.savez_compressed(str(output_path), **data)  # type: ignore[arg-type]
    print(f"  Corridor boundaries saved to {output_path}")


def load_corridor(path: Path) -> dict[str, npt.NDArray[np.float64]] | None:
    """Load corridor boundaries from a .npz file. Returns None if not found."""
    if not path.exists():
        return None
    data = np.load(str(path))
    return {k: data[k] for k in data.files}


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Compute aerocapture corridor boundaries")
    parser.add_argument("--toml", type=str, required=True, help="Mission TOML config (e.g., configs/missions/mars.toml)")
    parser.add_argument("--output", type=str, default="corridor_boundaries.npz", help="Output .npz file path")
    parser.add_argument("--tol", type=float, default=0.1, help="Bank angle bisection tolerance in degrees (default: 0.1)")
    args = parser.parse_args()

    print(f"Computing corridor boundaries for {args.toml}...")
    corridor = compute_corridor(args.toml, bank_tol=args.tol)
    save_corridor(corridor, Path(args.output))

    # Print summary
    for key in ["nominal", "undershoot", "overshoot"]:
        traj = corridor[key]
        if traj.size > 0:
            print(f"  {key}: {traj.shape[0]} timesteps, energy range [{traj[:, 8].min():.2f}, {traj[:, 8].max():.2f}] MJ/kg")
        else:
            print(f"  {key}: empty trajectory")


if __name__ == "__main__":
    main()
