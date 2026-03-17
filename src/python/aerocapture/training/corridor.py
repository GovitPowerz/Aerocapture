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
_COL_PERI_ALT = 14  # periapsis altitude (km)
_DV_CRASH_SENTINEL = 1e10  # above this, orbit is unusable (crash or degenerate)


def _is_viable_capture(fr: npt.NDArray[np.float64]) -> bool:
    """Check if a final_record represents a viable captured orbit.

    Requires: bound orbit (ecc < 1, energy < 0), periapsis above the surface,
    and correction DV below the crash sentinel.
    """
    ecc = float(fr[9])
    energy = float(fr[7])
    peri_alt = float(fr[_COL_PERI_ALT])
    dv = float(fr[_COL_DV_APO_PERI])
    return ecc < 1.0 and energy < 0.0 and peri_alt > 0.0 and dv < _DV_CRASH_SENTINEL


def _run_constant_bank(toml_path: str, bank_angle_deg: float) -> tuple[bool, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Run a single sim with constant bank angle (reference_trajectory mode).

    Returns (viable_capture, trajectory (T, 12), final_record (52,)).
    Uses run_mc with include_trajectories=True to get per-timestep data.
    """
    aero = _get_aero()

    overrides = {
        "guidance.type": "ftc",  # dummy — overridden by reference_trajectory=true
        "guidance.reference_trajectory": True,
        "guidance.reference_bank_angle": float(bank_angle_deg),
        "simulation.n_sims": 1,
    }
    results = aero.run_mc(toml_path=toml_path, overrides=overrides, include_trajectories=True)  # type: ignore[attr-defined]
    fr: npt.NDArray[np.float64] = results.final_records[0]
    trajs = results.trajectories
    traj: npt.NDArray[np.float64] = trajs[0] if trajs and len(trajs) > 0 else np.array([])
    return _is_viable_capture(fr), traj, fr


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

    # 1. Scan to find the viable capture range
    #    A "viable capture" requires bound orbit + periapsis above surface + DV below sentinel.
    #    Scan from 0° to 180° in coarse steps to find the captured region.
    print("  Scanning bank angle range for viable captures...")
    scan_step = 10.0
    scan_angles = np.arange(0.0, 180.0 + scan_step, scan_step)
    viable_angles: list[float] = []
    for bank in scan_angles:
        cap, _, _ = _run_constant_bank(toml_str, float(bank))
        status = "viable" if cap else "crash/escape"
        print(f"    bank={bank:>5.0f}°: {status}")
        if cap:
            viable_angles.append(float(bank))

    if not viable_angles:
        print("  ERROR: No viable captures found at any bank angle!")
        return {"nominal": np.array([]), "undershoot": np.array([]), "overshoot": np.array([]),
                "nominal_bank_deg": np.array([0.0]), "undershoot_bank_deg": np.array([0.0]), "overshoot_bank_deg": np.array([0.0])}

    scan_lo = min(viable_angles)
    scan_hi = max(viable_angles)
    print(f"  Viable capture range: ~{scan_lo:.0f}°—{scan_hi:.0f}°")

    # 2. Bisect for the undershoot boundary (low bank angle edge of the corridor)
    #    Below this bank angle, the trajectory crashes (too deep).
    if scan_lo <= 0.0:
        # 0° is viable → undershoot boundary is at 0°
        udr_bank = 0.0
        udr_traj = _run_constant_bank(toml_str, 0.0)[1]
        print("    Undershoot boundary: bank=0.00° (0° is viable)")
    else:
        print(f"  Bisecting undershoot boundary ({scan_lo - scan_step:.0f}°→{scan_lo:.0f}°)...")
        udr_bank, udr_traj = _bisect_bank_angle(toml_str, scan_lo - scan_step, scan_lo, captured_side="hi", tol=bank_tol)
        print(f"    Undershoot boundary: bank={udr_bank:.2f}°")

    # 3. Bisect for the overshoot boundary (high bank angle edge of the corridor)
    #    Above this bank angle, the trajectory crashes or escapes.
    if scan_hi >= 180.0:
        ovr_bank = 180.0
        ovr_traj = _run_constant_bank(toml_str, 180.0)[1]
        print("    Overshoot boundary: bank=180.00° (180° is viable)")
    else:
        print(f"  Bisecting overshoot boundary ({scan_hi:.0f}°→{scan_hi + scan_step:.0f}°)...")
        ovr_bank, ovr_traj = _bisect_bank_angle(toml_str, scan_hi, scan_hi + scan_step, captured_side="lo", tol=bank_tol)
        print(f"    Overshoot boundary: bank={ovr_bank:.2f}°")

    # 4. Find optimal nominal: minimize |dv1|+|dv2| within the captured corridor
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
