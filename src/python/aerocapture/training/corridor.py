"""Aerocapture corridor boundary computation.

Computes three boundary trajectories for a given mission config:
- Nominal: run with the configured guidance algorithm, no dispersions
- Undershoot: constant bank angle at the maximum lift-up angle that still captures
- Overshoot: constant bank angle at the maximum lift-down angle that still captures

The boundaries are found by bisecting on bank angle at the nominal entry FPA.

Usage (standalone):
    uv run python -m aerocapture.training.corridor \\
        --toml configs/nominal/msr_aller_ftc_nominal.toml \\
        --output corridor_boundaries.npz

The TOML must be a complete config with a [guidance] section (e.g., a nominal
or training config that inherits from a mission base), not a bare mission base.
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


def _run_constant_bank(toml_path: str, bank_angle_deg: float) -> tuple[bool, npt.NDArray[np.float64]]:
    """Run a single sim with constant bank angle (reference_trajectory mode).

    Returns (captured, trajectory) where trajectory is (T, 12) or empty.
    """
    aero = _get_aero()

    overrides = {
        "guidance.reference_trajectory": True,
        "guidance.reference_bank_angle": float(bank_angle_deg),
        "simulation.n_sims": 1,
    }
    result = aero.run(toml_path, overrides=overrides)  # type: ignore[attr-defined]
    traj: npt.NDArray[np.float64] = result.trajectory
    return bool(result.captured), traj


def _run_nominal(toml_path: str) -> tuple[bool, npt.NDArray[np.float64]]:
    """Run a single sim with the configured guidance, no dispersions.

    Returns (captured, trajectory) where trajectory is (T, 12) or empty.
    """
    aero = _get_aero()

    overrides = {
        "simulation.n_sims": 1,
    }
    result = aero.run(toml_path, overrides=overrides)  # type: ignore[attr-defined]
    traj: npt.NDArray[np.float64] = result.trajectory
    return bool(result.captured), traj


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
        captured, traj = _run_constant_bank(toml_path, mid)

        if captured:
            last_captured_traj = traj
            last_captured_bank = mid

        if abs(hi - lo) < tol:
            break

        if captured_side == "lo":
            # lo is captured, hi is escaped → if mid captures, move lo up
            if captured:
                lo = mid
            else:
                hi = mid
        else:
            # hi is captured, lo is escaped → if mid captures, move hi down
            if captured:
                hi = mid
            else:
                lo = mid

    return last_captured_bank, last_captured_traj


def compute_corridor(
    toml_path: str,
    bank_tol: float = 0.1,
) -> dict[str, npt.NDArray[np.float64]]:
    """Compute nominal + undershoot/overshoot corridor boundaries.

    The undershoot boundary is the constant bank angle closest to 0° (full lift-up)
    that still captures — this produces the highest pdyn trajectory.

    The overshoot boundary is the constant bank angle closest to 180° (full lift-down)
    that still captures — this produces the lowest pdyn trajectory.

    Args:
        toml_path: Path to mission TOML config.
        bank_tol: Bank angle bisection tolerance in degrees.

    Returns:
        Dict with keys: "nominal", "undershoot", "overshoot" → (T, 12) trajectory arrays,
        plus "undershoot_bank_deg" and "overshoot_bank_deg" scalar arrays.
    """
    toml_str = str(Path(toml_path).resolve())

    # 1. Nominal trajectory with configured guidance
    print("  Running nominal trajectory...")
    nom_captured, nom_traj = _run_nominal(toml_str)
    if not nom_captured:
        print("  WARNING: Nominal trajectory did not capture!")

    # 2. Find undershoot boundary: bisect bank angle in [0, 90]
    # At bank=0 (full lift-up), trajectory goes deepest → likely captured
    # At bank=90, less lift → may or may not capture
    # We want the HIGHEST bank angle that still captures (closest to escape)
    print("  Bisecting undershoot boundary (bank 0°→90°)...")
    cap_0, _ = _run_constant_bank(toml_str, 0.0)
    cap_90, _ = _run_constant_bank(toml_str, 90.0)

    if cap_0 and not cap_90:
        # Normal case: 0° captures, 90° escapes → bisect
        udr_bank, udr_traj = _bisect_bank_angle(toml_str, 0.0, 90.0, captured_side="lo", tol=bank_tol)
    elif cap_0 and cap_90:
        # Both capture → boundary is beyond 90°, use 90° as undershoot
        print("  Both 0° and 90° capture; using 90° as undershoot boundary")
        udr_bank, udr_traj = 90.0, _run_constant_bank(toml_str, 90.0)[1]
    else:
        # 0° doesn't capture → very steep entry, corridor may not exist
        print("  WARNING: Bank=0° does not capture; using 0° trajectory")
        udr_bank, udr_traj = 0.0, _run_constant_bank(toml_str, 0.0)[1]

    print(f"    Undershoot boundary: bank={udr_bank:.2f}°")

    # 3. Find overshoot boundary: bisect bank angle in [90, 180]
    # At bank=180 (full lift-down), trajectory is shallow → likely escapes
    # At bank=90, moderate → may capture
    print("  Bisecting overshoot boundary (bank 90°→180°)...")
    cap_180, _ = _run_constant_bank(toml_str, 180.0)

    if cap_90 and not cap_180:
        # Normal case: 90° captures, 180° escapes → bisect
        ovr_bank, ovr_traj = _bisect_bank_angle(toml_str, 90.0, 180.0, captured_side="lo", tol=bank_tol)
    elif not cap_90 and not cap_180:
        # Neither captures → try from nominal bank angle downward
        # Use the initial_bank_angle as known-captured reference
        print("  90° and 180° both escape; bisecting from 45°→180°...")
        cap_45, _ = _run_constant_bank(toml_str, 45.0)
        if cap_45:
            ovr_bank, ovr_traj = _bisect_bank_angle(toml_str, 45.0, 180.0, captured_side="lo", tol=bank_tol)
        else:
            print("  WARNING: Cannot find overshoot boundary")
            ovr_bank, ovr_traj = 180.0, _run_constant_bank(toml_str, 180.0)[1]
    else:
        # Both capture → boundary beyond 180° (unlikely)
        print("  Both 90° and 180° capture; using 180° as overshoot boundary")
        ovr_bank, ovr_traj = 180.0, _run_constant_bank(toml_str, 180.0)[1]

    print(f"    Overshoot boundary: bank={ovr_bank:.2f}°")

    return {
        "nominal": nom_traj,
        "undershoot": udr_traj,
        "overshoot": ovr_traj,
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
    parser.add_argument("--toml", type=str, required=True, help="TOML config with [guidance] section (e.g., nominal or training config)")
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
