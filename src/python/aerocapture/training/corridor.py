"""Aerocapture corridor boundary computation via Monte Carlo.

Computes corridor boundaries and optimal nominal trajectory using two MC runs:

1. **Full MC** (bank angle dispersed [0°,180°] + all mission dispersions):
   Filter viable captures, extract the energy-vs-pdyn envelope as corridor boundaries.
   The union of all captured trajectories defines the widest possible corridor.

2. **Bank-only MC** (bank angle dispersed [0°,180°], no other dispersions):
   Filter viable captures, pick the trajectory with lowest |dv1|+|dv2| as the
   optimal constant-bank nominal.

Usage (standalone):
    uv run python -m aerocapture.training.corridor \\
        --toml configs/missions/mars.toml \\
        --n-sims 5000 \\
        --output corridor_boundaries.npz

Any mission TOML works — the guidance scheme is irrelevant since all corridor
trajectories use constant bank angle (reference_trajectory mode).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

# Trajectory column indices (12-column format)
_TRAJ_COL_ENERGY = 8
_TRAJ_COL_PDYN = 9

# Final record column indices (52-column format)
_COL_ENERGY = 7
_COL_ECC = 9
_COL_PERI_ALT = 14
_COL_DV_APO_PERI = 40  # |dv1| + |dv2|
_DV_CRASH_SENTINEL = 1e10


def _viable_capture_mask(final_records: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    """Return boolean mask for viable captures in a batch of final records (N, 52).

    Viable = bound orbit (ecc < 1, energy < 0) + periapsis above surface + DV not sentinel.
    """
    ecc = final_records[:, _COL_ECC]
    energy = final_records[:, _COL_ENERGY]
    peri_alt = final_records[:, _COL_PERI_ALT]
    dv = final_records[:, _COL_DV_APO_PERI]
    return (ecc < 1.0) & (energy < 0.0) & (peri_alt > 0.0) & (dv < _DV_CRASH_SENTINEL)


def _run_mc_constant_bank(
    toml_path: str,
    n_sims: int,
    bank_lo: float,
    bank_hi: float,
    seed: int,
    with_dispersions: bool,
) -> tuple[npt.NDArray[np.float64], list[npt.NDArray[np.float64]]]:
    """Run MC with constant bank angle uniformly dispersed in [bank_lo, bank_hi].

    When with_dispersions=False, sets dispersion level to "none" to disable all
    mission dispersions (only bank angle varies).

    Returns (final_records (N, 52), trajectories list of (T_i, 12)).
    """
    try:
        import aerocapture_rs as aero  # type: ignore[import-not-found, import-untyped]
    except ImportError as e:
        msg = "PyO3 aerocapture_rs module required for corridor computation"
        raise ImportError(msg) from e

    # We can't directly disperse bank angle through the TOML dispersion system
    # (it disperses entry conditions, not guidance params). Instead, run n_sims
    # individual sims with bank angles drawn from uniform [bank_lo, bank_hi].
    rng = np.random.default_rng(seed)
    bank_angles = rng.uniform(bank_lo, bank_hi, n_sims)

    overrides_list = []
    for bank in bank_angles:
        ovr: dict[str, object] = {
            "guidance.type": "ftc",  # dummy — overridden by reference_trajectory
            "guidance.reference_trajectory": True,
            "guidance.reference_bank_angle": float(bank),
            "simulation.n_sims": 1,
        }
        if not with_dispersions:
            ovr["monte_carlo.dispersion_level"] = "none"
            ovr["monte_carlo.seed"] = seed
        overrides_list.append(ovr)

    results = aero.run_batch(  # type: ignore[attr-defined]
        toml_path,
        overrides_list,
        include_trajectories=True,
    )
    final_records: npt.NDArray[np.float64] = results.final_records
    trajectories: list[npt.NDArray[np.float64]] = results.trajectories
    return final_records, trajectories


def compute_corridor(
    toml_path: str,
    n_sims: int = 5000,
    seed: int = 42,
) -> dict[str, npt.NDArray[np.float64]]:
    """Compute corridor boundaries and optimal nominal via Monte Carlo.

    1. Full MC (bank [0°,180°] + all dispersions) → viable capture envelope = corridor.
    2. Bank-only MC (bank [0°,180°], no dispersions) → min-DV trajectory = nominal.

    Args:
        toml_path: Path to mission TOML config.
        n_sims: Number of MC sims per run (default 5000).
        seed: Random seed.

    Returns:
        Dict with keys:
        - "nominal": (T, 12) trajectory array (optimal constant-bank, min DV)
        - "nominal_bank_deg": scalar array with the nominal bank angle
        - "captured_trajectories": list of (T_i, 12) arrays for corridor envelope
        - "captured_final_records": (N_cap, 52) for captured sims
        - "n_sims": total sims run
        - "n_viable": number of viable captures
    """
    toml_str = str(Path(toml_path).resolve())

    # 1. Full MC: bank + all dispersions → corridor envelope
    print(f"  Running {n_sims}-sim full MC (bank [0°,180°] + dispersions)...")
    fr_full, traj_full = _run_mc_constant_bank(toml_str, n_sims, 0.0, 180.0, seed, with_dispersions=True)
    viable_full = _viable_capture_mask(fr_full)
    n_viable = int(viable_full.sum())
    print(f"    Viable captures: {n_viable}/{n_sims} ({100 * n_viable / n_sims:.1f}%)")

    captured_trajs = [traj_full[i] for i in np.where(viable_full)[0]]
    captured_frs = fr_full[viable_full]

    # 2. Bank-only MC: no dispersions → find optimal nominal
    n_nom = min(n_sims, 2000)  # fewer sims needed without dispersions
    print(f"  Running {n_nom}-sim bank-only MC (no dispersions)...")
    fr_nom, traj_nom = _run_mc_constant_bank(toml_str, n_nom, 0.0, 180.0, seed + 1, with_dispersions=False)
    viable_nom = _viable_capture_mask(fr_nom)
    n_viable_nom = int(viable_nom.sum())
    print(f"    Viable captures: {n_viable_nom}/{n_nom}")

    if n_viable_nom > 0:
        # Pick trajectory with lowest |dv1| + |dv2|
        dv_values = fr_nom[viable_nom, _COL_DV_APO_PERI]
        best_idx_in_viable = int(np.argmin(dv_values))
        best_idx = np.where(viable_nom)[0][best_idx_in_viable]
        nom_traj = traj_nom[best_idx]
        nom_dv = float(dv_values[best_idx_in_viable])
        # Recover bank angle from the initial bank (trajectory first timestep bank col)
        nom_bank = float(nom_traj[0, 10]) if nom_traj.size > 0 else 0.0  # col 10 = bank_deg
        print(f"    Nominal: bank≈{nom_bank:.1f}°, DV(apo+peri)={nom_dv:.1f} m/s")
    else:
        print("  WARNING: No viable captures in bank-only MC")
        nom_traj = np.array([])
        nom_bank = 0.0

    return {
        "nominal": nom_traj,
        "nominal_bank_deg": np.array([nom_bank]),
        "captured_trajectories_count": np.array([len(captured_trajs)]),
        "captured_final_records": captured_frs,
        "n_sims": np.array([n_sims]),
        "n_viable": np.array([n_viable]),
        # Store all captured trajectories concatenated with a separator scheme:
        # lengths array + flat concatenation, to fit in npz format
        **_pack_trajectories(captured_trajs),
    }


def _pack_trajectories(trajs: list[npt.NDArray[np.float64]]) -> dict[str, npt.NDArray[np.floating | np.signedinteger]]:
    """Pack variable-length trajectories into npz-compatible arrays.

    Returns {"traj_lengths": (N,), "traj_data": (total_rows, 12)}.
    """
    if not trajs:
        return {"traj_lengths": np.array([], dtype=np.int64), "traj_data": np.empty((0, 12))}
    lengths = np.array([len(t) for t in trajs], dtype=np.int64)
    data = np.vstack(trajs)
    return {"traj_lengths": lengths, "traj_data": data}


def _unpack_trajectories(data: dict[str, npt.NDArray[np.float64]]) -> list[npt.NDArray[np.float64]]:
    """Unpack trajectories from npz format back to list of arrays."""
    lengths = data.get("traj_lengths", np.array([]))
    flat = data.get("traj_data", np.empty((0, 12)))
    if lengths.size == 0:
        return []
    trajs: list[npt.NDArray[np.float64]] = []
    offset = 0
    for length in lengths:
        trajs.append(flat[offset : offset + int(length)])
        offset += int(length)
    return trajs


def save_corridor(data: dict[str, npt.NDArray[np.float64]], output_path: Path) -> None:
    """Save corridor data to a compressed .npz file."""
    np.savez_compressed(str(output_path), **data)  # type: ignore[arg-type]
    print(f"  Corridor data saved to {output_path}")


def load_corridor(path: Path) -> dict[str, npt.NDArray[np.float64]] | None:
    """Load corridor data from a .npz file. Returns None if not found."""
    if not path.exists():
        return None
    npz = np.load(str(path))
    return {k: npz[k] for k in npz.files}


def load_corridor_trajectories(path: Path) -> tuple[npt.NDArray[np.float64], list[npt.NDArray[np.float64]]] | None:
    """Load corridor nominal trajectory and captured trajectory list.

    Returns (nominal_trajectory, captured_trajectories) or None.
    """
    data = load_corridor(path)
    if data is None:
        return None
    nominal = data.get("nominal", np.array([]))
    captured = _unpack_trajectories(data)
    return nominal, captured


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Compute aerocapture corridor boundaries via Monte Carlo")
    parser.add_argument("--toml", type=str, required=True, help="Mission TOML config (e.g., configs/missions/mars.toml)")
    parser.add_argument("--output", type=str, default="corridor_boundaries.npz", help="Output .npz file path")
    parser.add_argument("--n-sims", type=int, default=5000, help="Number of MC sims (default: 5000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    print(f"Computing corridor boundaries for {args.toml}...")
    corridor = compute_corridor(args.toml, n_sims=args.n_sims, seed=args.seed)
    save_corridor(corridor, Path(args.output))

    # Print summary
    n_viable = int(corridor["n_viable"][0])
    n_total = int(corridor["n_sims"][0])
    print(f"\n  Summary: {n_viable}/{n_total} viable captures ({100 * n_viable / n_total:.1f}%)")
    nom = corridor["nominal"]
    if nom.size > 0:
        print(f"  Nominal: {nom.shape[0]} timesteps, energy [{nom[:, _TRAJ_COL_ENERGY].min():.2f}, {nom[:, _TRAJ_COL_ENERGY].max():.2f}] MJ/kg")
    n_cap_trajs = int(corridor["captured_trajectories_count"][0])
    print(f"  Captured trajectories stored: {n_cap_trajs}")


if __name__ == "__main__":
    main()
