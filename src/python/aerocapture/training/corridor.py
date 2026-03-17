"""Aerocapture corridor boundary computation via Monte Carlo.

Computes corridor boundaries (4 envelopes) and optimal nominal trajectory
using two MC phases:

1. **Phase 1** (bank angle dispersed [0deg,180deg] + all mission dispersions):
   Classify each trajectory (crash/undershoot/corridor/overshoot/hyperbolic),
   extract 4 pdyn envelopes at p99/p1 percentiles.

2. **Phase 2** (bank angle dispersed [0deg,180deg], no other dispersions):
   Find the trajectory with lowest |dv1|+|dv2| as the optimal constant-bank nominal.

Usage (standalone):
    uv run python -m aerocapture.training.corridor \\
        --toml configs/missions/mars.toml \\
        --output corridor_boundaries.npz
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.ndimage import uniform_filter1d

# Trajectory column indices (12-column format)
_TRAJ_COL_ENERGY = 8
_TRAJ_COL_PDYN = 9

# Final record column indices (52-column format)
_COL_ENERGY = 7
_COL_ECC = 9
_COL_PERI_ALT = 14
_COL_APO_ERR = 30
_COL_IFINAL = 31
_COL_DV_APO_PERI = 40  # |dv1| + |dv2|
_COL_DV_TOTAL = 41
_DV_CRASH_SENTINEL = 1e10

# Cache schema version
_SCHEMA_VERSION = 2

# Default corridor parameters
_DEFAULT_DELTA_ZA = 200.0
_DEFAULT_N_SIMS = 10000


def _viable_capture_mask(final_records: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    """Return boolean mask for viable captures in a batch of final records (N, 52).

    Viable = bound orbit (ecc < 1, energy < 0) + periapsis above surface + DV not sentinel.
    """
    ecc = final_records[:, _COL_ECC]
    energy = final_records[:, _COL_ENERGY]
    peri_alt = final_records[:, _COL_PERI_ALT]
    dv = final_records[:, _COL_DV_APO_PERI]
    return (ecc < 1.0) & (energy < 0.0) & (peri_alt > 0.0) & (dv < _DV_CRASH_SENTINEL)


def classify_trajectories(
    final_records: npt.NDArray[np.float64],
    delta_za: float = _DEFAULT_DELTA_ZA,
) -> npt.NDArray[np.str_]:
    """Classify each trajectory by outcome.

    Priority order: crash > timeout > hyperbolic > captured sub-categories.

    Returns array of strings: "crash", "undershoot", "corridor", "overshoot",
    "hyperbolic", or "timeout".
    """
    n = len(final_records)
    labels = np.empty(n, dtype="U12")

    if n == 0:
        return labels

    ifinal = final_records[:, _COL_IFINAL]
    energy = final_records[:, _COL_ENERGY]
    ecc = final_records[:, _COL_ECC]
    apo_err = final_records[:, _COL_APO_ERR]

    # Step 1: crash (ifinal == 1) — highest priority
    crash = ifinal == 1.0
    labels[crash] = "crash"

    # Step 2: timeout (ifinal == 2) — discard
    timeout = ifinal == 2.0
    labels[timeout] = "timeout"

    # Step 3: atmosphere exit (ifinal == 3)
    atm_exit = ifinal == 3.0
    captured = atm_exit & (ecc < 1.0) & (energy < 0.0)

    # Hyperbolic: atmosphere exit but not captured
    hyperbolic = atm_exit & ~captured & ~crash & ~timeout
    labels[hyperbolic] = "hyperbolic"

    # Captured sub-categories by apoapsis error
    undershoot = captured & (apo_err < -delta_za)
    overshoot = captured & (apo_err > delta_za)
    corridor = captured & ~undershoot & ~overshoot

    labels[undershoot] = "undershoot"
    labels[overshoot] = "overshoot"
    labels[corridor] = "corridor"

    return labels


def compute_envelopes(
    trajectories: list[npt.NDArray[np.float64]],
    labels: npt.NDArray[np.str_],
    delta_za: float = _DEFAULT_DELTA_ZA,
    n_bins: int = 200,
) -> dict[str, npt.NDArray[np.float64]]:
    """Extract 4 pdyn envelope curves from classified trajectories.

    Returns dict with keys: energy_bins, envelope_{undershoot,crash,overshoot,hyperbolic}_pdyn.
    Each envelope array has NaN where insufficient data exists.
    """
    # Shared energy axis from ALL trajectories
    all_energies: list[float] = []
    for t in trajectories:
        t_arr = np.asarray(t)
        if t_arr.ndim == 2 and t_arr.shape[0] > 0:
            all_energies.extend(t_arr[:, _TRAJ_COL_ENERGY].tolist())

    if not all_energies:
        empty = np.full(n_bins, np.nan)
        return {
            "energy_bins": np.linspace(-6, 4, n_bins),
            "envelope_undershoot_pdyn": empty.copy(),
            "envelope_crash_pdyn": empty.copy(),
            "envelope_overshoot_pdyn": empty.copy(),
            "envelope_hyperbolic_pdyn": empty.copy(),
        }

    e_all = np.array(all_energies)
    bins = np.linspace(e_all.min(), e_all.max(), n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    def _envelope(mask: npt.NDArray[np.bool_], percentile: float) -> npt.NDArray[np.float64]:
        """Compute percentile of pdyn per energy bin for trajectories matching mask."""
        result = np.full(n_bins, np.nan)
        if not mask.any():
            return result

        # Collect all (energy, pdyn) points from matching trajectories
        e_pts: list[float] = []
        p_pts: list[float] = []
        for i in np.where(mask)[0]:
            t_arr = np.asarray(trajectories[i])
            if t_arr.ndim != 2 or t_arr.shape[0] == 0:
                continue
            e_pts.extend(t_arr[:, _TRAJ_COL_ENERGY].tolist())
            p_pts.extend(t_arr[:, _TRAJ_COL_PDYN].tolist())

        if not e_pts:
            return result

        e_arr = np.array(e_pts)
        p_arr = np.array(p_pts)
        bin_idx = np.clip(np.digitize(e_arr, bins) - 1, 0, n_bins - 1)

        for b in range(n_bins):
            m = bin_idx == b
            count = int(m.sum())
            if count >= 3:  # minimum bin occupancy
                result[b] = np.percentile(p_arr[m], percentile)

        # Interpolate NaN gaps from neighbors
        valid = ~np.isnan(result)
        if valid.any() and (~valid).any():
            result[~valid] = np.interp(bin_centers[~valid], bin_centers[valid], result[valid])

        # Smooth to reduce jaggedness
        valid_after = ~np.isnan(result)
        if valid_after.sum() > 5:
            result[valid_after] = uniform_filter1d(result[valid_after], size=5)

        return result

    # Envelope A (undershoot boundary): p99 of captured trajectories with apo_err >= -delta_za
    # "corridor" and "overshoot" qualify; "undershoot" has apo_err < -delta_za so excluded
    mask_a = (labels == "corridor") | (labels == "overshoot")
    if not mask_a.any():
        warnings.warn("No captured trajectories with apo_err >= -delta_za — undershoot envelope empty", stacklevel=2)
    envelope_undershoot = _envelope(mask_a, 99)

    # Envelope B (crash boundary): p1 of crash trajectories (lower bound of crash zone)
    # When no crashes exist, the crash boundary is undefined (all NaN).
    mask_b = labels == "crash"
    if not mask_b.any():
        warnings.warn("No crash trajectories — crash envelope empty", stacklevel=2)
    envelope_crash = _envelope(mask_b, 1)

    # Envelope C (overshoot boundary): p1 of captured trajectories with apo_err <= +delta_za
    # These are: corridor + undershoot (captured with apo_err <= +delta_za)
    mask_c = (labels == "corridor") | (labels == "undershoot")
    if not mask_c.any():
        warnings.warn("No captured trajectories with apo_err <= +delta_za — overshoot envelope empty", stacklevel=2)
    envelope_overshoot = _envelope(mask_c, 1)

    # Envelope D (hyperbolic boundary): p1 of ALL captured trajectories
    mask_d = (labels == "corridor") | (labels == "undershoot") | (labels == "overshoot")
    if not mask_d.any():
        warnings.warn("No captured trajectories — hyperbolic envelope empty", stacklevel=2)
    envelope_hyperbolic = _envelope(mask_d, 1)

    return {
        "energy_bins": bin_centers,
        "envelope_undershoot_pdyn": envelope_undershoot,
        "envelope_crash_pdyn": envelope_crash,
        "envelope_overshoot_pdyn": envelope_overshoot,
        "envelope_hyperbolic_pdyn": envelope_hyperbolic,
    }


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


def _read_corridor_config(toml_path: str) -> tuple[float, int]:
    """Read delta_za and n_sims from TOML [corridor] section.

    Falls back to defaults if section is absent.
    """
    from aerocapture.training.toml_utils import load_toml_with_bases

    data = load_toml_with_bases(Path(toml_path))
    corridor = data.get("corridor", {})
    delta_za = float(corridor.get("delta_za", _DEFAULT_DELTA_ZA))
    n_sims = int(corridor.get("n_sims", _DEFAULT_N_SIMS))
    return delta_za, n_sims


def compute_corridor(
    toml_path: str,
    n_sims: int | None = None,
    seed: int = 42,
    delta_za: float | None = None,
) -> dict[str, npt.NDArray[np.float64]]:
    """Compute corridor boundaries and optimal nominal via Monte Carlo.

    1. Phase 1 (bank [0deg,180deg] + all dispersions): classify trajectories,
       extract 4 pdyn envelopes.
    2. Phase 2 (bank [0deg,180deg], no dispersions): find min-DV nominal.

    Args:
        toml_path: Path to mission TOML config.
        n_sims: Number of MC sims per phase (overrides TOML [corridor].n_sims).
        seed: Random seed.
        delta_za: Apoapsis error tolerance in km (overrides TOML [corridor].delta_za).

    Returns dict with corridor cache data (schema version 2).
    """
    toml_str = str(Path(toml_path).resolve())

    # Read config defaults, allow CLI overrides
    cfg_delta_za, cfg_n_sims = _read_corridor_config(toml_str)
    delta_za = delta_za if delta_za is not None else cfg_delta_za
    n_sims = n_sims if n_sims is not None else cfg_n_sims

    # Read target apoapsis
    from aerocapture.training.toml_utils import load_toml_with_bases

    toml_data = load_toml_with_bases(Path(toml_str))
    target_apo = float(toml_data.get("flight", {}).get("target_orbit", {}).get("apoapsis", 0.0))

    # Phase 1: Full MC (bank + all dispersions) -> envelopes
    print(f"  Phase 1: Running {n_sims}-sim full MC (bank [0deg,180deg] + dispersions)...")
    fr_full, traj_full = _run_mc_constant_bank(toml_str, n_sims, 0.0, 180.0, seed, with_dispersions=True)
    labels = classify_trajectories(fr_full, delta_za=delta_za)

    counts = np.array(
        [
            int((labels == "crash").sum()),
            int((labels == "undershoot").sum()),
            int((labels == "corridor").sum()),
            int((labels == "overshoot").sum()),
            int((labels == "hyperbolic").sum()),
        ]
    )
    print(f"    Classification: crash={counts[0]}, under={counts[1]}, corridor={counts[2]}, over={counts[3]}, hyper={counts[4]}")

    envelopes = compute_envelopes(traj_full, labels, delta_za=delta_za)

    # Phase 2: Bank-only MC (no dispersions) -> nominal
    print(f"  Phase 2: Running {n_sims}-sim bank-only MC (no dispersions)...")
    fr_nom, traj_nom = _run_mc_constant_bank(toml_str, n_sims, 0.0, 180.0, seed + 1, with_dispersions=False)
    viable_nom = _viable_capture_mask(fr_nom)
    n_viable_nom = int(viable_nom.sum())
    print(f"    Viable captures: {n_viable_nom}/{n_sims}")

    if n_viable_nom > 0:
        dv_values = fr_nom[viable_nom, _COL_DV_APO_PERI]
        best_idx_in_viable = int(np.argmin(dv_values))
        best_idx = np.where(viable_nom)[0][best_idx_in_viable]
        nom_traj = traj_nom[best_idx]
        nom_dv = float(dv_values[best_idx_in_viable])
        nom_bank = float(nom_traj[0, 10]) if nom_traj.size > 0 else 0.0
        nom_dv_total = float(fr_nom[best_idx, _COL_DV_TOTAL])
        print(f"    Nominal: bank={nom_bank:.1f}deg, DV(apo+peri)={nom_dv:.1f} m/s, DV(total)={nom_dv_total:.1f} m/s")
    else:
        print("  WARNING: No viable captures in bank-only MC")
        nom_traj = np.empty((0, 12))
        nom_bank = 0.0
        nom_dv = 0.0
        nom_dv_total = 0.0

    return {
        "schema_version": np.array([_SCHEMA_VERSION]),
        **envelopes,
        "nominal": nom_traj,
        "nominal_bank_deg": np.array([nom_bank]),
        "nominal_dv": np.array([nom_dv]),
        "nominal_dv_total": np.array([nom_dv_total]),
        "target_apoapsis_km": np.array([target_apo]),
        "delta_za_km": np.array([delta_za]),
        "n_sims": np.array([n_sims]),
        "classification_counts": counts,
    }


def save_corridor(data: dict[str, npt.NDArray[np.float64]], output_path: Path) -> None:
    """Save corridor data to a compressed .npz file."""
    np.savez_compressed(str(output_path), **data)  # type: ignore[arg-type]
    print(f"  Corridor data saved to {output_path}")


def load_corridor(path: Path) -> dict[str, npt.NDArray[np.float64]] | None:
    """Load corridor data from a .npz file.

    Returns None if file not found or schema version mismatch.
    """
    if not path.exists():
        return None
    npz = np.load(str(path))
    data = {k: npz[k] for k in npz.files}

    # Check schema version
    version = data.get("schema_version", np.array([0]))
    if int(version[0]) != _SCHEMA_VERSION:
        warnings.warn(
            f"Corridor cache {path} has schema version {int(version[0])}, expected {_SCHEMA_VERSION}. Recomputation required.",
            stacklevel=2,
        )
        return None

    return data


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Compute aerocapture corridor boundaries via Monte Carlo")
    parser.add_argument("--toml", type=str, required=True, help="Mission TOML config (e.g., configs/missions/mars.toml)")
    parser.add_argument("--output", type=str, default="corridor_boundaries.npz", help="Output .npz file path")
    parser.add_argument("--n-sims", type=int, default=None, help="Number of MC sims (default: from TOML or 10000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--delta-za", type=float, default=None, help="Apoapsis error tolerance km (default: from TOML or 200)")
    args = parser.parse_args()

    print(f"Computing corridor boundaries for {args.toml}...")
    corridor = compute_corridor(args.toml, n_sims=args.n_sims, seed=args.seed, delta_za=args.delta_za)
    save_corridor(corridor, Path(args.output))

    counts = corridor["classification_counts"]
    print(f"\n  Summary: crash={counts[0]}, undershoot={counts[1]}, corridor={counts[2]}, overshoot={counts[3]}, hyperbolic={counts[4]}")
    nom = corridor["nominal"]
    if nom.size > 0:
        print(f"  Nominal: {nom.shape[0]} timesteps, energy [{nom[:, _TRAJ_COL_ENERGY].min():.2f}, {nom[:, _TRAJ_COL_ENERGY].max():.2f}] MJ/kg")


if __name__ == "__main__":
    main()
