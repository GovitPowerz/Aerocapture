"""Aerocapture corridor boundary computation via Monte Carlo.

Computes the full capture corridor (crash-to-capture boundaries) and
±δZa restricted boundary trajectories using a single bank-angle sweep MC:

1. Bank angle dispersed [0deg,180deg], no mission dispersions.
2. Classify each trajectory (crash/undershoot/corridor/overshoot/hyperbolic).
3. Extract 2 fill envelopes: crash boundary (max pdyn of non-crash) and
   capture boundary (min pdyn of captured).
4. Find ±δZa boundary trajectories (closest to apo_err = ±delta_za).
5. Find min-DV nominal (lowest |dv_peri|+|dv_apo| among captured).

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

# Cache schema version — bumped from 2 to 3 for new corridor format
_SCHEMA_VERSION = 3

# Default corridor parameters
_DEFAULT_DELTA_ZA = 500.0
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
    n_bins: int = 200,
) -> dict[str, npt.NDArray[np.float64]]:
    """Extract crash and capture pdyn envelopes from classified trajectories.

    Returns dict with keys: energy_bins, envelope_crash_pdyn, envelope_capture_pdyn.
    - crash envelope: MAX pdyn of non-crash trajectories (above = crash zone)
    - capture envelope: MIN pdyn of captured trajectories (below = hyperbolic zone)
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
            "envelope_crash_pdyn": empty.copy(),
            "envelope_capture_pdyn": empty.copy(),
        }

    e_all = np.array(all_energies)
    bins = np.linspace(e_all.min(), e_all.max(), n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    def _envelope(mask: npt.NDArray[np.bool_], use_max: bool) -> npt.NDArray[np.float64]:
        """Compute envelope (max or min) of pdyn per energy bin for trajectories matching mask."""
        result = np.full(n_bins, np.nan)
        if not mask.any():
            return result

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

        agg_fn = np.max if use_max else np.min
        for b in range(n_bins):
            m = bin_idx == b
            count = int(m.sum())
            if count >= 3:  # minimum bin occupancy
                result[b] = agg_fn(p_arr[m])

        # Interpolate NaN gaps from neighbors
        valid = ~np.isnan(result)
        if valid.any() and (~valid).any():
            result[~valid] = np.interp(bin_centers[~valid], bin_centers[valid], result[valid])

        # Smooth to reduce jaggedness
        valid_after = ~np.isnan(result)
        if valid_after.sum() > 5:
            result[valid_after] = uniform_filter1d(result[valid_after], size=5)

        return result

    # Crash boundary: MAX pdyn of ALL non-crashing trajectories
    # The area ABOVE this envelope is the crash zone (red).
    mask_crash = (labels != "crash") & (labels != "timeout")
    has_crashes = (labels == "crash").any()
    if not has_crashes:
        warnings.warn("No crash trajectories observed — crash zone not drawn", stacklevel=2)
        envelope_crash = np.full(n_bins, np.nan)
    else:
        envelope_crash = _envelope(mask_crash, use_max=True)

    # Capture boundary: MIN pdyn of ALL captured trajectories
    # The area BELOW this envelope is the hyperbolic exit zone (red).
    mask_capture = (labels == "corridor") | (labels == "undershoot") | (labels == "overshoot")
    if not mask_capture.any():
        warnings.warn("No captured trajectories — capture envelope empty", stacklevel=2)
    envelope_capture = _envelope(mask_capture, use_max=False)

    return {
        "energy_bins": bin_centers,
        "envelope_crash_pdyn": envelope_crash,
        "envelope_capture_pdyn": envelope_capture,
    }


def _find_boundary_trajectory(
    final_records: npt.NDArray[np.float64],
    trajectories: list[npt.NDArray[np.float64]],
    target_apo_err: float,
) -> npt.NDArray[np.float64]:
    """Find the captured trajectory with apo_err closest to target_apo_err.

    Returns the trajectory array (T, 12), or empty (0, 12) if no captures.
    """
    ifinal = final_records[:, _COL_IFINAL]
    energy = final_records[:, _COL_ENERGY]
    ecc = final_records[:, _COL_ECC]
    captured = (ifinal == 3.0) & (ecc < 1.0) & (energy < 0.0)

    if not captured.any():
        return np.empty((0, 12))

    cap_idx = np.where(captured)[0]
    apo_err = final_records[cap_idx, _COL_APO_ERR]
    best = cap_idx[int(np.argmin(np.abs(apo_err - target_apo_err)))]
    return np.asarray(trajectories[best])


def _run_mc_constant_bank(
    toml_path: str,
    n_sims: int,
    bank_lo: float,
    bank_hi: float,
    seed: int,
    with_dispersions: bool = False,
) -> tuple[npt.NDArray[np.float64], list[npt.NDArray[np.float64]]]:
    """Run MC with constant bank angle uniformly dispersed in [bank_lo, bank_hi].

    When with_dispersions=True, uses whatever dispersions are configured in the
    TOML (e.g., initial_state, atmosphere from common.toml inheritance).
    When False, overrides all dispersions to "off".

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
            ovr["monte_carlo.initial_state.level"] = "off"
            ovr["monte_carlo.atmosphere.level"] = "off"
            ovr["monte_carlo.aerodynamics.level"] = "off"
            ovr["monte_carlo.navigation.level"] = "off"
            ovr["monte_carlo.mass.level"] = "off"
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
    """Compute corridor boundaries and optimal nominal via two-phase MC.

    Phase 1 (bank [0deg,180deg] + TOML-configured dispersions): classify
    trajectories, extract crash/capture envelopes. The TOML should inherit
    from common.toml (or equivalent) so that initial_state, atmosphere, etc.
    dispersions are active.

    Phase 2 (bank [0deg,180deg], no dispersions): find ±δZa boundary
    trajectories and min-DV nominal. Deterministic (each bank angle maps
    to exactly one outcome).

    Args:
        toml_path: Path to TOML config (training TOML recommended, so
            dispersions from common.toml are inherited).
        n_sims: Number of MC sims per phase (overrides TOML [corridor].n_sims).
        seed: Random seed.
        delta_za: Apoapsis error tolerance in km (overrides TOML [corridor].delta_za).

    Returns dict with corridor cache data (schema version 3).
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

    # Phase 1: bank [0deg,180deg] + TOML dispersions -> envelopes
    print(f"  Phase 1: Running {n_sims}-sim MC (bank [0deg,180deg] + TOML dispersions)...")
    fr_disp, traj_disp = _run_mc_constant_bank(toml_str, n_sims, 0.0, 180.0, seed, with_dispersions=True)
    labels = classify_trajectories(fr_disp, delta_za=delta_za)

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

    # Envelopes: crash and capture boundaries for red fill zones
    envelopes = compute_envelopes(traj_disp, labels)

    # Phase 2: bank [0deg,180deg], no dispersions -> boundary trajectories + nominal
    print(f"  Phase 2: Running {n_sims}-sim bank-angle sweep (no dispersions)...")
    fr_nom, traj_nom = _run_mc_constant_bank(toml_str, n_sims, 0.0, 180.0, seed + 1, with_dispersions=False)
    labels_nom = classify_trajectories(fr_nom, delta_za=delta_za)

    counts_nom = np.array(
        [
            int((labels_nom == "crash").sum()),
            int((labels_nom == "undershoot").sum()),
            int((labels_nom == "corridor").sum()),
            int((labels_nom == "overshoot").sum()),
            int((labels_nom == "hyperbolic").sum()),
        ]
    )
    print(f"    Classification (no disp): crash={counts_nom[0]}, under={counts_nom[1]}, corridor={counts_nom[2]}, over={counts_nom[3]}, hyper={counts_nom[4]}")

    # ±δZa boundary trajectories (dashed lines within corridor)
    boundary_undershoot = _find_boundary_trajectory(fr_nom, traj_nom, -delta_za)
    boundary_overshoot = _find_boundary_trajectory(fr_nom, traj_nom, +delta_za)

    if boundary_undershoot.size > 0:
        bank_u = float(boundary_undershoot[0, 10])
        apo_u = float(fr_nom[np.argmin(np.abs(fr_nom[:, _COL_APO_ERR] - (-delta_za))), _COL_APO_ERR])
        print(f"    Undershoot boundary: bank={bank_u:.1f}deg, apo_err={apo_u:.0f}km")
    if boundary_overshoot.size > 0:
        bank_o = float(boundary_overshoot[0, 10])
        apo_o = float(fr_nom[np.argmin(np.abs(fr_nom[:, _COL_APO_ERR] - delta_za)), _COL_APO_ERR])
        print(f"    Overshoot boundary:  bank={bank_o:.1f}deg, apo_err={apo_o:.0f}km")

    # Nominal: min |dv_peri| + |dv_apo| among captured (from Phase 2, no dispersions)
    viable = _viable_capture_mask(fr_nom)
    n_viable = int(viable.sum())
    print(f"    Viable captures: {n_viable}/{n_sims}")

    if n_viable > 0:
        dv_values = fr_nom[viable, _COL_DV_APO_PERI]
        best_idx_in_viable = int(np.argmin(dv_values))
        best_idx = np.where(viable)[0][best_idx_in_viable]
        nom_traj = np.asarray(traj_nom[best_idx])
        nom_dv = float(dv_values[best_idx_in_viable])
        nom_bank = float(nom_traj[0, 10]) if nom_traj.size > 0 else 0.0
        nom_dv_total = float(fr_nom[best_idx, _COL_DV_TOTAL])
        print(f"    Nominal: bank={nom_bank:.1f}deg, DV(apo+peri)={nom_dv:.1f} m/s, DV(total)={nom_dv_total:.1f} m/s")
    else:
        print("  WARNING: No viable captures in bank-angle sweep")
        nom_traj = np.empty((0, 12))
        nom_bank = 0.0
        nom_dv = 0.0
        nom_dv_total = 0.0

    return {
        "schema_version": np.array([_SCHEMA_VERSION]),
        **envelopes,
        "boundary_undershoot": boundary_undershoot,
        "boundary_overshoot": boundary_overshoot,
        "nominal": nom_traj,
        "nominal_bank_deg": np.array([nom_bank]),
        "nominal_dv": np.array([nom_dv]),
        "nominal_dv_total": np.array([nom_dv_total]),
        "target_apoapsis_km": np.array([target_apo]),
        "delta_za_km": np.array([delta_za]),
        "n_sims": np.array([n_sims]),
        "classification_counts": counts,
    }


class CorridorAccumulator:
    """Incremental corridor envelope accumulator.

    Maintains 4 running envelopes updated per generation:
    - crash_max_pdyn: max pdyn of non-crash trajectories (above = crash zone)
    - restricted_max_pdyn: max pdyn of corridor-classified (|apo_err| < delta_za)
    - restricted_min_pdyn: min pdyn of corridor-classified
    - capture_min_pdyn: min pdyn of all captured (below = hyperbolic zone)

    Designed for incremental updates across GA generations without storing
    all trajectory data.
    """

    def __init__(
        self,
        energy_min: float,
        energy_max: float,
        delta_za_restricted: float = 200.0,
        n_bins: int = 200,
    ) -> None:
        self.n_bins = n_bins
        self.delta_za_restricted = delta_za_restricted
        bins = np.linspace(energy_min, energy_max, n_bins + 1)
        self.energy_bins: npt.NDArray[np.float64] = (bins[:-1] + bins[1:]) / 2
        self._bin_edges: npt.NDArray[np.float64] = bins
        self.crash_max_pdyn: npt.NDArray[np.float64] = np.full(n_bins, np.nan)
        self.restricted_max_pdyn: npt.NDArray[np.float64] = np.full(n_bins, np.nan)
        self.restricted_min_pdyn: npt.NDArray[np.float64] = np.full(n_bins, np.nan)
        self.capture_min_pdyn: npt.NDArray[np.float64] = np.full(n_bins, np.nan)

    def update(
        self,
        trajectories: list[npt.NDArray[np.float64]],
        labels: npt.NDArray[np.str_],
    ) -> None:
        """Update running envelopes with a new batch of classified trajectories."""
        labels_arr = np.asarray(labels)
        non_crash = (labels_arr != "crash") & (labels_arr != "timeout")
        corridor = labels_arr == "corridor"
        captured = (labels_arr == "corridor") | (labels_arr == "undershoot") | (labels_arr == "overshoot")

        self._update_envelope(trajectories, non_crash, self.crash_max_pdyn, use_max=True)
        self._update_envelope(trajectories, corridor, self.restricted_max_pdyn, use_max=True)
        self._update_envelope(trajectories, corridor, self.restricted_min_pdyn, use_max=False)
        self._update_envelope(trajectories, captured, self.capture_min_pdyn, use_max=False)

    def _update_envelope(
        self,
        trajectories: list[npt.NDArray[np.float64]],
        mask: npt.NDArray[np.bool_],
        envelope: npt.NDArray[np.float64],
        use_max: bool,
    ) -> None:
        """Update one envelope array in-place for trajectories matching mask."""
        if not mask.any():
            return
        for i in np.where(mask)[0]:
            t = np.asarray(trajectories[i])
            if t.ndim != 2 or t.shape[0] == 0:
                continue
            e_arr = t[:, _TRAJ_COL_ENERGY]
            p_arr = t[:, _TRAJ_COL_PDYN]
            bin_idx = np.clip(np.digitize(e_arr, self._bin_edges) - 1, 0, self.n_bins - 1)
            for b in range(self.n_bins):
                m = bin_idx == b
                if not m.any():
                    continue
                val = float(np.max(p_arr[m]) if use_max else np.min(p_arr[m]))
                if np.isnan(envelope[b]):
                    envelope[b] = val
                elif use_max:
                    envelope[b] = max(envelope[b], val)
                else:
                    envelope[b] = min(envelope[b], val)

    def to_checkpoint(self) -> dict[str, npt.NDArray[np.float64]]:
        """Serialize accumulator state for checkpoint persistence."""
        return {
            "corridor_energy_bins": self.energy_bins,
            "corridor_crash_max_pdyn": self.crash_max_pdyn,
            "corridor_restricted_max_pdyn": self.restricted_max_pdyn,
            "corridor_restricted_min_pdyn": self.restricted_min_pdyn,
            "corridor_capture_min_pdyn": self.capture_min_pdyn,
            "corridor_delta_za": np.array([self.delta_za_restricted]),
        }

    @classmethod
    def from_checkpoint(cls, state: dict[str, npt.NDArray[np.float64]]) -> CorridorAccumulator:
        """Restore accumulator from a checkpoint dict."""
        energy_bins = state["corridor_energy_bins"]
        n_bins = len(energy_bins)
        delta_za = float(state["corridor_delta_za"][0])
        # Construct with dummy range; we overwrite bins directly below
        acc = cls(energy_min=0.0, energy_max=1.0, delta_za_restricted=delta_za, n_bins=n_bins)
        acc.energy_bins = energy_bins.copy()
        half = (energy_bins[1] - energy_bins[0]) / 2 if n_bins > 1 else 0.5
        acc._bin_edges = np.concatenate([[energy_bins[0] - half], energy_bins + half])
        acc.crash_max_pdyn = state["corridor_crash_max_pdyn"].copy()
        acc.restricted_max_pdyn = state["corridor_restricted_max_pdyn"].copy()
        acc.restricted_min_pdyn = state["corridor_restricted_min_pdyn"].copy()
        acc.capture_min_pdyn = state["corridor_capture_min_pdyn"].copy()
        return acc

    def to_corridor_data(
        self,
        nominal: npt.NDArray[np.float64] | None = None,
    ) -> dict[str, npt.NDArray[np.float64]]:
        """Export accumulated envelopes as a corridor data dict (schema v4).

        Fills NaN gaps via interpolation and applies a mild smoothing filter.
        """
        smoothed: dict[str, npt.NDArray[np.float64]] = {}
        for name, arr in [
            ("envelope_crash_pdyn", self.crash_max_pdyn),
            ("envelope_restricted_max_pdyn", self.restricted_max_pdyn),
            ("envelope_restricted_min_pdyn", self.restricted_min_pdyn),
            ("envelope_capture_pdyn", self.capture_min_pdyn),
        ]:
            s = arr.copy()
            valid = ~np.isnan(s)
            if valid.any() and (~valid).any():
                s[~valid] = np.interp(self.energy_bins[~valid], self.energy_bins[valid], s[valid])
            if valid.sum() > 5:
                s = uniform_filter1d(s, size=5)
            smoothed[name] = s

        return {
            "schema_version": np.array([4]),
            "energy_bins": self.energy_bins,
            **smoothed,
            "nominal": nominal if nominal is not None else np.empty((0, 12)),
            "delta_za_km": np.array([self.delta_za_restricted]),
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

    # Check schema version — accept v3 (2-layer) and v4 (4-layer with restricted envelopes)
    _ACCEPTED_VERSIONS = {_SCHEMA_VERSION, 4}
    version = data.get("schema_version", np.array([0]))
    if int(version[0]) not in _ACCEPTED_VERSIONS:
        warnings.warn(
            f"Corridor cache {path} has schema version {int(version[0])}, expected one of {_ACCEPTED_VERSIONS}. Recomputation required.",
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
    parser.add_argument("--delta-za", type=float, default=None, help="Apoapsis error tolerance km (default: from TOML or 500)")
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
