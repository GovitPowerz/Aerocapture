"""Aerocapture corridor boundary computation.

The corridor is built incrementally during piecewise_constant GA training
via `CorridorAccumulator`. Each generation's trajectories are classified
and their pdyn envelopes updated in a running max/min fashion.

The accumulator produces a schema-v4 `.npz` cache with 4 envelopes:
- crash boundary (max pdyn of non-crash)
- restricted corridor upper/lower (max/min pdyn of corridor-classified)
- capture boundary (min pdyn of all captured)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import numpy.typing as npt

# Trajectory column indices (12-column format)
_TRAJ_COL_ENERGY = 8
_TRAJ_COL_PDYN = 9

# Final record column indices (52-column format)
_COL_ENERGY = 7
_COL_ECC = 9
_COL_APO_ERR = 30
_COL_IFINAL = 31

# Default corridor parameters
_DEFAULT_DELTA_ZA = 500.0


def classify_trajectories(
    final_records: npt.NDArray[np.float64],
    delta_za: float = _DEFAULT_DELTA_ZA,
    delta_za_low: float | None = None,
    delta_za_high: float | None = None,
) -> npt.NDArray[np.str_]:
    """Classify each trajectory by outcome.

    Priority order: crash > timeout > hyperbolic > captured sub-categories.

    Corridor bounds are asymmetric: apo_err in [delta_za_low, delta_za_high].
    If delta_za_low/high are None, falls back to symmetric ±delta_za.

    Returns array of strings: "crash", "undershoot", "corridor", "overshoot",
    "hyperbolic", or "timeout".
    """
    lo = delta_za_low if delta_za_low is not None else -delta_za
    hi = delta_za_high if delta_za_high is not None else delta_za

    n = len(final_records)
    labels = np.empty(n, dtype="U12")

    if n == 0:
        return labels

    ifinal = final_records[:, _COL_IFINAL]
    energy = final_records[:, _COL_ENERGY]
    ecc = final_records[:, _COL_ECC]
    apo_err = final_records[:, _COL_APO_ERR]

    # Step 1: crash (ifinal == 1 or 4 = pending crash) — highest priority
    crash = (ifinal == 1.0) | (ifinal == 4.0)
    labels[crash] = "crash"

    # Step 2: timeout (ifinal == 2) — discard
    timeout = ifinal == 2.0
    labels[timeout] = "timeout"

    # Step 3: atmosphere exit (ifinal == 3)
    atm_exit = ifinal == 3.0
    captured = atm_exit & (ecc < 1.0) & (energy < 0.0)  # NOTE: stricter than charts.is_captured (adds energy<0) -- intentional.

    # Hyperbolic: atmosphere exit but not captured
    hyperbolic = atm_exit & ~captured & ~crash & ~timeout
    labels[hyperbolic] = "hyperbolic"

    # Captured sub-categories by apoapsis error (asymmetric bounds)
    undershoot = captured & (apo_err < lo)
    overshoot = captured & (apo_err > hi)
    corridor = captured & ~undershoot & ~overshoot

    labels[undershoot] = "undershoot"
    labels[overshoot] = "overshoot"
    labels[corridor] = "corridor"

    return labels


class CorridorAccumulator:
    """Incremental corridor envelope accumulator.

    Maintains 4 running envelopes updated per generation:
    - crash_max_pdyn: max pdyn of non-crash trajectories (above = crash zone)
    - restricted_max_pdyn: max pdyn of corridor-classified
    - restricted_min_pdyn: min pdyn of corridor-classified
    - capture_min_pdyn: min pdyn of all captured (below = hyperbolic zone)

    Corridor classification uses asymmetric bounds: apo_err in [delta_za_low, delta_za_high].

    Designed for incremental updates across GA generations without storing
    all trajectory data.
    """

    def __init__(
        self,
        energy_min: float,
        energy_max: float,
        delta_za_restricted: float = 200.0,
        delta_za_low: float | None = None,
        delta_za_high: float | None = None,
        n_bins: int = 200,
    ) -> None:
        self.n_bins = n_bins
        self.delta_za_restricted = delta_za_restricted
        self.delta_za_low = delta_za_low if delta_za_low is not None else -delta_za_restricted
        self.delta_za_high = delta_za_high if delta_za_high is not None else delta_za_restricted
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
            "corridor_delta_za_low": np.array([self.delta_za_low]),
            "corridor_delta_za_high": np.array([self.delta_za_high]),
        }

    @classmethod
    def from_checkpoint(cls, state: dict[str, npt.NDArray[np.float64]]) -> CorridorAccumulator:
        """Restore accumulator from a checkpoint dict."""
        energy_bins = state["corridor_energy_bins"]
        n_bins = len(energy_bins)
        delta_za = float(state["corridor_delta_za"][0])
        delta_za_low = float(state["corridor_delta_za_low"][0]) if "corridor_delta_za_low" in state else -delta_za
        delta_za_high = float(state["corridor_delta_za_high"][0]) if "corridor_delta_za_high" in state else delta_za
        # Construct with dummy range; we overwrite bins directly below
        acc = cls(energy_min=0.0, energy_max=1.0, delta_za_restricted=delta_za, delta_za_low=delta_za_low, delta_za_high=delta_za_high, n_bins=n_bins)
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

        Fills NaN gaps via interpolation and applies Gaussian smoothing.
        """
        from scipy.ndimage import gaussian_filter1d

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
            if valid.sum() > 10:
                s = gaussian_filter1d(s, sigma=3.0)
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

    Returns None if file not found or unsupported schema version.
    Accepts schema v4 (4-layer with restricted envelopes from CorridorAccumulator).
    """
    if not path.exists():
        return None
    npz = np.load(str(path))
    data = {k: npz[k] for k in npz.files}

    version = data.get("schema_version", np.array([0]))
    if int(version[0]) != 4:
        warnings.warn(
            f"Corridor cache {path} has schema version {int(version[0])}, expected 4. Recomputation required.",
            stacklevel=2,
        )
        return None

    return data
