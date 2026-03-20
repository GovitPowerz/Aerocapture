"""Tests for corridor boundary computation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from aerocapture.training.corridor import CorridorAccumulator, classify_trajectories, load_corridor, save_corridor


def _make_final_records(
    n_crash: int = 5,
    n_undershoot: int = 10,
    n_corridor: int = 50,
    n_overshoot: int = 10,
    n_hyperbolic: int = 5,
    n_timeout: int = 2,
    delta_za: float = 200.0,
) -> np.ndarray:
    """Create synthetic final_records with known classification counts."""
    n = n_crash + n_undershoot + n_corridor + n_overshoot + n_hyperbolic + n_timeout
    fr = np.zeros((n, 52))
    idx = 0

    fr[idx : idx + n_crash, 31] = 1.0
    idx += n_crash

    fr[idx : idx + n_undershoot, 31] = 3.0
    fr[idx : idx + n_undershoot, 7] = -1.0
    fr[idx : idx + n_undershoot, 9] = 0.5
    fr[idx : idx + n_undershoot, 30] = -(delta_za + 50)
    idx += n_undershoot

    fr[idx : idx + n_corridor, 31] = 3.0
    fr[idx : idx + n_corridor, 7] = -1.0
    fr[idx : idx + n_corridor, 9] = 0.5
    fr[idx : idx + n_corridor, 30] = np.linspace(-delta_za + 10, delta_za - 10, n_corridor)
    idx += n_corridor

    fr[idx : idx + n_overshoot, 31] = 3.0
    fr[idx : idx + n_overshoot, 7] = -1.0
    fr[idx : idx + n_overshoot, 9] = 0.5
    fr[idx : idx + n_overshoot, 30] = delta_za + 50
    idx += n_overshoot

    fr[idx : idx + n_hyperbolic, 31] = 3.0
    fr[idx : idx + n_hyperbolic, 7] = 1.0
    fr[idx : idx + n_hyperbolic, 9] = 1.5
    idx += n_hyperbolic

    fr[idx : idx + n_timeout, 31] = 2.0
    idx += n_timeout

    return fr


def _make_trajectories_with_labels(
    n_per_class: int = 20,
    n_steps: int = 50,
    seed: int = 42,
) -> tuple[list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    trajs: list[np.ndarray] = []
    labels_list: list[str] = []
    energy_range = np.linspace(4.0, -6.0, n_steps)

    for cls, pdyn_base in [("crash", 2.5), ("undershoot", 1.8), ("corridor", 1.0), ("overshoot", 0.5), ("hyperbolic", 0.2)]:
        for _ in range(n_per_class):
            t = np.zeros((n_steps, 12))
            t[:, 8] = energy_range
            t[:, 9] = pdyn_base + rng.normal(0, 0.05, n_steps)
            trajs.append(t)
            labels_list.append(cls)

    return trajs, np.array(labels_list)


class TestClassifyTrajectories:
    def test_correct_counts(self) -> None:
        fr = _make_final_records()
        labels = classify_trajectories(fr, delta_za=200.0)
        assert (labels == "crash").sum() == 5
        assert (labels == "undershoot").sum() == 10
        assert (labels == "corridor").sum() == 50
        assert (labels == "overshoot").sum() == 10
        assert (labels == "hyperbolic").sum() == 5
        assert (labels == "timeout").sum() == 2

    def test_crash_priority_over_captured(self) -> None:
        fr = np.zeros((1, 52))
        fr[0, 31] = 1.0
        fr[0, 7] = -1.0
        fr[0, 9] = 0.5
        fr[0, 30] = 0.0
        labels = classify_trajectories(fr, delta_za=200.0)
        assert labels[0] == "crash"

    def test_empty_input(self) -> None:
        fr = np.zeros((0, 52))
        labels = classify_trajectories(fr, delta_za=200.0)
        assert len(labels) == 0

    def test_all_crash(self) -> None:
        fr = np.zeros((10, 52))
        fr[:, 31] = 1.0
        labels = classify_trajectories(fr, delta_za=200.0)
        assert (labels == "crash").sum() == 10

    def test_boundary_values(self) -> None:
        fr = np.zeros((2, 52))
        fr[:, 31] = 3.0
        fr[:, 7] = -1.0
        fr[:, 9] = 0.5
        fr[0, 30] = -200.0
        fr[1, 30] = 200.0
        labels = classify_trajectories(fr, delta_za=200.0)
        assert labels[0] == "corridor"
        assert labels[1] == "corridor"

    def test_asymmetric_bounds(self) -> None:
        fr = np.zeros((3, 52))
        fr[:, 31] = 3.0
        fr[:, 7] = -1.0
        fr[:, 9] = 0.5
        fr[0, 30] = -150.0  # within [-200, +1000]
        fr[1, 30] = 800.0  # within [-200, +1000]
        fr[2, 30] = 1100.0  # outside (overshoot)
        labels = classify_trajectories(fr, delta_za_low=-200.0, delta_za_high=1000.0)
        assert labels[0] == "corridor"
        assert labels[1] == "corridor"
        assert labels[2] == "overshoot"


class TestCorridorAccumulator:
    def test_init_creates_nan_envelopes(self) -> None:
        acc = CorridorAccumulator(energy_min=-6e6, energy_max=5e6, delta_za_restricted=200.0)
        assert acc.energy_bins.shape == (200,)
        assert np.all(np.isnan(acc.crash_max_pdyn))
        assert np.all(np.isnan(acc.restricted_max_pdyn))
        assert np.all(np.isnan(acc.restricted_min_pdyn))
        assert np.all(np.isnan(acc.capture_min_pdyn))

    def test_update_populates_envelopes(self) -> None:
        acc = CorridorAccumulator(energy_min=-6e6, energy_max=5e6, delta_za_restricted=200.0)
        trajs, labels = _make_trajectories_with_labels()
        acc.update(trajs, labels)
        assert not np.all(np.isnan(acc.crash_max_pdyn))
        assert not np.all(np.isnan(acc.capture_min_pdyn))

    def test_update_is_incremental(self) -> None:
        acc = CorridorAccumulator(energy_min=-6e6, energy_max=5e6, delta_za_restricted=200.0)
        trajs, labels = _make_trajectories_with_labels()
        acc.update(trajs, labels)
        crash_after_first = acc.crash_max_pdyn.copy()
        acc.update(trajs, labels)
        np.testing.assert_array_equal(acc.crash_max_pdyn, crash_after_first)

    def test_checkpoint_roundtrip(self) -> None:
        acc = CorridorAccumulator(energy_min=-6e6, energy_max=5e6, delta_za_restricted=200.0, delta_za_low=-200.0, delta_za_high=1000.0)
        trajs, labels = _make_trajectories_with_labels()
        acc.update(trajs, labels)
        state = acc.to_checkpoint()
        acc2 = CorridorAccumulator.from_checkpoint(state)
        np.testing.assert_array_equal(acc.crash_max_pdyn, acc2.crash_max_pdyn)
        np.testing.assert_array_equal(acc.restricted_max_pdyn, acc2.restricted_max_pdyn)
        np.testing.assert_array_equal(acc.restricted_min_pdyn, acc2.restricted_min_pdyn)
        np.testing.assert_array_equal(acc.capture_min_pdyn, acc2.capture_min_pdyn)
        assert acc2.delta_za_low == -200.0
        assert acc2.delta_za_high == 1000.0

    def test_to_corridor_data(self) -> None:
        acc = CorridorAccumulator(energy_min=-6e6, energy_max=5e6, delta_za_restricted=200.0)
        trajs, labels = _make_trajectories_with_labels()
        acc.update(trajs, labels)
        data = acc.to_corridor_data(nominal=trajs[0])
        assert "schema_version" in data
        assert int(data["schema_version"][0]) == 4
        assert "envelope_crash_pdyn" in data
        assert "envelope_restricted_max_pdyn" in data
        assert "envelope_restricted_min_pdyn" in data
        assert "envelope_capture_pdyn" in data


class TestCorridorCache:
    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "corridor.npz"
        data = {
            "schema_version": np.array([4]),
            "energy_bins": np.linspace(-6, 4, 50),
            "envelope_crash_pdyn": np.random.default_rng(1).random(50),
            "envelope_capture_pdyn": np.random.default_rng(2).random(50),
            "envelope_restricted_max_pdyn": np.random.default_rng(3).random(50),
            "envelope_restricted_min_pdyn": np.random.default_rng(4).random(50),
            "nominal": np.random.default_rng(5).random((100, 12)),
            "delta_za_km": np.array([200.0]),
        }
        save_corridor(data, path)
        loaded = load_corridor(path)
        assert loaded is not None
        assert loaded["schema_version"][0] == 4
        np.testing.assert_array_equal(loaded["energy_bins"], data["energy_bins"])

    def test_load_old_cache_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "old_corridor.npz"
        np.savez_compressed(str(path), nominal=np.zeros((10, 12)), traj_lengths=np.array([10]))
        loaded = load_corridor(path)
        assert loaded is None

    def test_load_v3_cache_returns_none(self, tmp_path: Path) -> None:
        """v3 caches from old compute_corridor are no longer supported."""
        path = tmp_path / "v3_corridor.npz"
        np.savez_compressed(str(path), schema_version=np.array([3]), energy_bins=np.linspace(-6, 4, 50))
        loaded = load_corridor(path)
        assert loaded is None

    def test_load_missing_file_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "nonexistent.npz"
        loaded = load_corridor(path)
        assert loaded is None
