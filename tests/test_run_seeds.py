"""run_seeds byte-identity gates: one override set flown on a seed pool through the
seed-axis kernel must reproduce the per-seed `run_batch` path (one override dict per
seed, `monte_carlo.seed` applied last) byte for byte: final records, dispersions,
capture flags and trajectories, under both noise regimes and at any thread count.
Extends ADR-0004's chokepoint to deploy-side evaluation."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

aero = pytest.importorskip("aerocapture_rs")

EQGLIDE_TOML = "configs/training/msr_aller_eqglide_train.toml"
FNPAG_TOML = "configs/training/msr_aller_fnpag_train.toml"
MAMBA_TOML = "configs/training/sweep/mamba_p962.toml"
MAMBA_MODEL = "articles/paper/data/runs/headline/mamba_p962/best_model.json"

SEEDS = list(range(7_300_000, 7_300_000 + 6))
REGIMES = ["per_draw", "legacy"]

# (toml, the override set every seed shares)
CASES: dict[str, tuple[str, dict[str, object]]] = {
    # common.toml: dispersions + Gauss-Markov density perturbation.
    "eqglide_classical": (EQGLIDE_TOML, {"guidance.equilibrium_glide.k_hdot_scale": 0.35}),
    "eqglide_adaptive_events": (EQGLIDE_TOML, {"integration.mode": "adaptive"}),
    "fnpag_predictor": (FNPAG_TOML, {"navigation.density_filter_gain": 0.6}),
    # EKF navigation: the sensor-noise streams ride the same per-sim seed as the density noise.
    "eqglide_ekf_noise_streams": (EQGLIDE_TOML, {"navigation.mode": "ekf"}),
    # A deployed stateful NN: the model is read from disk once instead of once per seed.
    "mamba_nn_cell": (MAMBA_TOML, {"data.neural_network": MAMBA_MODEL, "guidance.command_shaping.max_bank_acceleration": 8.0}),
}


def _per_seed(toml: str, overrides: dict[str, object], seeds: list[int], *, include_trajectories: bool = True) -> object:
    """The path every deploy-side number took before run_seeds."""
    batch = [{"simulation.n_sims": 1, **overrides, "monte_carlo.seed": s} for s in seeds]
    return aero.run_batch(toml, batch, n_threads=1, include_trajectories=include_trajectories)


def _assert_same_bytes(new: object, old: object) -> None:
    assert len(new) == len(old)  # type: ignore[arg-type]
    for name in ("final_records", "dispersions", "captured"):
        a, b = np.asarray(getattr(new, name)), np.asarray(getattr(old, name))
        assert a.shape == b.shape and a.dtype == b.dtype, name
        assert a.tobytes() == b.tobytes(), f"{name} differs from the per-seed run_batch path"
    new_traj, old_traj = new.trajectories, old.trajectories  # type: ignore[attr-defined]
    assert len(new_traj) == len(old_traj)
    for i, (a, b) in enumerate(zip(new_traj, old_traj, strict=True)):
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape, f"trajectory {i} shape"
        assert a.tobytes() == b.tobytes(), f"trajectory {i} differs from the per-seed run_batch path"


@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize("case", list(CASES))
def test_run_seeds_byte_identical_to_per_seed_run_batch(case: str, regime: str) -> None:
    toml, overrides = CASES[case]
    overrides = {**overrides, "monte_carlo.noise_seeding": regime}
    old = _per_seed(toml, overrides, SEEDS)
    # Non-degenerate: every seed flew its own trajectory with a non-empty photo.
    rows = np.asarray(old.final_records)  # type: ignore[attr-defined]
    assert len({r.tobytes() for r in rows}) == len(SEEDS)
    assert all(np.asarray(t).shape[0] > 0 for t in old.trajectories)  # type: ignore[attr-defined]
    for n_threads in (1, 3, os.cpu_count() or 1, None):
        new = aero.run_seeds(toml, SEEDS, overrides, n_threads=n_threads, include_trajectories=True)
        _assert_same_bytes(new, old)


def test_run_seeds_without_trajectories_returns_empty_ones() -> None:
    toml, overrides = CASES["eqglide_classical"]
    old = _per_seed(toml, overrides, SEEDS, include_trajectories=False)
    new = aero.run_seeds(toml, SEEDS, overrides)
    _assert_same_bytes(new, old)
    assert all(np.asarray(t).shape == (0, 17) for t in new.trajectories)


def test_run_seeds_on_a_config_without_a_monte_carlo_section(tmp_path: Path) -> None:
    # The per-seed path's `monte_carlo.seed` override creates the section; the kernel
    # must build the same config, not fall back to "no dispersion config".
    cfg = tmp_path / "no_mc.toml"
    cfg.write_text(f'base = ["{Path("configs/missions/mars.toml").resolve()}"]\n[guidance]\ntype = "equilibrium_glide"\n')
    path = str(cfg)
    assert "monte_carlo" not in aero.load_config(path)
    old = _per_seed(path, {}, SEEDS[:2])
    new = aero.run_seeds(path, SEEDS[:2], include_trajectories=True)
    _assert_same_bytes(new, old)


def test_run_seeds_empty_pool() -> None:
    res = aero.run_seeds(EQGLIDE_TOML, [])
    assert len(res) == 0
    assert np.asarray(res.final_records).shape == (0, int(aero.FINAL_RECORD_LEN))
    with pytest.raises(RuntimeError, match="Cannot read"):  # the config is still read, as run_batch reads it
        aero.run_seeds("configs/test/does_not_exist_xyz.toml", [])


class TestRunSeedsOwnedKeys:
    """run_seeds owns the seed axis and one sim per seed, like run_grid."""

    @pytest.mark.parametrize("key", ["monte_carlo.seed", "simulation.n_sims"])
    def test_rejects_owned_key(self, key: str) -> None:
        with pytest.raises(ValueError, match=key):
            aero.run_seeds(EQGLIDE_TOML, [7_300_000], {key: 1})
