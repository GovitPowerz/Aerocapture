"""Behavioural gates for `[monte_carlo] noise_seeding` (PyO3 path).

The historical `legacy` mode seeds the per-sim stochastic streams (OU density
perturbation, EKF sensor noise) from `[simulation] random_seed + env_idx *
10_000`, so every n_sims=1 run shares ONE noise realization regardless of the
dispersion draw -- the conditioning defect the paper's Appendix E discloses.
`per_draw` derives the stream seed from the draw itself, so distinct scenarios
get distinct realizations while identical draws stay reproducible.
"""

from __future__ import annotations

import numpy as np
import pytest

aero = pytest.importorskip("aerocapture_rs")

# Base-inherits common.toml, whose [monte_carlo.density_perturbation] level is
# "low": the OU stream is active, so its realization is observable.
TOML = "configs/training/msr_aller_eqglide_train.toml"
DENSITY_PERTURBATION_COL = 16  # trajectory column (see BatchResults docs)
N_COMPARE = 100  # leading trajectory rows compared (runs differ in length)


def _perturbation_series(mode: str, seeds: list[int]) -> list[np.ndarray]:
    overrides = [{"simulation.n_sims": 1, "monte_carlo.seed": s, "monte_carlo.noise_seeding": mode} for s in seeds]
    res = aero.run_batch(TOML, overrides, include_trajectories=True)
    series = [np.asarray(t)[:N_COMPARE, DENSITY_PERTURBATION_COL] for t in res.trajectories]
    assert all(len(s) == N_COMPARE for s in series)
    return series


def test_legacy_shares_one_noise_path_across_seeds() -> None:
    a, b = _perturbation_series("legacy", [11, 12])
    assert np.array_equal(a, b), "legacy mode must reproduce the historical shared noise path"


def test_per_draw_gives_distinct_paths_to_distinct_draws() -> None:
    a, b = _perturbation_series("per_draw", [11, 12])
    assert not np.array_equal(a, b)


def test_per_draw_is_reproducible_for_an_identical_draw() -> None:
    (a,) = _perturbation_series("per_draw", [11])
    (b,) = _perturbation_series("per_draw", [11])
    assert np.array_equal(a, b)


def test_per_draw_differs_from_legacy_on_the_same_draw() -> None:
    (legacy,) = _perturbation_series("legacy", [11])
    (per_draw,) = _perturbation_series("per_draw", [11])
    assert not np.array_equal(legacy, per_draw)


def test_default_is_legacy() -> None:
    (explicit,) = _perturbation_series("legacy", [11])
    res = aero.run_batch(TOML, [{"simulation.n_sims": 1, "monte_carlo.seed": 11}], include_trajectories=True)
    implicit = np.asarray(res.trajectories[0])[:N_COMPARE, DENSITY_PERTURBATION_COL]
    assert np.array_equal(explicit, implicit)


def test_unknown_mode_is_rejected_at_load() -> None:
    with pytest.raises(RuntimeError, match="noise_seeding"):
        aero.run_batch(TOML, [{"simulation.n_sims": 1, "monte_carlo.seed": 11, "monte_carlo.noise_seeding": "per-draw"}])
