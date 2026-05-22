"""_select_best_teacher_per_seed picks lowest-DV captured trajectory per seed,
drops seeds with no captures."""

import numpy as np

from aerocapture.training.warm_start import _select_best_teacher_per_seed


def _traj(seed, dv, captured, n_steps=10):
    return {
        "seed": seed,
        "X": np.zeros((n_steps, 21)),
        "y_signed": np.zeros(n_steps),
        "dv": dv,
        "captured": captured,
    }


def test_picks_lowest_dv_captured_per_seed():
    by_scheme = {
        "ftc": [_traj(1, 100.0, True), _traj(2, 200.0, True)],
        "fnpag": [_traj(1, 50.0, True), _traj(2, 250.0, True)],
    }
    selected = _select_best_teacher_per_seed(by_scheme)
    by_seed = {s["seed"]: s for s in selected}
    assert by_seed[1]["scheme"] == "fnpag"
    assert by_seed[1]["dv"] == 50.0
    assert by_seed[2]["scheme"] == "ftc"
    assert by_seed[2]["dv"] == 200.0


def test_drops_seeds_with_no_captures():
    by_scheme = {
        "ftc": [_traj(1, 100.0, True), _traj(2, 999.0, False)],
        "fnpag": [_traj(1, 50.0, True), _traj(2, 888.0, False)],
    }
    selected = _select_best_teacher_per_seed(by_scheme)
    by_seed = {s["seed"]: s for s in selected}
    assert 1 in by_seed
    assert 2 not in by_seed


def test_mixed_capture_falls_back_to_captured_only():
    """A seed where one scheme captures and another fails: capture wins
    even if its DV is higher than the failure's nominal DV."""
    by_scheme = {
        "ftc": [_traj(1, 999.0, False)],   # failed, high DV
        "fnpag": [_traj(1, 500.0, True)],  # captured
    }
    selected = _select_best_teacher_per_seed(by_scheme)
    assert len(selected) == 1
    assert selected[0]["scheme"] == "fnpag"


def test_ignores_seeds_outside_intersection_gracefully():
    """If schemes have different seed coverage, union is taken."""
    by_scheme = {
        "ftc": [_traj(1, 100.0, True)],
        "fnpag": [_traj(2, 50.0, True)],
    }
    selected = _select_best_teacher_per_seed(by_scheme)
    assert len(selected) == 2
