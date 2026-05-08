"""Regression guard: NN scaffolding param names + ordering match the deploy chromosome layout.

The pin against literal expected_names ensures that reordering the source
_NAV_PARAMS, _LATERAL_PARAMS, _EXIT_PARAMS, _THERMAL_LIMITER_PARAMS, or
_SHAPING_PARAMS lists is detected — a comparison against `[*_NAV_PARAMS, ...]`
would change on both sides simultaneously and silently allow the reorder.

Why this matters: the chromosome layout for NN+optimize_scaffolding is
fixed by the order here. compare_guidance.py and report.py read
best_params.json and route by name prefix; if the order shifts after a
PSO run is checkpointed, the resume path's _check_resume_chromosome_shape
catches the width mismatch but a rename or reorder within the same width
would silently corrupt deploy.
"""

from __future__ import annotations

from aerocapture.training.param_spaces import _NN_SCAFFOLDING_PARAMS

# Pin to the literal expected ordering. Updating this list signals an
# intentional schema change that breaks compatibility with any cached
# warm_start_chromosome.npy or any best_params.json from prior training.
_EXPECTED_SCAFFOLDING_NAMES: list[str] = [
    # _NAV_PARAMS (2)
    "nav.density_filter_gain",
    "nav.density_gain_max_delta",
    # _LATERAL_PARAMS (6)
    "lateral.tau",
    "lateral.threshold",
    "lateral.min_reversal_interval",
    "lateral.lateral_activation",
    "lateral.lateral_inhibition",
    "lateral.max_reversals",
    # _EXIT_PARAMS (4)
    "exit.exit_velocity_threshold",
    "exit.exit_pdyn_margin",
    "exit.exit_radial_vel_gain",
    "exit.exit_altitude_threshold",
    # _THERMAL_LIMITER_PARAMS (4)
    "thermal.heat_flux_activation",
    "thermal.heat_load_activation",
    "thermal.heat_flux_ramp_exponent",
    "thermal.heat_load_ramp_exponent",
    # _SHAPING_PARAMS (1)
    "shaping.max_bank_acceleration",
]


def test_nn_scaffolding_matches_expected_names_in_expected_order():
    actual_names = [s.name for s in _NN_SCAFFOLDING_PARAMS]
    assert actual_names == _EXPECTED_SCAFFOLDING_NAMES, (
        "scaffolding chromosome layout drifted. If this is intentional, update "
        "_EXPECTED_SCAFFOLDING_NAMES AND understand that any cached "
        "warm_start_chromosome.npy or shipped best_params.json from prior runs "
        "is now incompatible. compare_guidance.py and report.py routing relies "
        "on stable name prefixes."
    )


def test_nn_scaffolding_has_seventeen_params():
    assert len(_NN_SCAFFOLDING_PARAMS) == 17


def test_nn_scaffolding_names_are_unique_and_prefixed():
    names = [s.name for s in _NN_SCAFFOLDING_PARAMS]
    assert len(set(names)) == len(names), "duplicate names"
    valid = ("nav.", "lateral.", "exit.", "thermal.", "shaping.")
    for name in names:
        assert name.startswith(valid), f"unexpected prefix in {name!r}"
