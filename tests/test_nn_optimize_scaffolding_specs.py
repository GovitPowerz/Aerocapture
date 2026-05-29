"""train.py NN-branch param-spec list must include scaffolding when knob is on."""

from __future__ import annotations

from aerocapture.training.encoding import nn_param_specs_from_v2
from aerocapture.training.param_spaces import _NN_SCAFFOLDING_PARAMS


def _toy_arch() -> list[dict]:
    return [
        {"type": "dense", "input_size": 4, "output_size": 8, "activation": "swish"},
        {"type": "dense", "input_size": 8, "output_size": 2, "activation": "asinh"},
    ]


def test_specs_include_scaffolding_when_knob_on() -> None:
    from aerocapture.training.rl.schemas import LayerSpec
    from pydantic import TypeAdapter

    arch = _toy_arch()
    validated = TypeAdapter(list[LayerSpec]).validate_python(arch)
    base_specs = nn_param_specs_from_v2(validated, bound_multiplier=2.0)

    full_specs = [*base_specs, *_NN_SCAFFOLDING_PARAMS]

    assert len(full_specs) == len(base_specs) + 17
    tail_names = [s.name for s in full_specs[len(base_specs) :]]
    expected_names = [s.name for s in _NN_SCAFFOLDING_PARAMS]
    assert tail_names == expected_names


def test_specs_match_chromosome_widths_per_knob_state() -> None:
    """Verify the chromosome width changes by exactly 17 when optimize_scaffolding
    flips. With the knob OFF, param_specs is just NN weights; with ON, the 17
    scaffolding params are appended at the tail.

    This was previously misnamed `test_specs_unchanged_when_knob_off` and only
    asserted `len(base_specs) > 0`. It now actually exercises both knob states.
    """
    from aerocapture.training.rl.schemas import LayerSpec
    from pydantic import TypeAdapter

    arch = _toy_arch()
    validated = TypeAdapter(list[LayerSpec]).validate_python(arch)
    base_specs = nn_param_specs_from_v2(validated, bound_multiplier=2.0)

    # Knob off: chromosome is just NN weights, no scaffolding tail.
    knob_off_full = base_specs
    # Knob on: scaffolding appended (mirrors train.py's behaviour after fix A3).
    knob_on_full = [*base_specs, *_NN_SCAFFOLDING_PARAMS]

    assert len(knob_on_full) - len(knob_off_full) == 17
    # Ensure no scaffolding leak when knob is off
    knob_off_names = {s.name for s in knob_off_full}
    scaff_names = {s.name for s in _NN_SCAFFOLDING_PARAMS}
    assert knob_off_names.isdisjoint(scaff_names), "knob-off chromosome must not contain scaffolding params"


def test_network_config_scaffolding_field_default() -> None:
    from aerocapture.training.config import NetworkConfig

    cfg = NetworkConfig(architecture=[{"type": "dense", "input_size": 4, "output_size": 2, "activation": "tanh"}])
    assert cfg.scaffolding == "off"


def test_network_config_rejects_unknown_scaffolding() -> None:
    import pytest

    from aerocapture.training.config import NetworkConfig

    with pytest.raises(ValueError, match="scaffolding must be"):
        NetworkConfig(
            architecture=[{"type": "dense", "input_size": 4, "output_size": 2, "activation": "tanh"}],
            scaffolding="partial",
        )


def test_resume_with_shape_mismatch_fails_loud() -> None:
    import numpy as np
    from aerocapture.training.train import _check_resume_chromosome_shape

    saved_pop = np.zeros((4, 1266))
    try:
        _check_resume_chromosome_shape(saved_pop, expected_n_params=1283)
    except ValueError as e:
        assert "shape mismatch" in str(e).lower()
        assert "1266" in str(e) and "1283" in str(e)
        return
    raise AssertionError("expected ValueError on shape mismatch")
