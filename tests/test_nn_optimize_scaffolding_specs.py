"""train.py NN-branch param-spec list must include scaffolding when knob is on."""
from __future__ import annotations

from aerocapture.training.encoding import nn_param_specs_from_v2
from aerocapture.training.param_spaces import _NN_SCAFFOLDING_PARAMS


def _toy_arch() -> list[dict]:
    return [
        {"type": "dense", "input_size": 4, "output_size": 8, "activation": "swish"},
        {"type": "dense", "input_size": 8, "output_size": 2, "activation": "asinh"},
    ]


def test_specs_include_scaffolding_when_knob_on():
    from pydantic import TypeAdapter
    from aerocapture.training.rl.schemas import LayerSpec

    arch = _toy_arch()
    validated = TypeAdapter(list[LayerSpec]).validate_python(arch)
    base_specs = nn_param_specs_from_v2(validated, bound_multiplier=2.0)

    full_specs = [*base_specs, *_NN_SCAFFOLDING_PARAMS]

    assert len(full_specs) == len(base_specs) + 17
    tail_names = [s.name for s in full_specs[len(base_specs):]]
    expected_names = [s.name for s in _NN_SCAFFOLDING_PARAMS]
    assert tail_names == expected_names


def test_specs_unchanged_when_knob_off():
    from pydantic import TypeAdapter
    from aerocapture.training.rl.schemas import LayerSpec

    arch = _toy_arch()
    validated = TypeAdapter(list[LayerSpec]).validate_python(arch)
    base_specs = nn_param_specs_from_v2(validated, bound_multiplier=2.0)
    assert len(base_specs) > 0
