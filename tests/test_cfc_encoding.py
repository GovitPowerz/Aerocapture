"""ParamSpec width + bound checks for the CfC probe layer."""

from __future__ import annotations

from aerocapture.training.config import _layer_n_params, _layer_output_size
from aerocapture.training.encoding import _layer_param_specs
from aerocapture.training.rl.schemas import CfcSpec


def test_cfc_spec_width_matches_n_params() -> None:
    spec = CfcSpec(type="cfc", input_size=3, hidden_size=4, backbone_units=5)
    specs = _layer_param_specs(spec, layer_idx=0, bound_multiplier=2.0)
    # B(I+H) + B + 4(HB + H) = 5*7 + 5 + 4*24 = 136
    assert len(specs) == 136
    assert _layer_n_params(spec.model_dump()) == 136
    assert _layer_output_size(spec.model_dump()) == 4


def test_cfc_spec_order_starts_with_backbone() -> None:
    spec = CfcSpec(type="cfc", input_size=3, hidden_size=4, backbone_units=5)
    specs = _layer_param_specs(spec, layer_idx=0, bound_multiplier=1.0)
    assert specs[0].name.startswith("w_bb")
    assert specs[5 * 7].name.startswith("b_bb")
    assert specs[-1].name.startswith("b_tb")
