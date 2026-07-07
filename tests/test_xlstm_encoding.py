"""ParamSpec width + forget-bound checks for the sLSTM/mLSTM probe layers."""

from __future__ import annotations

from aerocapture.training.config import _layer_n_params, _layer_output_size
from aerocapture.training.encoding import _layer_param_specs
from aerocapture.training.rl.schemas import MlstmSpec, SlstmSpec


def test_slstm_width_and_forget_slice_bounds() -> None:
    spec = SlstmSpec(type="slstm", input_size=3, hidden_size=4)
    specs = _layer_param_specs(spec, layer_idx=0, bound_multiplier=1.0)
    # 4HI + 4HH + 4H = 48 + 64 + 16 = 128
    assert len(specs) == 128
    assert _layer_n_params(spec.model_dump()) == 128
    assert _layer_output_size(spec.model_dump()) == 4
    bias = specs[48 + 64 :]
    h = 4
    for j, ps in enumerate(bias):
        if h <= j < 2 * h:  # forget slice (gate order i, f, z, o)
            assert ps.p_max == 3.0, f"forget bias {j} bound {ps.p_max}"
        else:
            assert ps.p_max == 0.1, f"bias {j} bound {ps.p_max}"


def test_mlstm_width_and_forget_bound() -> None:
    spec = MlstmSpec(type="mlstm", input_size=3, hidden_size=4)
    specs = _layer_param_specs(spec, layer_idx=0, bound_multiplier=1.0)
    # 4(HI + H) + 2(I + 1) = 64 + 8 = 72
    assert len(specs) == 72
    assert _layer_n_params(spec.model_dump()) == 72
    assert _layer_output_size(spec.model_dump()) == 4
    assert specs[-1].name.startswith("b_f")
    assert specs[-1].p_max == 3.0  # wide bound for the +2.0 forget center
    assert specs[-(3 + 1) - 1].name.startswith("b_i")  # b_i sits before w_f (len I=3) + b_f
    assert specs[-5].p_max == 0.1
