"""Encoding + config helper arms for TransformerSpec."""

from __future__ import annotations

from aerocapture.training.config import _layer_n_params, _layer_output_size
from aerocapture.training.encoding import _layer_param_specs
from aerocapture.training.rl.schemas import TransformerSpec


def test_transformer_n_params_formula() -> None:
    spec = TransformerSpec(type="transformer", d_model=32, n_heads=4, d_ffn=64, n_seq=64)
    # 4*d^2 + 2*ffn*d + ffn + 9*d
    expected = 4 * 32 * 32 + 2 * 64 * 32 + 64 + 9 * 32
    assert _layer_n_params(spec) == expected == 8544


def test_transformer_output_size_is_d_model() -> None:
    spec = TransformerSpec(type="transformer", d_model=32, n_heads=4, d_ffn=64, n_seq=64)
    assert _layer_output_size(spec) == 32


def test_transformer_param_specs_length_matches_n_params() -> None:
    spec = TransformerSpec(type="transformer", d_model=8, n_heads=2, d_ffn=16, n_seq=4)
    specs = _layer_param_specs(spec, bound_multiplier=1.0)
    assert len(specs) == _layer_n_params(spec)


def test_transformer_param_specs_xavier_bounds() -> None:
    from math import sqrt

    spec = TransformerSpec(type="transformer", d_model=8, n_heads=2, d_ffn=16, n_seq=4)
    specs = _layer_param_specs(spec, bound_multiplier=1.0)
    # First d_model*d_model = 64 entries are w_q; bound = sqrt(6/(2*d_model)) = sqrt(6/16)
    proj_bound = sqrt(6.0 / (2.0 * 8))
    assert abs(specs[0].p_min - (-proj_bound)) < 1e-12
    assert abs(specs[0].p_max - proj_bound) < 1e-12
    # After 4 projections (4 * (64 + 8) = 288 entries), we're at w_ffn1
    # w_ffn1 bound = sqrt(6 / (d_model + d_ffn)) = sqrt(6/24) = 0.5
    ffn_bound = sqrt(6.0 / (8 + 16))
    assert abs(specs[288].p_min - (-ffn_bound)) < 1e-12
    assert abs(specs[288].p_max - ffn_bound) < 1e-12
