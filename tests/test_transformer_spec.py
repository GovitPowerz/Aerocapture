"""TransformerSpec pydantic schema + LayerSpec discriminated-union tests."""

from __future__ import annotations

import pytest
from aerocapture.training.rl.schemas import LayerSpec, TransformerSpec
from pydantic import TypeAdapter, ValidationError


def test_transformer_spec_validates_shapes() -> None:
    spec = TransformerSpec(type="transformer", d_model=32, n_heads=4, d_ffn=64, n_seq=64)
    assert spec.d_model == 32
    assert spec.n_heads == 4


def test_transformer_spec_rejects_non_divisible_heads() -> None:
    with pytest.raises(ValidationError):
        TransformerSpec(type="transformer", d_model=33, n_heads=4, d_ffn=64, n_seq=64)


def test_transformer_spec_rejects_zero_fields() -> None:
    for kwargs in [
        dict(d_model=0, n_heads=1, d_ffn=1, n_seq=1),
        dict(d_model=4, n_heads=0, d_ffn=1, n_seq=1),
        dict(d_model=4, n_heads=2, d_ffn=0, n_seq=1),
        dict(d_model=4, n_heads=2, d_ffn=8, n_seq=0),
    ]:
        with pytest.raises(ValidationError):
            TransformerSpec(type="transformer", **kwargs)  # type: ignore[arg-type]


def test_layerspec_discriminates_transformer() -> None:
    raw = {"type": "transformer", "d_model": 8, "n_heads": 2, "d_ffn": 16, "n_seq": 4}
    parsed: LayerSpec = TypeAdapter(LayerSpec).validate_python(raw)
    assert isinstance(parsed, TransformerSpec)
