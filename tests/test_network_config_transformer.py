"""NetworkConfig end-to-end through __post_init__ with Transformer layers.

Regression for the Phase 3a critical bug: _layer_input_size did not recognize
transformer entries (which have d_model, not input_size). Both dict-form and
Pydantic-form entries must work, since train.py uses dicts and other callers
may use Pydantic LayerSpec objects.
"""

from __future__ import annotations

from aerocapture.training.config import (
    NetworkConfig,
    _layer_input_size,
    _layer_output_size,
    describe_architecture,
)
from aerocapture.training.rl.schemas import DenseSpec, TransformerSpec


def _transformer_arch() -> list[dict]:
    return [
        {"type": "dense", "input_size": 23, "output_size": 32, "activation": "linear"},
        {"type": "transformer", "d_model": 32, "n_heads": 4, "d_ffn": 64, "n_seq": 64},
        {"type": "dense", "input_size": 32, "output_size": 2, "activation": "linear"},
    ]


def test_network_config_post_init_accepts_transformer_dict_arch() -> None:
    nc = NetworkConfig()
    nc.architecture = _transformer_arch()
    # Must not raise -- the Phase 3a training config takes this exact path.
    nc.__post_init__()


def test_layer_input_size_transformer_dict() -> None:
    entry = {"type": "transformer", "d_model": 32, "n_heads": 4, "d_ffn": 64, "n_seq": 64}
    assert _layer_input_size(entry) == 32  # d_model


def test_layer_input_size_transformer_pydantic() -> None:
    spec = TransformerSpec(type="transformer", d_model=32, n_heads=4, d_ffn=64, n_seq=64)
    assert _layer_input_size(spec) == 32


def test_layer_output_size_transformer_dict() -> None:
    entry = {"type": "transformer", "d_model": 32, "n_heads": 4, "d_ffn": 64, "n_seq": 64}
    assert _layer_output_size(entry) == 32


def test_describe_architecture_transformer_dict() -> None:
    desc = describe_architecture(_transformer_arch())
    assert "Transformer" in desc or "transformer" in desc
    assert "d_model=32" in desc
    assert "n_heads=4" in desc


def test_describe_architecture_transformer_pydantic() -> None:
    architecture = [
        DenseSpec(type="dense", input_size=23, output_size=32, activation="linear"),
        TransformerSpec(type="transformer", d_model=32, n_heads=4, d_ffn=64, n_seq=64),
        DenseSpec(type="dense", input_size=32, output_size=2, activation="linear"),
    ]
    desc = describe_architecture(architecture)
    assert "transformer" in desc
    assert "d_model=32" in desc
    assert "n_heads=4" in desc
