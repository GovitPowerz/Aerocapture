"""build_layer accepts all six layer specs for warm-start training."""

import pytest
from aerocapture.training.rl.layers import build_layer
from aerocapture.training.rl.schemas import (
    DenseSpec,
    GruSpec,
    LstmSpec,
    MambaSpec,
    TransformerSpec,
    WindowSpec,
)


@pytest.mark.parametrize(
    "spec",
    [
        DenseSpec(type="dense", input_size=4, output_size=2, activation="tanh"),
        GruSpec(type="gru", input_size=4, hidden_size=8),
        LstmSpec(type="lstm", input_size=4, hidden_size=8),
        WindowSpec(type="window", input_size=4, n_steps=3),
        TransformerSpec(type="transformer", d_model=8, n_heads=2, d_ffn=16, n_seq=4),
        MambaSpec(type="mamba", input_size=8, d_state=4, dt_rank=2),
    ],
)
def test_build_layer_constructs_all_types(spec):
    layer = build_layer(spec)
    assert layer is not None


@pytest.mark.parametrize(
    "spec",
    [
        DenseSpec(type="dense", input_size=4, output_size=2, activation="tanh"),
        GruSpec(type="gru", input_size=4, hidden_size=8),
        LstmSpec(type="lstm", input_size=4, hidden_size=8),
        WindowSpec(type="window", input_size=4, n_steps=3),
        TransformerSpec(type="transformer", d_model=8, n_heads=2, d_ffn=16, n_seq=4),
        MambaSpec(type="mamba", input_size=8, d_state=4, dt_rank=2),
    ],
)
def test_new_state_accepts_batch_size_and_device(spec):
    layer = build_layer(spec)
    state = layer.new_state(batch_size=2, device=None)
    assert state is None or state is not None  # contract: callable without error
