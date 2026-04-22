"""Assert build_layer raises NotImplementedError for TransformerSpec."""

from __future__ import annotations

import pytest
from aerocapture.training.rl.layers import build_layer
from aerocapture.training.rl.schemas import TransformerSpec


def test_build_layer_rejects_transformer_spec() -> None:
    spec = TransformerSpec(type="transformer", d_model=4, n_heads=2, d_ffn=8, n_seq=4)
    with pytest.raises(NotImplementedError, match="Transformer is PSO-only in Phase 3a"):
        build_layer(spec)
