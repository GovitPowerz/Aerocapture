"""Phase 2b Python WindowSpec + WindowLayer unit tests.

Covers: WindowSpec pydantic validation, LayerSpec discriminated union dispatch,
WindowLayer torch module forward + new_state dtype tracking, zero-parameter
invariant, build_layer PPO-rejection guard.
"""

from __future__ import annotations

import pytest
import torch
from aerocapture.training.rl.layers import WindowLayer, build_layer
from aerocapture.training.rl.schemas import LayerSpec, WindowSpec
from pydantic import TypeAdapter, ValidationError

# ── WindowSpec schema ───────────────────────────────────────────────────


def test_window_spec_constructs_with_required_fields() -> None:
    spec = WindowSpec(type="window", input_size=4, n_steps=8)
    assert spec.input_size == 4
    assert spec.n_steps == 8


def test_window_spec_rejects_zero_fields() -> None:
    with pytest.raises(ValidationError):
        WindowSpec(type="window", input_size=0, n_steps=8)
    with pytest.raises(ValidationError):
        WindowSpec(type="window", input_size=4, n_steps=0)


def test_window_spec_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        WindowSpec(type="window", input_size=4, n_steps=8, activation="tanh")  # type: ignore[call-arg]


def test_layer_spec_discriminator_dispatches_to_window() -> None:
    adapter: TypeAdapter[LayerSpec] = TypeAdapter(LayerSpec)
    parsed = adapter.validate_python({"type": "window", "input_size": 4, "n_steps": 8})
    assert isinstance(parsed, WindowSpec)
    assert parsed.input_size == 4
    assert parsed.n_steps == 8


# ── WindowLayer torch module ────────────────────────────────────────────


def test_window_forward_rolls_buffer_and_concatenates() -> None:
    layer = WindowLayer(input_size=2, n_steps=3).double()
    state = layer.new_state(batch_size=1)
    assert state.shape == (1, 3, 2)
    assert torch.all(state == 0.0)

    x0 = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    out0, state = layer.forward(x0, state)
    assert out0.shape == (1, 6)
    assert torch.equal(out0, torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 2.0]], dtype=torch.float64))

    x1 = torch.tensor([[3.0, 4.0]], dtype=torch.float64)
    out1, state = layer.forward(x1, state)
    assert torch.equal(out1, torch.tensor([[0.0, 0.0, 1.0, 2.0, 3.0, 4.0]], dtype=torch.float64))

    x2 = torch.tensor([[5.0, 6.0]], dtype=torch.float64)
    out2, state = layer.forward(x2, state)
    assert torch.equal(out2, torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=torch.float64))


def test_window_new_state_respects_module_dtype() -> None:
    layer = WindowLayer(input_size=4, n_steps=2)
    state_f32 = layer.new_state(batch_size=2)
    assert state_f32.dtype == torch.float32

    layer.double()
    state_f64 = layer.new_state(batch_size=2)
    assert state_f64.dtype == torch.float64


def test_window_has_zero_trainable_parameters() -> None:
    layer = WindowLayer(input_size=4, n_steps=8)
    n_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    assert n_params == 0


def test_window_rejects_invalid_input_shape() -> None:
    layer = WindowLayer(input_size=4, n_steps=3).double()
    state = layer.new_state(batch_size=1)
    bad_x = torch.zeros(1, 5, dtype=torch.float64)  # wrong input_size
    with pytest.raises(AssertionError):
        layer.forward(bad_x, state)


def test_window_rejects_invalid_state_shape() -> None:
    layer = WindowLayer(input_size=4, n_steps=3).double()
    x = torch.zeros(1, 4, dtype=torch.float64)
    bad_state = torch.zeros(1, 2, 4, dtype=torch.float64)  # wrong n_steps
    with pytest.raises(AssertionError):
        layer.forward(x, bad_state)


def test_window_layer_rejects_zero_construction_fields() -> None:
    with pytest.raises(ValueError):
        WindowLayer(input_size=0, n_steps=4)
    with pytest.raises(ValueError):
        WindowLayer(input_size=4, n_steps=0)


# ── build_layer PPO-rejection guard ─────────────────────────────────────


def test_window_layer_is_exported_from_layers_module() -> None:
    # Exposed for the cross-language equivalence test.
    assert WindowLayer is not None


def test_build_layer_raises_on_window_spec() -> None:
    spec = WindowSpec(type="window", input_size=4, n_steps=8)
    with pytest.raises(NotImplementedError) as exc_info:
        build_layer(spec)
    msg = str(exc_info.value)
    assert "Window-MLP is PSO-only" in msg
    assert "docs/superpowers/specs/2026-04-20-phase-2b-window-mlp-design.md" in msg
