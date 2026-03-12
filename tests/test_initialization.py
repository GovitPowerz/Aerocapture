"""Tests for NN weight initialization functions."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aerocapture.training.initialization import compute_layer_bound, generate_initialized_weights


class TestComputeLayerBound:
    def test_xavier_tanh(self) -> None:
        assert compute_layer_bound(6, 12, "tanh") == pytest.approx(math.sqrt(6 / 18))

    def test_xavier_sigmoid(self) -> None:
        assert compute_layer_bound(6, 12, "sigmoid") == pytest.approx(math.sqrt(6 / 18))

    def test_xavier_asinh(self) -> None:
        assert compute_layer_bound(12, 2, "asinh") == pytest.approx(math.sqrt(6 / 14))

    def test_he_relu(self) -> None:
        assert compute_layer_bound(6, 64, "relu") == pytest.approx(math.sqrt(6 / 6))

    def test_lecun_linear(self) -> None:
        assert compute_layer_bound(32, 2, "linear") == pytest.approx(math.sqrt(3 / 32))

    def test_unknown_activation_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown activation"):
            compute_layer_bound(6, 12, "swish")
