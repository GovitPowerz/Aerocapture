"""Tests for real-valued encoding roundtrip.

Covers:
- encode_to_normalized / decode_normalized roundtrip
- boundary conditions (all-zeros -> p_min, all-ones -> p_max)
- property: decoded values always within [p_min, p_max]
"""

from __future__ import annotations

import numpy as np
import pytest
from aerocapture.training.encoding import decode_normalized, encode_to_normalized
from aerocapture.training.param_spaces import PARAM_SPACES
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.fixtures.factories import make_normalized_individual

ALL_SCHEMES = list(PARAM_SPACES.keys())


class TestEncodingRoundtrip:
    @pytest.mark.parametrize("scheme", ALL_SCHEMES)
    def test_roundtrip_preserves_values(self, scheme: str) -> None:
        """Encode default params -> decode -> values match exactly."""
        specs = PARAM_SPACES[scheme]
        defaults = {s.name: s.default for s in specs}

        x = encode_to_normalized(defaults, specs)
        decoded = decode_normalized(x, specs)

        for spec in specs:
            expected = defaults[spec.name]
            actual = decoded[spec.name]
            # Real-valued encoding should be exact (no quantization)
            rel_tol = 1e-10
            abs_tol = max(rel_tol * abs(expected), 1e-12)
            assert abs(actual - expected) <= abs_tol, f"scheme={scheme} param={spec.name}: expected {expected}, got {actual}"

    @pytest.mark.parametrize("scheme", ALL_SCHEMES)
    def test_all_zeros_gives_minimum(self, scheme: str) -> None:
        """All-zero normalized vector should decode to p_min for each parameter."""
        specs = PARAM_SPACES[scheme]
        n_params = len(specs)
        x = make_normalized_individual(n_params, strategy="zeros")

        decoded = decode_normalized(x, specs)

        for spec in specs:
            expected = spec.p_min
            actual = decoded[spec.name]
            assert abs(actual - expected) <= 1e-12 + 1e-10 * abs(expected), (
                f"scheme={scheme} param={spec.name}: zeros -> expected p_min={expected}, got {actual}"
            )

    @pytest.mark.parametrize("scheme", ALL_SCHEMES)
    def test_all_ones_gives_maximum(self, scheme: str) -> None:
        """All-one normalized vector should decode to p_max for each parameter."""
        specs = PARAM_SPACES[scheme]
        n_params = len(specs)
        x = make_normalized_individual(n_params, strategy="ones")

        decoded = decode_normalized(x, specs)

        for spec in specs:
            expected = spec.p_max
            actual = decoded[spec.name]
            assert abs(actual - expected) <= 1e-12 + 1e-10 * abs(expected), (
                f"scheme={scheme} param={spec.name}: ones -> expected p_max={expected}, got {actual}"
            )


class TestEncodingProperties:
    @pytest.mark.parametrize("scheme", ALL_SCHEMES)
    @given(data=st.data())
    @settings(max_examples=30)
    def test_decoded_params_respect_bounds(self, scheme: str, data: st.DataObject) -> None:
        """Any normalized vector in [0, 1] decodes to values within [p_min, p_max]."""
        specs = PARAM_SPACES[scheme]
        n_params = len(specs)

        values = data.draw(
            st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=n_params, max_size=n_params),
        )
        x = np.array(values, dtype=np.float64)

        decoded = decode_normalized(x, specs)

        for spec in specs:
            val = decoded[spec.name]
            assert spec.p_min <= val <= spec.p_max + 1e-9, f"scheme={scheme} param={spec.name}: {val} outside [{spec.p_min}, {spec.p_max}]"
