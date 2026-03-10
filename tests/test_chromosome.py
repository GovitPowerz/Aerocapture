"""Tests for chromosome encode/decode roundtrip.

Covers:
- encode_params_to_chromosome / decode_params_from_chromosome roundtrip
- boundary conditions (all-zeros → p_min, all-ones → p_max)
- property: decoded values always within [p_min, p_max]
"""

from __future__ import annotations

import numpy as np
import pytest
from aerocapture.training.evaluate import decode_params_from_chromosome
from aerocapture.training.param_spaces import PARAM_SPACES
from aerocapture.training.population import encode_params_to_chromosome
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.fixtures.factories import make_chromosome, make_training_config

ALL_SCHEMES = list(PARAM_SPACES.keys())


class TestChromosomeRoundtrip:
    @pytest.mark.parametrize("scheme", ALL_SCHEMES)
    def test_roundtrip_preserves_values(self, scheme: str) -> None:
        """Encode default params → decode → values match within 1% relative tolerance."""
        config = make_training_config(scheme)
        specs = PARAM_SPACES[scheme]
        defaults = {s.name: s.default for s in specs}

        chrom = encode_params_to_chromosome(defaults, config)
        decoded = decode_params_from_chromosome(chrom, config)

        for spec in specs:
            expected = defaults[spec.name]
            actual = decoded[spec.name]
            # Relative tolerance 1% plus absolute floor — handles params whose default is 0.
            # Quantisation error from n_bit=16 is ~0.002% of the full range.
            p_range = spec.p_max - spec.p_min
            abs_tol = max(0.01 * abs(expected), 0.001 * p_range)
            assert abs(actual - expected) <= abs_tol, f"scheme={scheme} param={spec.name}: expected {expected}, got {actual} (tol={abs_tol:.2e})"

    @pytest.mark.parametrize("scheme", ALL_SCHEMES)
    def test_all_zeros_gives_minimum(self, scheme: str) -> None:
        """All-zero chromosome should decode to (near) p_min for each parameter."""
        config = make_training_config(scheme)
        specs = PARAM_SPACES[scheme]
        chrom_len = len(specs) * config.ga.n_bit
        chrom = make_chromosome(chrom_len, strategy="zeros")

        decoded = decode_params_from_chromosome(chrom, config)

        for spec in specs:
            expected = spec.p_min  # log-scale: 10^log_min = p_min
            actual = decoded[spec.name]
            assert abs(actual - expected) <= 0.01 * abs(expected) + 1e-12, f"scheme={scheme} param={spec.name}: zeros → expected p_min={expected}, got {actual}"

    @pytest.mark.parametrize("scheme", ALL_SCHEMES)
    def test_all_ones_gives_maximum(self, scheme: str) -> None:
        """All-one chromosome should decode to (near) p_max for each parameter."""
        config = make_training_config(scheme)
        specs = PARAM_SPACES[scheme]
        chrom_len = len(specs) * config.ga.n_bit
        chrom = make_chromosome(chrom_len, strategy="ones")

        decoded = decode_params_from_chromosome(chrom, config)

        for spec in specs:
            expected = spec.p_max
            actual = decoded[spec.name]
            assert abs(actual - expected) <= 0.01 * abs(expected) + 1e-12, f"scheme={scheme} param={spec.name}: ones → expected p_max={expected}, got {actual}"


class TestChromosomeProperties:
    @pytest.mark.parametrize("scheme", ALL_SCHEMES)
    @given(data=st.data())
    @settings(max_examples=30)
    def test_decoded_params_respect_bounds(self, scheme: str, data: st.DataObject) -> None:
        """Any binary chromosome decodes to values within [p_min, p_max]."""
        config = make_training_config(scheme)
        specs = PARAM_SPACES[scheme]
        chrom_len = len(specs) * config.ga.n_bit

        bits = data.draw(
            st.lists(st.integers(0, 1), min_size=chrom_len, max_size=chrom_len),
        )
        chrom = np.array(bits, dtype=np.int8)

        decoded = decode_params_from_chromosome(chrom, config)

        for spec in specs:
            val = decoded[spec.name]
            assert spec.p_min <= val <= spec.p_max + 1e-9, f"scheme={scheme} param={spec.name}: {val} outside [{spec.p_min}, {spec.p_max}]"
