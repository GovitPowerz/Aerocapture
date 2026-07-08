"""Mamba3 population init: correct slab width + finite, for all flag combos."""

from __future__ import annotations

import numpy as np
from aerocapture.training.config import _layer_n_params
from aerocapture.training.initialization_v2 import init_v2_population


def test_init_v2_mamba3_shape_and_finite() -> None:
    for disc in ("euler", "trapezoidal"):
        for sm in ("real", "complex"):
            arch = [
                {"type": "dense", "input_size": 23, "output_size": 16, "activation": "swish"},
                {"type": "mamba3", "input_size": 16, "d_state": 8, "dt_rank": 1, "discretization": disc, "state_mode": sm},
                {"type": "dense", "input_size": 16, "output_size": 2, "activation": "linear"},
            ]
            expected = sum(_layer_n_params(e) for e in arch)
            pop = init_v2_population(arch, n_pop=8, bound_multiplier=2.0, rng=np.random.default_rng(0))
            assert pop.shape == (8, expected), (disc, sm)
            assert np.isfinite(pop).all(), (disc, sm)
