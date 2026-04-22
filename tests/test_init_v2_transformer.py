"""init_v2_population Transformer arm."""

from __future__ import annotations

import numpy as np
from aerocapture.training.initialization_v2 import init_v2_population
from aerocapture.training.rl.schemas import DenseSpec, TransformerSpec


def test_init_v2_population_transformer_slab_shape_and_bounds() -> None:
    architecture = [
        DenseSpec(type="dense", input_size=8, output_size=4, activation="linear"),
        TransformerSpec(type="transformer", d_model=4, n_heads=2, d_ffn=8, n_seq=4),
        DenseSpec(type="dense", input_size=4, output_size=2, activation="linear"),
    ]
    n_pop = 16
    rng = np.random.default_rng(42)
    pop = init_v2_population(architecture, n_pop, bound_multiplier=1.0, rng=rng)
    # Dense 0: 8*4 + 4 = 36
    # Transformer: 4*16 + 2*32 + 8 + 36 = 172
    # Dense 2: 4*2 + 2 = 10
    # Total: 218
    assert pop.shape == (n_pop, 218)
    assert np.all(np.isfinite(pop))


def test_init_v2_population_transformer_only_matches_specs_length() -> None:
    # Pure Transformer -- slab width must equal _transformer_specs length.
    from aerocapture.training.encoding import _layer_param_specs

    spec = TransformerSpec(type="transformer", d_model=8, n_heads=2, d_ffn=16, n_seq=4)
    architecture = [spec]
    n_pop = 8
    rng = np.random.default_rng(0)
    pop = init_v2_population(architecture, n_pop, bound_multiplier=1.0, rng=rng)
    expected = len(_layer_param_specs(spec, bound_multiplier=1.0))
    assert pop.shape == (n_pop, expected)
