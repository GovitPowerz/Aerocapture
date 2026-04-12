"""Real-valued initial population generation for pymoo optimization."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from aerocapture.training.encoding import encode_to_normalized, nn_param_specs_from_architecture
from aerocapture.training.initialization import generate_initialized_weights
from aerocapture.training.param_spaces import ParamSpec


def create_initial_population(
    specs: list[ParamSpec],
    n_pop: int,
    rng: np.random.Generator,
    seed_defaults: bool = True,
    seed_params: dict[str, float] | None = None,
    perturbation_scale: float = 0.05,
) -> npt.NDArray[np.float64]:
    """Create real-valued initial population in [0, 1] for non-NN schemes.

    Args:
        specs: Parameter specifications with bounds.
        n_pop: Population size.
        rng: Random number generator.
        seed_defaults: If True, seed first individual from defaults.
        seed_params: Optional known-good params to seed first individual.
        perturbation_scale: Scale of perturbation around seeded individual.

    Returns:
        Array of shape (n_pop, n_params) with values in [0, 1].
    """
    n_params = len(specs)
    pop = rng.random((n_pop, n_params))

    if seed_params is not None:
        seed_x = encode_to_normalized(seed_params, specs)
        pop[0] = np.clip(seed_x, 0.0, 1.0)
        n_seeded = min(n_pop // 2, n_pop - 1)
        for i in range(1, 1 + n_seeded):
            noise = rng.normal(0.0, perturbation_scale, size=n_params)
            pop[i] = np.clip(seed_x + noise, 0.0, 1.0)
    elif seed_defaults:
        defaults = {s.name: s.default for s in specs}
        seed_x = encode_to_normalized(defaults, specs)
        pop[0] = np.clip(seed_x, 0.0, 1.0)
        n_seeded = min(n_pop // 2, n_pop - 1)
        for i in range(1, 1 + n_seeded):
            noise = rng.normal(0.0, perturbation_scale, size=n_params)
            pop[i] = np.clip(seed_x + noise, 0.0, 1.0)

    return pop


def create_nn_initial_population(
    layer_sizes: list[int],
    activations: list[str],
    n_pop: int,
    rng: np.random.Generator,
    bound_multiplier: float = 2.0,
    seed_weights: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """Create real-valued initial population in [0, 1] for NN weight optimization.

    Uses activation-aware weight initialization (Xavier/He/LeCun) to generate
    weights that are well-scaled for each layer, then normalizes to [0, 1].

    Args:
        layer_sizes: NN layer sizes.
        activations: Activation functions per layer transition.
        n_pop: Population size.
        rng: Random number generator.
        bound_multiplier: Multiplier for weight bounds (default: 2.0).
        seed_weights: Optional known-good weights to seed first individual.

    Returns:
        Array of shape (n_pop, n_params) with values in [0, 1].
    """
    specs = nn_param_specs_from_architecture(layer_sizes, activations, bound_multiplier)
    n_params = len(specs)
    pop = np.empty((n_pop, n_params), dtype=np.float64)

    for i in range(n_pop):
        weights = generate_initialized_weights(layer_sizes, activations, rng)
        # Normalize each weight to [0, 1] using its ParamSpec bounds
        for j, s in enumerate(specs):
            pop[i, j] = np.clip((weights[j] - s.p_min) / (s.p_max - s.p_min), 0.0, 1.0)

    if seed_weights is not None:
        # Encode known weights as first individual
        for j, s in enumerate(specs):
            if j < len(seed_weights):
                pop[0, j] = np.clip((seed_weights[j] - s.p_min) / (s.p_max - s.p_min), 0.0, 1.0)

    return pop
