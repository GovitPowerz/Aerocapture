"""GA + NN hyperparameters for aerocapture guidance training.

Replaces MATLAB Param_Struct_Aerocap.m.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


@dataclass
class NetworkConfig:
    """Neural network architecture configuration."""

    n_input: int = 7
    n_hidden: int = 24
    n_output: int = 2

    @property
    def n_base_coef(self) -> int:
        """Base number of weights + biases."""
        return (self.n_input + self.n_output) * self.n_hidden + self.n_hidden + self.n_output

    @property
    def n_coef(self) -> int:
        """Total coefficients including sign bits (doubled)."""
        return self.n_base_coef * 2


@dataclass
class GAConfig:
    """Genetic algorithm configuration."""

    n_bit: int = 32
    p_max: float = 1.0
    p_min: float = -1.0
    variation: float = 0.1
    n_pop: int = 20
    n_subpop: int = 1
    migration_interval: int = 10
    n_gen: int = 100
    mutation_rate: float = 0.01
    n_runs: int = 100


@dataclass
class SimConfig:
    """Simulation configuration for cost evaluation."""

    executable: str = "aerocap_nn"
    init_file: str = "aerocap.in_msr_aller_64_nn"
    nn_param_file: str = "../donnees/nn_param.temp"
    final_file: str = "../sorties/final.temp"
    exec_dir: str = "old_codebase/exec"
    n_sims: int = 50


@dataclass
class TrainingConfig:
    """Complete training configuration."""

    network: NetworkConfig = field(default_factory=NetworkConfig)
    ga: GAConfig = field(default_factory=GAConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    save_dir: str = "save_net"

    def build_conversion_matrix(self) -> npt.NDArray[np.float64]:
        """Build binary-to-decimal conversion matrix.

        Returns:
            Array of shape (n_coef, n_bit) for converting binary chromosomes
            to decimal parameter values.
        """
        n_coef = self.network.n_coef
        n_bit = self.ga.n_bit
        p_range = self.ga.p_max - self.ga.p_min
        # Each row: [2^(nbit-1), 2^(nbit-2), ..., 2^0] / (2^nbit - 1) * range
        bit_weights = np.power(2.0, np.arange(n_bit - 1, -1, -1))
        conv = np.tile(bit_weights, (n_coef, 1)) / (2**n_bit - 1) * p_range
        return conv

    def random_network(self, rng: np.random.Generator | None = None) -> npt.NDArray[np.float64]:
        """Generate random initial network weights.

        Returns:
            Array of shape (n_coef,) with values in [-0.1, 0.1].
        """
        if rng is None:
            rng = np.random.default_rng()
        return 0.1 * (2 * rng.random(self.network.n_coef) - 1)
