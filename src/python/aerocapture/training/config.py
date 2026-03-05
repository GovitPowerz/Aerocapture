"""GA + NN hyperparameters for aerocapture guidance training.

Replaces MATLAB Param_Struct_Aerocap.m.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt


@dataclass
class NetworkConfig:
    """Neural network architecture configuration."""

    n_input: int = 6
    n_hidden: int = 12
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

    n_bit: int = 16
    p_max: float = 3.0
    p_min: float = -3.0
    variation: float = 0.1
    direct_encoding: bool = True
    n_pop: int = 20
    n_subpop: int = 1
    migration_interval: int = 10
    n_gen: int = 100
    mutation_rate: float = 0.02
    n_runs: int = 100


@dataclass
class SimConfig:
    """Simulation configuration for cost evaluation."""

    executable: str = "./aerocap_nn"
    init_file: str = "train_nn.in"
    nn_param_file: str = "../donnees/nn_param.temp"
    final_file: str = "../sorties/final.train_nn_temp"
    exec_dir: str = "old_codebase/exec"
    n_sims: int = 10


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

    def load_base_network(self, filepath: str | Path) -> npt.NDArray[np.float64]:
        """Load base network weights from a Fortran nn_param file.

        Reads past the 6-line header, extracts n_base_coef weights,
        and pads to n_coef (doubled) for the GA perturbation scheme.

        Returns:
            Array of shape (n_coef,) with loaded weights (padded with 1.0).
        """
        from aerocapture.io._fortran import parse_fortran_line

        weights = []
        with open(filepath) as f:
            # Skip 6 header lines
            for _ in range(6):
                next(f)
            for line in f:
                vals = parse_fortran_line(line.strip())
                if vals:
                    weights.extend(vals)

        n_base = self.network.n_base_coef
        base = np.array(weights[:n_base], dtype=np.float64)
        # Pad to n_coef with 1.0 (second half used as multiplicative identity)
        padded = np.ones(self.network.n_coef, dtype=np.float64)
        padded[:n_base] = base
        return padded
