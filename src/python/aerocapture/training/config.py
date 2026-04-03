"""GA hyperparameters for aerocapture guidance training.

Supports both NN weight optimization and generic guidance parameter optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt


@dataclass
class NetworkConfig:
    """Neural network architecture configuration.

    Supports arbitrary layer configurations via `layer_sizes` and `activations`.
    Default [16, 24, 2] with ["tanh", "asinh"] matches the 16-input Rust architecture.
    """

    layer_sizes: list[int] = field(default_factory=lambda: [16, 24, 2])
    activations: list[str] = field(default_factory=lambda: ["tanh", "asinh"])

    def __post_init__(self) -> None:
        n_layers = len(self.layer_sizes) - 1
        if len(self.activations) != n_layers:
            msg = f"activations length ({len(self.activations)}) must equal len(layer_sizes)-1 ({n_layers})"
            raise ValueError(msg)

    @property
    def n_input(self) -> int:
        return self.layer_sizes[0]

    @property
    def n_output(self) -> int:
        return self.layer_sizes[-1]

    @property
    def n_hidden(self) -> int:
        """First hidden layer size (backward compat for 3-layer networks)."""
        return self.layer_sizes[1]

    @property
    def n_base_coef(self) -> int:
        """Total weights + biases across all layers."""
        return sum(self.layer_sizes[i] * self.layer_sizes[i + 1] + self.layer_sizes[i + 1] for i in range(len(self.layer_sizes) - 1))

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
    rotate_seeds: bool = False
    adaptive_seeds: bool = False
    seed_pool_cap: int = 100
    cost_alpha: float = 0.7
    cvar_percentile: int = 20
    stress_interval: int = 5
    stress_probes: int = 200
    stress_inject: int = 20


@dataclass
class SimConfig:
    """Simulation configuration for cost evaluation."""

    executable: str = "src/rust/target/release/aerocapture"
    nn_param_file: str = "data/neural_network/nn_model.json"
    final_file: str = "output/final.train_nn_temp"
    exec_dir: str = "."
    n_sims: int = 10
    toml_config: str | None = None  # TOML config path (relative to exec_dir); if set, passed as CLI arg
    sim_timeout_secs: float | None = None  # wall-clock timeout per simulation (seconds); None = no limit


@dataclass
class TrainingConfig:
    """Complete training configuration."""

    network: NetworkConfig = field(default_factory=NetworkConfig)
    ga: GAConfig = field(default_factory=GAConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    save_dir: str = "training_output"
    guidance_type: str = "neural_network"

    @property
    def n_params(self) -> int:
        """Number of parameters to optimize (depends on guidance type)."""
        if self.guidance_type == "neural_network":
            return self.network.n_base_coef
        from aerocapture.training.param_spaces import PARAM_SPACES

        return len(PARAM_SPACES[self.guidance_type])

    @property
    def chrom_length(self) -> int:
        """Binary chromosome length."""
        return self.n_params * self.ga.n_bit

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
        conv: npt.NDArray[np.float64] = np.tile(bit_weights, (n_coef, 1)) / (2**n_bit - 1) * p_range
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
        """Load base network weights from a JSON nn_param file.

        Returns:
            Array of shape (n_coef,) with loaded weights (padded with 1.0).
        """
        import json

        filepath = Path(filepath)
        content = filepath.read_text().strip()

        data = json.loads(content)
        weights = []
        for i in range(len(data["architecture"]["layers"]) - 1):
            layer = data["weights"][f"layer_{i}"]
            for row in layer["w"]:
                weights.extend(row)
            weights.extend(layer["b"])

        n_base = self.network.n_base_coef
        base = np.array(weights[:n_base], dtype=np.float64)
        padded = np.ones(self.network.n_coef, dtype=np.float64)
        padded[:n_base] = base
        return padded
