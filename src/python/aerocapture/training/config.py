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
    """Neural network architecture configuration.

    Supports arbitrary layer configurations via `layer_sizes` and `activations`.
    Default [6, 12, 2] with ["tanh", "asinh"] matches the legacy Fortran architecture.
    """

    layer_sizes: list[int] = field(default_factory=lambda: [6, 12, 2])
    activations: list[str] = field(default_factory=lambda: ["tanh", "asinh"])

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
        return sum(
            self.layer_sizes[i] * self.layer_sizes[i + 1] + self.layer_sizes[i + 1]
            for i in range(len(self.layer_sizes) - 1)
        )

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

    executable: str = "../../src/rust/target/release/aerocapture"
    init_file: str = "train_nn.in"
    nn_param_file: str = "../donnees/nn_param.temp"
    final_file: str = "../sorties/final.train_nn_temp"
    exec_dir: str = "old_codebase/exec"
    n_sims: int = 10
    toml_config: str | None = None  # TOML config path (relative to exec_dir); if set, passed as CLI arg


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
        """Load base network weights from a JSON or legacy Fortran nn_param file.

        Auto-detects format: JSON files start with '{', legacy files have a 6-line header.

        Returns:
            Array of shape (n_coef,) with loaded weights (padded with 1.0).
        """
        import json

        filepath = Path(filepath)
        content = filepath.read_text().strip()

        if content.startswith("{"):
            # JSON: weights already in row-major order
            data = json.loads(content)
            weights = []
            for i in range(len(data["architecture"]["layers"]) - 1):
                layer = data["weights"][f"layer_{i}"]
                for row in layer["w"]:
                    weights.extend(row)
                weights.extend(layer["b"])
        else:
            # Legacy Fortran: weights in column-major order, convert to row-major
            from aerocapture.io._fortran import parse_fortran_line

            raw_values: list[float] = []
            with open(filepath) as f:
                for _ in range(6):
                    next(f)
                for line in f:
                    vals = parse_fortran_line(line.strip())
                    if vals:
                        raw_values.extend(vals)

            # Reorder column-major to row-major for each layer pair
            weights = []
            idx = 0
            sizes = self.network.layer_sizes
            for k in range(len(sizes) - 1):
                n_in, n_out = sizes[k], sizes[k + 1]
                # Fortran: for i in 0..n_in: for j in 0..n_out: w[j][i]
                col_major = raw_values[idx : idx + n_in * n_out]
                idx += n_in * n_out
                # Convert to row-major: w[j][i] for j in 0..n_out, i in 0..n_in
                for j in range(n_out):
                    for i in range(n_in):
                        weights.append(col_major[i * n_out + j])
                # Biases (no reordering needed)
                weights.extend(raw_values[idx : idx + n_out])
                idx += n_out

        n_base = self.network.n_base_coef
        base = np.array(weights[:n_base], dtype=np.float64)
        padded = np.ones(self.network.n_coef, dtype=np.float64)
        padded[:n_base] = base
        return padded
