"""Training configuration for aerocapture guidance optimization.

Supports both NN weight optimization and generic guidance parameter optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

from aerocapture.training.optimizer import OptimizerConfig


@dataclass
class NetworkConfig:
    """Neural network architecture configuration.

    Supports two encodings:
      - v1 (dense-only, backward compat): `layer_sizes` + `activations` fields.
      - v2 (heterogeneous): `architecture` list of dicts with per-layer type+shape,
        mirroring the Rust `LayerSpec` enum and the TOML `[[network.architecture]]`
        array-of-tables (dense | gru | ...). When set, `architecture` takes
        precedence over `layer_sizes`/`activations`.
    """

    layer_sizes: list[int] = field(default_factory=lambda: [16, 24, 2])
    activations: list[str] = field(default_factory=lambda: ["tanh", "asinh"])
    input_mask: list[int] | None = None
    architecture: list[dict] | None = None

    def __post_init__(self) -> None:
        if self.architecture is not None:
            # v2: validate shapes via _layer_n_params (raises on unknown type).
            for entry in self.architecture:
                _layer_n_params(entry)
            first_input = self.architecture[0].get("input_size")
            if self.input_mask is not None and first_input is not None:
                first_input_int = int(first_input)
                if len(self.input_mask) != first_input_int:
                    msg = f"input_mask length ({len(self.input_mask)}) must equal architecture[0].input_size ({first_input_int})"
                    raise ValueError(msg)
            return
        n_layers = len(self.layer_sizes) - 1
        if len(self.activations) != n_layers:
            msg = f"activations length ({len(self.activations)}) must equal len(layer_sizes)-1 ({n_layers})"
            raise ValueError(msg)
        if self.input_mask is not None and len(self.input_mask) != self.layer_sizes[0]:
            msg = f"input_mask length ({len(self.input_mask)}) must equal layer_sizes[0] ({self.layer_sizes[0]})"
            raise ValueError(msg)

    @property
    def n_input(self) -> int:
        if self.architecture is not None:
            return int(self.architecture[0]["input_size"])
        return self.layer_sizes[0]

    @property
    def n_output(self) -> int:
        if self.architecture is not None:
            last = self.architecture[-1]
            size: object = last["output_size"] if "output_size" in last else last["hidden_size"]
            assert isinstance(size, int)
            return size
        return self.layer_sizes[-1]

    @property
    def n_hidden(self) -> int:
        """First hidden layer size (backward compat for 3-layer networks)."""
        return self.layer_sizes[1]

    @property
    def n_base_coef(self) -> int:
        """Total weights + biases across all layers."""
        if self.architecture is not None:
            return sum(_layer_n_params(entry) for entry in self.architecture)
        return sum(self.layer_sizes[i] * self.layer_sizes[i + 1] + self.layer_sizes[i + 1] for i in range(len(self.layer_sizes) - 1))

    @property
    def n_coef(self) -> int:
        """Total coefficients (same as n_base_coef; sign bits removed in pymoo migration)."""
        return self.n_base_coef


def _layer_n_params(entry: dict) -> int:
    """Parameter count for a single v2 architecture entry. Mirrors Rust LayerWeights::n_params."""
    ltype = entry["type"]
    if ltype == "dense":
        return int(entry["input_size"]) * int(entry["output_size"]) + int(entry["output_size"])
    if ltype == "gru":
        h = int(entry["hidden_size"])
        i = int(entry["input_size"])
        return 3 * h * i + 3 * h * h + 2 * 3 * h
    raise ValueError(f"Unknown v2 layer type: {ltype!r}")


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
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
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

    def random_network(self, rng: np.random.Generator | None = None) -> npt.NDArray[np.float64]:
        """Generate random initial network weights.

        Returns:
            Array of shape (n_base_coef,) with values in [-0.1, 0.1].
        """
        if rng is None:
            rng = np.random.default_rng()
        return 0.1 * (2 * rng.random(self.network.n_base_coef) - 1)

    def load_base_network(self, filepath: str | Path) -> npt.NDArray[np.float64]:
        """Load base network weights from a JSON nn_param file.

        Returns:
            Array of shape (n_base_coef,) with loaded weights.
        """
        import json

        filepath = Path(filepath)
        content = filepath.read_text().strip()

        data = json.loads(content)
        weights: list[float] = []
        for i in range(len(data["architecture"]["layers"]) - 1):
            layer = data["weights"][f"layer_{i}"]
            for row in layer["w"]:
                weights.extend(row)
            weights.extend(layer["b"])

        n_base = self.network.n_base_coef
        return np.array(weights[:n_base], dtype=np.float64)
