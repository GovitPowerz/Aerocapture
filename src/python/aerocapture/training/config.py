"""Training configuration for aerocapture guidance optimization.

Supports both NN weight optimization and generic guidance parameter optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
            # v2: validate per-entry shapes + chain consistency (layer i's output size
            # must equal layer i+1's input size).
            for entry in self.architecture:
                _layer_n_params(entry)
            for i in range(len(self.architecture) - 1):
                prev_out = _layer_output_size(self.architecture[i])
                next_in = _layer_input_size(self.architecture[i + 1])
                if prev_out != next_in:
                    msg = (
                        f"architecture chain mismatch at layer {i}->{i + 1}: "
                        f"layer {i} ({self.architecture[i]['type']}) produces "
                        f"output_size={prev_out}, but layer {i + 1} "
                        f"({self.architecture[i + 1]['type']}) expects input_size={next_in}"
                    )
                    raise ValueError(msg)
            if self.input_mask is not None:
                first_input_int = _layer_input_size(self.architecture[0])
                if len(self.input_mask) != first_input_int:
                    msg = f"input_mask length ({len(self.input_mask)}) must equal architecture[0] input size ({first_input_int})"
                    raise ValueError(msg)
            return
        n_layers = len(self.layer_sizes) - 1
        if len(self.activations) != n_layers:
            msg = f"activations length ({len(self.activations)}) must equal len(layer_sizes)-1 ({n_layers})"
            raise ValueError(msg)
        if self.input_mask is not None and len(self.input_mask) != self.layer_sizes[0]:
            msg = f"input_mask length ({len(self.input_mask)}) must equal layer_sizes[0] ({self.layer_sizes[0]})"
            raise ValueError(msg)

    def describe(self) -> str:
        """Human-readable multi-line architecture summary.

        Example:
            Network architecture (6946 params):
              layer 0: dense   23 -> 32   tanh
              layer 1: gru     32 -> 32   hidden_size=32
              layer 2: dense   32 -> 2    linear
            input_mask: [0..22] (23 inputs)
            output: atan2 (bank angle from (out0, out1))
        """
        return describe_architecture(self)

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


def _layer_n_params(entry: Any) -> int:
    """Parameter count for a single v2 architecture entry. Mirrors Rust LayerWeights::n_params."""
    from aerocapture.training.rl.schemas import TransformerSpec

    if isinstance(entry, TransformerSpec):
        return 4 * entry.d_model * entry.d_model + 2 * entry.d_ffn * entry.d_model + entry.d_ffn + 9 * entry.d_model
    # Normalise other Pydantic models to plain dicts.
    if hasattr(entry, "model_dump"):
        entry = entry.model_dump()
    ltype = entry["type"]
    if ltype == "dense":
        return int(entry["input_size"]) * int(entry["output_size"]) + int(entry["output_size"])
    if ltype == "gru":
        h = int(entry["hidden_size"])
        i = int(entry["input_size"])
        return 3 * h * i + 3 * h * h + 2 * 3 * h
    if ltype == "lstm":
        h = int(entry["hidden_size"])
        i = int(entry["input_size"])
        return 4 * h * i + 4 * h * h + 2 * 4 * h
    if ltype == "window":
        return 0  # zero trainable parameters
    if ltype == "transformer":
        d = int(entry["d_model"])
        f = int(entry["d_ffn"])
        return 4 * d * d + 2 * f * d + f + 9 * d
    if ltype == "mamba":
        d_inner = int(entry["input_size"])
        d_state = int(entry["d_state"])
        dt_rank = int(entry["dt_rank"])
        return d_inner * (3 * d_state + 2 * dt_rank + 2)
    raise ValueError(f"Unknown v2 layer type: {ltype!r}")


def _layer_input_size(entry: Any) -> int:
    """Input size of a v2 layer entry. Accepts dict OR Pydantic LayerSpec."""
    from aerocapture.training.rl.schemas import TransformerSpec

    if isinstance(entry, TransformerSpec):
        return entry.d_model
    if isinstance(entry, dict):
        if entry.get("type") == "transformer":
            return int(entry["d_model"])
        return int(entry["input_size"])
    # Other Pydantic specs (Dense/Gru/Lstm/Window): all have input_size.
    return int(entry.input_size)


def _layer_output_size(entry: Any) -> int:
    """Output size of a v2 layer entry. Dense: output_size. GRU/LSTM: hidden_size
    (the cell emits its hidden state to the next layer). Window: n_steps * input_size
    (flattened ring buffer). Transformer: d_model. Mamba: input_size (d_inner)."""
    from aerocapture.training.rl.schemas import MambaSpec, TransformerSpec

    if isinstance(entry, TransformerSpec):
        return entry.d_model
    if isinstance(entry, MambaSpec):
        return entry.input_size
    # Normalise other Pydantic models to plain dicts.
    if hasattr(entry, "model_dump"):
        entry = entry.model_dump()
    ltype = entry["type"]
    if ltype == "dense":
        return int(entry["output_size"])
    if ltype == "gru":
        return int(entry["hidden_size"])
    if ltype == "lstm":
        return int(entry["hidden_size"])
    if ltype == "window":
        return int(entry["input_size"]) * int(entry["n_steps"])
    if ltype == "transformer":
        return int(entry["d_model"])
    if ltype == "mamba":
        return int(entry["input_size"])
    raise ValueError(f"Unknown v2 layer type: {ltype!r}")


def describe_architecture(network: NetworkConfig | list[Any]) -> str:
    """Format a human-readable architecture summary for stdout at training start.

    Accepts either a NetworkConfig or a bare list[dict|LayerSpec] (the latter
    is used by tests and callers that only have the raw architecture list).
    """
    from aerocapture.training.rl.schemas import TransformerSpec

    arch: list[Any] | None
    if isinstance(network, list):
        arch = network
        n_params = sum(_layer_n_params(e) for e in arch)
        input_mask = None
    else:
        arch = network.architecture
        n_params = network.n_base_coef
        input_mask = network.input_mask

    if arch is not None:
        lines = [f"Network architecture ({n_params} params):"]
        for i, entry in enumerate(arch):
            # Normalise Pydantic models to a dict for uniform access, but keep
            # TransformerSpec separate since it lacks input_size/output_size.
            if isinstance(entry, TransformerSpec):
                ltype = "transformer"
                tail = f"d_model={entry.d_model}, n_heads={entry.n_heads}, d_ffn={entry.d_ffn}, n_seq={entry.n_seq}"
            else:
                if hasattr(entry, "model_dump"):
                    entry = entry.model_dump()
                ltype = entry["type"]
                if ltype == "dense":
                    tail = entry.get("activation", "?")
                elif ltype in ("gru", "lstm"):
                    tail = f"hidden_size={entry['hidden_size']}"
                elif ltype == "window":
                    tail = f"n_steps={entry['n_steps']}"
                elif ltype == "transformer":
                    tail = f"d_model={entry['d_model']}, n_heads={entry['n_heads']}, d_ffn={entry['d_ffn']}, n_seq={entry['n_seq']}"
                elif ltype == "mamba":
                    tail = f"d_state={entry['d_state']}, dt_rank={entry['dt_rank']}"
                else:
                    tail = ltype
            in_size = _layer_input_size(entry)
            out_size = _layer_output_size(entry)
            lines.append(f"  layer {i}: {ltype:<11} {in_size:>4} -> {out_size:<4} {tail}")
    else:
        assert not isinstance(network, list)
        lines = [f"Network architecture ({n_params} params):"]
        for i, act in enumerate(network.activations):
            fan_in = network.layer_sizes[i]
            fan_out = network.layer_sizes[i + 1]
            lines.append(f"  layer {i}: dense       {fan_in:>4} -> {fan_out:<4} {act}")

    if input_mask is not None:
        n = len(input_mask)
        lines.append(f"  input_mask: {n} indices {input_mask if n <= 8 else f'[{input_mask[0]}..{input_mask[-1]}]'}")

    return "\n".join(lines)


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
