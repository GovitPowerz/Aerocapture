"""Pydantic schemas for NN model JSON v2 format.

Mirror of the Rust serde types in src/rust/src/data/neural.rs.
Adding a new layer type means: add a *Spec class, list it in LayerSpec, and
add the matching Rust variant. No other file in this module changes.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Field, model_validator

Activation = Literal["tanh", "relu", "sigmoid", "asinh", "linear", "swish", "mish"]


class DenseSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["dense"]
    input_size: int = Field(ge=1)
    output_size: int = Field(ge=1)
    activation: Activation


class GruSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["gru"]
    input_size: int = Field(ge=1)
    hidden_size: int = Field(ge=1)


class LstmSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["lstm"]
    input_size: int = Field(ge=1)
    hidden_size: int = Field(ge=1)


class WindowSpec(BaseModel):
    """Zero-parameter FIFO ring-buffer layer (Phase 2b, PSO-only).

    Maintains a buffer of the last `n_steps` inputs and concatenates them into
    a vector of length `n_steps * input_size` for the next Dense layer.
    `build_layer(WindowSpec)` raises NotImplementedError -- Window is PSO-only,
    PPO deferred to a future phase.
    """

    model_config = ConfigDict(extra="forbid")
    type: Literal["window"]
    input_size: int = Field(ge=1)
    n_steps: int = Field(ge=1)


class TransformerSpec(BaseModel):
    """Causal self-attention Transformer layer (Phase 3a, PSO-only initially).

    d_model must be divisible by n_heads (multi-head attention constraint).
    `build_layer(TransformerSpec)` raises NotImplementedError -- PPO support deferred.
    """

    model_config = ConfigDict(extra="forbid")
    type: Literal["transformer"]
    d_model: int = Field(ge=1)
    n_heads: int = Field(ge=1)
    d_ffn: int = Field(ge=1)
    n_seq: int = Field(ge=1)

    @model_validator(mode="after")
    def _validate_head_divisibility(self) -> TransformerSpec:
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}")
        return self


LayerSpec = Annotated[DenseSpec | GruSpec | LstmSpec | WindowSpec | TransformerSpec, Discriminator("type")]


class LayerWeights(BaseModel):
    model_config = ConfigDict(extra="allow")  # per-layer-type schema-free bag
    w: list[list[float]] | None = None
    b: list[float] | None = None


class ArchitectureV2(BaseModel):
    # `extra="ignore"` so legacy JSON files that still carry `output_interpretation`
    # keep loading. The field is obsolete -- bank is always atan2(out[0], out[1]).
    model_config = ConfigDict(extra="ignore")
    format_version: Literal[2]
    architecture: list[LayerSpec]
    weights: dict[str, LayerWeights]
    input_mask: list[int] | None = None
    ablated_input: int | None = None
