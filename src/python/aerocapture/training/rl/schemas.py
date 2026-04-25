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


class MambaSpec(BaseModel):
    """Selective SSM (Mamba S6) layer (Phase 4a, PSO-only).

    Input/output dims are both `input_size` (d_inner). `dt_rank` is the
    bottleneck rank for the delta projection; if None, resolves to
    `max(1, input_size // 16)` (paper default). After validation, `spec.dt_rank`
    is always the resolved int value.

    `build_layer(MambaSpec)` raises NotImplementedError -- PPO support deferred
    to Phase 4b (see docs/superpowers/specs/2026-04-24-phase-4a-mamba-ssm-mvp-design.md).
    """

    model_config = ConfigDict(extra="forbid")
    type: Literal["mamba"]
    input_size: int = Field(ge=1)
    d_state: int = Field(ge=1)
    dt_rank: int | None = None

    @model_validator(mode="after")
    def _resolve_and_validate_dt_rank(self) -> MambaSpec:
        rank = self.dt_rank if self.dt_rank is not None else max(1, self.input_size // 16)
        object.__setattr__(self, "dt_rank", rank)
        if rank < 1:
            raise ValueError(f"dt_rank must be >= 1, got {rank}")
        if rank > self.input_size:
            raise ValueError(f"dt_rank ({rank}) must be <= input_size ({self.input_size})")
        return self


LayerSpec = Annotated[DenseSpec | GruSpec | LstmSpec | WindowSpec | TransformerSpec | MambaSpec, Discriminator("type")]


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
