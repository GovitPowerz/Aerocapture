"""Pydantic schemas for NN model JSON v2 format.

Mirror of the Rust serde types in src/rust/src/data/neural.rs.
Adding a new layer type means: add a *Spec class, list it in LayerSpec, and
add the matching Rust variant. No other file in this module changes.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Activation = Literal["tanh", "relu", "sigmoid", "asinh", "linear", "swish", "mish"]


class DenseSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["dense"]
    input_size: int = Field(ge=1)
    output_size: int = Field(ge=1)
    activation: Activation


# Phase 0 has a single layer type. Phase 1+ variants (GruSpec, LstmSpec, AttentionSpec,
# LayerNormSpec, SsmSpec, WindowSpec) will turn this into a discriminated union:
#   LayerSpec = Annotated[DenseSpec | GruSpec | ..., Discriminator("type")]
# Until a second variant exists, the Discriminator wrapper is semantically a no-op and
# ruff (UP007) rejects `Union[DenseSpec]` — so we use the bare type alias for now.
LayerSpec = DenseSpec


class LayerWeights(BaseModel):
    model_config = ConfigDict(extra="allow")  # per-layer-type schema-free bag
    w: list[list[float]] | None = None
    b: list[float] | None = None


class ArchitectureV2(BaseModel):
    model_config = ConfigDict(extra="forbid")
    format_version: Literal[2]
    architecture: list[LayerSpec]
    weights: dict[str, LayerWeights]
    output_interpretation: Literal["atan2", "direct"]
    input_mask: list[int] | None = None
    ablated_input: int | None = None
