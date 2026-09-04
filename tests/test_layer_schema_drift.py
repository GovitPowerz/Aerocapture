"""Drift test: the pure-Python layer-schema fallback must equal the Rust tensor table.

`aerocapture_rs.layer_schema` IS the Rust `tensor_table!` (flat order, JSON
keys, parameter count). The fallback in `layer_schema.py` exists only for the
extension-less CI job; this asserts it element-wise for every layer type.
"""

from __future__ import annotations

import json
import math

import pytest
from aerocapture.training.layer_schema import _fallback_layer_schema, layer_entry_dict, layer_n_params, layer_schema

aero = pytest.importorskip("aerocapture_rs")

CASES: dict[str, dict] = {
    "dense": {"type": "dense", "input_size": 5, "output_size": 3, "activation": "tanh"},
    "gru": {"type": "gru", "input_size": 5, "hidden_size": 3},
    "lstm": {"type": "lstm", "input_size": 5, "hidden_size": 3},
    "window": {"type": "window", "input_size": 5, "n_steps": 4},
    "transformer": {"type": "transformer", "d_model": 6, "n_heads": 2, "d_ffn": 9, "n_seq": 4},
    "mamba": {"type": "mamba", "input_size": 5, "d_state": 3, "dt_rank": 2},
    "mamba_dt_rank_resolved": {"type": "mamba", "input_size": 40, "d_state": 3},
    "cfc": {"type": "cfc", "input_size": 5, "hidden_size": 3, "backbone_units": 7},
    "slstm": {"type": "slstm", "input_size": 5, "hidden_size": 3},
    "mlstm": {"type": "mlstm", "input_size": 5, "hidden_size": 3},
}
for _disc in ("euler", "trapezoidal"):
    for _sm in ("real", "complex"):
        CASES[f"mamba3_{_disc}_{_sm}"] = {"type": "mamba3", "input_size": 5, "d_state": 3, "dt_rank": 2, "discretization": _disc, "state_mode": _sm}


@pytest.mark.parametrize("case", sorted(CASES))
def test_fallback_schema_matches_rust(case: str) -> None:
    entry = layer_entry_dict(CASES[case])
    rust = [(name, tuple(shape)) for name, shape in aero.layer_schema(json.dumps(entry))]
    assert _fallback_layer_schema(entry) == rust
    assert layer_schema(CASES[case]) == rust
    assert layer_n_params(CASES[case]) == sum(math.prod(s) for _, s in rust)


def test_quantize_tensor_names_exist_in_schema() -> None:
    from aerocapture.training.quantize import _quantizable_tensors

    model = {"architecture": [CASES["dense"], CASES["mamba"]]}
    for key, i, field, _ in _quantizable_tensors(model, "all"):
        names = [n for n, _ in layer_schema(model["architecture"][i])]
        assert field in names, f"{key} not in schema {names}"


def test_invalid_dims_rejected_by_rust() -> None:
    with pytest.raises(ValueError, match="divisible"):
        aero.layer_schema(json.dumps({"type": "transformer", "d_model": 5, "n_heads": 2, "d_ffn": 4, "n_seq": 2}))
    with pytest.raises(ValueError, match="positive"):
        aero.layer_schema(json.dumps({"type": "window", "input_size": 0, "n_steps": 2}))
