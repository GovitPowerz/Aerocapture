"""Frozen `nn_param_specs_from_v2` output for every layer type.

The ParamSpec list (names, bounds, centers, ORDER) is the PSO chromosome
contract: `warm_start_bounds.json` and every checkpoint population are encoded
under it. These fixtures were generated before the tensor-table refactor and
pin the exact per-parameter specs for each layer type at two bound multipliers,
including stacked Mamba layers (layer_idx-dependent dt-bias centers).

Regenerate (only for a DELIBERATE contract change) with
`NN_SPECS_FIXTURE_WRITE=1 uv run pytest tests/test_nn_param_specs_frozen.py`.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import pytest
from aerocapture.training.encoding import nn_param_specs_from_v2
from aerocapture.training.rl.schemas import (
    CfcSpec,
    DenseSpec,
    GruSpec,
    LstmSpec,
    Mamba3Spec,
    MambaSpec,
    MlstmSpec,
    SlstmSpec,
    TransformerSpec,
    WindowSpec,
)

FIXTURE_DIR = Path(__file__).parent / "reference_data" / "nn_param_specs_v2"


def _dense(i: int, o: int, act: str = "linear") -> DenseSpec:
    return DenseSpec(type="dense", input_size=i, output_size=o, activation=act)  # type: ignore[arg-type]


CASES: dict[str, list] = {
    "dense": [_dense(3, 2, "tanh")],
    "gru": [GruSpec(type="gru", input_size=3, hidden_size=2)],
    "lstm": [LstmSpec(type="lstm", input_size=3, hidden_size=2)],
    "window": [WindowSpec(type="window", input_size=2, n_steps=3), _dense(6, 2)],
    "transformer": [TransformerSpec(type="transformer", d_model=4, n_heads=2, d_ffn=6, n_seq=3), _dense(4, 2)],
    "mamba": [MambaSpec(type="mamba", input_size=3, d_state=2, dt_rank=1), _dense(3, 2)],
    "mamba_stacked": [
        _dense(3, 3, "swish"),
        MambaSpec(type="mamba", input_size=3, d_state=2, dt_rank=1),
        MambaSpec(type="mamba", input_size=3, d_state=2, dt_rank=1),
        _dense(3, 2),
    ],
    "cfc": [CfcSpec(type="cfc", input_size=3, hidden_size=2, backbone_units=4)],
    "slstm": [SlstmSpec(type="slstm", input_size=3, hidden_size=2)],
    "mlstm": [MlstmSpec(type="mlstm", input_size=3, hidden_size=2)],
}
for _disc in ("euler", "trapezoidal"):
    for _sm in ("real", "complex"):
        CASES[f"mamba3_{_disc}_{_sm}"] = [
            _dense(3, 3, "swish"),
            Mamba3Spec(type="mamba3", input_size=3, d_state=2, dt_rank=1, discretization=_disc, state_mode=_sm),  # type: ignore[arg-type]
            _dense(3, 2),
        ]


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("mul", [1.0, 2.0])
def test_param_specs_match_frozen_fixture(case: str, mul: float) -> None:
    specs = [asdict(s) for s in nn_param_specs_from_v2(CASES[case], bound_multiplier=mul)]
    path = FIXTURE_DIR / f"{case}_m{mul:g}.json"
    if os.environ.get("NN_SPECS_FIXTURE_WRITE"):
        FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(specs, indent=1) + "\n")
    frozen = json.loads(path.read_text())
    assert len(specs) == len(frozen), f"{case}: spec count {len(specs)} != frozen {len(frozen)}"
    for i, (got, want) in enumerate(zip(specs, frozen, strict=True)):
        assert got == want, f"{case} m={mul}: spec {i} differs: {got} != {want}"
