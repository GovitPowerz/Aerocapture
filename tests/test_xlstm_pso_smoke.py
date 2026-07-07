"""PSO plumbing smoke for the sLSTM/mLSTM probe layers.

slstm arch: Dense(23 -> 8, tanh) -> Slstm(8, 6) -> Dense(6 -> 2, linear) = 192 + 360 + 14 = 566.
mlstm arch: Dense(23 -> 8, tanh) -> Mlstm(8, 6) -> Dense(6 -> 2, linear) = 192 + 234 + 14 = 440.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("aerocapture_rs")
import aerocapture_rs  # type: ignore[import-not-found]  # noqa: E402

CASES = [
    ("slstm", {"type": "slstm", "input_size": 8, "hidden_size": 6}, 566, ("weight_ih", "weight_hh", "bias")),
    ("mlstm", {"type": "mlstm", "input_size": 8, "hidden_size": 6}, 440, ("w_q", "b_q", "w_k", "b_k", "w_v", "b_v", "w_o", "b_o", "w_i", "b_i", "w_f", "b_f")),
]


@pytest.mark.slow
@pytest.mark.parametrize(("name", "mid", "total", "keys"), CASES, ids=[c[0] for c in CASES])
def test_xlstm_pso_smoke(name: str, mid: dict, total: int, keys: tuple, tmp_path: Path) -> None:
    from aerocapture.training.encoding import nn_param_specs_from_v2
    from aerocapture.training.initialization_v2 import init_v2_population
    from aerocapture.training.rl.schemas import LayerSpec
    from pydantic import TypeAdapter

    architecture_dicts = [
        {"type": "dense", "input_size": 23, "output_size": 8, "activation": "tanh"},
        mid,
        {"type": "dense", "input_size": 6, "output_size": 2, "activation": "linear"},
    ]
    adapter = TypeAdapter(list[LayerSpec])
    architecture_specs = adapter.validate_python(architecture_dicts)

    param_specs = nn_param_specs_from_v2(architecture_specs, bound_multiplier=2.0)
    assert len(param_specs) == total, f"Expected {total} params, got {len(param_specs)}"

    rng = np.random.default_rng(42)
    pop = init_v2_population(architecture_dicts, n_pop=4, bound_multiplier=2.0, rng=rng)
    assert pop.shape == (4, total)
    assert np.all(np.isfinite(pop))

    json_path = tmp_path / f"{name}_pso_best.json"
    aerocapture_rs.flat_weights_to_json(pop[0].astype(np.float64).tolist(), json.dumps(architecture_dicts), str(json_path), None)

    loaded = json.loads(json_path.read_text())
    assert [e["type"] for e in loaded["architecture"]] == ["dense", name, "dense"]
    layer_1 = loaded["weights"]["layer_1"]
    for key in keys:
        assert key in layer_1, f"missing {name} weight key: {key!r}"

    out = np.asarray(aerocapture_rs.nn_forward(str(json_path), np.zeros(23, dtype=np.float64).tolist()), dtype=np.float64)
    assert out.shape == (2,)
    assert all(math.isfinite(v) for v in out)
