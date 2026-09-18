"""PSO serialization round-trip for every NN layer type (`tests/nn_archs.py`), the
2-generation `train()` smoke for the rows that carry a training TOML, and the
coverage guard that turns "a new layer type without a gate" into a red build.

Round-trip: `nn_param_specs_from_v2` -> `init_v2_population` ->
`aerocapture_rs.flat_weights_to_json` -> JSON keys/shapes from `layer_schema` ->
`nn_forward` finite. It never trains; `test_train_two_gens` does (real PSO,
~16 sims). Real end-to-end CLI training stays in `test_mamba_pso_end_to_end.py`.
"""

from __future__ import annotations

import json
import typing
from pathlib import Path

import numpy as np
import pytest
from aerocapture.training.layer_schema import layer_entry_dict, layer_schema
from aerocapture.training.torch_mirror.schemas import LayerSpec
from pydantic import TypeAdapter

from tests.nn_archs import ARCHS, ArchCase

aerocapture_rs = pytest.importorskip("aerocapture_rs")

from aerocapture.training.encoding import nn_param_specs_from_v2  # noqa: E402
from aerocapture.training.initialization_v2 import init_v2_population  # noqa: E402

ROWS = [pytest.param(c, id=c.name) for c in ARCHS.values()]


def test_archs_cover_every_layer_type() -> None:
    """One row per `LayerSpec` union member; the mamba3 flag combos share the type."""
    union = typing.get_args(LayerSpec)[0]
    spec_types = {typing.get_args(m.model_fields["type"].annotation)[0] for m in typing.get_args(union)}
    row_types = {e["type"] for c in ARCHS.values() for e in c.arch}
    assert spec_types <= row_types, f"LayerSpec types without an ArchCase row: {sorted(spec_types - row_types)}"


@pytest.mark.parametrize("case", ROWS)
def test_serialization_roundtrip(case: ArchCase, tmp_path: Path) -> None:
    specs = TypeAdapter(list[LayerSpec]).validate_python(case.arch)
    assert len(nn_param_specs_from_v2(specs, bound_multiplier=2.0)) == case.n_params

    pop = init_v2_population(case.arch, n_pop=4, bound_multiplier=2.0, rng=np.random.default_rng(42))
    assert pop.shape == (4, case.n_params)
    assert np.all(np.isfinite(pop))

    path = tmp_path / f"{case.name}.json"
    aerocapture_rs.flat_weights_to_json(pop[0].astype(np.float64).tolist(), json.dumps(case.arch), str(path), None)
    loaded = json.loads(path.read_text())
    assert loaded["format_version"] == 2
    assert loaded["architecture"] == [layer_entry_dict(e) for e in case.arch]
    for i, entry in enumerate(case.arch):
        schema = layer_schema(entry)
        if not schema:
            assert f"layer_{i}" not in loaded.get("weights", {}), "zero-param layer must be spec-only"
            continue
        tensors = loaded["weights"][f"layer_{i}"]
        assert set(tensors) == {name for name, _ in schema}, f"{case.name} layer {i}: keys {sorted(tensors)} vs schema"
        for name, shape in schema:
            assert np.asarray(tensors[name], dtype=np.float64).shape == tuple(shape), f"{case.name} layer {i}.{name}"

    out = np.asarray(aerocapture_rs.nn_forward(str(path), [0.0] * case.n_inputs), dtype=np.float64)
    assert out.shape == (2,)
    assert np.all(np.isfinite(out)), f"non-finite output: {out}"


@pytest.mark.slow
@pytest.mark.parametrize("case", [pytest.param(c, id=c.name) for c in ARCHS.values() if c.train is not None])
def test_train_two_gens(case: ArchCase, tmp_path: Path) -> None:
    """The full stack (config parse, architecture construction, PSO eval, Rust runtime,
    JSON write) runs 2 PSO generations end to end. Not a convergence test."""
    from aerocapture.training.config import NetworkConfig, SimConfig, TrainingConfig
    from aerocapture.training.optimizer import OptimizerConfig, PSOSettings
    from aerocapture.training.train import train

    assert case.train is not None
    save_dir = tmp_path / f"neural_network_{case.name}_pso_smoke"
    cfg = TrainingConfig(
        network=NetworkConfig(architecture=case.train.arch, input_mask=list(range(case.train.n_inputs))),
        optimizer=OptimizerConfig(algorithm="pso", n_pop=8, n_gen=2, seed_strategy="fixed", training_n_sims=2, validation_n_sims=2, pso=PSOSettings()),
        sim=SimConfig(executable="src/rust/target/release/aerocapture", nn_param_file=str(save_dir / "best_model.json"), toml_config=case.train.toml, n_sims=2),
        save_dir=str(save_dir),
        guidance_type="neural_network",
    )
    result = train(cfg, seed=1, cwd=".", verbose=False, no_tui=True, from_scratch=True)
    assert result is not None
    assert not result.get("interrupted", False)
    assert result.get("best_individual") is not None

    best_model = save_dir / "best_model.json"
    assert best_model.exists(), f"best_model.json missing under {save_dir}"
    raw = json.loads(best_model.read_text())
    assert raw["format_version"] == 2
    assert [e["type"] for e in raw["architecture"]] == [e["type"] for e in case.train.arch]
    out = aerocapture_rs.nn_forward(str(best_model), [0.0] * case.train.n_inputs)
    assert len(out) == 2
    assert all(isinstance(v, float) for v in out)


def test_window_rejects_zero_fields(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        aerocapture_rs.flat_weights_to_json([], json.dumps([{"type": "window", "input_size": 4, "n_steps": 0}]), str(tmp_path / "bad.json"), None)
