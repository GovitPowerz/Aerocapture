"""`warm_start_compare` flies the warm-started network from a cell written by the trainer's
own artifact writer, so under the config's `[network] normalization` table (#222: until then
it wrote the model JSON itself without the table, and the atan2 family flew the comparison
under the Rust default scales)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
ARCH = [
    {"type": "dense", "input_size": 17, "output_size": 4, "activation": "swish"},
    {"type": "dense", "input_size": 4, "output_size": 2, "activation": "asinh"},
]


def test_warm_started_network_flies_under_the_config_normalization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("aerocapture_rs")
    from aerocapture.training import warm_start_compare as wsc
    from aerocapture.training.training_config import _setup_param_specs, build_training_config_from_toml

    from tests.fixtures.factories import leaf_toml_with_architecture

    toml_path = tmp_path / "config.toml"
    toml_path.write_text(leaf_toml_with_architecture(REPO / "configs/training/msr_aller_nn_atan2_train.toml", ARCH))
    config, toml_data = build_training_config_from_toml(str(toml_path))
    config.save_dir = str(tmp_path / "run")
    config.optimizer.validation_n_sims = 2
    config.warm_start.supervisor_schemes = ["ftc"]
    config.warm_start.n_warm_seeds = 2
    supervisor = tmp_path / "ftc_best_params.json"
    supervisor.write_text(json.dumps({"capture_damping": 0.05}))
    config.warm_start.params_paths = {"ftc": str(supervisor)}
    specs = _setup_param_specs(config, toml_data, verbose=False)

    flown_models: list[dict] = []

    def fake_evaluate_cell(cell_dir: Path | None, base_toml: Path, seeds: list[int], *, model: Path | None = None, **kwargs: object) -> None:
        if model is not None:
            flown_models.append(json.loads(model.read_text()))
        raise RuntimeError("no flight in this test")

    monkeypatch.setattr(wsc, "evaluate_cell", fake_evaluate_cell)
    manifest = wsc.render_trajectory_comparison(config, base_mc_seed=42, warm_chromo=np.full(len(specs), 0.5), param_specs=specs)

    assert len(flown_models) == 2  # the NN side on both pools
    expected = toml_data["network"]["normalization"]
    assert len(expected) == 35
    for model in flown_models:
        assert model["normalization"] == expected
        assert [layer["input_size"] for layer in model["architecture"]] == [17, 4]
    assert set(manifest["pools"]) == {"train", "val"}
