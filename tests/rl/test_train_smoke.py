"""End-to-end smoke test: tiny PPO run, checks artifacts exist."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("aerocapture_rs")
pytest.importorskip("torch")


def _make_dummy_model(path: Path, config_path: Path) -> None:
    """Write a minimal valid NeuralNetModel JSON v2 matching the rl_train TOML architecture.

    Reads [network] from the resolved TOML so the dummy stays in sync
    when the config changes. Post-Task-5, PPO warm-start uses the v2 loader
    (aerocapture.training.model_io.load_policy_from_json), so the dummy must
    be format_version=2.
    """
    from aerocapture.training.toml_utils import load_toml_with_bases

    cfg = load_toml_with_bases(config_path)
    net = cfg.get("network", {})
    toml_layers: list[int] = net.get("layer_sizes", [16, 64, 64, 2])
    activations: list[str] = net.get("activations", ["tanh", "tanh", "linear"])
    input_mask: list[int] = net.get("input_mask", list(range(toml_layers[0])))

    input_dim = toml_layers[0]
    layer_sizes = toml_layers[1:]

    architecture: list[dict[str, object]] = []
    weights_dict: dict[str, object] = {}
    prev = input_dim
    for i, (out_dim, act) in enumerate(zip(layer_sizes, activations, strict=True)):
        architecture.append(
            {
                "type": "dense",
                "input_size": prev,
                "output_size": out_dim,
                "activation": act,
            }
        )
        weights_dict[f"layer_{i}"] = {
            "w": [[0.0] * prev for _ in range(out_dim)],
            "b": [0.0] * out_dim,
        }
        prev = out_dim

    doc = {
        "format_version": 2,
        "architecture": architecture,
        "weights": weights_dict,
        "input_mask": input_mask,
        "ablated_input": None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(doc, f)


@pytest.mark.slow
def test_ppo_smoke_produces_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = Path("configs/training/msr_aller_rl_train.toml")
    out = tmp_path / "rl_smoke"
    dummy_model = tmp_path / "dummy_model.json"
    _make_dummy_model(dummy_model, config_path)

    from aerocapture.training.rl.train import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            str(config_path),
            "--total-steps",
            "512",
            "--n-envs",
            "2",
            "--rollout-steps",
            "64",
            "--validation-n-sims",
            "4",
            "--validation-interval-updates",
            "1",
            "--data-neural-network",
            str(dummy_model),
            "--no-tui",
            "--skip-report",
            "--output-dir",
            str(out),
        ],
    )
    main()

    assert (out / "best_model.json").exists(), "best_model.json missing"
    assert (out / "config_resolved.toml").exists(), "config_resolved.toml missing"
    assert any(out.glob("rl_training_*.jsonl")), "no rl_training_*.jsonl found"

    with (out / "best_model.json").open() as f:
        doc = json.load(f)
    assert "architecture" in doc
    assert "weights" in doc
    assert "output_interpretation" not in doc
