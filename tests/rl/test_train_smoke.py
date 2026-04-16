"""End-to-end smoke test: tiny PPO run, checks artifacts exist."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("aerocapture_rs")
pytest.importorskip("torch")


def _make_dummy_model(path: Path) -> None:
    """Write a minimal valid NeuralNetModel JSON matching the rl_train TOML architecture.

    Architecture: [23, 16, 8, 2] (23 inputs), mish/mish/linear activations.
    Must match the [network] section in msr_aller_rl_train.toml.
    Uses the Rust NnJsonFile format. Weights are all zeros.
    """
    input_dim = 23
    layer_sizes = [16, 8, 2]
    activations = ["mish", "mish", "linear"]
    input_mask = list(range(input_dim))

    weights_dict: dict[str, object] = {}
    prev = input_dim
    for i, out_dim in enumerate(layer_sizes):
        weights_dict[f"layer_{i}"] = {
            "w": [[0.0] * prev for _ in range(out_dim)],
            "b": [0.0] * out_dim,
        }
        prev = out_dim

    doc = {
        "format_version": 1,
        "architecture": {
            "layers": [input_dim] + layer_sizes,
            "activations": activations,
        },
        "weights": weights_dict,
        "output_interpretation": "atan2",
        "input_mask": input_mask,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(doc, f)


@pytest.mark.slow
def test_ppo_smoke_produces_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = Path("configs/training/msr_aller_rl_train.toml")
    out = tmp_path / "rl_smoke"
    dummy_model = tmp_path / "dummy_model.json"
    _make_dummy_model(dummy_model)

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
    assert doc["output_interpretation"] == "atan2"
