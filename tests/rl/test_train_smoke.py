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


def test_linear_anneal_schedule() -> None:
    """_linear_anneal: unchanged before start, linear to 0 over [start, 1], off at start=1."""
    from aerocapture.training.rl.train import _linear_anneal

    assert _linear_anneal(0.01, 0.5, 1.0) == 0.01  # anneal_start=1.0 disables
    assert _linear_anneal(3e-4, 0.4, 0.5) == 3e-4  # before start: unchanged
    assert _linear_anneal(1.0, 0.5, 0.5) == pytest.approx(1.0)  # at start: full
    assert _linear_anneal(1.0, 0.75, 0.5) == pytest.approx(0.5)  # halfway: half
    assert _linear_anneal(1.0, 1.0, 0.5) == pytest.approx(0.0)  # end: zero


def test_linear_anneal_final_update_overshoot() -> None:
    """env_steps overshoots total_env_steps on the last update (frac_done > 1.0):
    the default anneal_start = 1.0 ('off') must return base, not divide by zero."""
    from aerocapture.training.rl.train import _linear_anneal

    assert _linear_anneal(0.01, 1.022, 1.0) == 0.01  # default entropy config, final update
    assert _linear_anneal(3e-4, 1.2, 0.5) == pytest.approx(0.0)  # active anneal floors at 0


@pytest.mark.parametrize("interrupt", [True, False])
def test_interrupted_run_writes_no_final_eval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupt: bool) -> None:
    """A Ctrl+C'd run must not reach the report: its final_eval.parquet is 18_rl_baseline.sh's
    done marker, so an interrupted cell would be skipped on rerun instead of resumed."""
    import aerocapture.training.rl.report_rl as report_rl
    import aerocapture.training.rl.train as train_mod

    calls: list[str] = []

    def fake_run_ppo(cfg, toml_path, output_dir, logger, display, interrupted, *_args) -> None:  # type: ignore[no-untyped-def]
        (output_dir / "best_model.json").write_text("{}")
        interrupted["v"] = interrupt

    monkeypatch.setattr(train_mod, "_run_ppo", fake_run_ppo)
    monkeypatch.setattr(train_mod, "_run_final_eval", lambda *_a: calls.append("final_eval"))
    monkeypatch.setattr(report_rl, "generate_report", lambda *_a: calls.append("report"))
    toml = "configs/training/paper/rl/hl_dense_p515_ppo_scratch.toml"
    monkeypatch.setattr(sys, "argv", ["train.py", toml, "--no-tui", "--output-dir", str(tmp_path)])
    train_mod.main()

    assert calls == ([] if interrupt else ["final_eval", "report"])


def _tiny_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, torch_seed: int) -> tuple[list[dict], bytes]:
    from aerocapture.training.rl.train import main

    toml = tmp_path / f"{name}.toml"
    toml.write_text(f'base = ["{Path("configs/training/msr_aller_nn_atan2_ppo_train.toml").resolve()}"]\n\n[rl]\ntorch_seed = {torch_seed}\n')
    out = tmp_path / name
    argv = ["train.py", str(toml), "--from-scratch", "--no-tui", "--skip-report", "--output-dir", str(out), "--total-steps", "256"]
    argv += ["--n-envs", "2", "--rollout-steps", "64", "--validation-n-sims", "2", "--validation-interval-updates", "1"]
    monkeypatch.setattr(sys, "argv", argv)
    main()
    volatile = ("timestamp", "wallclock_seconds", "config_hash")
    records = [{k: v for k, v in json.loads(line).items() if k not in volatile} for f in sorted(out.glob("rl_training_*.jsonl")) for line in f.open()]
    return records, (out / "best_model.json").read_bytes()


def test_torch_seed_makes_a_run_reproducible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """[rl] torch_seed seeds the policy init and exploration noise: the same seed replays the run
    exactly, another seed does not (until 2026-10-02 torch was never seeded, so two PPO-scratch
    runs of one config ended at 180 and 446 m/s mean)."""
    a = _tiny_run(tmp_path, monkeypatch, "a", 0)
    assert a[0], "no training records"
    assert _tiny_run(tmp_path, monkeypatch, "b", 0) == a
    assert _tiny_run(tmp_path, monkeypatch, "c", 1)[1] != a[1]
