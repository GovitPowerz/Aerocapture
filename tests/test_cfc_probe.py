"""Unit tests for the CfC probe driver + shared probe machinery."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from aerocapture.training.experiments.probe_common import _completion, aggregate, arch_toml, cvar95


def _write_arm(tmp: Path, arch: list[dict], checkpoint_gens: list[int]) -> tuple[Path, Path]:
    """Fabricate an arm output dir + a self-contained config for _completion tests.

    The config is standalone (no base chain) so load_toml_with_bases resolves it
    without touching the repo; its [network].architecture is what _completion
    compares the deployed best_model.json against.
    """
    out_dir = tmp / "arm"
    out_dir.mkdir(parents=True)
    (out_dir / "best_model.json").write_text(json.dumps({"format_version": 2, "architecture": arch, "weights": {}}))
    for g in checkpoint_gens:
        (out_dir / f"checkpoint_g{g:05d}.json").write_text("{}")
    blocks = "\n\n".join("[[network.architecture]]\n" + "\n".join(f'{k} = "{v}"' if isinstance(v, str) else f"{k} = {v}" for k, v in e.items()) for e in arch)
    config = tmp / "arm.toml"
    config.write_text(f'[guidance]\ntype = "neural_network"\n\n[network]\ninput_mask = [0]\n\n{blocks}\n')
    return out_dir, config


_ARCH = [
    {"type": "dense", "input_size": 17, "output_size": 11, "activation": "swish"},
    {"type": "gru", "input_size": 11, "hidden_size": 11},
    {"type": "dense", "input_size": 11, "output_size": 2, "activation": "asinh"},
]


def test_completion_done(tmp_path: Path) -> None:
    out_dir, cfg = _write_arm(tmp_path, _ARCH, [10, 5000])
    assert _completion(out_dir, cfg, n_gen=5000) == "done"


def test_completion_partial_when_under_trained(tmp_path: Path) -> None:
    # best_model.json present (mid-run promotion) but the latest checkpoint is < n_gen.
    out_dir, cfg = _write_arm(tmp_path, _ARCH, [10, 14])
    assert _completion(out_dir, cfg, n_gen=5000) == "partial"


def test_completion_stale_when_arch_differs(tmp_path: Path) -> None:
    # Deployed H=32 model, config now says H=11 (the gru_s0 landmine) -> stale even at g5000.
    stale = [
        {"type": "dense", "input_size": 17, "output_size": 32, "activation": "swish"},
        {"type": "gru", "input_size": 32, "hidden_size": 32},
        {"type": "dense", "input_size": 32, "output_size": 2, "activation": "asinh"},
    ]
    out_dir = tmp_path / "arm"
    out_dir.mkdir()
    (out_dir / "best_model.json").write_text(json.dumps({"format_version": 2, "architecture": stale, "weights": {}}))
    (out_dir / "checkpoint_g05000.json").write_text("{}")
    _, cfg = _write_arm(tmp_path / "cfgonly", _ARCH, [])  # config carries the CURRENT H=11 arch
    assert _completion(out_dir, cfg, n_gen=5000) == "stale"


def test_completion_absent(tmp_path: Path) -> None:
    _, cfg = _write_arm(tmp_path, _ARCH, [])
    assert _completion(tmp_path / "nonexistent", cfg, n_gen=5000) == "absent"


def test_train_jobs_skips_done_resumes_partial_wipes_stale(tmp_path: Path, monkeypatch) -> None:
    """The resumability contract: done -> no subprocess; partial -> resume (no
    --from-scratch); stale -> relaunch WITH --from-scratch; absent -> fresh."""
    import aerocapture.training.experiments.probe_common as pc

    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    out_root = tmp_path / "out"
    out_root.mkdir()

    def _cfg_text(arch: list[dict]) -> str:
        blocks = "\n\n".join(
            "[[network.architecture]]\n" + "\n".join(f'{k} = "{v}"' if isinstance(v, str) else f"{k} = {v}" for k, v in e.items()) for e in arch
        )
        return f'[guidance]\ntype = "neural_network"\n\n[network]\ninput_mask = [0]\n\n{blocks}\n'

    stale_arch = [
        {"type": "dense", "input_size": 17, "output_size": 32, "activation": "swish"},
        {"type": "gru", "input_size": 32, "hidden_size": 32},
        {"type": "dense", "input_size": 32, "output_size": 2, "activation": "asinh"},
    ]

    def make(arm: str, deployed: list[dict] | None, gens: list[int]) -> None:
        # Every arm's CONFIG carries the current _ARCH; only the deployed model varies.
        (cfg_dir / f"{arm}_s0.toml").write_text(_cfg_text(_ARCH))
        if deployed is not None:
            d = out_root / f"{arm}_s0"
            d.mkdir()
            (d / "best_model.json").write_text(json.dumps({"format_version": 2, "architecture": deployed, "weights": {}}))
            for g in gens:
                (d / f"checkpoint_g{g:05d}.json").write_text("{}")

    make("done", _ARCH, [5000])
    make("partial", _ARCH, [14])
    make("stale", stale_arch, [5000])
    make("absent", None, [])

    launched: dict[str, list[str]] = {}

    def fake_run(cmd: list[str], **_: object) -> object:
        # cmd[3] is the config path -> recover the arm name.
        arm = Path(cmd[3]).stem.removesuffix("_s0")
        launched[arm] = cmd
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(pc.subprocess, "run", fake_run)
    pc.train_jobs(["done", "partial", "stale", "absent"], 1, cfg_dir, out_root, 5000, 2, None, force=False, from_scratch=False)

    assert "done" not in launched  # skipped, never launched
    assert set(launched) == {"partial", "stale", "absent"}
    assert "--from-scratch" not in launched["partial"]  # resume preserves the checkpoint
    assert "--from-scratch" in launched["stale"]  # unusable checkpoints wiped
    assert "--from-scratch" not in launched["absent"]  # fresh dir, nothing to wipe


def test_cvar95_is_worst_5pct_mean() -> None:
    x = np.arange(100.0)
    cv = cvar95(x)
    assert cv > float(np.percentile(x, 95))
    assert cv == float(np.mean(x[x >= np.percentile(x, 95)]))


def test_cvar95_empty_is_nan() -> None:
    assert np.isnan(cvar95(np.array([])))


def test_aggregate_mean_std() -> None:
    per_rep = [
        {"rms_cost": 10.0, "capture_rate": 1.0, "dv_p50": 100.0, "dv_p95": 200.0, "cvar95": 250.0},
        {"rms_cost": 12.0, "capture_rate": 0.9, "dv_p50": 110.0, "dv_p95": 220.0, "cvar95": 270.0},
    ]
    agg = aggregate(per_rep)
    assert agg["n_repeats"] == 2
    assert agg["dv_p95"]["mean"] == 210.0
    assert agg["dv_p95"]["std"] == 10.0


def test_arch_toml_renders_blocks() -> None:
    arch = [
        {"type": "dense", "input_size": 21, "output_size": 32, "activation": "swish"},
        {"type": "cfc", "input_size": 32, "hidden_size": 32, "backbone_units": 32},
    ]
    s = arch_toml(arch)
    assert s.count("[[network.architecture]]") == 2
    assert 'type = "cfc"' in s
    assert "backbone_units = 32" in s
    assert 'activation = "swish"' in s


def test_probe_offset_alias() -> None:
    from aerocapture.training.evaluate import MAMBA3_EVAL_SEED_OFFSET, PROBE_EVAL_SEED_OFFSET

    assert PROBE_EVAL_SEED_OFFSET == 10_000_000
    assert MAMBA3_EVAL_SEED_OFFSET == PROBE_EVAL_SEED_OFFSET


def test_cfc_arms_and_budget_within_2pct() -> None:
    from aerocapture.training.config import _layer_n_params
    from aerocapture.training.experiments.cfc_probe import ARMS

    assert set(ARMS) == {"gru", "cfc"}
    totals = {arm: sum(_layer_n_params(e) for e in arch) for arm, arch in ARMS.items()}
    assert totals["gru"] == 1014  # 198 + 792 + 24 == the sweep cell gru_p1014, verbatim
    assert totals["cfc"] == 1003  # 198 + 781 + 24 (B=11 backbone)
    assert abs(totals["cfc"] - totals["gru"]) / totals["gru"] < 0.02


def test_cfc_leaf_toml_carries_layer_and_seed() -> None:
    from pathlib import Path

    from aerocapture.training.experiments.cfc_probe import ARMS, BASE_SEED
    from aerocapture.training.experiments.probe_common import leaf_toml

    toml = leaf_toml("cfc_probe", "cfc", ARMS["cfc"], BASE_SEED + 2, BASE_SEED, Path("training_output/cfc_probe/cfc_s2"), 500, 10)
    assert 'base = ["../msr_aller_nn_atan2_train.toml"]' in toml
    assert "input_mask" not in toml  # inherited from the atan2 base, not respecified
    assert "n_pop = 300" in toml
    # algorithm / seed_strategy / curation are NOT overridden: the sweep's
    # ga + adaptive + bucket=max inherit from common.toml via the atan2 base.
    assert "algorithm = " not in toml
    assert "seed_strategy = " not in toml
    assert 'type = "cfc"' in toml
    assert "backbone_units = 11" in toml
    assert f"seed = {BASE_SEED + 2}" in toml
    assert ".cfc_probe_cfc_s2" in toml
