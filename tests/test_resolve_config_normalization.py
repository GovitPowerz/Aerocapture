"""Unit tests for `_resolve_config_normalization` (train.py).

These lock the dark exception branches that gate deployed-model correctness:
a config-level `[network.normalization]` override must reach the deployed
`best_model.json`, and a missing / unreadable TOML must degrade to None rather
than raise. Pure (no PyO3 / MC); only `config.sim.{toml_config,exec_dir}` are
touched, so a SimpleNamespace stub suffices.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

from aerocapture.training.config import TrainingConfig
from aerocapture.training.train import _resolve_config_normalization

REPO_ROOT = Path(__file__).resolve().parents[1]


def _stub(toml_config: str | None, exec_dir: str = ".") -> TrainingConfig:
    # _resolve_config_normalization only reads config.sim.{toml_config,exec_dir}.
    ns = SimpleNamespace(sim=SimpleNamespace(toml_config=toml_config, exec_dir=exec_dir))
    return cast(TrainingConfig, ns)


def test_falsy_toml_config_returns_none() -> None:
    assert _resolve_config_normalization(_stub(None), REPO_ROOT) is None
    assert _resolve_config_normalization(_stub(""), REPO_ROOT) is None


def test_missing_toml_path_returns_none_not_raises() -> None:
    cfg = _stub("does_not_exist_anywhere.toml")
    assert _resolve_config_normalization(cfg, REPO_ROOT) is None


def test_config_with_normalization_override_returns_len_35() -> None:
    cfg = _stub("configs/training/msr_aller_nn_train_consolidated.toml")
    norm = _resolve_config_normalization(cfg, REPO_ROOT)
    assert isinstance(norm, list)
    assert len(norm) == 35


def test_config_without_override_returns_none(tmp_path: Path) -> None:
    # Self-contained: a config with a [network] table but no `normalization`
    # key must resolve to None. Don't rely on a real training config staying
    # override-free -- they accrue calibrated [network.normalization] blocks.
    cfg_toml = tmp_path / "no_override.toml"
    cfg_toml.write_text('[guidance]\ntype = "neural_network"\n\n[network]\ninput_mask = [0, 1, 2]\n')
    cfg = _stub(str(cfg_toml))
    assert _resolve_config_normalization(cfg, REPO_ROOT) is None


def test_cwd_none_falls_back_to_exec_dir() -> None:
    # exec_dir carries the repo root; cwd=None must resolve through it.
    cfg = _stub("configs/training/msr_aller_nn_train_consolidated.toml", exec_dir=str(REPO_ROOT))
    norm = _resolve_config_normalization(cfg, None)
    assert isinstance(norm, list)
    assert len(norm) == 35
