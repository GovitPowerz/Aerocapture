"""Every training TOML must resolve to its own training_output/<scheme>/ directory.

Prevents the class of bugs where:
- All NN variants (gru_pso, lstm_pso, window_pso, gru_ppo, lstm_ppo) collapse onto
  `training_output/neural_network/` because `[guidance] type = "neural_network"`
  is shared.
- All PPO variants collapse onto `training_output/neural_network_rl/` because
  the RL trainer hardcoded that as OUT_DIR_DEFAULT.

Both failure modes broke warm-start / resume (scheme B overwrites scheme A's
best_model.json) and broke the post-PPO nominal run (looks for its own deploy
path which training didn't write to).

The fix: save_dir is derived from `[data] neural_network` parent for NN schemes,
and from `{guidance_type}` for non-NN schemes. This test locks that contract in.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from aerocapture.training.compare_guidance import SCHEME_TRAINING_CONFIGS
from aerocapture.training.toml_utils import load_toml_with_bases


def _resolve_save_dir_for_toml(toml_path: Path) -> Path:
    """Replicates the save_dir resolution logic in train.py (NN path) and
    rl/train.py (_resolve_output_dir). NN schemes derive from [data]
    neural_network parent; non-NN schemes derive from [guidance] type.
    """
    raw = load_toml_with_bases(toml_path)
    guidance_type: str | None = raw.get("guidance", {}).get("type")
    assert guidance_type, f"{toml_path}: missing [guidance] type"
    if guidance_type == "neural_network":
        nn_path: str | None = raw.get("data", {}).get("neural_network")
        assert nn_path, f"{toml_path}: NN scheme missing [data] neural_network"
        return Path(nn_path).parent
    return Path("training_output") / guidance_type


# ── Per-scheme expected paths ──────────────────────────────────────────


EXPECTED_SAVE_DIRS = {
    "equilibrium_glide": Path("training_output/equilibrium_glide"),
    "energy_controller": Path("training_output/energy_controller"),
    "pred_guid": Path("training_output/pred_guid"),
    "fnpag": Path("training_output/fnpag"),
    "ftc": Path("training_output/ftc"),
    "neural_network": Path("training_output/neural_network_islands"),
    "neural_network_rl": Path("training_output/neural_network_rl"),
    "neural_network_gru_pso": Path("training_output/neural_network_gru_pso"),
    "neural_network_gru_pso_magonly": Path("training_output/neural_network_gru_pso_magonly"),
    "neural_network_gru_ppo": Path("training_output/neural_network_gru_ppo"),
    "neural_network_lstm_pso": Path("training_output/neural_network_lstm_pso"),
    "neural_network_lstm_ppo": Path("training_output/neural_network_lstm_ppo"),
    "neural_network_window_pso": Path("training_output/neural_network_window_pso"),
    "piecewise_constant": Path("training_output/piecewise_constant"),
}


@pytest.mark.parametrize("scheme,expected", EXPECTED_SAVE_DIRS.items())
def test_training_toml_resolves_to_expected_save_dir(scheme: str, expected: Path) -> None:
    toml_path = Path(SCHEME_TRAINING_CONFIGS[scheme])
    assert toml_path.exists(), f"missing training config: {toml_path}"
    resolved = _resolve_save_dir_for_toml(toml_path)
    assert resolved == expected, (
        f"{scheme}: expected save_dir {expected}, resolved {resolved}. Check {toml_path}'s [data] neural_network (NN) or [guidance] type (non-NN)."
    )


def test_all_schemes_have_unique_save_dirs() -> None:
    """No two schemes may share a save_dir (would cause overwrites on
    back-to-back training, broken warm-start/resume, and broken deploy paths)."""
    seen: dict[Path, str] = {}
    for scheme, toml_path_str in SCHEME_TRAINING_CONFIGS.items():
        toml_path = Path(toml_path_str)
        if not toml_path.exists():
            continue
        resolved = _resolve_save_dir_for_toml(toml_path)
        if resolved in seen:
            pytest.fail(f"save_dir collision: {scheme} and {seen[resolved]} both resolve to {resolved}")
        seen[resolved] = scheme


def test_nn_schemes_use_training_output_prefix() -> None:
    """NN scheme TOMLs must point [data] neural_network under training_output/
    (not data/) so that training artifacts + deploy paths line up and the
    startup guards in train.py / rl/train.py don't reject them.
    """
    for scheme, toml_path_str in SCHEME_TRAINING_CONFIGS.items():
        toml_path = Path(toml_path_str)
        if not toml_path.exists():
            continue
        raw = load_toml_with_bases(toml_path)
        if raw.get("guidance", {}).get("type") != "neural_network":
            continue
        nn_path = raw.get("data", {}).get("neural_network")
        assert nn_path, f"{scheme}: NN TOML missing [data] neural_network"
        assert nn_path.startswith("training_output/"), (
            f"{scheme}: [data] neural_network = '{nn_path}' must start with 'training_output/' for save_dir derivation to land in the right place"
        )
