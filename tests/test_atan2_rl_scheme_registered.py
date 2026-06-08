"""neural_network_atan2_rl is wired into compare_guidance."""

from __future__ import annotations


def test_atan2_rl_scheme_registered() -> None:
    from aerocapture.training import compare_guidance as cg

    assert "neural_network_atan2_rl" in cg.SCHEMES
    assert cg.SCHEME_TRAINING_CONFIGS["neural_network_atan2_rl"] == "configs/training/msr_aller_nn_atan2_ppo_train.toml"
    assert "neural_network_atan2_rl" in cg._NN_DEPLOY_SCHEMES
