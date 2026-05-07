"""End-to-end smoke for warm_start.build_warm_start_chromosome."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.mark.slow
def test_build_warm_start_chromosome_returns_correctly_shaped_normalized_vector(tmp_path: Path):
    repo_root = Path(__file__).parents[1]
    ftc_params = repo_root / "training_output" / "ftc" / "best_params.json"
    if not ftc_params.exists():
        pytest.skip("FTC training output absent")

    from aerocapture.training.config import NetworkConfig, TrainingConfig
    from aerocapture.training.warm_start import build_warm_start_chromosome

    cfg = TrainingConfig()
    cfg.guidance_type = "neural_network"
    cfg.network = NetworkConfig(
        architecture=[
            {"type": "dense", "input_size": 21, "output_size": 8, "activation": "swish"},
            {"type": "dense", "input_size": 8, "output_size": 1, "activation": "tanh"},
        ],
        input_mask=list(range(21)),
        output_parameterization="acos_tanh",
        optimize_scaffolding=False,
    )
    cfg.sim.toml_config = "configs/training/msr_aller_ftc_train.toml"
    cfg.sim.exec_dir = str(repo_root)
    cfg.save_dir = str(tmp_path / "warm")
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    chromo = build_warm_start_chromosome(
        cfg=cfg,
        n_warm_seeds=4,
        n_epochs=2,
        rng=rng,
    )
    # 21*8 + 8 + 8*1 + 1 = 185
    assert chromo.shape == (185,), chromo.shape
    assert (chromo >= 0.0).all() and (chromo <= 1.0).all()
    assert (Path(cfg.save_dir) / "warm_start_chromosome.npy").exists()
    assert (Path(cfg.save_dir) / "warm_start_cache_key.json").exists()
