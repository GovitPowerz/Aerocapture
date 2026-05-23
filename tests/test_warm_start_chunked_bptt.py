"""_chunked_bptt_train runs end-to-end on a synthetic 2-trajectory corpus,
returns a trained V2Policy + per-epoch losses."""

import numpy as np
import pytest
from aerocapture.training.config import NetworkConfig
from aerocapture.training.warm_start import _chunked_bptt_train


def _make_trajectories(n_trajectories: int = 2, T: int = 64, input_dim: int = 4) -> list[dict]:
    """Synthetic: y_signed = sin(X[:, 0]) so a small MLP can fit it quickly."""
    trajs = []
    rng = np.random.default_rng(0)
    for i in range(n_trajectories):
        X = rng.standard_normal((T, input_dim))
        y = np.sin(X[:, 0])
        trajs.append({"seed": i, "X": X, "y_signed": y, "dv": 100.0, "captured": True, "scheme": "ftc"})
    return trajs


def test_dense_runs_and_loss_decreases() -> None:
    trajs = _make_trajectories()
    network = NetworkConfig(
        architecture=[
            {"type": "dense", "input_size": 4, "output_size": 8, "activation": "tanh"},
            {"type": "dense", "input_size": 8, "output_size": 1, "activation": "tanh"},
        ],
        input_mask=list(range(4)),
        output_parameterization="acos_tanh",
    )
    policy, losses, _ = _chunked_bptt_train(
        trajectories=trajs,
        network=network,
        bptt_length=16,
        n_epochs=5,
        lr=1e-2,
    )
    assert len(losses) == 5
    assert all(np.isfinite(losses))
    assert losses[-1] < losses[0]  # loss decreased


def test_gru_runs_and_loss_finite() -> None:
    trajs = _make_trajectories(T=32)
    network = NetworkConfig(
        architecture=[
            {"type": "dense", "input_size": 4, "output_size": 8, "activation": "tanh"},
            {"type": "gru", "input_size": 8, "hidden_size": 8},
            {"type": "dense", "input_size": 8, "output_size": 1, "activation": "tanh"},
        ],
        input_mask=list(range(4)),
        output_parameterization="acos_tanh",
    )
    policy, losses, _ = _chunked_bptt_train(
        trajectories=trajs,
        network=network,
        bptt_length=8,
        n_epochs=3,
        lr=1e-2,
    )
    assert len(losses) == 3
    assert all(np.isfinite(losses))


def test_bptt_length_greater_than_n_seq_raises_for_transformer() -> None:
    trajs = _make_trajectories(T=32, input_dim=8)
    network = NetworkConfig(
        architecture=[
            {"type": "dense", "input_size": 8, "output_size": 8, "activation": "tanh"},
            {"type": "transformer", "d_model": 8, "n_heads": 2, "d_ffn": 16, "n_seq": 4},
            {"type": "dense", "input_size": 8, "output_size": 1, "activation": "tanh"},
        ],
        input_mask=list(range(8)),
        output_parameterization="acos_tanh",
    )
    with pytest.raises(ValueError, match="bptt_length.*n_seq"):
        _chunked_bptt_train(
            trajectories=trajs,
            network=network,
            bptt_length=16,  # > n_seq=4
            n_epochs=1,
            lr=1e-2,
        )
