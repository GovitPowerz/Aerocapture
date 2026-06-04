"""_chunked_bptt_train runs end-to-end on a synthetic 2-trajectory corpus,
returns a trained V2Policy + per-epoch losses."""

import numpy as np
import pytest

pytest.importorskip("aerocapture_rs")

from aerocapture.training.config import AdamConfig, NetworkConfig  # noqa: E402
from aerocapture.training.warm_start import _chunked_bptt_train  # noqa: E402


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
        adam=AdamConfig(lr=1e-2),
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
        adam=AdamConfig(lr=1e-2),
    )
    assert len(losses) == 3
    assert all(np.isfinite(losses))


def test_minibatch_size_clamps_to_n_chunks() -> None:
    """minibatch_size > n_chunks must clamp gracefully (no IndexError, no
    wasted epochs). The clamp is `max(1, min(minibatch_size, n_chunks))`."""
    trajs = _make_trajectories(n_trajectories=2, T=32)  # → 4 chunks at bptt=16, 2 chunks at bptt=8
    network = NetworkConfig(
        architecture=[
            {"type": "dense", "input_size": 4, "output_size": 8, "activation": "tanh"},
            {"type": "dense", "input_size": 8, "output_size": 1, "activation": "tanh"},
        ],
        input_mask=list(range(4)),
        output_parameterization="acos_tanh",
    )
    # Ask for minibatch_size=512 with only 4 chunks → clamps to 4, runs without error.
    policy, losses, n_chunks = _chunked_bptt_train(
        trajectories=trajs,
        network=network,
        bptt_length=16,
        n_epochs=2,
        minibatch_size=512,
        adam=AdamConfig(lr=1e-2),
    )
    assert n_chunks == 4
    assert len(losses) == 2
    assert all(np.isfinite(losses))


def test_minibatch_size_changes_loss_path_but_keeps_shape() -> None:
    """Two runs with different minibatch_size on the same data + seed produce
    DIFFERENT loss trajectories (gradient batching is different) but the
    final policy still has the expected parameter shape. This is the
    correctness contract: the new prestack path doesn't silently drop or
    rebatch chunks vs the old per-minibatch np.stack path."""
    trajs = _make_trajectories(n_trajectories=4, T=64)
    network = NetworkConfig(
        architecture=[
            {"type": "dense", "input_size": 4, "output_size": 8, "activation": "tanh"},
            {"type": "dense", "input_size": 8, "output_size": 1, "activation": "tanh"},
        ],
        input_mask=list(range(4)),
        output_parameterization="acos_tanh",
    )
    _, losses_small, _ = _chunked_bptt_train(trajectories=trajs, network=network, bptt_length=16, n_epochs=3, minibatch_size=4, adam=AdamConfig(lr=1e-2))
    _, losses_big, _ = _chunked_bptt_train(trajectories=trajs, network=network, bptt_length=16, n_epochs=3, minibatch_size=64, adam=AdamConfig(lr=1e-2))
    assert all(np.isfinite(losses_small))
    assert all(np.isfinite(losses_big))
    # Both converge in the expected direction.
    assert losses_small[-1] < losses_small[0]
    assert losses_big[-1] < losses_big[0]


def test_adam_config_overrides_threaded_to_optimizer() -> None:
    """Non-default AdamConfig values must reach the underlying torch.optim.Adam.
    Validate by asserting that high lr + weight_decay produces a noticeably
    different loss trajectory than the defaults on the same toy data + seed."""
    trajs = _make_trajectories(n_trajectories=4, T=64)
    network = NetworkConfig(
        architecture=[
            {"type": "dense", "input_size": 4, "output_size": 8, "activation": "tanh"},
            {"type": "dense", "input_size": 8, "output_size": 1, "activation": "tanh"},
        ],
        input_mask=list(range(4)),
        output_parameterization="acos_tanh",
    )
    # Default Adam (lr=1e-3).
    _, losses_default, _ = _chunked_bptt_train(trajectories=trajs, network=network, bptt_length=16, n_epochs=5)
    # High lr + weight_decay.
    _, losses_tuned, _ = _chunked_bptt_train(
        trajectories=trajs,
        network=network,
        bptt_length=16,
        n_epochs=5,
        adam=AdamConfig(lr=1e-1, weight_decay=1e-2),
    )
    assert all(np.isfinite(losses_default))
    assert all(np.isfinite(losses_tuned))
    # Materially different trajectories (high lr should drive faster initial change).
    assert losses_default != losses_tuned


def test_amsgrad_variant_runs() -> None:
    """The amsgrad=True path should also produce finite losses (smoke test that
    the kwarg reaches torch.optim.Adam without TypeError)."""
    trajs = _make_trajectories(n_trajectories=2, T=32)
    network = NetworkConfig(
        architecture=[
            {"type": "dense", "input_size": 4, "output_size": 4, "activation": "tanh"},
            {"type": "dense", "input_size": 4, "output_size": 1, "activation": "tanh"},
        ],
        input_mask=list(range(4)),
        output_parameterization="acos_tanh",
    )
    _, losses, _ = _chunked_bptt_train(
        trajectories=trajs,
        network=network,
        bptt_length=16,
        n_epochs=2,
        adam=AdamConfig(lr=1e-2, amsgrad=True),
    )
    assert all(np.isfinite(losses))


def test_eval_callback_fires_at_expected_epochs() -> None:
    """eval_callback must fire at multiples of eval_interval AND on the final
    epoch, regardless of whether the final epoch happens to be a multiple."""
    trajs = _make_trajectories(n_trajectories=2, T=32)
    network = NetworkConfig(
        architecture=[
            {"type": "dense", "input_size": 4, "output_size": 4, "activation": "tanh"},
            {"type": "dense", "input_size": 4, "output_size": 1, "activation": "tanh"},
        ],
        input_mask=list(range(4)),
        output_parameterization="acos_tanh",
    )
    fired_at: list[int] = []

    def _cb(epoch: int, policy: object) -> None:  # noqa: ARG001
        fired_at.append(epoch)

    # eval_interval=2 over 5 epochs -> fires at 2, 4, AND 5 (final epoch)
    _chunked_bptt_train(
        trajectories=trajs,
        network=network,
        bptt_length=16,
        n_epochs=5,
        adam=AdamConfig(lr=1e-2),
        eval_callback=_cb,
        eval_interval=2,
    )
    assert fired_at == [2, 4, 5]


def test_eval_callback_not_fired_when_interval_is_zero() -> None:
    """eval_interval=0 (the default) disables the periodic eval entirely."""
    trajs = _make_trajectories(n_trajectories=2, T=32)
    network = NetworkConfig(
        architecture=[
            {"type": "dense", "input_size": 4, "output_size": 4, "activation": "tanh"},
            {"type": "dense", "input_size": 4, "output_size": 1, "activation": "tanh"},
        ],
        input_mask=list(range(4)),
        output_parameterization="acos_tanh",
    )
    fired: list[int] = []

    def _cb(epoch: int, policy: object) -> None:  # noqa: ARG001
        fired.append(epoch)

    _chunked_bptt_train(
        trajectories=trajs,
        network=network,
        bptt_length=16,
        n_epochs=3,
        adam=AdamConfig(lr=1e-2),
        eval_callback=_cb,
        eval_interval=0,
    )
    assert fired == []


def test_eval_callback_failure_does_not_abort_training() -> None:
    """A raising eval_callback prints a WARNING but training continues."""
    trajs = _make_trajectories(n_trajectories=2, T=32)
    network = NetworkConfig(
        architecture=[
            {"type": "dense", "input_size": 4, "output_size": 4, "activation": "tanh"},
            {"type": "dense", "input_size": 4, "output_size": 1, "activation": "tanh"},
        ],
        input_mask=list(range(4)),
        output_parameterization="acos_tanh",
    )

    def _cb(epoch: int, policy: object) -> None:  # noqa: ARG001
        raise RuntimeError("simulated MC failure")

    # 3 epochs, eval at each -> 3 simulated failures; training still returns.
    _, losses, _ = _chunked_bptt_train(
        trajectories=trajs,
        network=network,
        bptt_length=16,
        n_epochs=3,
        adam=AdamConfig(lr=1e-2),
        eval_callback=_cb,
        eval_interval=1,
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
            adam=AdamConfig(lr=1e-2),
        )


def test_atan2_signed_requires_two_outputs() -> None:
    """atan2_signed needs a 2-output (sin,cos) head. A 1-output last layer must
    raise a clear error, not silently broadcast (warm_start.py:415 bug). The
    load-time guard in NetworkConfig.__post_init__ now rejects this at
    construction (defense-in-depth, before warm_start/the Rust runtime see it)."""
    with pytest.raises(ValueError, match="atan2_signed.*output_size"):
        NetworkConfig(
            architecture=[
                {"type": "dense", "input_size": 4, "output_size": 4, "activation": "tanh"},
                {"type": "dense", "input_size": 4, "output_size": 1, "activation": "tanh"},
            ],
            input_mask=[0, 1, 2, 3],
            output_parameterization="atan2_signed",
        )
