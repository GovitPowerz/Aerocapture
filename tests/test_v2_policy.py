import torch
from aerocapture.training.rl.policy import V2Policy
from aerocapture.training.rl.schemas import DenseSpec


def _two_layer_policy() -> V2Policy:
    architecture = [
        DenseSpec(type="dense", input_size=3, output_size=4, activation="tanh"),
        DenseSpec(type="dense", input_size=4, output_size=2, activation="linear"),
    ]
    return V2Policy(architecture=architecture, input_mask=None)


def test_v2_policy_forward_shape() -> None:
    policy = _two_layer_policy()
    state = policy.new_state(batch_size=1, device="cpu")
    x = torch.tensor([[0.5, -0.3, 0.1]])
    y, new_state = policy(x, state)
    assert y.shape == (1, 2)
    assert len(new_state) == 2


def test_v2_policy_forward_is_deterministic() -> None:
    torch.manual_seed(0)
    policy = _two_layer_policy()
    state1 = policy.new_state(1, "cpu")
    state2 = policy.new_state(1, "cpu")
    x = torch.tensor([[0.5, -0.3, 0.1]])
    y1, _ = policy(x, state1)
    y2, _ = policy(x, state2)
    torch.testing.assert_close(y1, y2)


def test_v2_policy_log_std_not_in_state_dict_export_contract() -> None:
    policy = _two_layer_policy()
    sd = policy.state_dict()
    assert "log_std" in sd  # log_std IS in state_dict
    # but exporter filters it out; separately tested in export round-trip.


def test_v2_policy_forward_mean_logstd_dense_shapes() -> None:
    from aerocapture.training.rl.policy import V2Policy
    from aerocapture.training.rl.schemas import DenseSpec

    arch: list[DenseSpec] = [
        DenseSpec(type="dense", input_size=4, output_size=8, activation="tanh"),
        DenseSpec(type="dense", input_size=8, output_size=2, activation="linear"),
    ]
    p = V2Policy(architecture=arch, input_mask=None)
    obs = torch.zeros(3, 4)
    state = p.new_state(3, "cpu")
    mean, log_std, new_state = p.forward_mean_logstd(obs, state)
    assert mean.shape == (3, 2)
    assert log_std.shape == (2,)
    assert len(new_state) == len(arch)


def test_v2_policy_sample_dense_shapes() -> None:
    from aerocapture.training.rl.policy import V2Policy
    from aerocapture.training.rl.schemas import DenseSpec

    arch: list[DenseSpec] = [
        DenseSpec(type="dense", input_size=4, output_size=8, activation="tanh"),
        DenseSpec(type="dense", input_size=8, output_size=2, activation="linear"),
    ]
    p = V2Policy(architecture=arch, input_mask=None)
    obs = torch.zeros(3, 4)
    state = p.new_state(3, "cpu")
    bank, raw, log_prob, new_state = p.sample(obs, state)
    assert bank.shape == (3,)
    assert raw.shape == (3, 2)
    assert log_prob.shape == (3,)
    # Determinism check: the same seed + same obs + same state yields the same raw.
    torch.manual_seed(0)
    b1, r1, _, _ = p.sample(obs, state)
    torch.manual_seed(0)
    b2, r2, _, _ = p.sample(obs, state)
    torch.testing.assert_close(r1, r2, rtol=0, atol=0)
    torch.testing.assert_close(b1, b2, rtol=0, atol=0)


def test_v2_policy_evaluate_dense_shapes_and_grad() -> None:
    from aerocapture.training.rl.policy import V2Policy
    from aerocapture.training.rl.schemas import DenseSpec

    arch: list[DenseSpec] = [
        DenseSpec(type="dense", input_size=4, output_size=8, activation="tanh"),
        DenseSpec(type="dense", input_size=8, output_size=2, activation="linear"),
    ]
    p = V2Policy(architecture=arch, input_mask=None)
    T, B = 5, 3
    obs_seq = torch.randn(T, B, 4)
    raw_seq = torch.randn(T, B, 2)
    dones_seq = torch.zeros(T, B, dtype=torch.bool)
    state_0 = p.new_state(B, "cpu")
    log_probs_seq, entropy_seq = p.evaluate(obs_seq, state_0, dones_seq, raw_seq)
    assert log_probs_seq.shape == (T, B)
    assert entropy_seq.shape == (T, B)
    # Gradient flows through all T timesteps into policy parameters.
    loss = log_probs_seq.sum()
    loss.backward()
    grads = [p_.grad for p_ in p.parameters() if p_.grad is not None]
    assert len(grads) > 0
    assert any(g.abs().sum().item() > 0 for g in grads)


def test_v2_policy_evaluate_with_gru_grad_flows_through_time() -> None:
    from aerocapture.training.rl.policy import V2Policy
    from aerocapture.training.rl.schemas import DenseSpec, GruSpec

    arch: list[DenseSpec | GruSpec] = [
        DenseSpec(type="dense", input_size=3, output_size=4, activation="tanh"),
        GruSpec(type="gru", input_size=4, hidden_size=4),
        DenseSpec(type="dense", input_size=4, output_size=2, activation="linear"),
    ]
    p = V2Policy(architecture=arch, input_mask=None)
    T, B = 6, 2
    obs_seq = torch.randn(T, B, 3)
    raw_seq = torch.randn(T, B, 2)
    dones_seq = torch.zeros(T, B, dtype=torch.bool)
    state_0 = p.new_state(B, "cpu")
    log_probs_seq, _ = p.evaluate(obs_seq, state_0, dones_seq, raw_seq)
    assert log_probs_seq.shape == (T, B)
    # log_probs_seq[-1] depends on obs_seq[0..T-1] through the GRU state; gradient
    # w.r.t. obs_seq[0] must be non-zero.
    obs_seq.requires_grad_(True)
    lp, _ = p.evaluate(obs_seq, state_0, dones_seq, raw_seq)
    lp[-1].sum().backward()
    assert obs_seq.grad is not None
    assert obs_seq.grad[0].abs().sum().item() > 0


def test_v2_policy_evaluate_resets_state_on_done() -> None:
    """When dones_seq[t] is True, the state at step t+1 is zeroed per-env."""
    import torch
    from aerocapture.training.rl.policy import V2Policy
    from aerocapture.training.rl.schemas import DenseSpec, GruSpec

    arch: list[DenseSpec | GruSpec] = [
        DenseSpec(type="dense", input_size=2, output_size=4, activation="tanh"),
        GruSpec(type="gru", input_size=4, hidden_size=4),
        DenseSpec(type="dense", input_size=4, output_size=2, activation="linear"),
    ]
    p = V2Policy(architecture=arch, input_mask=None)

    # Two env batch: env 0 sees done at t=1, env 1 never dones.
    # Build an observation sequence that would otherwise push GRU state far from zero.
    T, B = 3, 2
    obs_seq = torch.ones(T, B, 2) * 0.5
    raw_seq = torch.zeros(T, B, 2)
    dones_seq = torch.zeros(T, B, dtype=torch.bool)
    dones_seq[1, 0] = True  # env 0 terminates after step 1

    state_0 = p.new_state(B, "cpu")
    lp_with_done, _ = p.evaluate(obs_seq, state_0, dones_seq, raw_seq)
    lp_no_done, _ = p.evaluate(obs_seq, state_0, torch.zeros_like(dones_seq), raw_seq)

    # env 1 (no done) must match the baseline at every step.
    torch.testing.assert_close(lp_with_done[:, 1], lp_no_done[:, 1], rtol=0, atol=0)
    # env 0 after done: log_prob at step 2 differs from the baseline (state was zeroed).
    assert (lp_with_done[2, 0] - lp_no_done[2, 0]).abs().item() > 1e-6


def test_zero_state_where_done_handles_lstm_tuple_state() -> None:
    """`_zero_state_where_done` now recurses into tuple state entries (Task 8 /
    Phase 2a LSTM MVP). The Phase 0 guard that raised on tuples is retired;
    non-tuple, non-Tensor, non-None entries still raise TypeError.
    """
    from aerocapture.training.rl.policy import _zero_state_where_done

    # LSTM state: tuple of (h, c) tensors.
    h = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    c = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    state = [None, (h, c)]  # dense layer (None) + LSTM layer
    done_mask = torch.tensor([True, False])

    new_state = _zero_state_where_done(state, done_mask)
    assert new_state[0] is None
    new_h, new_c = new_state[1]
    # done env (index 0) zeroed in both h and c
    assert torch.all(new_h[0] == 0.0)
    assert torch.all(new_c[0] == 0.0)
    # not-done env (index 1) unchanged
    assert torch.equal(new_h[1], h[1])
    assert torch.equal(new_c[1], c[1])
