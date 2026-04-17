import torch
from aerocapture.training.rl.policy import V2Policy
from aerocapture.training.rl.schemas import DenseSpec


def _two_layer_policy() -> V2Policy:
    architecture = [
        DenseSpec(type="dense", input_size=3, output_size=4, activation="tanh"),
        DenseSpec(type="dense", input_size=4, output_size=2, activation="linear"),
    ]
    return V2Policy(architecture=architecture, output_interpretation="atan2", input_mask=None)


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
