"""collect_rollout at episode boundaries, driven by a scripted env.

The env contract (BatchedSimulation.step): a done env's returned obs/aux rows are
already its new episode's s_0; the ended episode's s_T is in
info["terminal_observation"] / info["terminal_aux"].
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

pytest.importorskip("aerocapture_rs")
torch = pytest.importorskip("torch")

from aerocapture.training.rl import train as rl_train  # noqa: E402
from aerocapture.training.rl.config import PPOConfig, RLConfig  # noqa: E402
from aerocapture.training.rl.policy import ValueNetwork  # noqa: E402
from aerocapture.training.rl.ppo import RolloutBuffer  # noqa: E402
from aerocapture.training.torch_mirror.policy import V2Policy  # noqa: E402
from aerocapture.training.torch_mirror.schemas import DenseSpec  # noqa: E402

OBS_DIM, AUX_WIDTH = 3, 7


def _rows(tags: list[float], width: int) -> npt.NDArray[np.float32]:
    return np.repeat(np.array(tags, dtype=np.float32)[:, None], width, axis=1)


class _ScriptedEnv:
    """2 envs, 3 steps. Every obs/aux row is filled with a tag naming its state.

    env 1 is truncated at step 1 (terminal tag 12, reset tag 20); env 0 terminates
    at step 2 (terminal tag 3, reset tag 30).
    """

    n_envs = 2
    obs_dim = OBS_DIM
    # (next tags, done, truncated, terminal tags) per step
    script: list[tuple[list[float], list[bool], list[bool], list[float | None]]] = [
        ([1.0, 11.0], [False, False], [False, False], [None, None]),
        ([2.0, 20.0], [False, True], [False, True], [None, 12.0]),
        ([30.0, 21.0], [True, False], [False, False], [3.0, None]),
    ]

    def __init__(self) -> None:
        self.k = 0

    def step(self, actions: Any) -> tuple[Any, ...]:
        tags, done, truncated, term = self.script[self.k]
        self.k += 1
        info: list[dict[str, Any]] = []
        for i in range(self.n_envs):
            d: dict[str, Any] = {}
            if done[i]:
                d = {
                    "truncated": truncated[i],
                    "final_record": [0.0],
                    "dv_m_s": 0.0,
                    "captured": False,
                    "terminal_observation": [term[i]] * OBS_DIM,
                    "terminal_aux": [term[i]] * AUX_WIDTH,
                }
            info.append(d)
        return _rows(tags, OBS_DIM), np.zeros(2, np.float32), np.array(done), info, _rows(tags, AUX_WIDTH)


class _RecordingShaper:
    cost_kwargs = None

    def __init__(self) -> None:
        self.calls: list[tuple[npt.NDArray[Any], ...]] = []

    def step_reward(self, obs_cur: Any, obs_next: Any, aux_cur: Any, aux_next: Any, absorbing: Any) -> npt.NDArray[np.float64]:
        self.calls.append((obs_cur[:, 0].copy(), obs_next[:, 0].copy(), aux_cur[:, 0].copy(), aux_next[:, 0].copy(), absorbing.copy()))
        return np.zeros(len(obs_cur))


def test_collect_rollout_pairs_each_step_with_its_own_episode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rl_train, "compute_terminal_cost", lambda fr, cost_kwargs: 0.0)
    cfg = RLConfig(n_envs=2, ppo=PPOConfig(rollout_steps=3))
    policy = V2Policy(architecture=[DenseSpec(type="dense", input_size=OBS_DIM, output_size=2, activation="linear")], input_mask=list(range(OBS_DIM)))
    value = ValueNetwork(OBS_DIM, [4], ["tanh", "linear"])
    buf = RolloutBuffer.create(3, 2, OBS_DIM, hidden_shapes=[None])
    shaper = _RecordingShaper()

    rl_train.collect_rollout(
        _ScriptedEnv(),
        policy,
        value,
        buf,
        np.zeros((3, 2), np.float32),
        _rows([0.0, 10.0], OBS_DIM),
        _rows([0.0, 10.0], AUX_WIDTH),
        obs_norm=None,
        ret_norm=None,
        step_calc=shaper,
        cfg=cfg,
        episodic_returns=[],
        episodic_dvs=[],
        episodic_captures=[],
    )

    # (s, s') tags per step for obs and aux: s' of an ending step is its terminal
    # state, and the next step's s is the reset state -- never the terminal one.
    # Only the termination is absorbing; the truncation keeps Phi(s_T).
    expected = [([0, 10], [1, 11], [False, False]), ([1, 11], [2, 12], [False, False]), ([2, 20], [3, 21], [True, False])]
    for (obs_cur, obs_next, aux_cur, aux_next, absorbing), (cur, nxt, absorb) in zip(shaper.calls, expected, strict=True):
        np.testing.assert_array_equal(obs_cur, cur)
        np.testing.assert_array_equal(aux_cur, cur)
        np.testing.assert_array_equal(obs_next, nxt)
        np.testing.assert_array_equal(aux_next, nxt)
        np.testing.assert_array_equal(absorbing, absorb)

    # dones: any episode end (GAE trace cut, BPTT state reset); terminated: absorbing only.
    np.testing.assert_array_equal(buf.dones, [[False, False], [False, True], [True, False]])
    np.testing.assert_array_equal(buf.terminated, [[False, False], [False, False], [True, False]])
