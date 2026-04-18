"""PPO update rule and rollout buffer for aerocapture RL training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from aerocapture.training.rl.policy import V2Policy, ValueNetwork

if TYPE_CHECKING:
    from aerocapture.training.rl.normalizers import ObsNormalizer


@dataclass
class RolloutBuffer:
    """Fixed-size per-env rollout buffer; (n_steps, n_envs, ...) tensors.

    ``raw_actions`` stores the 2D Gaussian sample (n_steps, n_envs, 2) so that
    ``ppo_update`` replays the *exact* point at which ``old_log_probs`` were
    evaluated. The scalar bank angle sent to the env is atan2(raw[0], raw[1]).

    Hidden-state fields (Phase 1.5):
        h_initial: list[ndarray | None], one entry per layer. Dense layers have
                   entry == None; recurrent layers have shape (n_envs, H).
        h_final:   same shape as h_initial; state at t=n_steps (seed for next rollout).
        states:    list[ndarray | None], one per layer. Dense layers None;
                   recurrent layers have shape (n_steps, n_envs, H). states[t]
                   stores the state *before* step t was computed.
    """

    n_steps: int
    n_envs: int
    obs_dim: int
    obs: npt.NDArray[np.float32]
    raw_actions: npt.NDArray[np.float32]
    log_probs: npt.NDArray[np.float32]
    rewards: npt.NDArray[np.float32]
    values: npt.NDArray[np.float32]
    dones: npt.NDArray[np.bool_]
    h_initial: list[npt.NDArray[np.float32] | None]
    h_final: list[npt.NDArray[np.float32] | None]
    states: list[npt.NDArray[np.float32] | None]

    @classmethod
    def create(
        cls,
        n_steps: int,
        n_envs: int,
        obs_dim: int,
        hidden_shapes: list[tuple[int, ...] | None] | None = None,
    ) -> RolloutBuffer:
        """Create a rollout buffer.

        hidden_shapes: list of per-layer hidden-state shapes (excluding the
                       batch axis). None entries are dense/stateless layers.
                       If hidden_shapes is None, defaults to a zero-length
                       list (feedforward-only, no state tracking).
        """
        if hidden_shapes is None:
            hidden_shapes = []
        h_initial: list[npt.NDArray[np.float32] | None] = [None if s is None else np.zeros((n_envs,) + s, dtype=np.float32) for s in hidden_shapes]
        h_final: list[npt.NDArray[np.float32] | None] = [None if s is None else np.zeros((n_envs,) + s, dtype=np.float32) for s in hidden_shapes]
        states: list[npt.NDArray[np.float32] | None] = [None if s is None else np.zeros((n_steps, n_envs) + s, dtype=np.float32) for s in hidden_shapes]
        return cls(
            n_steps=n_steps,
            n_envs=n_envs,
            obs_dim=obs_dim,
            obs=np.zeros((n_steps, n_envs, obs_dim), dtype=np.float32),
            raw_actions=np.zeros((n_steps, n_envs, 2), dtype=np.float32),
            log_probs=np.zeros((n_steps, n_envs), dtype=np.float32),
            rewards=np.zeros((n_steps, n_envs), dtype=np.float32),
            values=np.zeros((n_steps, n_envs), dtype=np.float32),
            dones=np.zeros((n_steps, n_envs), dtype=np.bool_),
            h_initial=h_initial,
            h_final=h_final,
            states=states,
        )


def compute_gae(
    rewards: npt.NDArray[np.float32],
    values: npt.NDArray[np.float32],
    next_values: npt.NDArray[np.float32],
    dones: npt.NDArray[np.bool_],
    gamma: float,
    lam: float,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """GAE-lambda with per-step next-value bootstrap.

    `values[t]`: V(s_t). `next_values[t]`: V(s_{t+1}) -- caller supplies
    V(terminal_obs) for truncated steps and V(reset_obs) for continuing
    steps where the episode did not end. `dones[t]` should be True *only*
    for true terminations; truncation sets `done=False` so the bootstrap
    is kept.
    """
    n = rewards.shape[0]
    adv = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(n)):
        not_done = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_values[t] * not_done - values[t]
        gae = delta + gamma * lam * not_done * gae
        adv[t] = gae
    ret = adv + values
    return adv, ret


def ppo_update_bptt(
    policy: V2Policy,
    value: ValueNetwork,
    optim: torch.optim.Optimizer,
    buf: RolloutBuffer,
    advantages: npt.NDArray[np.float32],
    returns: npt.NDArray[np.float32],
    bptt_length: int,
    clip_range: float,
    update_epochs: int,
    minibatches: int,
    entropy_coef: float,
    value_coef: float,
    max_grad_norm: float,
    target_kl: float | None = None,
    obs_norm: ObsNormalizer | None = None,
) -> dict[str, float]:
    """Chunked truncated-BPTT PPO update.

    Splits each env's rollout into rollout_steps // bptt_length chunks.
    Minibatches partition the env axis; within each minibatch, the time axis
    stays intact and gradients flow through `bptt_length` timesteps per chunk.
    """
    n_steps, n_envs = buf.rewards.shape
    assert n_steps % bptt_length == 0, "rollout_steps must be divisible by bptt_length"
    n_chunks = n_steps // bptt_length

    envs_per_minibatch = max(1, n_envs // minibatches)

    # Normalize advantages once over the full rollout.
    adv = advantages.astype(np.float32)
    adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)

    # Pre-normalize observations if an ObsNormalizer is active (same as feedforward path).
    obs_for_eval = buf.obs
    if obs_norm is not None:
        obs_for_eval = obs_norm.normalize(obs_for_eval.reshape(-1, buf.obs_dim)).reshape(buf.obs.shape)

    metrics_acc: dict[str, list[float]] = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "approx_kl": [],
        "clip_frac": [],
    }
    epochs_run = 0

    for _ in range(update_epochs):
        # Shuffle both axes each epoch to decorrelate gradient updates (Ni et al. 2021
        # recommend joint env+chunk shuffle for recurrent PPO). Chunks are independent
        # under truncated BPTT because each chunk's seed state is detached from the
        # stored snapshot, so chunk order is free to vary.
        env_indices = np.arange(n_envs)
        np.random.shuffle(env_indices)
        chunk_indices = np.arange(n_chunks)
        np.random.shuffle(chunk_indices)
        epoch_kls: list[float] = []

        for mb_start in range(0, n_envs, envs_per_minibatch):
            mb = env_indices[mb_start : mb_start + envs_per_minibatch]
            if len(mb) == 0:
                continue

            for c in chunk_indices:
                lo, hi = c * bptt_length, (c + 1) * bptt_length
                mb_obs = torch.from_numpy(obs_for_eval[lo:hi, mb]).float()
                mb_raw = torch.from_numpy(buf.raw_actions[lo:hi, mb]).float()
                mb_old_lp = torch.from_numpy(buf.log_probs[lo:hi, mb]).float()
                mb_adv = torch.from_numpy(adv_norm[lo:hi, mb]).float()
                mb_ret = torch.from_numpy(returns[lo:hi, mb].astype(np.float32)).float()
                mb_dones = torch.from_numpy(buf.dones[lo:hi, mb])

                # Seed chunk c's initial state from the snapshot stored during rollout.
                # buf.states[li][lo] is the state *before* step `lo`, which is exactly
                # what chunk c (starting at timestep `lo`) needs. Detach to stop
                # gradient flow across chunks.
                h_chunk_detached: list = []
                for layer_s in buf.states:
                    if layer_s is None:
                        h_chunk_detached.append(None)
                    else:
                        h_chunk_detached.append(torch.from_numpy(layer_s[lo, mb]).float().detach())

                new_lp_seq, entropy_seq = policy.evaluate(
                    mb_obs,
                    h_chunk_detached,
                    mb_dones,
                    mb_raw,
                )  # shapes (L, |mb|) each

                # Feedforward critic -- flatten time x env for the value predictions.
                mb_obs_flat = mb_obs.reshape(-1, buf.obs_dim)
                v_pred_flat = value(mb_obs_flat)
                v_pred = v_pred_flat.reshape(bptt_length, -1)

                # PPO clipped surrogate across the (L, |mb|) sample axis.
                ratio = (new_lp_seq - mb_old_lp).exp()
                s1 = ratio * mb_adv
                s2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * mb_adv
                policy_loss = -torch.min(s1, s2).mean()
                value_loss = 0.5 * ((v_pred - mb_ret) ** 2).mean()
                entropy = entropy_seq.mean()

                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(policy.parameters()) + list(value.parameters()), max_grad_norm)
                optim.step()

                with torch.no_grad():
                    approx_kl = (mb_old_lp - new_lp_seq).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > clip_range).float().mean().item()
                metrics_acc["policy_loss"].append(policy_loss.item())
                metrics_acc["value_loss"].append(value_loss.item())
                metrics_acc["entropy"].append(entropy.item())
                metrics_acc["approx_kl"].append(approx_kl)
                metrics_acc["clip_frac"].append(clip_frac)
                epoch_kls.append(approx_kl)

        epochs_run += 1
        if target_kl is not None and epoch_kls and float(np.mean(epoch_kls)) > target_kl:
            break

    result = {k: float(np.mean(v)) for k, v in metrics_acc.items()}
    result["epochs_run"] = float(epochs_run)
    return result
