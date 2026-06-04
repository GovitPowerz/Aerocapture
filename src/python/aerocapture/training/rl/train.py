"""RL training CLI and outer loop.

Usage:
    python -m aerocapture.training.rl.train <config.toml> \\
        [--algorithm ppo|sac] [--total-steps N] [--no-tui] [--skip-report]

Produces training_output/neural_network_rl/ with best_model.json, rl_training_*.jsonl,
checkpoint.pt, and optionally report.pdf.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import tomli_w
import torch

from aerocapture.training.rl.config import RLConfig
from aerocapture.training.rl.display import make_display
from aerocapture.training.rl.env import AerocaptureVecEnv
from aerocapture.training.rl.export import export_policy_to_json, export_v2_policy_to_json
from aerocapture.training.rl.logger import RLLogger
from aerocapture.training.rl.normalizers import ObsNormalizer, ReturnNormalizer
from aerocapture.training.rl.policy import GaussianPolicy, V2Policy, ValueNetwork
from aerocapture.training.rl.policy import np_state_to_torch as _np_state_to_torch
from aerocapture.training.rl.policy import torch_state_to_np as _torch_state_to_np
from aerocapture.training.rl.ppo import RolloutBuffer, compute_gae, ppo_update_bptt
from aerocapture.training.rl.rewards import StepRewardCalculator, compute_terminal_cost
from aerocapture.training.rl.sac import SACAgent


def _resolve_output_dir(cfg: RLConfig) -> Path:
    """Derive the per-scheme output directory from the TOML's `[data] neural_network`
    path. Mirrors `train.py`'s PSO/GA path so every (variant × algorithm) tuple
    lands in its own folder automatically and compare_guidance / deploy paths
    line up with where training actually wrote.
    """
    nn_path = cfg.raw_toml.get("data", {}).get("neural_network")
    if not nn_path:
        raise SystemExit(
            'ERROR: RL TOML must set `[data] neural_network = "training_output/<scheme>/best_model.json"` so the output dir can be derived from it.'
        )
    parent = Path(nn_path).parent
    if not str(parent).startswith("training_output/"):
        raise SystemExit(
            f"ERROR: [data] neural_network = '{nn_path}' must live under 'training_output/' so checkpoints and report artifacts land alongside the deploy JSON."
        )
    return parent


# Column indices in the 52-element final_record array (see runner.rs).
_IDX_ECC = 9
_IDX_IFINAL = 31


# ---------------------------------------------------------------------------
# Shared helpers (used by both PPO and SAC loops)
# ---------------------------------------------------------------------------


def _parse_network_config(cfg: RLConfig) -> tuple[list[int], list[Any], int]:
    """Parse [network] from the RL TOML.

    Returns (input_mask, architecture, input_dim).

    Supports both v1 (layer_sizes + activations, dense-only) and v2
    ([[network.architecture]] array-of-tables with dense + gru layers) formats.
    v2 takes precedence when present.
    """
    from pydantic import TypeAdapter

    from aerocapture.training.rl.schemas import LayerSpec

    net = cfg.raw_toml.get("network", {})
    input_mask: list[int] = net.get("input_mask", list(range(16)))

    arch_raw = net.get("architecture")
    if arch_raw is not None:
        adapter: TypeAdapter[list[LayerSpec]] = TypeAdapter(list[LayerSpec])
        architecture: list[Any] = adapter.validate_python(list(arch_raw))
        input_dim = architecture[0].input_size
        if len(input_mask) != input_dim:
            raise ValueError(f"[network] input_mask length ({len(input_mask)}) must equal architecture[0].input_size ({input_dim})")
        return input_mask, architecture, input_dim

    # v1 path: layer_sizes + activations
    toml_layers: list[int] = net.get("layer_sizes", [16, 64, 64, 2])
    activations: list[str] = net.get("activations", ["tanh", "tanh", "linear"])
    input_dim = len(input_mask)
    if toml_layers[0] != input_dim:
        raise ValueError(f"layer_sizes[0]={toml_layers[0]} must equal len(input_mask)={input_dim}")
    if len(toml_layers) - 1 != len(activations):
        raise ValueError(f"len(layer_sizes)-1={len(toml_layers) - 1} must equal len(activations)={len(activations)}")
    adapter_v1: TypeAdapter[list[LayerSpec]] = TypeAdapter(list[LayerSpec])
    architecture_v1: list[Any] = adapter_v1.validate_python(
        [
            {
                "type": "dense",
                "input_size": toml_layers[i],
                "output_size": toml_layers[i + 1],
                "activation": activations[i],
            }
            for i in range(len(toml_layers) - 1)
        ]
    )
    return input_mask, architecture_v1, input_dim


def _dense_only_shapes(architecture: list[Any]) -> tuple[list[int], list[str]]:
    """Derive v1-shaped (layer_sizes, activations) from a DenseSpec-only architecture.

    Still load-bearing for the SAC path (_run_sac), which keeps GaussianPolicy
    until Phase 1.6's SAC-GRU migration. PPO uses V2Policy directly (Task 5).
    Raises NotImplementedError if any non-dense layer is present.
    """
    from aerocapture.training.rl.schemas import DenseSpec

    if not all(isinstance(s, DenseSpec) for s in architecture):
        raise NotImplementedError("SAC / GaussianPolicy path requires dense-only architecture; SAC-GRU lands in Phase 1.6.")
    layer_sizes = [s.output_size for s in architecture]
    activations = [s.activation for s in architecture]
    return layer_sizes, activations


def _generate_seed_model(cfg: RLConfig, path: Path) -> None:
    """Export a randomly-initialized V2Policy as a seed model JSON for BatchedSimulation."""
    input_mask, architecture, _input_dim = _parse_network_config(cfg)
    policy = V2Policy(
        architecture=architecture,
        input_mask=input_mask,
        initial_log_std=cfg.ppo.initial_log_std,
        min_log_std=cfg.ppo.min_log_std,
    )
    export_v2_policy_to_json(policy, str(path), obs_normalizer=None)


def _describe_rl_architecture(cfg: RLConfig) -> None:
    """Fail-fast chain check + stdout description of the RL architecture.

    _parse_network_config already Pydantic-validates per-spec; this helper
    adds the chain check (layer i output -> layer i+1 input) and prints a
    human-readable summary for operator feedback at training start.
    """
    from aerocapture.training.config import _layer_output_size, describe_architecture

    input_mask, architecture, _input_dim = _parse_network_config(cfg)

    # Chain consistency: prev.output == next.input.
    for i in range(len(architecture) - 1):
        prev_out = _layer_output_size(architecture[i])
        next_in = architecture[i + 1].input_size
        if prev_out != next_in:
            raise ValueError(
                f"[network.architecture] chain mismatch at layer {i}->{i + 1}: "
                f"layer {i} ({architecture[i].type}) produces output={prev_out}, "
                f"but layer {i + 1} ({architecture[i + 1].type}) expects input={next_in}"
            )

    print(describe_architecture(architecture), file=sys.stderr)
    print(f"  input_mask: {len(input_mask)} indices", file=sys.stderr)


def _build_shaper_and_norms(
    cfg: RLConfig, input_mask: list[int], gamma: float, toml_path: Path
) -> tuple[StepRewardCalculator, ReturnNormalizer | None, ObsNormalizer | None]:
    from aerocapture.training.report import read_cost_kwargs

    step_calc = StepRewardCalculator(
        input_mask=input_mask,
        gamma=gamma,
        corridor_weight=cfg.reward.corridor_weight,
        energy_rate_weight=cfg.reward.energy_rate_weight,
        constraint_weight=cfg.reward.constraint_weight,
        apoapsis_weight=cfg.reward.apoapsis_weight,
        eccentricity_weight=cfg.reward.eccentricity_weight,
        energy_scale=cfg.reward.energy_scale,
        cost_kwargs=read_cost_kwargs(toml_path),
    )
    ret_norm = ReturnNormalizer(gamma=gamma, warmup_steps=cfg.reward.norm_warmup_steps) if cfg.reward.normalize_returns else None
    obs_norm = ObsNormalizer(obs_dim=len(input_mask)) if cfg.reward.normalize_obs else None
    return step_calc, ret_norm, obs_norm


def _terminal_observations(info: list[dict[str, Any]], done: npt.NDArray[np.bool_], obs_dim: int) -> npt.NDArray[np.float32]:
    """Extract per-env terminal observation from info dicts. Fallback: zeros."""
    out = np.zeros((len(info), obs_dim), dtype=np.float32)
    for i, d in enumerate(done):
        if d and "terminal_observation" in info[i]:
            out[i] = np.asarray(info[i]["terminal_observation"], dtype=np.float32)
    return out


def _validate_deterministic(
    policy: V2Policy,
    toml_path: Path,
    output_dir: Path,
    cfg: RLConfig,
    input_mask: list[int],
    obs_norm: ObsNormalizer | None = None,
) -> dict[str, Any]:
    """Export deterministic V2Policy + run validation batch; return RMS cost + capture rate."""
    import aerocapture_rs  # type: ignore[import]

    from aerocapture.training.evaluate import VALIDATION_SEED_OFFSET, compute_cost, make_reserved_seeds
    from aerocapture.training.report import read_cost_kwargs

    tmp_json = output_dir / "gen_current_model.json"
    export_v2_policy_to_json(policy, str(tmp_json), obs_normalizer=obs_norm)

    base_seed = int(cfg.raw_toml.get("monte_carlo", {}).get("seed", 42))
    seeds = make_reserved_seeds(base_seed, VALIDATION_SEED_OFFSET, cfg.validation_n_sims)

    overrides_list = [{"data.neural_network": str(tmp_json), "monte_carlo.seed": s, "simulation.n_sims": 1} for s in seeds]
    results = aerocapture_rs.run_batch(str(toml_path), overrides_list)
    fr = results.final_records

    cost_kwargs = read_cost_kwargs(toml_path)
    rms_cost = float(compute_cost(fr, **cost_kwargs))
    capture_rate = float(np.mean((fr[:, _IDX_IFINAL] == 3) & (fr[:, _IDX_ECC] < 1.0)))
    return {"val_rms_cost": rms_cost, "val_capture_rate": capture_rate}


def _validate_deterministic_v1(
    policy: GaussianPolicy,
    toml_path: Path,
    output_dir: Path,
    cfg: RLConfig,
    input_mask: list[int],
    obs_norm: ObsNormalizer | None = None,
) -> dict[str, Any]:
    """Legacy v1 validate path for the SAC loop (still on GaussianPolicy).

    SAC will migrate to V2Policy in Phase 1.6; until then we keep a v1-shaped
    twin so the SAC path doesn't have to resurrect itself through Task 5's
    V2-only validation helper.
    """
    import aerocapture_rs  # type: ignore[import]

    from aerocapture.training.evaluate import VALIDATION_SEED_OFFSET, compute_cost, make_reserved_seeds
    from aerocapture.training.report import read_cost_kwargs

    tmp_json = output_dir / "gen_current_model.json"
    export_policy_to_json(policy, tmp_json, input_mask, obs_normalizer=obs_norm)

    base_seed = int(cfg.raw_toml.get("monte_carlo", {}).get("seed", 42))
    seeds = make_reserved_seeds(base_seed, VALIDATION_SEED_OFFSET, cfg.validation_n_sims)

    overrides_list = [{"data.neural_network": str(tmp_json), "monte_carlo.seed": s, "simulation.n_sims": 1} for s in seeds]
    results = aerocapture_rs.run_batch(str(toml_path), overrides_list)
    fr = results.final_records

    cost_kwargs = read_cost_kwargs(toml_path)
    rms_cost = float(compute_cost(fr, **cost_kwargs))
    capture_rate = float(np.mean((fr[:, _IDX_IFINAL] == 3) & (fr[:, _IDX_ECC] < 1.0)))
    return {"val_rms_cost": rms_cost, "val_capture_rate": capture_rate}


def _run_final_eval(toml_path: Path, best_model: Path, cfg: RLConfig) -> None:
    import aerocapture_rs  # type: ignore[import]

    from aerocapture.training.evaluate import FINAL_EVAL_SEED_OFFSET, make_reserved_seeds
    from aerocapture.training.report import print_eval_summary, read_cost_kwargs

    n_sims = cfg.validation_n_sims
    base_seed = int(cfg.raw_toml.get("monte_carlo", {}).get("seed", 42))
    seeds = make_reserved_seeds(base_seed, FINAL_EVAL_SEED_OFFSET, n_sims)

    overrides_list = [{"data.neural_network": str(best_model), "monte_carlo.seed": s, "simulation.n_sims": 1} for s in seeds]
    print(f"\nRunning {n_sims}-sim final evaluation...", file=sys.stderr)
    results = aerocapture_rs.run_batch(str(toml_path), overrides_list)

    cost_kwargs = read_cost_kwargs(toml_path)
    print_eval_summary(results.final_records, n_sims, cost_kwargs=cost_kwargs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Train neural_network guidance via PPO or SAC.")
    ap.add_argument("toml_path")
    ap.add_argument("--algorithm", choices=["ppo", "sac"], default=None)
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--n-envs", type=int, default=None)
    ap.add_argument("--rollout-steps", type=int, default=None)
    ap.add_argument("--validation-n-sims", type=int, default=None)
    ap.add_argument("--validation-interval-updates", type=int, default=None)
    ap.add_argument("--data-neural-network", type=Path, default=None, help="Override path to neural network model JSON")
    ap.add_argument("--from-scratch", "-fs", action="store_true", help="Initialize with random weights (no seed model required)")
    ap.add_argument("--learning-rate", type=float, default=None, help="Override PPO/SAC learning rate")
    ap.add_argument("--clip-range", type=float, default=None, help="Override PPO clip range")
    ap.add_argument("--entropy-coef", type=float, default=None, help="Override PPO entropy coefficient")
    ap.add_argument("--min-log-std", type=float, default=None, help="Override PPO min_log_std floor")
    ap.add_argument("--update-epochs", type=int, default=None, help="Override PPO update epochs per rollout")
    ap.add_argument("--lr-anneal-start", type=float, default=None, help="Override PPO LR anneal start fraction")
    ap.add_argument("--target-kl", type=float, default=None, help="Override PPO target_kl early-stop threshold")
    ap.add_argument("--no-tui", action="store_true")
    ap.add_argument("--skip-report", action="store_true")
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output dir. Default: derived from TOML [data] neural_network parent.",
    )
    args = ap.parse_args()

    overrides: dict[str, Any] = {}
    if args.algorithm:
        overrides["algorithm"] = args.algorithm
    if args.total_steps is not None:
        overrides["total_env_steps"] = args.total_steps
    if args.n_envs is not None:
        overrides["n_envs"] = args.n_envs
    if args.validation_n_sims is not None:
        overrides["validation_n_sims"] = args.validation_n_sims
    if args.validation_interval_updates is not None:
        overrides["validation_interval_updates"] = args.validation_interval_updates

    ppo_overrides: dict[str, Any] = {}
    if args.rollout_steps is not None:
        ppo_overrides["rollout_steps"] = args.rollout_steps
    if args.learning_rate is not None:
        ppo_overrides["learning_rate"] = args.learning_rate
    if args.clip_range is not None:
        ppo_overrides["clip_range"] = args.clip_range
    if args.entropy_coef is not None:
        ppo_overrides["entropy_coef"] = args.entropy_coef
    if args.min_log_std is not None:
        ppo_overrides["min_log_std"] = args.min_log_std
    if args.update_epochs is not None:
        ppo_overrides["update_epochs"] = args.update_epochs
    if args.lr_anneal_start is not None:
        ppo_overrides["lr_anneal_start"] = args.lr_anneal_start
    if args.target_kl is not None:
        ppo_overrides["target_kl"] = args.target_kl

    cfg = RLConfig.from_toml(Path(args.toml_path), overrides=overrides or None, ppo_overrides=ppo_overrides or None)

    if args.from_scratch and args.data_neural_network is not None:
        ap.error("--from-scratch and --data-neural-network are mutually exclusive")

    # Derive output_dir from the TOML [data] neural_network parent if not overridden.
    # Each (variant × algorithm) gets its own folder automatically.
    if args.output_dir is None:
        args.output_dir = _resolve_output_dir(cfg)

    env_overrides: dict[str, Any] | None = None
    if args.data_neural_network is not None:
        env_overrides = {"data.neural_network": str(args.data_neural_network)}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    warmstart_json: Path | None = None
    if args.data_neural_network is not None and not args.from_scratch:
        warmstart_json = args.data_neural_network
        for stale in ("checkpoint.pt", "best_model.json"):
            p = args.output_dir / stale
            if p.exists():
                p.unlink()
                print(f"Cleared stale {stale} for warm-start", file=sys.stderr)

    if args.from_scratch:
        for stale in ("checkpoint.pt", "best_model.json"):
            p = args.output_dir / stale
            if p.exists():
                p.unlink()
        seed_model_path = args.output_dir / "seed_model.json"
        _generate_seed_model(cfg, seed_model_path)
        print(f"Generated seed model: {seed_model_path}", file=sys.stderr)
        env_overrides = env_overrides or {}
        env_overrides["data.neural_network"] = str(seed_model_path)

    # Architecture summary (fail-fast chain check + stdout description).
    _describe_rl_architecture(cfg)

    config_hash = hashlib.sha256(json.dumps(cfg.raw_toml, sort_keys=True).encode()).hexdigest()[:12]
    (args.output_dir / "config_resolved.toml").write_bytes(tomli_w.dumps(cfg.raw_toml).encode())

    logger = RLLogger(args.output_dir, config_hash)
    display = make_display(cfg.total_env_steps, enabled=not args.no_tui and sys.stdout.isatty())

    interrupted = {"v": False}

    def _on_sigint(_s: int, _f: Any) -> None:
        interrupted["v"] = True

    prev_handler = signal.signal(signal.SIGINT, _on_sigint)
    try:
        if cfg.algorithm == "ppo":
            _run_ppo(cfg, Path(args.toml_path), args.output_dir, logger, display, interrupted, args.resume, env_overrides, warmstart_json)
        elif cfg.algorithm == "sac":
            _run_sac(cfg, Path(args.toml_path), args.output_dir, logger, display, interrupted, env_overrides)
        else:
            raise NotImplementedError(f"algorithm {cfg.algorithm!r} not supported")
    finally:
        signal.signal(signal.SIGINT, prev_handler)
        display.close()
        logger.close()

    best_model = args.output_dir / "best_model.json"
    if best_model.exists():
        _run_final_eval(Path(args.toml_path), best_model, cfg)

    if not args.skip_report:
        from aerocapture.training.rl.report_rl import generate_report

        generate_report(args.output_dir, Path(args.toml_path))


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------


def _save_ppo_checkpoint(
    output_dir: Path,
    policy: V2Policy,
    value: ValueNetwork,
    optim: torch.optim.Optimizer,
    update_idx: int,
    env_steps: int,
    best_val_cost: float,
    ret_norm: ReturnNormalizer | None,
    obs_norm: ObsNormalizer | None,
) -> None:
    torch.save(
        {
            "policy": policy.state_dict(),
            "value": value.state_dict(),
            "optim": optim.state_dict(),
            "update_idx": update_idx,
            "env_steps": env_steps,
            "best_val_cost": best_val_cost,
            "ret_norm": ret_norm.state_dict() if ret_norm is not None else None,
            "obs_norm": obs_norm.state_dict() if obs_norm is not None else None,
        },
        output_dir / "checkpoint.pt",
    )


def build_critic_from_architecture(architecture: list[Any], input_dim: int) -> ValueNetwork:
    """Feedforward critic trunk whose widths mirror the policy architecture.

    This is a deliberate Phase 1.5 simplification: the critic sees raw obs and its
    widths mirror the policy trunk so training cost roughly matches the policy. It is
    NOT a structural match -- the critic has no recurrence. For a recurrent arch,
    GRU hidden_size is used as a plain feedforward width (tanh activation). A
    dedicated [value_network] TOML section with its own MLP spec is a reasonable
    future lift if critics under-fit when GRU capacity grows.
    ValueNetwork contract: len(activations) == len(hidden_sizes) + 1 (the final
    activation is the action-head's and ValueNetwork replaces it with linear).
    """
    from aerocapture.training.rl.schemas import DenseSpec as _DS

    critic_hidden_sizes: list[int] = []
    critic_activations: list[str] = []
    for spec in architecture[:-1]:
        if isinstance(spec, _DS):
            critic_hidden_sizes.append(spec.output_size)
            critic_activations.append(spec.activation)
        else:  # GruSpec -> treat as a tanh-activated hidden layer of width hidden_size
            critic_hidden_sizes.append(spec.hidden_size)
            critic_activations.append("tanh")
    # Append the action head's activation so len(activations) == len(hidden_sizes) + 1.
    final_spec = architecture[-1]
    if isinstance(final_spec, _DS):
        critic_activations.append(final_spec.activation)
    else:
        critic_activations.append("tanh")
    return ValueNetwork(input_dim, critic_hidden_sizes, critic_activations)


def collect_rollout(
    env: Any,
    policy: V2Policy,
    value: ValueNetwork,
    buf: RolloutBuffer,
    next_values: npt.NDArray[np.float32],
    obs: npt.NDArray[np.float32],
    aux_cur: npt.NDArray[np.float32],
    *,
    obs_norm: Any,
    ret_norm: Any,
    step_calc: Any,
    cfg: RLConfig,
    episodic_returns: list[float],
    episodic_dvs: list[float],
    episodic_captures: list[bool],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], int]:
    """Collect one rollout: fill `buf` + `next_values`, append episodic_* in place.

    Returns (obs, aux_cur, env_steps_delta) so the caller advances its loop state.
    """
    env_steps_delta = 0
    # Hidden-state tracking: seed from previous rollout's final state.
    h_current: list = [None if s is None else s.copy() for s in buf.h_final]
    # Snapshot of rollout-start state for chunk 0 of the BPTT update.
    buf.h_initial = [None if s is None else s.copy() for s in h_current]

    for t in range(cfg.ppo.rollout_steps):
        if obs_norm is not None:
            obs_norm.update(obs)
            obs_policy = obs_norm.normalize(obs)
        else:
            obs_policy = obs
        obs_t = torch.from_numpy(obs_policy).float()

        # Store the per-step pre-state so chunk c of the update loop can seed
        # from buf.states[c * bptt_length] (same indexing contract as the spec).
        for li, s in enumerate(h_current):
            if s is not None:
                states_li = buf.states[li]
                assert states_li is not None  # layer has state <=> buf.states[li] is populated
                states_li[t] = s

        state_t = _np_state_to_torch(h_current)
        with torch.no_grad():
            bank, raw, log_prob, state_next = policy.sample(obs_t, state_t)
            v_pred = value(obs_t)

        actions_np = bank.cpu().numpy().astype(np.float32)
        next_obs, _rust_reward, done, info, aux_next = env.step(actions_np)

        # Terminal-obs-aware next obs: unchanged PBRS + value bootstrap logic.
        term_obs = _terminal_observations(info, done, env.obs_dim)
        next_obs_for_shape = np.where(done[:, None], term_obs, next_obs)

        shaped = step_calc.step_reward(obs, next_obs_for_shape, aux_cur, aux_next).astype(np.float32)

        for i, d in enumerate(done):
            if d:
                fr = np.array(info[i]["final_record"], dtype=np.float64)
                term_cost = compute_terminal_cost(fr, cost_kwargs=step_calc.cost_kwargs)
                shaped[i] += float(-term_cost)
                episodic_returns.append(float(-term_cost))
                episodic_dvs.append(float(info[i].get("dv_m_s", float("nan"))))
                episodic_captures.append(bool(info[i].get("captured", False)))

        if ret_norm is not None:
            ret_norm.update(shaped.astype(np.float64), done)
            shaped = ret_norm.normalize(shaped.astype(np.float64)).astype(np.float32)

        with torch.no_grad():
            nv_obs = term_obs.copy()
            nv_obs = np.where(done[:, None], nv_obs, next_obs)
            nv_obs_policy = obs_norm.normalize(nv_obs) if obs_norm is not None else nv_obs
            nv = value(torch.from_numpy(nv_obs_policy).float()).cpu().numpy()

        truncated = np.array([bool(info[i].get("truncated", False)) for i in range(cfg.n_envs)], dtype=np.bool_)

        buf.obs[t] = obs
        buf.raw_actions[t] = raw.cpu().numpy()
        buf.log_probs[t] = log_prob.cpu().numpy()
        buf.rewards[t] = shaped
        buf.values[t] = v_pred.cpu().numpy()
        buf.dones[t] = done & ~truncated
        next_values[t] = nv

        # Advance hidden state; zero per-env on done (matches Rust auto-reset).
        h_next_np = _torch_state_to_np(state_next)
        for li in range(len(h_current)):
            if h_current[li] is not None:
                h_next_np[li][done] = 0.0
                h_current[li] = h_next_np[li]

        obs = next_obs
        aux_cur = aux_next
        env_steps_delta += cfg.n_envs

    # End of rollout: snapshot for next rollout's h_initial.
    buf.h_final = [None if s is None else s.copy() for s in h_current]

    return obs, aux_cur, env_steps_delta


def _run_ppo(
    cfg: RLConfig,
    toml_path: Path,
    output_dir: Path,
    logger: RLLogger,
    display: Any,
    interrupted: dict[str, bool],
    resume_dir: Path | None,
    env_overrides: dict[str, Any] | None = None,
    warmstart_json: Path | None = None,
) -> None:
    input_mask, architecture, input_dim = _parse_network_config(cfg)
    step_calc, ret_norm, obs_norm = _build_shaper_and_norms(cfg, input_mask, gamma=cfg.ppo.gamma, toml_path=toml_path)

    env = AerocaptureVecEnv(
        toml_path=str(toml_path),
        n_envs=cfg.n_envs,
        seed_base=cfg.seed_base,
        overrides=env_overrides,
    )

    policy = V2Policy(
        architecture=architecture,
        input_mask=input_mask,
        initial_log_std=cfg.ppo.initial_log_std,
        min_log_std=cfg.ppo.min_log_std,
    )
    if warmstart_json is not None:
        from aerocapture.training.model_io import load_policy_from_json

        warm_loaded = load_policy_from_json(str(warmstart_json), device="cpu")
        if len(warm_loaded.layers) != len(policy.layers):
            raise ValueError(
                f"Warm-start architecture mismatch: {warmstart_json} has "
                f"{len(warm_loaded.layers)} layers, TOML [[network.architecture]] "
                f"declares {len(policy.layers)}. Either update the TOML to match the "
                f"checkpoint, or train from scratch."
            )
        type_mismatches = [
            (i, type(a).__name__, type(b).__name__) for i, (a, b) in enumerate(zip(warm_loaded.layers, policy.layers, strict=True)) if type(a) is not type(b)
        ]
        if type_mismatches:
            diffs = ", ".join(f"layer {i}: checkpoint={a} vs TOML={b}" for i, a, b in type_mismatches)
            raise ValueError(
                f"Warm-start layer-type mismatch: {warmstart_json} has incompatible "
                f"layer types -- {diffs}. load_state_dict would silently write mismatched "
                f"weight tensors. Either update the TOML to match the checkpoint, or train "
                f"from scratch."
            )
        policy.load_state_dict(warm_loaded.state_dict())
        print(f"Warm-started policy from {warmstart_json}", file=sys.stderr)

    # TODO(Phase 2+): consider [value_network] block if critic under-fitting is observed.
    value = build_critic_from_architecture(architecture, input_dim)
    optim = torch.optim.Adam(
        list(policy.parameters()) + list(value.parameters()),
        lr=cfg.ppo.learning_rate,
    )

    env_steps = 0
    update_idx = 0
    best_val_cost = float("inf")
    ckpt_path = (resume_dir or output_dir) / "checkpoint.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, weights_only=True)
        policy.load_state_dict(ckpt["policy"])
        value.load_state_dict(ckpt["value"])
        optim.load_state_dict(ckpt["optim"])
        update_idx = int(ckpt["update_idx"])
        env_steps = int(ckpt["env_steps"])
        best_val_cost = float(ckpt["best_val_cost"])
        if ret_norm is not None and ckpt.get("ret_norm") is not None:
            ret_norm.load_state_dict(ckpt["ret_norm"])
        if obs_norm is not None and ckpt.get("obs_norm") is not None:
            obs_norm.load_state_dict(ckpt["obs_norm"])
        print(f"Resumed from checkpoint: update {update_idx}, {env_steps} env steps", file=sys.stderr)

    # Derive per-layer hidden shapes from the architecture.
    # Dense: None (stateless). GRU: (H,). LSTM: (2, H) -- packs (h, c) as a single array.
    from aerocapture.training.rl.schemas import DenseSpec as _DS
    from aerocapture.training.rl.schemas import GruSpec as _GS
    from aerocapture.training.rl.schemas import LstmSpec as _LS
    from aerocapture.training.rl.schemas import TransformerSpec as _TS
    from aerocapture.training.rl.schemas import WindowSpec as _WS

    hidden_shapes: list = []
    for spec in architecture:
        if isinstance(spec, _DS):
            hidden_shapes.append(None)
        elif isinstance(spec, _GS):
            hidden_shapes.append((spec.hidden_size,))
        elif isinstance(spec, _LS):
            hidden_shapes.append((2, spec.hidden_size))
        elif isinstance(spec, _WS):
            raise NotImplementedError(
                "Window-MLP is PSO-only in Phase 2b; PPO use deferred. See docs/superpowers/specs/2026-04-20-phase-2b-window-mlp-design.md"
            )
        elif isinstance(spec, _TS):
            raise NotImplementedError(
                "Transformer is PSO-only in Phase 3a; PPO use deferred. See docs/superpowers/specs/2026-04-22-phase-3a-transformer-mvp-design.md"
            )
        else:
            raise ValueError(f"Unknown layer spec type in hidden_shapes derivation: {type(spec).__name__}")

    buf = RolloutBuffer.create(
        cfg.ppo.rollout_steps,
        cfg.n_envs,
        env.obs_dim,
        hidden_shapes=hidden_shapes,
    )
    # Bootstrap values for each step: value network's estimate of V(next_obs) per env.
    # We store this alongside the rollout so GAE can use V(terminal_obs) on truncated
    # episodes instead of V(reset_obs) which would leak across episode boundaries.
    next_values = np.zeros((cfg.ppo.rollout_steps, cfg.n_envs), dtype=np.float32)

    obs, aux_cur = env.reset()
    episodic_returns: list[float] = []
    episodic_dvs: list[float] = []
    episodic_captures: list[bool] = []
    start_time = time.time()

    while env_steps < cfg.total_env_steps and not interrupted["v"]:
        obs, aux_cur, steps = collect_rollout(
            env,
            policy,
            value,
            buf,
            next_values,
            obs,
            aux_cur,
            obs_norm=obs_norm,
            ret_norm=ret_norm,
            step_calc=step_calc,
            cfg=cfg,
            episodic_returns=episodic_returns,
            episodic_dvs=episodic_dvs,
            episodic_captures=episodic_captures,
        )
        env_steps += steps

        advantages = np.zeros_like(buf.rewards)
        returns = np.zeros_like(buf.rewards)
        for e in range(cfg.n_envs):
            adv, ret = compute_gae(
                buf.rewards[:, e],
                buf.values[:, e],
                next_values[:, e],
                buf.dones[:, e],
                gamma=cfg.ppo.gamma,
                lam=cfg.ppo.gae_lambda,
            )
            advantages[:, e] = adv
            returns[:, e] = ret

        frac_done = env_steps / cfg.total_env_steps
        anneal_start = cfg.ppo.lr_anneal_start
        lr = cfg.ppo.learning_rate if frac_done <= anneal_start else cfg.ppo.learning_rate * max((1.0 - frac_done) / (1.0 - anneal_start), 0.0)
        for pg in optim.param_groups:
            pg["lr"] = lr

        metrics = ppo_update_bptt(
            policy,
            value,
            optim,
            buf,
            advantages,
            returns,
            bptt_length=cfg.ppo.bptt_length,
            clip_range=cfg.ppo.clip_range,
            update_epochs=cfg.ppo.update_epochs,
            minibatches=cfg.ppo.minibatches,
            entropy_coef=cfg.ppo.entropy_coef,
            value_coef=cfg.ppo.value_coef,
            max_grad_norm=cfg.ppo.max_grad_norm,
            target_kl=cfg.ppo.target_kl,
            obs_norm=obs_norm,
        )

        update_idx += 1

        val_attempted = update_idx % cfg.validation_interval_updates == 0
        val_record: dict[str, Any] = {}
        if val_attempted:
            val_record = _validate_deterministic(policy, toml_path, output_dir, cfg, input_mask, obs_norm=obs_norm)
            if val_record["val_rms_cost"] < best_val_cost:
                best_val_cost = val_record["val_rms_cost"]
                export_v2_policy_to_json(policy, str(output_dir / "best_model.json"), obs_normalizer=obs_norm)
                val_record["val_promoted"] = True
            else:
                val_record["val_promoted"] = False

        if update_idx % cfg.checkpoint_interval_updates == 0:
            _save_ppo_checkpoint(output_dir, policy, value, optim, update_idx, env_steps, best_val_cost, ret_norm, obs_norm)

        record: dict[str, Any] = {
            "update_idx": update_idx,
            "env_steps": env_steps,
            "episodic_return_mean": float(np.mean(episodic_returns[-64:])) if episodic_returns else float("nan"),
            "episodic_dv_m_s_mean": float(np.mean(episodic_dvs[-64:])) if episodic_dvs else float("nan"),
            "episodic_capture_rate": float(np.mean(episodic_captures[-64:])) if episodic_captures else float("nan"),
            "policy_loss": metrics["policy_loss"],
            "value_loss": metrics["value_loss"],
            "entropy": metrics["entropy"],
            "approx_kl": metrics["approx_kl"],
            "clip_frac": metrics["clip_frac"],
            "epochs_run": metrics.get("epochs_run", float(cfg.ppo.update_epochs)),
            "learning_rate": lr,
            "val_attempted": val_attempted,
            "val_promoted": val_record.get("val_promoted", False),
            "val_rms_cost": val_record.get("val_rms_cost"),
            "val_capture_rate": val_record.get("val_capture_rate"),
            "best_val_cost": best_val_cost,
            "wallclock_seconds": time.time() - start_time,
        }
        logger.log_update(record)
        display.update(record)

    _save_ppo_checkpoint(output_dir, policy, value, optim, update_idx, env_steps, best_val_cost, ret_norm, obs_norm)
    if best_val_cost == float("inf"):
        export_v2_policy_to_json(policy, str(output_dir / "best_model.json"), obs_normalizer=obs_norm)

    env.close()


# ---------------------------------------------------------------------------
# SAC
# ---------------------------------------------------------------------------


def _save_sac_checkpoint(
    output_dir: Path,
    agent: SACAgent,
    update_idx: int,
    env_steps: int,
    best_val_cost: float,
    ret_norm: ReturnNormalizer | None,
    obs_norm: ObsNormalizer | None,
) -> None:
    torch.save(
        {
            "policy": agent.policy.state_dict(),
            "q1": agent.q1.state_dict(),
            "q2": agent.q2.state_dict(),
            "q1_target": agent.q1_target.state_dict(),
            "q2_target": agent.q2_target.state_dict(),
            "log_alpha": agent.log_alpha.data,
            "replay_buffer": agent.replay_buffer.state_dict(),
            "update_idx": update_idx,
            "env_steps": env_steps,
            "best_val_cost": best_val_cost,
            "ret_norm": ret_norm.state_dict() if ret_norm is not None else None,
            "obs_norm": obs_norm.state_dict() if obs_norm is not None else None,
        },
        output_dir / "checkpoint.pt",
    )


def _run_sac(
    cfg: RLConfig,
    toml_path: Path,
    output_dir: Path,
    logger: RLLogger,
    display: Any,
    interrupted: dict[str, bool],
    env_overrides: dict[str, Any] | None = None,
) -> None:
    input_mask, architecture, input_dim = _parse_network_config(cfg)
    layer_sizes, activations = _dense_only_shapes(architecture)
    step_calc, ret_norm, obs_norm = _build_shaper_and_norms(cfg, input_mask, gamma=cfg.sac.gamma, toml_path=toml_path)

    env = AerocaptureVecEnv(
        toml_path=str(toml_path),
        n_envs=cfg.n_envs,
        seed_base=cfg.seed_base,
        overrides=env_overrides,
    )

    sac_cfg = cfg.sac
    agent = SACAgent(
        obs_dim=input_dim,
        layer_sizes=layer_sizes,
        activations=activations,
        buffer_size=sac_cfg.buffer_size,
        batch_size=sac_cfg.batch_size,
        gamma=sac_cfg.gamma,
        tau=sac_cfg.tau,
        learning_rate=sac_cfg.learning_rate,
        target_entropy=sac_cfg.target_entropy,
        initial_alpha=sac_cfg.initial_alpha,
    )

    env_steps = 0
    update_idx = 0
    best_val_cost = float("inf")
    ckpt_path = output_dir / "checkpoint.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, weights_only=False)
        agent.policy.load_state_dict(ckpt["policy"])
        agent.q1.load_state_dict(ckpt["q1"])
        agent.q2.load_state_dict(ckpt["q2"])
        agent.q1_target.load_state_dict(ckpt["q1_target"])
        agent.q2_target.load_state_dict(ckpt["q2_target"])
        agent.log_alpha.data.copy_(ckpt["log_alpha"])
        if ckpt.get("replay_buffer") is not None:
            agent.replay_buffer.load_state_dict(ckpt["replay_buffer"])
        update_idx = int(ckpt["update_idx"])
        env_steps = int(ckpt["env_steps"])
        best_val_cost = float(ckpt["best_val_cost"])
        if ret_norm is not None and ckpt.get("ret_norm") is not None:
            ret_norm.load_state_dict(ckpt["ret_norm"])
        if obs_norm is not None and ckpt.get("obs_norm") is not None:
            obs_norm.load_state_dict(ckpt["obs_norm"])
        print(f"SAC resumed: update {update_idx}, {env_steps} env steps, buffer={len(agent.replay_buffer)}", file=sys.stderr)

    obs, aux_cur = env.reset()
    episodic_returns: list[float] = []
    episodic_dvs: list[float] = []
    episodic_captures: list[bool] = []
    start_time = time.time()
    metrics: dict[str, Any] = {}

    while env_steps < cfg.total_env_steps and not interrupted["v"]:
        if obs_norm is not None:
            obs_norm.update(obs)
            obs_policy = obs_norm.normalize(obs)
        else:
            obs_policy = obs
        obs_t = torch.from_numpy(obs_policy).float()
        with torch.no_grad():
            bank_t, raw_t, _ = agent.policy.sample(obs_t)
        actions_np = bank_t.cpu().numpy().astype(np.float32)
        raw_np = raw_t.cpu().numpy().astype(np.float32)

        next_obs, _rust_reward, done, info, aux_next = env.step(actions_np)

        term_obs = _terminal_observations(info, done, env.obs_dim)
        next_obs_for_shape = np.where(done[:, None], term_obs, next_obs)
        shaped = step_calc.step_reward(obs, next_obs_for_shape, aux_cur, aux_next).astype(np.float32)

        for i, d in enumerate(done):
            if d:
                fr = np.array(info[i]["final_record"], dtype=np.float64)
                term_cost = compute_terminal_cost(fr, cost_kwargs=step_calc.cost_kwargs)
                shaped[i] += float(-term_cost)
                episodic_returns.append(float(-term_cost))
                episodic_dvs.append(float(info[i].get("dv_m_s", float("nan"))))
                episodic_captures.append(bool(info[i].get("captured", False)))

        if ret_norm is not None:
            ret_norm.update(shaped.astype(np.float64), done)
            shaped_norm = ret_norm.normalize(shaped.astype(np.float64)).astype(np.float32)
        else:
            shaped_norm = shaped

        # SAC stores normalized obs in replay buffer for policy/critic consistency.
        # For truncated steps, the Q-target bootstraps via (1-done)*Q(next), so
        # `next_obs` must be the *terminal* observation (pre-reset), not the reset
        # observation of a freshly-drawn episode (which would leak cross-episode state).
        truncated = np.array([bool(info[i].get("truncated", False)) for i in range(cfg.n_envs)], dtype=np.bool_)
        true_next = np.where(done[:, None], term_obs, next_obs)
        next_obs_policy = obs_norm.normalize(true_next) if obs_norm is not None else true_next
        done_for_buffer = done & ~truncated
        agent.replay_buffer.push(obs_policy, raw_np, shaped_norm, next_obs_policy, done_for_buffer)
        obs = next_obs
        aux_cur = aux_next
        env_steps += cfg.n_envs

        buffer_ready = len(agent.replay_buffer) >= max(sac_cfg.batch_size, sac_cfg.warmup_steps)
        if buffer_ready and env_steps % (sac_cfg.train_every * cfg.n_envs) == 0:
            for _ in range(sac_cfg.gradient_steps):
                batch_obs, batch_raw, batch_rew, batch_next, batch_done = agent.replay_buffer.sample(sac_cfg.batch_size)
                metrics = agent.update(batch_obs, batch_raw, batch_rew, batch_next, batch_done)
            update_idx += 1

            val_attempted = update_idx % cfg.validation_interval_updates == 0
            val_record: dict[str, Any] = {}
            if val_attempted:
                val_record = _validate_deterministic_v1(agent.policy, toml_path, output_dir, cfg, input_mask, obs_norm=obs_norm)
                if val_record["val_rms_cost"] < best_val_cost:
                    best_val_cost = val_record["val_rms_cost"]
                    export_policy_to_json(agent.policy, output_dir / "best_model.json", input_mask, obs_normalizer=obs_norm)
                    val_record["val_promoted"] = True
                else:
                    val_record["val_promoted"] = False

            if update_idx % cfg.checkpoint_interval_updates == 0:
                _save_sac_checkpoint(output_dir, agent, update_idx, env_steps, best_val_cost, ret_norm, obs_norm)

            record: dict[str, Any] = {
                "update_idx": update_idx,
                "env_steps": env_steps,
                "episodic_return_mean": float(np.mean(episodic_returns[-64:])) if episodic_returns else float("nan"),
                "episodic_dv_m_s_mean": float(np.mean(episodic_dvs[-64:])) if episodic_dvs else float("nan"),
                "episodic_capture_rate": float(np.mean(episodic_captures[-64:])) if episodic_captures else float("nan"),
                "policy_loss": metrics.get("policy_loss", float("nan")),
                "value_loss": metrics.get("q_loss", float("nan")),
                "entropy": metrics.get("mean_log_prob", float("nan")),
                "alpha": metrics.get("alpha", float("nan")),
                "learning_rate": sac_cfg.learning_rate,
                "val_attempted": val_attempted,
                "val_promoted": val_record.get("val_promoted", False),
                "val_rms_cost": val_record.get("val_rms_cost"),
                "val_capture_rate": val_record.get("val_capture_rate"),
                "best_val_cost": best_val_cost,
                "wallclock_seconds": time.time() - start_time,
            }
            logger.log_update(record)
            display.update(record)

    _save_sac_checkpoint(output_dir, agent, update_idx, env_steps, best_val_cost, ret_norm, obs_norm)
    if best_val_cost == float("inf"):
        export_policy_to_json(agent.policy, output_dir / "best_model.json", input_mask, obs_normalizer=obs_norm)

    env.close()


if __name__ == "__main__":
    main()
