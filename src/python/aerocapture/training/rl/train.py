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
from aerocapture.training.rl.export import export_policy_to_json
from aerocapture.training.rl.logger import RLLogger
from aerocapture.training.rl.normalizers import ObsNormalizer, ReturnNormalizer
from aerocapture.training.rl.policy import GaussianPolicy, ValueNetwork
from aerocapture.training.rl.ppo import RolloutBuffer, compute_gae, ppo_update
from aerocapture.training.rl.rewards import StepRewardCalculator, compute_terminal_cost
from aerocapture.training.rl.sac import SACAgent

OUT_DIR_DEFAULT = Path("training_output/neural_network_rl")

# Column indices in the 52-element final_record array (see runner.rs).
_IDX_ECC = 9
_IDX_IFINAL = 31


# ---------------------------------------------------------------------------
# Shared helpers (used by both PPO and SAC loops)
# ---------------------------------------------------------------------------


def _parse_network_config(cfg: RLConfig) -> tuple[list[int], list[int], list[str], int]:
    """Extract (input_mask, layer_sizes, activations, input_dim) from TOML [network].

    TOML layer_sizes always includes the input dim as the first element; the
    policy expects hidden+output only, so the first element is stripped.
    """
    network_cfg = cfg.raw_toml.get("network", {})
    input_mask: list[int] = network_cfg.get("input_mask", list(range(16)))
    toml_layers: list[int] = network_cfg.get("layer_sizes", [16, 64, 64, 2])
    activations: list[str] = network_cfg.get("activations", ["tanh", "tanh", "linear"])
    input_dim = len(input_mask)
    if toml_layers[0] != input_dim:
        raise ValueError(f"layer_sizes[0]={toml_layers[0]} must equal len(input_mask)={input_dim}")
    layer_sizes = toml_layers[1:]
    if len(layer_sizes) != len(activations):
        raise ValueError(f"len(layer_sizes[1:])={len(layer_sizes)} must equal len(activations)={len(activations)}")
    return input_mask, layer_sizes, activations, input_dim


def _generate_seed_model(cfg: RLConfig, path: Path) -> None:
    """Export a randomly-initialized policy as a seed model JSON for BatchedSimulation."""
    input_mask, layer_sizes, activations, input_dim = _parse_network_config(cfg)
    policy = GaussianPolicy(input_dim, layer_sizes, activations)
    export_policy_to_json(policy, path, input_mask)


def _build_shaper_and_norms(cfg: RLConfig, input_mask: list[int], gamma: float) -> tuple[StepRewardCalculator, ReturnNormalizer | None, ObsNormalizer | None]:
    step_calc = StepRewardCalculator(
        input_mask=input_mask,
        gamma=gamma,
        corridor_weight=cfg.reward.corridor_weight,
        energy_rate_weight=cfg.reward.energy_rate_weight,
        constraint_weight=cfg.reward.constraint_weight,
        apoapsis_weight=cfg.reward.apoapsis_weight,
        eccentricity_weight=cfg.reward.eccentricity_weight,
        energy_scale=cfg.reward.energy_scale,
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
    policy: GaussianPolicy,
    toml_path: Path,
    output_dir: Path,
    cfg: RLConfig,
    input_mask: list[int],
    obs_norm: ObsNormalizer | None = None,
) -> dict[str, Any]:
    """Export deterministic policy + run validation batch; return RMS cost + capture rate."""
    import aerocapture_rs  # type: ignore[import]

    from aerocapture.training.evaluate import VALIDATION_SEED_OFFSET, compute_cost, make_reserved_seeds

    tmp_json = output_dir / "gen_current_model.json"
    export_policy_to_json(policy, tmp_json, input_mask, obs_normalizer=obs_norm)

    base_seed = int(cfg.raw_toml.get("monte_carlo", {}).get("seed", 42))
    seeds = make_reserved_seeds(base_seed, VALIDATION_SEED_OFFSET, cfg.validation_n_sims)

    overrides_list = [{"data.neural_network": str(tmp_json), "monte_carlo.seed": s, "simulation.n_sims": 1} for s in seeds]
    results = aerocapture_rs.run_batch(str(toml_path), overrides_list)
    fr = results.final_records

    rms_cost = float(compute_cost(fr))
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
    ap.add_argument("--output-dir", type=Path, default=OUT_DIR_DEFAULT)
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
    policy: GaussianPolicy,
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
    input_mask, layer_sizes, activations, input_dim = _parse_network_config(cfg)
    step_calc, ret_norm, obs_norm = _build_shaper_and_norms(cfg, input_mask, gamma=cfg.ppo.gamma)

    env = AerocaptureVecEnv(
        toml_path=str(toml_path),
        n_envs=cfg.n_envs,
        seed_base=cfg.seed_base,
        overrides=env_overrides,
    )

    policy = GaussianPolicy(input_dim, layer_sizes, activations, cfg.ppo.initial_log_std, cfg.ppo.min_log_std)
    if warmstart_json is not None:
        policy.load_weights_from_json(warmstart_json)
        print(f"Warm-started policy from {warmstart_json}", file=sys.stderr)
    value = ValueNetwork(input_dim, layer_sizes[:-1], activations)
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

    buf = RolloutBuffer.create(cfg.ppo.rollout_steps, cfg.n_envs, env.obs_dim)
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
        for t in range(cfg.ppo.rollout_steps):
            if obs_norm is not None:
                obs_norm.update(obs)
                obs_policy = obs_norm.normalize(obs)
            else:
                obs_policy = obs
            obs_t = torch.from_numpy(obs_policy).float()
            with torch.no_grad():
                mean, log_std = policy.forward_mean_logstd(obs_t)
                std = log_std.exp()
                eps = torch.randn_like(mean)
                raw = mean + std * eps
                bank = torch.atan2(raw[..., 0], raw[..., 1])
                dist = torch.distributions.Normal(mean, std)
                log_prob = dist.log_prob(raw).sum(-1)
                v_pred = value(obs_t)

            actions_np = bank.cpu().numpy().astype(np.float32)
            next_obs, _rust_reward, done, info, aux_next = env.step(actions_np)

            # Terminal-obs-aware next obs: for done envs, use the pre-reset obs
            # so PBRS and value bootstrap see the final physical state.
            term_obs = _terminal_observations(info, done, env.obs_dim)
            next_obs_for_shape = np.where(done[:, None], term_obs, next_obs)

            shaped = step_calc.step_reward(obs, next_obs_for_shape, aux_cur, aux_next).astype(np.float32)

            for i, d in enumerate(done):
                if d:
                    fr = np.array(info[i]["final_record"], dtype=np.float64)
                    term_cost = compute_terminal_cost(fr)
                    shaped[i] += float(-term_cost)
                    episodic_returns.append(float(-term_cost))
                    episodic_dvs.append(float(info[i].get("dv_m_s", float("nan"))))
                    episodic_captures.append(bool(info[i].get("captured", False)))

            # Update return normalizer with raw shaped rewards + per-env done mask.
            # Rewards are normalized per-step (same as SAC) using the just-updated std,
            # so advantages at GAE time use a stable scale within each rollout.
            if ret_norm is not None:
                ret_norm.update(shaped.astype(np.float64), done)
                shaped = ret_norm.normalize(shaped.astype(np.float64)).astype(np.float32)

            # Per-env bootstrap value: for truncated envs bootstrap off terminal obs;
            # for proper terminations bootstrap is zeroed by `done` mask in GAE below.
            with torch.no_grad():
                nv_obs = term_obs.copy()
                # Non-done envs: next_obs is the actual next state.
                nv_obs = np.where(done[:, None], nv_obs, next_obs)
                nv_obs_policy = obs_norm.normalize(nv_obs) if obs_norm is not None else nv_obs
                nv = value(torch.from_numpy(nv_obs_policy).float()).cpu().numpy()

            # Distinguish truncation (keep bootstrap) from termination (zero bootstrap).
            truncated = np.array([bool(info[i].get("truncated", False)) for i in range(cfg.n_envs)], dtype=np.bool_)

            buf.obs[t] = obs
            buf.raw_actions[t] = raw.cpu().numpy()
            buf.log_probs[t] = log_prob.cpu().numpy()
            buf.rewards[t] = shaped
            buf.values[t] = v_pred.cpu().numpy()
            # Zero bootstrap on true terminations; keep on truncations and normal transitions.
            buf.dones[t] = done & ~truncated
            next_values[t] = nv

            obs = next_obs
            aux_cur = aux_next
            env_steps += cfg.n_envs

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

        raw_obs_flat = buf.obs.reshape(-1, env.obs_dim)
        if obs_norm is not None:
            raw_obs_flat = obs_norm.normalize(raw_obs_flat)
        flat_obs = torch.from_numpy(raw_obs_flat).float()
        flat_raw = torch.from_numpy(buf.raw_actions.reshape(-1, 2)).float()
        flat_old_lp = torch.from_numpy(buf.log_probs.reshape(-1)).float()
        flat_adv = torch.from_numpy(advantages.reshape(-1)).float()
        flat_ret = torch.from_numpy(returns.reshape(-1)).float()

        metrics = ppo_update(
            policy,
            value,
            optim,
            flat_obs,
            flat_raw,
            flat_old_lp,
            flat_adv,
            flat_ret,
            clip_range=cfg.ppo.clip_range,
            update_epochs=cfg.ppo.update_epochs,
            minibatches=cfg.ppo.minibatches,
            entropy_coef=cfg.ppo.entropy_coef,
            value_coef=cfg.ppo.value_coef,
            max_grad_norm=cfg.ppo.max_grad_norm,
            target_kl=cfg.ppo.target_kl,
        )

        update_idx += 1

        val_attempted = update_idx % cfg.validation_interval_updates == 0
        val_record: dict[str, Any] = {}
        if val_attempted:
            val_record = _validate_deterministic(policy, toml_path, output_dir, cfg, input_mask, obs_norm=obs_norm)
            if val_record["val_rms_cost"] < best_val_cost:
                best_val_cost = val_record["val_rms_cost"]
                export_policy_to_json(policy, output_dir / "best_model.json", input_mask, obs_normalizer=obs_norm)
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
        export_policy_to_json(policy, output_dir / "best_model.json", input_mask, obs_normalizer=obs_norm)

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
    input_mask, layer_sizes, activations, input_dim = _parse_network_config(cfg)
    step_calc, ret_norm, obs_norm = _build_shaper_and_norms(cfg, input_mask, gamma=cfg.sac.gamma)

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
                term_cost = compute_terminal_cost(fr)
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
                val_record = _validate_deterministic(agent.policy, toml_path, output_dir, cfg, input_mask, obs_norm=obs_norm)
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
