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
import torch

from aerocapture.training.rl.config import RLConfig
from aerocapture.training.rl.display import make_display
from aerocapture.training.rl.env import AerocaptureVecEnv
from aerocapture.training.rl.export import export_policy_to_json
from aerocapture.training.rl.logger import RLLogger
from aerocapture.training.rl.policy import GaussianPolicy, ValueNetwork
from aerocapture.training.rl.ppo import RolloutBuffer, compute_gae, ppo_update
from aerocapture.training.rl.rewards import PBRSShaper
from aerocapture.training.rl.sac import SACAgent

OUT_DIR_DEFAULT = Path("training_output/neural_network_rl")


def _dict_to_toml(d: dict[str, Any], prefix: str = "") -> str:
    """Minimal recursive TOML serializer (no tomli_w dependency)."""
    scalars: list[str] = []
    sections: list[str] = []
    for k, v in d.items():
        if isinstance(v, dict):
            header = f"[{prefix}{k}]" if prefix else f"[{k}]"
            body = _dict_to_toml(v, prefix=f"{prefix}{k}.")
            sections.append(f"{header}\n{body}")
        elif isinstance(v, bool):
            scalars.append(f"{k} = {'true' if v else 'false'}")
        elif isinstance(v, str):
            scalars.append(f'{k} = "{v}"')
        elif isinstance(v, list):
            scalars.append(f"{k} = {json.dumps(v)}")
        else:
            scalars.append(f"{k} = {v}")
    parts = scalars + ([""] if scalars and sections else []) + sections
    return "\n".join(parts) + ("\n" if parts else "")


def _parse_network_config(cfg: RLConfig) -> tuple[list[int], list[int], list[str], int]:
    """Extract (input_mask, layer_sizes, activations, input_dim) from TOML [network].

    TOML layer_sizes includes the input dim (e.g. [23, 32, 16, 8, 2]),
    but GaussianPolicy expects hidden+output only (e.g. [32, 16, 8, 2]).
    """
    network_cfg = cfg.raw_toml.get("network", {})
    input_mask: list[int] = network_cfg.get("input_mask", list(range(16)))
    toml_layers: list[int] = network_cfg.get("layer_sizes", [16, 64, 64, 2])
    activations: list[str] = network_cfg.get("activations", ["tanh", "tanh", "linear"])
    input_dim = len(input_mask)
    layer_sizes = toml_layers[1:] if toml_layers[0] == input_dim and len(toml_layers) - 1 == len(activations) else toml_layers
    return input_mask, layer_sizes, activations, input_dim


def _generate_seed_model(cfg: RLConfig, path: Path) -> None:
    """Export a randomly-initialized policy as a seed model JSON for BatchedSimulation."""
    input_mask, layer_sizes, activations, input_dim = _parse_network_config(cfg)
    policy = GaussianPolicy(input_dim, layer_sizes, activations)
    export_policy_to_json(policy, path, input_mask)


# Column indices in the 52-element final_record array (verified against
# src/rust/src/simulation/runner.rs final_record layout):
#   index 9  = eccentricity
#   index 31 = ifinal (1=hyperbolic, 2=crash, 3=captured, 4=pending_crash, 5=timeout)
_IDX_ECC = 9
_IDX_IFINAL = 31


def main() -> None:
    ap = argparse.ArgumentParser(description="Train neural_network guidance via PPO.")
    ap.add_argument("toml_path")
    ap.add_argument("--algorithm", choices=["ppo", "sac"], default=None)
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--n-envs", type=int, default=None)
    ap.add_argument("--rollout-steps", type=int, default=None)
    ap.add_argument("--validation-n-sims", type=int, default=None)
    ap.add_argument("--validation-interval-updates", type=int, default=None)
    ap.add_argument("--data-neural-network", type=Path, default=None, help="Override path to neural network model JSON")
    ap.add_argument("--from-scratch", "-fs", action="store_true", help="Initialize with random weights (no seed model required)")
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

    cfg = RLConfig.from_toml(Path(args.toml_path), overrides=overrides or None, ppo_overrides=ppo_overrides or None)

    env_overrides: dict[str, Any] | None = None
    if args.data_neural_network is not None:
        env_overrides = {"data.neural_network": str(args.data_neural_network)}
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
    (args.output_dir / "config_resolved.toml").write_text(_dict_to_toml(cfg.raw_toml))

    logger = RLLogger(args.output_dir, config_hash)
    display = make_display(cfg.total_env_steps, enabled=not args.no_tui and sys.stdout.isatty())

    interrupted = {"v": False}

    def _on_sigint(_s: int, _f: Any) -> None:
        interrupted["v"] = True

    prev_handler = signal.signal(signal.SIGINT, _on_sigint)
    try:
        if cfg.algorithm == "ppo":
            _run_ppo(cfg, Path(args.toml_path), args.output_dir, logger, display, interrupted, args.resume, env_overrides)
        elif cfg.algorithm == "sac":
            _run_sac(cfg, Path(args.toml_path), args.output_dir, logger, display, interrupted, env_overrides)
        else:
            raise NotImplementedError(f"algorithm {cfg.algorithm!r} not supported")
    finally:
        signal.signal(signal.SIGINT, prev_handler)
        display.close()
        logger.close()

    if not args.skip_report:
        try:
            from aerocapture.training.rl.report_rl import generate_report  # type: ignore[import]

            generate_report(args.output_dir, Path(args.toml_path))
        except ImportError:
            print("report_rl not yet implemented — skipping PDF generation", file=sys.stderr)


def _run_ppo(
    cfg: RLConfig,
    toml_path: Path,
    output_dir: Path,
    logger: RLLogger,
    display: Any,
    interrupted: dict[str, bool],
    resume_dir: Path | None,
    env_overrides: dict[str, Any] | None = None,
) -> None:
    # Note: v1 uses pure terminal-cost rewards (no PBRS shaping). The shaper
    # needs a side channel from BatchedSimulation to extract (energy, pdyn)
    # from the env state — not wired up yet. Training will converge more
    # slowly but the optimum is unchanged (PBRS is policy-invariant by
    # construction; disabling it only affects sample efficiency).
    # TODO: wire up obs -> (energy, pdyn) extraction once BatchedSimulation
    # exposes a side channel for the raw sim state scalars.
    shaper = PBRSShaper(enabled=False)

    input_mask, layer_sizes, activations, input_dim = _parse_network_config(cfg)

    env = AerocaptureVecEnv(
        toml_path=str(toml_path),
        n_envs=cfg.n_envs,
        seed_base=cfg.seed_base,
        overrides=env_overrides,
    )

    policy = GaussianPolicy(input_dim, layer_sizes, activations, cfg.ppo.initial_log_std, cfg.ppo.min_log_std)
    value = ValueNetwork(input_dim, layer_sizes[:-1], activations)
    optim = torch.optim.Adam(
        list(policy.parameters()) + list(value.parameters()),
        lr=cfg.ppo.learning_rate,
    )

    # --- Checkpoint resume ---
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
        print(f"Resumed from checkpoint: update {update_idx}, {env_steps} env steps", file=sys.stderr)

    buf = RolloutBuffer.create(cfg.ppo.rollout_steps, cfg.n_envs, env.obs_dim)

    obs = env.reset()
    episodic_returns: list[float] = []
    episodic_dvs: list[float] = []
    episodic_captures: list[bool] = []
    start_time = time.time()

    while env_steps < cfg.total_env_steps and not interrupted["v"]:
        # --- Rollout collection ---
        for t in range(cfg.ppo.rollout_steps):
            obs_t = torch.from_numpy(obs).float()
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
            next_obs, _rust_reward, done, info = env.step(actions_np)

            # Step shaping (disabled in v1 — returns zeros).
            shaped = shaper.step_reward(obs, next_obs, cfg.ppo.gamma).astype(np.float32)

            # On terminal steps, write the episode cost as reward.
            for i, d in enumerate(done):
                if d:
                    fr = np.array(info[i]["final_record"], dtype=np.float64)
                    from aerocapture.training.rl.rewards import compute_terminal_cost

                    term_cost = compute_terminal_cost(fr)
                    # boundary PBRS term: gamma*phi(s_{T+1}=0) - phi(s_T);
                    # with shaping disabled phi=0 so this collapses to -term_cost.
                    shaped[i] = float(-term_cost)
                    episodic_returns.append(float(-term_cost))
                    episodic_dvs.append(float(info[i].get("dv_m_s", float("nan"))))
                    episodic_captures.append(bool(info[i].get("captured", False)))

            buf.obs[t] = obs
            buf.actions[t] = actions_np
            buf.log_probs[t] = log_prob.cpu().numpy()
            buf.rewards[t] = shaped
            buf.values[t] = v_pred.squeeze(-1).cpu().numpy()
            buf.dones[t] = done

            obs = next_obs
            env_steps += cfg.n_envs

        # --- Bootstrap value for last obs ---
        with torch.no_grad():
            last_v = value(torch.from_numpy(obs).float()).squeeze(-1).cpu().numpy()

        # --- GAE per env ---
        advantages = np.zeros_like(buf.rewards)
        returns = np.zeros_like(buf.rewards)
        for e in range(cfg.n_envs):
            vs = np.concatenate([buf.values[:, e], last_v[e : e + 1]])
            adv, ret = compute_gae(
                buf.rewards[:, e],
                vs,
                buf.dones[:, e],
                gamma=cfg.ppo.gamma,
                lam=cfg.ppo.gae_lambda,
            )
            advantages[:, e] = adv
            returns[:, e] = ret

        # --- LR anneal (constant until lr_anneal_start, then linear decay to 0) ---
        frac_done = env_steps / cfg.total_env_steps
        anneal_start = cfg.ppo.lr_anneal_start
        lr = cfg.ppo.learning_rate if frac_done <= anneal_start else cfg.ppo.learning_rate * max((1.0 - frac_done) / (1.0 - anneal_start), 0.0)
        for pg in optim.param_groups:
            pg["lr"] = lr

        flat_obs = torch.from_numpy(buf.obs.reshape(-1, env.obs_dim)).float()
        flat_actions = torch.from_numpy(buf.actions.reshape(-1)).float()
        flat_old_lp = torch.from_numpy(buf.log_probs.reshape(-1)).float()
        flat_adv = torch.from_numpy(advantages.reshape(-1)).float()
        flat_ret = torch.from_numpy(returns.reshape(-1)).float()

        metrics = ppo_update(
            policy,
            value,
            optim,
            flat_obs,
            flat_actions,
            flat_old_lp,
            flat_adv,
            flat_ret,
            clip_range=cfg.ppo.clip_range,
            update_epochs=cfg.ppo.update_epochs,
            minibatches=cfg.ppo.minibatches,
            entropy_coef=cfg.ppo.entropy_coef,
            value_coef=cfg.ppo.value_coef,
            max_grad_norm=cfg.ppo.max_grad_norm,
        )

        update_idx += 1

        # --- Validation gate ---
        val_attempted = update_idx % cfg.validation_interval_updates == 0
        val_record: dict[str, Any] = {}
        if val_attempted:
            val_record = _validate_deterministic(policy, toml_path, output_dir, cfg, input_mask)
            if val_record["val_rms_cost"] < best_val_cost:
                best_val_cost = val_record["val_rms_cost"]
                export_policy_to_json(policy, output_dir / "best_model.json", input_mask)
                val_record["val_promoted"] = True
            else:
                val_record["val_promoted"] = False

        # --- Periodic checkpoint ---
        if update_idx % cfg.checkpoint_interval_updates == 0:
            _save_checkpoint(output_dir, policy, value, optim, update_idx, env_steps, best_val_cost)

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

    # --- Final checkpoint + export ---
    _save_checkpoint(output_dir, policy, value, optim, update_idx, env_steps, best_val_cost)
    if best_val_cost == float("inf"):
        # No validation fired yet; export current policy so the caller has something.
        export_policy_to_json(policy, output_dir / "best_model.json", input_mask)

    env.close()


def _save_checkpoint(
    output_dir: Path,
    policy: GaussianPolicy,
    value: ValueNetwork,
    optim: torch.optim.Optimizer,
    update_idx: int,
    env_steps: int,
    best_val_cost: float,
) -> None:
    torch.save(
        {
            "policy": policy.state_dict(),
            "value": value.state_dict(),
            "optim": optim.state_dict(),
            "update_idx": update_idx,
            "env_steps": env_steps,
            "best_val_cost": best_val_cost,
        },
        output_dir / "checkpoint.pt",
    )


def _validate_deterministic(
    policy: GaussianPolicy,
    toml_path: Path,
    output_dir: Path,
    cfg: RLConfig,
    input_mask: list[int],
) -> dict[str, Any]:
    """Export deterministic policy to JSON, run validation batch, return RMS cost + capture rate."""
    import aerocapture_rs  # type: ignore[import]

    from aerocapture.training.evaluate import VALIDATION_SEED_OFFSET, compute_cost, make_reserved_seeds

    tmp_json = output_dir / "gen_current_model.json"
    export_policy_to_json(policy, tmp_json, input_mask)

    base_seed = int(cfg.raw_toml.get("monte_carlo", {}).get("mc_seed", 42))
    seeds = make_reserved_seeds(base_seed, VALIDATION_SEED_OFFSET, cfg.validation_n_sims)

    overrides_list = [{"data.neural_network": str(tmp_json), "monte_carlo.mc_seed": s, "simulation.n_sims": 1} for s in seeds]
    results = aerocapture_rs.run_batch(str(toml_path), overrides_list)
    fr = results.final_records  # (N, 52)

    rms_cost = float(compute_cost(fr))
    # Captured: ifinal==3 AND ecc<1.0 (energy<0 implicit for valid orbits).
    # Indices verified against runner.rs final_record layout:
    #   _IDX_IFINAL=31, _IDX_ECC=9
    capture_rate = float(np.mean((fr[:, _IDX_IFINAL] == 3) & (fr[:, _IDX_ECC] < 1.0)))

    return {"val_rms_cost": rms_cost, "val_capture_rate": capture_rate}


def _run_sac(
    cfg: RLConfig,
    toml_path: Path,
    output_dir: Path,
    logger: RLLogger,
    display: Any,
    interrupted: dict[str, bool],
    env_overrides: dict[str, Any] | None = None,
) -> None:
    """SAC outer loop (experimental). Shares GaussianPolicy + export path with PPO."""
    input_mask, layer_sizes, activations, input_dim = _parse_network_config(cfg)

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

    # --- Checkpoint resume ---
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
        update_idx = int(ckpt["update_idx"])
        env_steps = int(ckpt["env_steps"])
        best_val_cost = float(ckpt["best_val_cost"])
        print(f"SAC resumed from checkpoint: update {update_idx}, {env_steps} env steps", file=sys.stderr)

    obs = env.reset()
    episodic_returns: list[float] = []
    episodic_dvs: list[float] = []
    episodic_captures: list[bool] = []
    start_time = time.time()

    while env_steps < cfg.total_env_steps and not interrupted["v"]:
        # --- Collect one step per env ---
        obs_t = torch.from_numpy(obs).float()
        with torch.no_grad():
            bank_t, _ = agent.policy.sample(obs_t)
        actions_np = bank_t.cpu().numpy().astype(np.float32)

        next_obs, _rust_reward, done, info = env.step(actions_np)

        rewards_np = np.zeros(cfg.n_envs, dtype=np.float32)
        for i, d in enumerate(done):
            if d:
                fr = np.array(info[i]["final_record"], dtype=np.float64)
                from aerocapture.training.rl.rewards import compute_terminal_cost

                term_cost = compute_terminal_cost(fr)
                rewards_np[i] = float(-term_cost)
                episodic_returns.append(float(-term_cost))
                episodic_dvs.append(float(info[i].get("dv_m_s", float("nan"))))
                episodic_captures.append(bool(info[i].get("captured", False)))

        agent.replay_buffer.push(obs, actions_np, rewards_np, next_obs, done)
        obs = next_obs
        env_steps += cfg.n_envs

        # --- Update every train_every steps if buffer warm ---
        if len(agent.replay_buffer) >= sac_cfg.batch_size and env_steps % (sac_cfg.train_every * cfg.n_envs) == 0:
            for _ in range(sac_cfg.gradient_steps):
                batch_obs, batch_act, batch_rew, batch_next, batch_done = agent.replay_buffer.sample(sac_cfg.batch_size)
                metrics = agent.update(batch_obs, batch_act, batch_rew, batch_next, batch_done)
            update_idx += 1

            # --- Validation gate ---
            val_attempted = update_idx % cfg.validation_interval_updates == 0
            val_record: dict[str, Any] = {}
            if val_attempted:
                val_record = _validate_deterministic(agent.policy, toml_path, output_dir, cfg, input_mask)
                if val_record["val_rms_cost"] < best_val_cost:
                    best_val_cost = val_record["val_rms_cost"]
                    export_policy_to_json(agent.policy, output_dir / "best_model.json", input_mask)
                    val_record["val_promoted"] = True
                else:
                    val_record["val_promoted"] = False

            # --- Periodic checkpoint ---
            if update_idx % cfg.checkpoint_interval_updates == 0:
                torch.save(
                    {
                        "policy": agent.policy.state_dict(),
                        "q1": agent.q1.state_dict(),
                        "q2": agent.q2.state_dict(),
                        "q1_target": agent.q1_target.state_dict(),
                        "q2_target": agent.q2_target.state_dict(),
                        "log_alpha": agent.log_alpha.data,
                        "update_idx": update_idx,
                        "env_steps": env_steps,
                        "best_val_cost": best_val_cost,
                    },
                    output_dir / "checkpoint.pt",
                )

            record: dict[str, Any] = {
                "update_idx": update_idx,
                "env_steps": env_steps,
                "episodic_return_mean": float(np.mean(episodic_returns[-64:])) if episodic_returns else float("nan"),
                "episodic_dv_m_s_mean": float(np.mean(episodic_dvs[-64:])) if episodic_dvs else float("nan"),
                "episodic_capture_rate": float(np.mean(episodic_captures[-64:])) if episodic_captures else float("nan"),
                "policy_loss": metrics.get("policy_loss", float("nan")),
                "value_loss": metrics.get("q_loss", float("nan")),  # map q_loss to value_loss for display compat
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

    # --- Final checkpoint + export ---
    torch.save(
        {
            "policy": agent.policy.state_dict(),
            "q1": agent.q1.state_dict(),
            "q2": agent.q2.state_dict(),
            "q1_target": agent.q1_target.state_dict(),
            "q2_target": agent.q2_target.state_dict(),
            "log_alpha": agent.log_alpha.data,
            "update_idx": update_idx,
            "env_steps": env_steps,
            "best_val_cost": best_val_cost,
        },
        output_dir / "checkpoint.pt",
    )
    if best_val_cost == float("inf"):
        export_policy_to_json(agent.policy, output_dir / "best_model.json", input_mask)

    env.close()


if __name__ == "__main__":
    main()
