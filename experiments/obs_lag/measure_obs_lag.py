"""Fly one NN cell through run_batch (deploy) and through BatchedSimulation (the RL env).

Paired arms on the fresh 8M pool, legacy noise regime. The env seeds its per-sim noise with
`env_idx = seed`; run_batch uses `env_idx = 0`, so the deploy arms override
`simulation.random_seed = seed * 10_000` to fly the env's noise path (legacy base seed =
`random_seed + env_idx * 10_000`):
  deploy_quoted   run_batch on the requote's own noise path (reproduces fresh_pool_requote.json)
  deploy          run_batch, env-matched noise
  deploy_noshape  run_batch, env-matched noise, command shaper off
  env             BatchedSimulation + the torch-mirror policy on the obs step() returns; measures
                  whatever env the installed extension implements (see README.md)

    uv run python experiments/obs_lag/measure_obs_lag.py env --tag parity
    uv run python experiments/obs_lag/measure_obs_lag.py deploy \\
        --cell articles/paper/data/runs/rl/dense_p515_ppo_warm --toml configs/training/paper/rl/dense_p515_ppo_warm.toml
"""

import argparse
from pathlib import Path

import aerocapture_rs
import numpy as np
import numpy.typing as npt
import torch
from aerocapture.training.cell_eval import FR_DV_TOTAL, _resolve_cell, evaluate_cell, is_captured, reserved_pool
from aerocapture.training.deploy_overrides import LEGACY_NOISE_REGIME
from aerocapture.training.model_io import load_policy_from_json
from aerocapture.training.seeds import HEADLINE_REQUOTE_SEED_OFFSET

OUT = Path("/tmp/obs_lag")


def fly_env(model: Path, eval_toml: Path, overrides: dict[str, object], seeds: list[int]) -> npt.NDArray[np.float64]:
    policy = load_policy_from_json(str(model)).double().eval()
    env = aerocapture_rs.BatchedSimulation(str(eval_toml), n_envs=len(seeds), overrides=overrides)
    obs, _ = env.reset(np.asarray(seeds, dtype=np.int64))
    records: dict[int, npt.NDArray[np.float64]] = {}
    state = policy.new_state(len(seeds), "cpu")
    with torch.no_grad():
        while len(records) < len(seeds):
            out, state = policy(torch.from_numpy(np.asarray(obs, dtype=np.float64)), state)
            bank = torch.atan2(out[:, 0], out[:, 1]).numpy().astype(np.float32)
            obs, _, done, info, _ = env.step(bank)
            for i in np.flatnonzero(done):
                records.setdefault(int(i), np.asarray(info[i]["final_record"], dtype=np.float64))
    return np.stack([records[i] for i in range(len(seeds))])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("arm", choices=["deploy_quoted", "deploy", "deploy_noshape", "env"])
    p.add_argument("--tag", default="")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--cell", type=Path, default=Path("articles/paper/data/runs/headline/dense_p515"))
    p.add_argument("--toml", type=Path, default=Path("configs/training/msr_aller_nn_atan2_best_paper.toml"))
    a = p.parse_args()

    eval_toml, cell_ov = _resolve_cell(a.cell, a.toml, None)
    seeds = reserved_pool(eval_toml, HEADLINE_REQUOTE_SEED_OFFSET, a.n)
    matched = [{"simulation.random_seed": float(s * 10_000)} for s in seeds]
    if a.arm == "deploy_quoted":
        fr = evaluate_cell(a.cell, a.toml, pool=(HEADLINE_REQUOTE_SEED_OFFSET, a.n), extra_overrides=LEGACY_NOISE_REGIME).final_records
    elif a.arm in ("deploy", "deploy_noshape"):
        extra = {**LEGACY_NOISE_REGIME, **({"guidance.command_shaping.enabled": False} if a.arm == "deploy_noshape" else {})}
        fr = evaluate_cell(a.cell, a.toml, seeds, extra_overrides=extra, per_seed_overrides=matched).final_records
    else:
        fr = fly_env(a.cell / "best_model.json", eval_toml, {"simulation.n_sims": 1, **cell_ov, **LEGACY_NOISE_REGIME}, seeds)

    OUT.mkdir(exist_ok=True)
    name = a.cell.name + "_" + a.arm + (f"_{a.tag}" if a.tag else "")
    np.savez(OUT / f"{name}.npz", seeds=np.asarray(seeds), final_records=fr)
    cap = is_captured(fr)
    dv = np.sort(fr[cap, FR_DV_TOTAL])
    tail = dv[-max(1, len(dv) // 20) :].mean()
    print(
        f"{name}: n={len(fr)} capture={100 * cap.mean():.1f}% mean={dv.mean():.2f} p50={np.percentile(dv, 50):.2f} "
        f"p95={np.percentile(dv, 95):.2f} cvar95={tail:.2f} max={dv.max():.2f}"
    )


if __name__ == "__main__":
    main()
