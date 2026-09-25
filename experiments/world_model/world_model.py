"""Learned dynamics (world model) of the aerocapture plant (#113).

Stages (all by default, each resumable): data, train, plan, eval, plot. Datasets and model
checkpoints live under training_output/world_model/ (regenerable, not tracked); results go to
world_model.json and the six figures next to this file. Question, design and findings:
experiments/world_model/README.md.

Usage: uv run python experiments/world_model/world_model.py [--stages data train plan eval plot] [--force] [--arms ...]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aerocapture_rs
import matplotlib as mpl
import numpy as np
import torch
import wm_metrics
import wm_model
import wm_planner
import wm_plant
from aerocapture.training.deploy_overrides import overrides_from_params
from aerocapture.training.paper_stats import capture_mask, paired_comparison, run_stats
from aerocapture.training.seeds import WORLD_MODEL_SEED_OFFSET, base_mc_seed_from_toml, make_reserved_seeds
from aerocapture.training.toml_utils import load_toml_with_bases
from scipy.stats import spearmanr

mpl.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "world_model.json"
WORK = REPO / "training_output/world_model"
WM_MEDIUM = HERE / "configs/wm_medium.toml"
WM_HIGH = HERE / "configs/wm_high.toml"
BASE_MC_SEED = base_mc_seed_from_toml(load_toml_with_bases(WM_MEDIUM))
POOLS = {"train": 10_000, "val": 1_000, "test_id": 1_000, "test_ood": 1_000}
POOL_TOML = {"train": WM_MEDIUM, "val": WM_MEDIUM, "test_id": WM_MEDIUM, "test_ood": WM_HIGH}
N_PLAN = 1_000  # the planning pool: the next slice of the same reserved stream
MAX_TICKS = 2_000  # [simulation] max_time 2000 s at the 1 s guidance period
KINDS = ("gru", "mlp")
MODEL_SEEDS = (0, 1, 2)
MODELS = [f"{k}_s{s}" for k in KINDS for s in MODEL_SEEDS]
EPOCHS = 40
FNPAG_TOML = "configs/training/msr_aller_fnpag_train.toml"
FNPAG_PARAMS = REPO / "articles/paper/data/runs/classical_baselines/fnpag/best_params.json"
REPLAN_EVERY = 10  # ticks; FNPAG replans every 2 s, the clairvoyant arm's replays make 10 the affordable period


def _stream() -> list[int]:
    return make_reserved_seeds(BASE_MC_SEED, WORLD_MODEL_SEED_OFFSET, sum(POOLS.values()) + N_PLAN)


def pool_seeds() -> dict[str, list[int]]:
    stream, out, lo = _stream(), {}, 0
    for name, n in POOLS.items():
        out[name] = stream[lo : lo + n]
        lo += n
    return out


def planning_seeds() -> list[int]:
    return _stream()[-N_PLAN:]


def _sh(*cmd: str) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()


def meta() -> dict[str, object]:
    """The machine and commit a stage ran on (its timings are only comparable on the same machine)."""
    return {
        "cpu": _sh("sysctl", "-n", "machdep.cpu.brand_string") or platform.processor(),
        "logical_cpus": os.cpu_count(),
        "memory_gib": round(int(_sh("sysctl", "-n", "hw.memsize") or 0) / 2**30),
        "torch_threads": torch.get_num_threads(),
        "commit": _sh("git", "rev-parse", "HEAD"),
        "date": time.strftime("%Y-%m-%d"),
    }


@dataclass
class Pool:
    """Flights of one pool: per-flight arrays plus terminal outcomes."""

    seeds: np.ndarray
    obs: list[np.ndarray]
    aux: list[np.ndarray]
    actions: list[np.ndarray]
    ifinal: np.ndarray
    captured: np.ndarray
    dv_m_s: np.ndarray
    apoapsis_km: np.ndarray
    ecc: np.ndarray
    final_record: np.ndarray

    def __len__(self) -> int:
        return len(self.seeds)


def save_pool(path: Path, eps: list[wm_plant.Episode]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        seeds=np.array([e.seed for e in eps], np.int64),
        lengths=np.array([len(e) for e in eps], np.int64),
        obs=np.concatenate([e.obs for e in eps]),
        aux=np.concatenate([e.aux for e in eps]),
        actions=np.concatenate([e.actions for e in eps]),
        ifinal=np.array([e.ifinal for e in eps]),
        captured=np.array([e.captured for e in eps]),
        dv_m_s=np.array([e.dv_m_s for e in eps]),
        apoapsis_km=np.array([e.apoapsis_km for e in eps]),
        ecc=np.array([e.ecc for e in eps]),
        final_record=np.stack([e.final_record for e in eps]),
    )


def load_pool(path: Path) -> Pool:
    z = np.load(path)
    cuts = np.cumsum(z["lengths"])[:-1]
    return Pool(
        z["seeds"],
        np.split(z["obs"], cuts),
        np.split(z["aux"], cuts),
        np.split(z["actions"], cuts),
        z["ifinal"],
        z["captured"],
        z["dv_m_s"],
        z["apoapsis_km"],
        z["ecc"],
        z["final_record"],
    )


def stage_data(args: argparse.Namespace) -> dict[str, object]:
    wm_plant.write_stub_model(WM_MEDIUM)
    block: dict[str, object] = {"meta": meta()}
    for name, seeds in pool_seeds().items():
        path = WORK / f"data/{name}.npz"
        if not path.exists() or args.force:
            t0 = time.perf_counter()
            eps = wm_plant.fly(POOL_TOML[name], seeds, np.stack([wm_plant.bank_schedule(s, MAX_TICKS) for s in seeds]))
            save_pool(path, eps)
            path.with_suffix(".json").write_text(json.dumps({"wall_s": time.perf_counter() - t0}) + "\n")
            print(f"data {name}: {len(eps)} flights in {time.perf_counter() - t0:.0f} s")
        pool = load_pool(path)
        lengths = np.array([len(a) for a in pool.actions])
        wall = json.loads(path.with_suffix(".json").read_text())["wall_s"]
        block[name] = {
            "config": str(POOL_TOML[name].relative_to(REPO)),
            "wall_s": round(wall, 1),
            "env_ticks_per_s": round(float(lengths.sum()) / wall),
            "n_flights": len(pool),
            "n_steps": int(lengths.sum()),
            "length_p5_p50_p95": np.percentile(lengths, [5, 50, 95]).tolist(),
            "crash_pct": round(100 * float(np.mean(pool.ifinal == 1)), 2),
            "capture_pct": round(100 * float(np.mean(pool.captured)), 2),
            "captured_apoapsis_200_1500_km_pct": round(100 * float(np.mean(pool.captured & (pool.apoapsis_km > 200) & (pool.apoapsis_km < 1500))), 2),
        }
    return block


def model_path(kind: str, seed: int) -> Path:
    return WORK / f"models/{kind}_s{seed}.pt"


def stage_train(args: argparse.Namespace) -> dict[str, object]:
    train, val = _pool_arrays("data/train.npz")[1], _pool_arrays("data/val.npz")[1]
    block: dict[str, object] = {
        "meta": meta(),
        "epochs": EPOCHS,
        "batch_flights": wm_model.BATCH_FLIGHTS,
        "optimizer": f"Adam {wm_model.LR:g}, cosine to {wm_model.LR / 10:g}, grad clip 1",
        "device": "cpu",
    }
    for kind in KINDS:
        for seed in MODEL_SEEDS:
            path = model_path(kind, seed)
            if not path.exists() or args.force:
                t0 = time.perf_counter()
                model, norm, history = wm_model.fit(kind, train, val, seed, EPOCHS)
                history[-1]["total_wall_s"] = time.perf_counter() - t0
                wm_model.save(path, model, norm, history)
            model, _, history = wm_model.load(path)
            best = min(history, key=lambda r: r["val_nll"])
            block[f"{kind}_s{seed}"] = {
                "params": sum(p.numel() for p in model.parameters()),
                "best_epoch": int(best["epoch"]),
                "best_val_nll": round(best["val_nll"], 4),
                "train_wall_min": round(history[-1].get("total_wall_s", float("nan")) / 60, 1),
                "epoch_wall_s_p50": round(float(np.median([r["wall_s"] for r in history])), 1),
                "val_nll_by_epoch": [round(r["val_nll"], 4) for r in history],
            }
    return block


def planner_config() -> wm_planner.PlannerConfig:
    target_km = aerocapture_rs.load_config(str(WM_MEDIUM))["flight"]["target_orbit"]["apoapsis"]
    return wm_planner.PlannerConfig.from_fnpag(json.loads(FNPAG_PARAMS.read_text()), target_km, REPLAN_EVERY)


def readout() -> wm_planner.Readout:
    exit_km = aerocapture_rs.load_config(str(WM_MEDIUM))["flight"]["final_conditions"]["altitude"]
    return wm_planner.Readout(wm_plant.load_normalization(), exit_km)


def predictor(label: str) -> wm_model.Predictor:
    kind, seed = label.split("_s")
    model, norm, _ = wm_model.load(model_path(kind, int(seed)))
    return wm_model.Predictor(model, norm)


def _outcomes(ifinal: np.ndarray, ecc: np.ndarray, dv: np.ndarray, fr: np.ndarray) -> dict[str, object]:
    idx = aerocapture_rs.final_record_indices()
    cap = capture_mask(ifinal, ecc)
    stats = {k: v for k, v in run_stats(ifinal, ecc, dv).items() if k in ("n", "capture_pct", "dv_p50", "dv_p95", "dv_p99", "dv_cvar95", "dv_cvar95_ci")}
    return {
        **stats,
        "apoapsis_err_km_p50_abs": round(float(np.median(np.abs(fr[cap, idx["apoapsis_err_km"]]))), 2),
        "apoapsis_err_km_p95_abs": round(float(np.percentile(np.abs(fr[cap, idx["apoapsis_err_km"]]), 95)), 2),
        "periapsis_alt_km_p50": round(float(np.median(fr[cap, idx["periapsis_alt_km"]])), 2),
        "dv1_ms_p50": round(float(np.median(fr[cap, idx["dv1_ms"]])), 2),
    }


def _curves(ifinal: np.ndarray, ecc: np.ndarray, dv: np.ndarray, fr: np.ndarray) -> dict[str, object]:
    """Sorted captured DV and |apoapsis error| (the survival curves of fig_planning)."""
    cap = capture_mask(ifinal, ecc)
    apo = np.abs(fr[cap, aerocapture_rs.final_record_indices()["apoapsis_err_km"]])
    return {"dv_sorted": np.round(np.sort(dv[cap]), 2).tolist(), "apoapsis_err_abs_sorted": np.round(np.sort(apo), 3).tolist()}


def stage_plan(args: argparse.Namespace) -> dict[str, object]:
    seeds, cfg = planning_seeds(), planner_config()
    block: dict[str, object] = {
        "meta": meta(),
        "pool": f"{N_PLAN} seeds after the model pools in the WORLD_MODEL_SEED_OFFSET stream (base MC seed {BASE_MC_SEED})",
        "config": str(WM_MEDIUM.relative_to(REPO)),
        "noise_seeding": aerocapture_rs.load_config(str(WM_MEDIUM))["monte_carlo"].get("noise_seeding", "per_draw"),
        "replan_every_ticks": REPLAN_EVERY,
        "target_apoapsis_km": cfg.target_apoapsis_m / 1e3,
        "planner": "FNPAG corrector (bisection on a constant bank to exit, 11 predictions per replan) + lateral port, FNPAG deployed gains",
    }
    runs: dict[str, object] = {}
    for arm in args.arms:
        path = WORK / f"plan/{arm}.npz"
        if not path.exists() or args.force:
            print(f"plan {arm}: {len(seeds)} flights")
            t0 = time.perf_counter()
            predict: wm_planner.ModelArm | wm_planner.OracleArm
            if arm == "oracle":
                predict = wm_planner.OracleArm(WM_MEDIUM, seeds, wm_plant.STUB_MODEL)
            else:
                predict = wm_planner.ModelArm(predictor(arm), len(seeds), readout())
            eps, timing = wm_planner.run_mpc(WM_MEDIUM, seeds, cfg, predict, readout(), wm_plant.STUB_MODEL)
            timing["total_wall_s"] = time.perf_counter() - t0
            save_pool(path, eps)
            path.with_suffix(".json").write_text(json.dumps(timing, indent=1) + "\n")
    fnpag = WORK / "plan/fnpag.npy"
    if not fnpag.exists() or args.force:
        t0 = time.perf_counter()
        ovr = overrides_from_params(json.loads(FNPAG_PARAMS.read_text()), "fnpag")
        np.save(fnpag, aerocapture_rs.run_grid(str(REPO / FNPAG_TOML), [ovr], seeds)[0])
        fnpag.with_suffix(".json").write_text(json.dumps({"total_wall_s": time.perf_counter() - t0}) + "\n")
    idx = aerocapture_rs.final_record_indices()
    fr = np.load(fnpag)
    fn = (fr[:, idx["ifinal"]], fr[:, idx["ecc"]], fr[:, idx["dv_total_ms"]], fr)
    fn_wall = json.loads(fnpag.with_suffix(".json").read_text())["total_wall_s"]
    runs["fnpag_deployed"] = {"config": FNPAG_TOML, "total_wall_s": round(fn_wall, 1), **_outcomes(*fn), **_curves(*fn)}
    oracle = load_pool(WORK / "plan/oracle.npz") if (WORK / "plan/oracle.npz").exists() else None
    for arm in ["oracle", *MODELS]:
        path = WORK / f"plan/{arm}.npz"
        if not path.exists():
            continue
        pool = load_pool(path)
        timing = json.loads(path.with_suffix(".json").read_text())
        row: dict[str, object] = {
            **_outcomes(pool.ifinal, pool.ecc, pool.dv_m_s, pool.final_record),
            "ms_per_flight_replan": round(timing["ms_per_flight_replan"], 2),
            "plan_wall_s": round(timing["plan_wall_s"], 1),
            "total_wall_s": round(timing["total_wall_s"], 1),
            "at_bank_min_pct_ticks_0_150": round(
                100 * float(np.mean(np.concatenate([np.isclose(np.abs(a[:150]), cfg.bank_min, atol=1e-3) for a in pool.actions]))), 1
            ),
            **_curves(pool.ifinal, pool.ecc, pool.dv_m_s, pool.final_record),
        }
        if oracle is not None and arm != "oracle":
            row["vs_oracle"] = paired_comparison(pool.dv_m_s, capture_mask(pool.ifinal, pool.ecc), oracle.dv_m_s, capture_mask(oracle.ifinal, oracle.ecc))
        runs[arm] = row
    block["runs"] = runs
    return block


# ---- eval ----

H = 200
START_EVERY = 50
H_PROB = (1, 2, 5, 10, 20, 50, 100, 200)
LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
N_PROB_FLIGHTS, PROB_START_EVERY, N_SAMPLES = 250, 100, 32
PHYS = {"altitude_km": 8, "velocity_m_s": 4, "fpa_deg": 9, "energy_mj_kg": wm_planner.ENERGY}
EVAL_POOLS = {"test_id": "data/test_id.npz", "test_ood": "data/test_ood.npz", "closed_loop": "plan/oracle.npz", "self_planned": "plan/gru_s0.npz"}
TAIL_STARTS = (100, 200, 300, 400)  # absolute ticks, bounce to late ascent: a start at a fraction of each flight would leak its length
CF_FRACS = (0.25, 0.5, 0.75, 1.0)  # of the tick of the bounce
CF_DELTA_DEG, CF_WINDOW = 10.0, 20
CHUNK = 1500


def _pool_arrays(rel: str) -> tuple[Pool, wm_model.Sequences]:
    pool = load_pool(WORK / rel)
    return pool, wm_model.Sequences.from_arrays(pool.obs, pool.aux, pool.actions)


def _phys(ro: wm_planner.Readout, x: np.ndarray, name: str) -> np.ndarray:
    j = PHYS[name]
    if j == wm_planner.ENERGY:
        return x[..., j].astype(np.float64)
    v = ro.raw(x, j)
    return np.rad2deg(v) if name == "fpa_deg" else v


Filtered = list[tuple[np.ndarray, np.ndarray | None]]


def _filter(pred: wm_model.Predictor, seqs: wm_model.Sequences) -> Filtered:
    """Teacher-forced pass over every flight: (one-step means, recurrent state after each step)."""
    out: Filtered = []
    for x, a in zip(seqs.x, seqs.a_next, strict=True):
        mu, _, hs = pred.teacher_forced(x, a)
        out.append((mu, hs))
    return out


def _start_states(
    filt: Filtered, seqs: wm_model.Sequences, actions: list[np.ndarray], pairs: list[tuple[int, int]], horizon: int, with_truth: bool = True
) -> dict[str, np.ndarray]:
    """For (flight, t0) pairs: the filtered state at t0, the flown banks a_{t0+1..} (last one held), and optionally
    the truth x_{t0+1..} and teacher-forced means with their validity mask."""
    b = len(pairs)
    out: dict[str, np.ndarray] = {
        "h0": np.zeros((b, wm_model.HIDDEN), np.float32),
        "x0": np.empty((b, wm_model.N_X), np.float32),
        "banks": np.empty((b, horizon)),
    }
    if with_truth:
        out["truth"] = np.full((b, horizon, wm_model.N_X), np.nan, np.float32)
        out["tf"] = np.full((b, horizon, wm_model.N_X), np.nan, np.float32)
        out["mask"] = np.zeros((b, horizon), bool)
    for r, (i, t0) in enumerate(pairs):
        mu, hs = filt[i]
        x, a = seqs.x[i], actions[i]
        if hs is not None and t0 > 0:
            out["h0"][r] = hs[t0 - 1]
        out["x0"][r] = x[t0]
        tail = a[t0 + 1 : t0 + 1 + horizon]
        out["banks"][r, : len(tail)] = tail
        out["banks"][r, len(tail) :] = tail[-1] if len(tail) else a[-1]
        if with_truth:
            n = min(horizon, len(x) - 1 - t0)
            out["truth"][r, :n] = x[t0 + 1 : t0 + 1 + n]
            out["tf"][r, :n] = mu[t0 : t0 + n]
            out["mask"][r, :n] = True
    return out


def _rmse(pred: np.ndarray, truth: np.ndarray, sd: np.ndarray, active: np.ndarray) -> np.ndarray:
    return np.asarray(np.sqrt(np.mean(((pred - truth) / sd)[..., active] ** 2, axis=-1)))


def _curve(e: np.ndarray) -> dict[str, list[float]]:
    return {"median": np.round(np.nanmedian(e, axis=0), 4).tolist(), "mean": np.round(np.nanmean(e, axis=0), 4).tolist()}


def eval_deterministic(
    pred: wm_model.Predictor, seqs: wm_model.Sequences, actions: list[np.ndarray], ref: wm_model.Normalizer, ro: wm_planner.Readout
) -> dict[str, object]:
    """Free-running (mean) and teacher-forced error vs horizon from starts every START_EVERY ticks, plus persistence."""
    filt = _filter(pred, seqs)
    pairs = [(i, t0) for i in range(len(seqs.x)) for t0 in range(0, len(seqs.x[i]) - 1, START_EVERY)]
    err: dict[str, list[np.ndarray]] = {k: [] for k in ("free", "tf", "persist", "kin_free", "kin_true", *(f"abs_{p}" for p in PHYS))}
    for lo in range(0, len(pairs), CHUNK):
        st = _start_states(filt, seqs, actions, pairs[lo : lo + CHUNK], H)
        free = pred.free_run(st["h0"], st["x0"], st["banks"])
        err["free"].append(_rmse(free, st["truth"], ref.sd_x, ref.active))
        err["tf"].append(_rmse(st["tf"], st["truth"], ref.sd_x, ref.active))
        err["persist"].append(_rmse(np.broadcast_to(st["x0"][:, None], free.shape), st["truth"], ref.sd_x, ref.active))
        for p in PHYS:
            err[f"abs_{p}"].append(np.abs(_phys(ro, free, p) - _phys(ro, st["truth"], p)))
        for key, traj in (("kin_free", free), ("kin_true", st["truth"])):
            err[key].append(_kinematic_residual(ro, st["x0"], traj))
    return {"n_pairs": len(pairs), **{k: _curve(np.concatenate(v)) for k, v in err.items()}}


def _kinematic_residual(ro: wm_planner.Readout, x0: np.ndarray, traj: np.ndarray) -> np.ndarray:
    """|altitude step - V sin(fpa) dt| per step (km): how far a trajectory is from the kinematic identity h' = V sin(gamma)."""
    full = np.concatenate([x0[:, None], traj], axis=1)
    alt, v, fpa = ro.raw(full, wm_planner.ALT), ro.raw(full, wm_planner.VEL), ro.raw(full, wm_planner.FPA)
    return np.asarray(np.abs(np.diff(alt, axis=1) - v[:, :-1] * np.sin(fpa[:, :-1]) / 1e3))


BANK_SWEEP_DEG = (20, 30, 40, 50, 60, 65, 70, 75, 80, 90, 100, 110)
N_SWEEP = 200


def eval_bank_sweep(preds: dict[str, wm_model.Predictor], ro: wm_planner.Readout) -> dict[str, object]:
    """The planner's first query: exit apoapsis of a constant bank flown from the first tick, model vs plant."""
    pool, seqs = _pool_arrays(EVAL_POOLS["test_id"])
    seeds = [int(s) for s in pool.seeds[:N_SWEEP]]
    x0 = np.stack([seqs.x[i][0] for i in range(N_SWEEP)])
    first = np.stack([pool.actions[i][:1] for i in range(N_SWEEP)])
    out: dict[str, dict[str, list[float]]] = {}
    for b in np.deg2rad(BANK_SWEEP_DEG):
        rows = {"plant": wm_planner.plant_apoapsis(wm_plant.fly(WM_MEDIUM, seeds, np.concatenate([first, np.full((N_SWEEP, 1), b)], axis=1)))}
        h0 = np.zeros((N_SWEEP, wm_model.HIDDEN), np.float32)
        for name, pred in preds.items():
            rows[name] = ro.apoapsis(pred.free_run(h0, x0, np.full((N_SWEEP, wm_planner.ROLLOUT_CAP), b), stop=ro.terminated))
        for name, apo in rows.items():
            r = out.setdefault(name, {"apoapsis_km_p50": [], "crash_frac": []})
            r["apoapsis_km_p50"].append(round(float(np.median(apo)) / 1e3, 1))
            r["crash_frac"].append(round(float(np.mean(apo == 0.0)), 3))
    return {"banks_deg": list(BANK_SWEEP_DEG), "n_flights": N_SWEEP, "start": "tick 0 of the in-distribution test pool", **out}


def _sampled(
    pred: wm_model.Predictor, seqs: wm_model.Sequences, actions: list[np.ndarray], ref: wm_model.Normalizer, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """N_SAMPLES sampled free runs per start: (samples (N_SAMPLES, pairs, horizons, 40), truth, valid), standardized."""
    n = min(N_PROB_FLIGHTS, len(seqs.x))
    sub = wm_model.Sequences(seqs.x[:n], seqs.a_next[:n])
    pairs = [(i, t0) for i in range(n) for t0 in range(0, len(seqs.x[i]) - 1, PROB_START_EVERY)]
    st = _start_states(_filter(pred, sub), sub, actions, pairs, H)
    rep = lambda a: np.repeat(a[None], N_SAMPLES, axis=0).reshape(-1, *a.shape[1:])  # noqa: E731
    samples = pred.free_run(rep(st["h0"]), rep(st["x0"]), rep(st["banks"]), rng=np.random.default_rng(seed), keep=H_PROB)
    samples = ((samples.reshape(N_SAMPLES, len(pairs), len(H_PROB), -1) - ref.mu_x) / ref.sd_x)[..., ref.active]
    truth = ((st["truth"][:, [h - 1 for h in H_PROB]] - ref.mu_x) / ref.sd_x)[..., ref.active]
    return samples, truth, st["mask"][:, [h - 1 for h in H_PROB]]


def _prob_scores(samples: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> dict[str, object]:
    """Ensemble CRPS and central-interval coverage at every H_PROB horizon."""
    crps, coverage, spread, mean_abs_err = [], [], [], []
    for k in range(len(H_PROB)):
        v = valid[:, k]
        s, t = samples[:, v, k], truth[v, k]
        crps.append(round(float(wm_metrics.crps_ensemble(s, t).mean()), 4))
        coverage.append(np.round(wm_metrics.central_coverage(s, t, np.array(LEVELS)), 4).tolist())
        spread.append(round(float(np.median(s.std(axis=0))), 5))
        mean_abs_err.append(round(float(np.median(np.abs(s.mean(axis=0) - t))), 5))
    return {
        "n_pairs": len(truth),
        "n_members": len(samples),
        "horizons": list(H_PROB),
        "levels": list(LEVELS),
        "crps": crps,
        "coverage": coverage,
        "spread_median": spread,
        "mean_abs_err_median": mean_abs_err,
    }


def eval_probabilistic(
    preds: dict[str, wm_model.Predictor], seqs: wm_model.Sequences, actions: list[np.ndarray], ref: wm_model.Normalizer
) -> dict[str, object]:
    """Each model's sampled free runs, then per kind the pooled samples of its three training seeds (a deep ensemble)."""
    out: dict[str, object] = {}
    by_kind: dict[str, list[np.ndarray]] = {}
    for k, (name, pred) in enumerate(preds.items()):
        samples, truth, valid = _sampled(pred, seqs, actions, ref, k)
        out[name] = _prob_scores(samples, truth, valid)
        by_kind.setdefault(name.split("_s")[0], []).append(samples)
    for kind, members in by_kind.items():
        out[f"{kind}_ensemble"] = _prob_scores(np.concatenate(members), truth, valid)
    return out


def _bounce_tick(ro: wm_planner.Readout, x: np.ndarray) -> int:
    b = ro.raw(x, wm_planner.BOUNCE) > 0.5
    return int(b.argmax()) if b.any() else len(x) - 1


def eval_tail(
    preds: dict[str, wm_model.Predictor],
    pool: Pool,
    seqs: wm_model.Sequences,
    ro: wm_planner.Readout,
    label: np.ndarray,
    persistence: Callable[[np.ndarray], np.ndarray],
) -> dict[str, object]:
    """Does a free run on the flown banks from tick t0 rank the flights that end badly (label) first?

    A crash reads as an infinite predicted delta-v. Flights already over at t0 are dropped. `persistence`
    scores the state at t0 alone (x0 -> score, higher = worse), the no-model baseline.
    """
    starts = [[(i, t0) for i in range(len(pool)) if t0 < len(seqs.x[i]) - 1] for t0 in TAIL_STARTS]
    idx = [np.array([i for i, _ in pairs]) for pairs in starts]
    auc = {
        "persistence": [
            round(wm_metrics.roc_auc(persistence(np.stack([seqs.x[i][t0] for i, t0 in pairs])), label[j]), 4) for pairs, j in zip(starts, idx, strict=True)
        ]
    }
    crash: dict[str, list[float]] = {}
    for name, pred in preds.items():
        filt = _filter(pred, seqs)
        auc[name], crash[name] = [], []
        for pairs, j in zip(starts, idx, strict=True):
            st = _start_states(filt, seqs, pool.actions, pairs, wm_planner.ROLLOUT_CAP, with_truth=False)
            dv = ro.dv(pred.free_run(st["h0"], st["x0"], st["banks"], stop=ro.terminated))
            auc[name].append(round(wm_metrics.roc_auc(dv, label[j]), 4))
            crash[name].append(round(float(np.mean(np.isinf(dv))), 4))
    return {
        "starts": list(TAIL_STARTS),
        "n_flights": [len(j) for j in idx],
        "n_positive": [int(label[j].sum()) for j in idx],
        "true_crash_frac": round(float(np.mean(pool.ifinal == 1)), 4),
        "auc": auc,
        "predicted_crash_frac": crash,
    }


def tail_evals(preds: dict[str, wm_model.Predictor], ro: wm_planner.Readout) -> dict[str, object]:
    """The delta-v tail of the clairvoyant planner's flights, and crashes in the in-distribution test pool."""
    pool, seqs = _pool_arrays(EVAL_POOLS["closed_loop"])
    cap = capture_mask(pool.ifinal, pool.ecc)
    threshold = float(np.percentile(pool.dv_m_s[cap], 95))
    dv_tail = eval_tail(preds, pool, seqs, ro, ~cap | (pool.dv_m_s >= threshold), lambda x0: (100.0 * np.sinh(x0[:, wm_planner.DV])).sum(axis=1))
    test, test_seqs = _pool_arrays(EVAL_POOLS["test_id"])
    crash = eval_tail(preds, test, test_seqs, ro, test.ifinal == 1, lambda x0: -ro.raw(x0, wm_planner.PERI))
    return {
        "dv_tail": {"pool": "closed_loop", "label": "not captured or delta-v >= p95 of captured", "dv_threshold_m_s": round(threshold, 2), **dv_tail},
        "crash": {"pool": "test_id", "label": "crashed", "persistence": "lowest current osculating periapsis", **crash},
    }


CF_SHORT = 30  # ticks after the pulse starts: the short-horizon readout (the pulse lasts CF_WINDOW)


def _agreement(t: np.ndarray, p: np.ndarray, eps: float) -> dict[str, float]:
    ok = np.isfinite(t) & np.isfinite(p)
    big = ok & (np.abs(t) > eps)
    if ok.sum() < 3:
        return {"n": int(ok.sum()), "sign_agreement": float("nan"), "slope": float("nan"), "spearman": float("nan")}
    return {
        "n": int(ok.sum()),
        "sign_agreement": round(float(np.mean(np.sign(p[big]) == np.sign(t[big]))), 4) if big.any() else float("nan"),
        "slope": round(float(np.polyfit(t[ok], p[ok], 1)[0]), 4),
        "spearman": round(float(spearmanr(t[ok], p[ok]).statistic), 4),
    }


def eval_counterfactual(preds: dict[str, wm_model.Predictor], pool: Pool, seqs: wm_model.Sequences, ro: wm_planner.Readout) -> dict[str, object]:
    """Pulse the flown bank magnitude by +-10 deg for 20 ticks at a fraction of the bounce tick, the rest of the flown banks unchanged.

    Readouts of the effect, predicted vs true (the true one by replaying the seed): CF_SHORT ticks after the pulse starts,
    the orbital energy and the osculating apoapsis (where the orbit is bound on both branches); and the exit apoapsis
    (where both model rollouts end in a bound exit).
    """
    nominal = wm_planner.apoapsis_m(pool.ifinal, pool.ecc, pool.apoapsis_km)
    t_ps = [[max(2, int(frac * _bounce_tick(ro, seqs.x[i]))) for i in range(len(pool))] for frac in CF_FRACS]
    nan_x = np.full(wm_model.N_X, np.nan, np.float32)

    def short(x: np.ndarray, t: int) -> np.ndarray:
        return x[t + CF_SHORT] if t + CF_SHORT < len(x) else nan_x

    def deltas(moved: np.ndarray, base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(energy kJ/kg, apoapsis km) change between two (B, 42) states; apoapsis only where both orbits are bound."""
        bound = (moved[:, wm_planner.ENERGY] < 0.0) & (base[:, wm_planner.ENERGY] < 0.0)
        d_apo = (ro.raw(moved, wm_planner.APO) - ro.raw(base, wm_planner.APO)) / 1e3
        return 1e3 * (moved[:, wm_planner.ENERGY] - base[:, wm_planner.ENERGY]).astype(np.float64), np.where(bound, d_apo, np.nan)

    true: dict[str, list[np.ndarray]] = {"energy": [], "apoapsis30": [], "apoapsis_exit": []}
    for t_p in t_ps:
        per_sign: dict[str, list[np.ndarray]] = {k: [] for k in true}
        base_x = np.stack([short(seqs.x[i], t) for i, t in enumerate(t_p)])
        for sign in (1.0, -1.0):
            rows = np.stack([_perturbed(np.concatenate([a, np.full(MAX_TICKS - len(a), a[-1])]), t, sign) for a, t in zip(pool.actions, t_p, strict=True)])
            eps = wm_plant.fly(WM_MEDIUM, [int(s) for s in pool.seeds], rows)
            flown = wm_planner.plant_apoapsis(eps)
            d_e, d_a = deltas(np.stack([short(wm_model.features(e.obs, e.aux), t) for e, t in zip(eps, t_p, strict=True)]), base_x)
            per_sign["energy"].append(d_e)
            per_sign["apoapsis30"].append(d_a)
            per_sign["apoapsis_exit"].append(np.where(_finite(flown) & _finite(nominal), (flown - nominal) / 1e3, np.nan))
        for k in true:
            true[k].append(np.concatenate(per_sign[k]))
    pred_d: dict[str, dict[str, list[np.ndarray]]] = {}
    for name, pred in preds.items():
        filt = _filter(pred, seqs)
        pred_d[name] = {k: [] for k in true}
        for t_p in t_ps:
            st = _start_states(filt, seqs, pool.actions, [(i, t - 1) for i, t in enumerate(t_p)], wm_planner.ROLLOUT_CAP, with_truth=False)
            keep = (CF_SHORT + 1,)  # x_{t_p + CF_SHORT} from the state at t_p - 1, fixed horizon (no early stop)
            base_short = pred.free_run(st["h0"], st["x0"], st["banks"][:, : CF_SHORT + 1], keep=keep)[:, 0]
            base_exit = ro.apoapsis(pred.free_run(st["h0"], st["x0"], st["banks"], stop=ro.terminated))
            per_sign = {k: [] for k in true}
            for sign in (1.0, -1.0):
                banks = np.stack([_perturbed(b, 0, sign) for b in st["banks"]])
                d_e, d_a = deltas(pred.free_run(st["h0"], st["x0"], banks[:, : CF_SHORT + 1], keep=keep)[:, 0], base_short)
                moved_exit = ro.apoapsis(pred.free_run(st["h0"], st["x0"], banks, stop=ro.terminated))
                per_sign["energy"].append(d_e)
                per_sign["apoapsis30"].append(d_a)
                per_sign["apoapsis_exit"].append(np.where(_finite(base_exit) & _finite(moved_exit), (moved_exit - base_exit) / 1e3, np.nan))
            for k in true:
                pred_d[name][k].append(np.concatenate(per_sign[k]))
    eps_by_key = {"energy": 0.1, "apoapsis30": 1.0, "apoapsis_exit": 1.0}  # |true change| below this is not scored for sign
    metrics: dict[str, dict[str, list[float]]] = {}
    scatter: dict[str, dict[str, dict[str, list[float]]]] = {}
    for fi, frac in enumerate(CF_FRACS):
        for name in preds:
            m = metrics.setdefault(name, {})
            for key, eps_ in eps_by_key.items():
                for stat, v in _agreement(true[key][fi], pred_d[name][key][fi], eps_).items():
                    m.setdefault(f"{key}_{stat}", []).append(v)
            if name.endswith("_s0"):
                t, p = true["energy"][fi], pred_d[name]["energy"][fi]
                ok = np.isfinite(t) & np.isfinite(p)
                scatter.setdefault(name, {})[str(frac)] = {"true_kj_kg": np.round(t[ok], 2).tolist(), "pred_kj_kg": np.round(p[ok], 2).tolist()}
    return {
        "fractions_of_bounce_tick": list(CF_FRACS),
        "delta_deg": CF_DELTA_DEG,
        "window_ticks": CF_WINDOW,
        "short_readout_ticks": CF_SHORT,
        "true_energy_delta_kj_kg_p50_abs": [round(float(np.nanmedian(np.abs(t))), 2) for t in true["energy"]],
        "true_apoapsis30_delta_km_p50_abs": [round(float(np.nanmedian(np.abs(t))), 1) for t in true["apoapsis30"]],
        "true_apoapsis30_n": [int(np.isfinite(t).sum()) for t in true["apoapsis30"]],
        "true_apoapsis_exit_delta_km_p50_abs": [round(float(np.nanmedian(np.abs(t))), 1) for t in true["apoapsis_exit"]],
        "metrics": metrics,
        "scatter": scatter,
    }


def _finite(apo: np.ndarray) -> np.ndarray:
    return np.asarray((apo > 0.0) & (apo < wm_planner.UNBOUND_APOAPSIS_M))


def _perturbed(actions: np.ndarray, t_p: int, sign: float) -> np.ndarray:
    a = actions.astype(np.float64).copy()
    w = slice(t_p, t_p + CF_WINDOW)
    a[w] = np.sign(a[w]) * np.clip(np.abs(a[w]) + sign * np.deg2rad(CF_DELTA_DEG), 0.0, np.pi)
    return a


def stage_eval(args: argparse.Namespace) -> dict[str, object]:
    t0 = time.perf_counter()
    for name in ("test_id", "closed_loop"):  # the tail, sweep and counterfactual evals need these two
        if not (WORK / EVAL_POOLS[name]).exists():
            raise FileNotFoundError(f"eval needs {EVAL_POOLS[name]}: run the data and plan stages first")
    preds = {m: predictor(m) for m in MODELS}
    ref = preds["gru_s0"].norm
    ro = readout()
    block: dict[str, object] = {
        "meta": meta(),
        "horizon": H,
        "start_every": START_EVERY,
        "reference_normalizer": "gru_s0 (every model trains on the same pool)",
    }
    det: dict[str, object] = {}
    prob: dict[str, object] = {}
    for pool_name, rel in EVAL_POOLS.items():
        if not (WORK / rel).exists():
            print(f"eval: {rel} missing, pool {pool_name} skipped")
            continue
        pool, seqs = _pool_arrays(rel)
        det[pool_name] = {name: eval_deterministic(pred, seqs, pool.actions, ref, ro) for name, pred in preds.items()}
        prob[pool_name] = eval_probabilistic(preds, seqs, pool.actions, ref)
        print(f"eval {pool_name}: done")
    block["rollout"], block["calibration"] = det, prob
    block["tail"] = tail_evals(preds, ro)
    block["bank_sweep"] = eval_bank_sweep(preds, ro)
    pool, seqs = _pool_arrays(EVAL_POOLS["closed_loop"])
    block["counterfactual"] = eval_counterfactual(preds, pool, seqs, ro)
    block["wall_s"] = round(time.perf_counter() - t0, 1)
    return block


# ---- plot ----

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"  # experiments/throughput palette (validated)
GRU_C, MLP_C, ORACLE_C, FNPAG_C, BASE_C = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#b4b3ad"
KIND_STYLE: dict[str, dict[str, Any]] = {
    "gru": {"color": GRU_C, "ls": "-", "label": "GRU"},
    "mlp": {"color": MLP_C, "ls": (0, (5, 2)), "label": "MLP (one-step)"},
}
POOL_LABELS = {
    "test_id": "in-dist.\n(medium)",
    "test_ood": "high\ndispersions",
    "closed_loop": "closed loop\n(oracle)",
    "self_planned": "self-planned\n(GRU s0)",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "svg.hashsalt": "aerocapture-world-model",
            "svg.fonttype": "path",
            "font.size": 9,
            "axes.edgecolor": INK2,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 10,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "legend.frameon": False,
        }
    )


def _save(fig: Any, name: str) -> None:
    fig.savefig(HERE / name, metadata={"Date": None}, dpi=200)
    plt.close(fig)


def _per_seed(block: dict[str, Any], kind: str) -> list[Any]:
    return [block[f"{kind}_s{s}"] for s in MODEL_SEEDS if f"{kind}_s{s}" in block]


def _band(ax: Any, x: Any, runs: list[list[float]], color: str, ls: Any, label: str, lw: float = 2.0) -> None:
    """Median of the training seeds, band = their min..max."""
    y = np.array(runs, dtype=float)
    ax.fill_between(x, y.min(0), y.max(0), color=color, alpha=0.18, lw=0)
    ax.plot(x, np.median(y, 0), color=color, ls=ls, lw=lw, label=label)


def _seed_lines(ax: Any, x: Any, runs: list[list[float]], color: str, ls: Any, label: str) -> None:
    """One thin line per training seed, where seeds disagree qualitatively and a band's middle would be no model's value."""
    for k, y in enumerate(runs):
        ax.plot(x, np.array(y, dtype=float), color=color, ls=ls, lw=1.4, marker="o", ms=4, mec=SURFACE, label=label if k == 0 else None)


def plot_rollout(block: dict[str, Any]) -> None:
    det = block["rollout"]["test_id"]
    h = np.arange(1, H + 1)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), layout="constrained")
    for ax, (key, ylabel, title) in zip(
        axes[:2],
        [("free", "standardized RMSE, 40 channels (log)", "State error vs horizon"), ("abs_altitude_km", "altitude error, km (log)", "Altitude")],
        strict=True,
    ):
        for kind, st in KIND_STYLE.items():
            _band(ax, h, [r[key]["median"] for r in _per_seed(det, kind)], st["color"], st["ls"], f"{st['label']} free-running")
            if key == "free":
                ax.plot(
                    h,
                    np.nanmean(np.array([r["tf"]["median"] for r in _per_seed(det, kind)], dtype=float), axis=0),
                    color=st["color"],
                    ls=":",
                    lw=1.2,
                    label=f"{st['label']} teacher-forced",
                )
        if key == "free":
            ax.plot(h, np.array(det["gru_s0"]["persist"]["median"], dtype=float), color=BASE_C, lw=2, label="persistence (x_t0)")
        ax.set(xlabel="horizon, ticks (1 s)", ylabel=ylabel, title=title, yscale="log", xlim=(1, H), ylim=(5e-3, 1e3) if key == "free" else (1e-3, 1e4))
    ax = axes[2]
    ax.plot(h, np.array(det["gru_s0"]["kin_true"]["median"], dtype=float), color=BASE_C, lw=2, label="flown trajectories")
    for kind, st in KIND_STYLE.items():
        _band(ax, h, [r["kin_free"]["median"] for r in _per_seed(det, kind)], st["color"], st["ls"], f"{st['label']} free-running")
    ax.set(
        xlabel="horizon, ticks (1 s)",
        ylabel="|altitude step - V sin(fpa) dt|, km (log)",
        title="Kinematic consistency",
        yscale="log",
        xlim=(1, H),
        ylim=(1e-3, 1e3),
    )
    ax.legend(fontsize=7.5, loc="upper left")
    for ax in axes:
        for hz in (10, 50, 200):
            ax.axvline(hz, color=INK2, lw=0.6, ls=":")
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.suptitle(
        f"Median over {det['gru_s0']['n_pairs']} (flight, start) pairs of the in-distribution test pool; line = median of 3 training seeds,"
        " band = min..max; diverged MLP runs leave the axes",
        color=INK2,
        fontsize=8,
        x=0.99,
        ha="right",
    )
    _save(fig, "fig_rollout_error.svg")


def plot_calibration(block: dict[str, Any]) -> None:
    cal = block["calibration"]["test_id"]
    first = cal["gru_s0"]
    levels, horizons = np.array(first["levels"]), first["horizons"]
    shown = [1, 10, 50, 200]
    i90 = first["levels"].index(0.9)
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.4), layout="constrained")
    for ax, (kind, st) in zip(axes[:2], KIND_STYLE.items(), strict=True):
        ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls=(0, (4, 3)))
        for k, hz in enumerate(shown):
            j = horizons.index(hz)
            cov = np.mean([r["coverage"][j] for r in _per_seed(cal, kind)], axis=0)
            ax.plot(levels, cov, color=st["color"], alpha=0.35 + 0.65 * k / (len(shown) - 1), lw=2, marker="o", ms=4, mec=SURFACE)
            ax.annotate(f"h={hz}", (levels[-1], cov[-1]), xytext=(4, 0), textcoords="offset points", fontsize=7.5, color=INK, va="center")
        ax.set(xlabel="nominal central-interval level", ylabel="empirical coverage", title=f"{st['label']}: reliability", xlim=(0, 1.08), ylim=(0, 1))
    for kind, st in KIND_STYLE.items():
        _band(axes[2], horizons, [r["crps"] for r in _per_seed(cal, kind)], st["color"], st["ls"], f"{st['label']}, one model")
        axes[2].plot(
            horizons, cal[f"{kind}_ensemble"]["crps"], color=st["color"], ls=":", lw=1.6, marker="o", ms=4, mec=SURFACE, label=f"{st['label']}, 3-seed ensemble"
        )
        _band(axes[3], horizons, [[c[i90] for c in r["coverage"]] for r in _per_seed(cal, kind)], st["color"], st["ls"], f"{st['label']}, one model")
        axes[3].plot(
            horizons,
            [c[i90] for c in cal[f"{kind}_ensemble"]["coverage"]],
            color=st["color"],
            ls=":",
            lw=1.6,
            marker="o",
            ms=4,
            mec=SURFACE,
            label=f"{st['label']}, 3-seed ensemble",
        )
    axes[2].set(xlabel="horizon, ticks (log)", ylabel="CRPS, standardized units (log)", title="Ensemble CRPS vs horizon", xscale="log", yscale="log")
    axes[3].axhline(0.9, color=INK2, lw=1, ls=(0, (4, 3)))
    axes[3].set(xlabel="horizon, ticks (log)", ylabel="coverage of the 90% interval", title="One model vs 3-seed ensemble", xscale="log", ylim=(0, 1))
    axes[2].legend(fontsize=7)
    axes[3].legend(fontsize=7, loc="lower left")
    fig.suptitle(
        f"{N_SAMPLES} sampled free runs per start and model, {first['n_pairs']} starts; bands = 3 training seeds", color=INK2, fontsize=8, x=0.99, ha="right"
    )
    _save(fig, "fig_calibration.svg")


def plot_ood(block: dict[str, Any]) -> None:
    pools = [p for p in POOL_LABELS if p in block["rollout"]]
    hz = 50
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), layout="constrained")
    j = H_PROB.index(hz)
    metrics = [
        ("NRMSE", lambda p, r: block["rollout"][p][r]["free"]["median"][hz - 1], "standardized RMSE (log)", True),
        ("CRPS", lambda p, r: block["calibration"][p][r]["crps"][j], "CRPS, standardized (log)", True),
        ("90% coverage", lambda p, r: block["calibration"][p][r]["coverage"][j][LEVELS.index(0.9)], "empirical coverage of the 90% interval", False),
    ]
    x = np.arange(len(pools))
    for ax, (name, get, ylabel, log) in zip(axes, metrics, strict=True):
        for off, (kind, st) in zip((-0.19, 0.19), KIND_STYLE.items(), strict=True):
            vals = np.array([[get(p, f"{kind}_s{s}") for s in MODEL_SEEDS] for p in pools])
            mean = vals.mean(1)
            ax.bar(x + off, mean, width=0.36, color=st["color"], edgecolor=SURFACE, linewidth=2, label=st["label"])
            ax.errorbar(x + off, mean, yerr=[mean - vals.min(1), vals.max(1) - mean], fmt="none", ecolor=INK, elinewidth=1, capsize=2)
        if name == "90% coverage":
            ax.axhline(0.9, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.set_xticks(x, [POOL_LABELS[p] for p in pools], fontsize=7.5)
        ax.grid(axis="x", visible=False)
        ax.set(ylabel=ylabel, title=f"{name} at h = {hz}", yscale="log" if log else "linear")
    axes[2].legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, 0.95))
    fig.suptitle("Bars = mean over 3 training seeds, whiskers = min..max", color=INK2, fontsize=8, x=0.99, ha="right")
    _save(fig, "fig_ood.svg")


def plot_tail(block: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), layout="constrained")
    titles = {"dv_tail": "Delta-v tail, clairvoyant planner's flights", "crash": "Crash, in-distribution test pool"}
    for ax, (key, title) in zip(axes, titles.items(), strict=True):
        tail = block["tail"][key]
        starts = tail["starts"]
        ax.axhline(0.5, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.annotate("chance", (starts[0], 0.5), xytext=(0, 3), textcoords="offset points", fontsize=7.5, color=INK2)
        for kind, st in KIND_STYLE.items():
            _seed_lines(ax, starts, [tail["auc"][f"{kind}_s{s}"] for s in MODEL_SEEDS], st["color"], st["ls"], f"{st['label']}, one line per seed")
        ax.plot(starts, np.array(tail["auc"]["persistence"], dtype=float), color=BASE_C, lw=2, marker="o", ms=5, mec=SURFACE, label="persistence")
        pos = ", ".join(f"{p}/{n}" for p, n in zip(tail["n_positive"], tail["n_flights"], strict=True))
        ax.set(xlabel="prediction start, tick", ylabel="ROC AUC", title=f"{title}\n(positive / flying per start: {pos})", ylim=(0.0, 1.02))
        ax.legend(fontsize=7.5, loc="lower right")
    fig.suptitle(
        "Free run on the flown banks until a predicted exit or crash; persistence = current navigation delta-v (tail), lowest current periapsis (crash)",
        color=INK2,
        fontsize=8,
        x=0.99,
        ha="right",
    )
    _save(fig, "fig_tail.svg")


def plot_counterfactual(block: dict[str, Any]) -> None:
    cf = block["counterfactual"]
    fracs = cf["fractions_of_bounce_tick"]
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.5), layout="constrained")
    for ax, (kind, st) in zip(axes[:2], KIND_STYLE.items(), strict=True):
        pts = cf["scatter"][f"{kind}_s0"]
        lim = max(max([abs(v) for v in pts[str(f)]["true_kj_kg"]] + [1.0]) for f in fracs)
        ax.plot([-lim, lim], [-lim, lim], color=INK2, lw=1, ls=(0, (4, 3)))
        for k, f in enumerate(fracs):
            p = pts[str(f)]
            ax.scatter(
                p["true_kj_kg"],
                p["pred_kj_kg"],
                s=9,
                color=st["color"],
                alpha=0.25 + 0.75 * k / (len(fracs) - 1),
                lw=0,
                label=f"{f:g} x bounce tick",
                rasterized=True,  # 16k points: an embedded bitmap, axes and text stay vector
            )
        ax.set(
            xscale="symlog",
            yscale="symlog",
            xlabel=f"true energy change at +{cf['short_readout_ticks']} ticks, kJ/kg",
            ylabel="predicted change, kJ/kg",
            title=f"{st['label']} s0",
        )
        ax.legend(fontsize=7, loc="upper left", markerscale=1.5)
    for ax, key, what in ((axes[2], "energy", "energy"), (axes[3], "apoapsis30", "apoapsis")):
        for kind, st in KIND_STYLE.items():
            _seed_lines(
                ax,
                fracs,
                [cf["metrics"][f"{kind}_s{s}"][f"{key}_sign_agreement"] for s in MODEL_SEEDS],
                st["color"],
                st["ls"],
                f"{st['label']}, one line per seed",
            )
        ax.axhline(0.5, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.set(
            xlabel="pulse start, fraction of the bounce tick",
            ylabel=f"sign agreement, {what} at +{cf['short_readout_ticks']} ticks",
            title=f"Direction: {what}",
            ylim=(-0.02, 1.02),
            xlim=(0.2, 1.05),
        )
        ax.legend(fontsize=7.5, loc="lower right")
    fig.suptitle(
        f"Bank magnitude pulsed by +-{cf['delta_deg']:g} deg for {cf['window_ticks']} ticks on the closed-loop pool, rest of the flown banks unchanged;"
        " apoapsis scored where the orbit is bound on both branches",
        color=INK2,
        fontsize=8,
        x=0.99,
        ha="right",
    )
    _save(fig, "fig_counterfactual.svg")


def plot_planning(block: dict[str, Any], sweep: dict[str, Any] | None) -> None:
    runs = block["runs"]
    order: list[tuple[str, str, Any, str | None]] = [
        ("oracle", ORACLE_C, "-", "clairvoyant plant"),
        ("fnpag_deployed", FNPAG_C, (0, (1, 1.5)), "FNPAG deployed (context)"),
    ]
    order += [(f"{k}_s{s}", st["color"], st["ls"], f"{st['label']} model" if s == 0 else None) for k, st in KIND_STYLE.items() for s in MODEL_SEEDS]
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.6), layout="constrained")
    for arm, color, ls, label in order:
        if arm not in runs:
            continue
        for ax, key in zip(axes[:2], ("dv_sorted", "apoapsis_err_abs_sorted"), strict=True):
            v = np.array(runs[arm][key])
            ax.plot(
                v,
                1.0 - np.arange(len(v)) / runs[arm]["n"],
                color=color,
                ls=ls,
                lw=1.6,
                label=None if label is None else f"{label} ({runs[arm]['capture_pct']:g}% capture)",
            )
    axes[0].set(
        xlabel="correction delta-v, m/s (captured flights)",
        ylabel="fraction of flights exceeding (log)",
        yscale="log",
        title="Delta-v survival",
        xlim=(90, None),
    )
    axes[1].set(xlabel="|exit apoapsis error|, km (captured, log)", ylabel="fraction exceeding (log)", xscale="log", yscale="log", title="Apoapsis targeting")
    axes[0].legend(fontsize=7, loc="lower left")
    if sweep is not None:
        ax = axes[2]
        ax.plot(sweep["banks_deg"], sweep["plant"]["apoapsis_km_p50"], color=ORACLE_C, lw=2, marker="o", ms=5, mec=SURFACE, label="plant")
        for kind, st in KIND_STYLE.items():
            runs_ = [sweep[f"{kind}_s{s}"]["apoapsis_km_p50"] for s in MODEL_SEEDS]
            _seed_lines(ax, sweep["banks_deg"], runs_, st["color"], st["ls"], f"{st['label']}, one line per seed")
        ax.axhline(block["target_apoapsis_km"], color=INK2, lw=1, ls=(0, (4, 3)))
        ax.annotate(
            "target", (sweep["banks_deg"][-1], block["target_apoapsis_km"]), xytext=(0, 3), textcoords="offset points", fontsize=7.5, color=INK2, ha="right"
        )
        ax.set_yscale("symlog", linthresh=10.0)
        ax.set(xlabel="constant bank from the first tick, deg", ylabel="median exit apoapsis, km (unbound = 1e6, crash = 0)", ylim=(0, 3e6))
        ax.set_title("The planner's first query")
        ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(0, 0.3))
    arms = [a for a, *_ in order if a in runs and a != "fnpag_deployed"]
    ms = [runs[a]["ms_per_flight_replan"] for a in arms]
    colors = [c for a, c, *_ in order if a in arms]
    axes[3].barh(range(len(arms)), ms, color=colors, edgecolor=SURFACE, linewidth=2, height=0.7)
    for y, v in enumerate(ms):
        axes[3].text(v, y, f" {v:.1f}", va="center", fontsize=7.5, color=INK)
    axes[3].set_yticks(range(len(arms)), arms, fontsize=7.5)
    axes[3].invert_yaxis()
    axes[3].grid(axis="y", visible=False)
    axes[3].set(xscale="log", xlabel="wall ms per flight-replan (batched)", title="Planning cost", xlim=(0.3, 40))
    fig.suptitle(f"{block['pool']}, per_draw noise, replan every {block['replan_every_ticks']} ticks", color=INK2, fontsize=8, x=0.99, ha="right")
    _save(fig, "fig_planning.svg")


def stage_plot(args: argparse.Namespace) -> dict[str, object]:
    results = json.loads(OUT.read_text())
    _style()
    if "eval" in results:
        plot_rollout(results["eval"])
        plot_calibration(results["eval"])
        plot_ood(results["eval"])
        plot_tail(results["eval"])
        plot_counterfactual(results["eval"])
    if "plan" in results:
        plot_planning(results["plan"], results.get("eval", {}).get("bank_sweep"))
    return {"figures": sorted(p.name for p in HERE.glob("fig_*.svg"))}


STAGE_FNS = {"data": stage_data, "train": stage_train, "plan": stage_plan, "eval": stage_eval, "plot": stage_plot}


def _json_safe(obj: Any) -> Any:
    """nan / inf -> null, so strict JSON parsers (jq) read the results."""
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", choices=list(STAGE_FNS), default=list(STAGE_FNS))
    ap.add_argument("--force", action="store_true", help="recompute cached datasets / models / runs")
    ap.add_argument("--arms", nargs="+", default=["oracle", *MODELS], help="plan stage: which predictors fly the planning pool")
    args = ap.parse_args(argv)
    for stage in args.stages:
        with np.errstate(over="ignore", invalid="ignore"):  # diverged free runs reach inf / nan: nanmedian skips nan, ranks inf last
            block = STAGE_FNS[stage](args)
        results = json.loads(OUT.read_text()) if OUT.exists() else {}  # re-read: stages may run in parallel processes
        results[stage] = block
        OUT.write_text(json.dumps(_json_safe(results), indent=1, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
