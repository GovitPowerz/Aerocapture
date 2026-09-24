"""Throughput and scaling study of the batched simulator (#114).

Five stages, all run by default. Each rewrites its own block of throughput.json
(with the machine and commit it ran on; blocks of stages not run are kept), then
both figures are redrawn from the JSON:

  scaling  sims/s vs Rayon thread count per scheme: one run_grid call flying one
           deployed cell on n seeds of the final-eval pool, repeats interleaved
           across thread counts so thermal drift spreads over all of them
  seams    per-sim cost at 1 thread of the three entry points: run_grid, run_batch
           (per-sim TOML patch + parse + SimData build) and the Rust CLI (process,
           config + table load, CSV output; it flies its own Monte Carlo draws)
  grid     run_grid wall at the paper allocation (the converged headline
           population, 512 in-memory Mamba weight vectors) vs n_seeds: the
           intercept is the per-call SimData build, the slope the cells
  memory   peak RSS added by a run_grid call (a fresh process per point)
  profile  per-generation wall time of the real GA loop resumed from the headline
           checkpoint, split by loop phase and by Rust vs Python, plus the
           cProfile top-5 Python hotspots

Cells are the paper's deployed operating points from the committed bundle
(articles/paper/data/runs/), so scaling, seams and memory reproduce on a fresh
clone; grid and profile resume the headline run's local checkpoint
(training_output/mamba_p962_long). Noise regime: per_draw (the default,
ADR-0006); timing does not depend on it. Build the release CLI and the
extension first (./build.sh); close other workloads.

Usage: uv run python experiments/throughput/throughput.py [--stages scaling ...] [--plot-only]
"""

from __future__ import annotations

import argparse
import cProfile
import json
import os
import platform
import pstats
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aerocapture_rs
import matplotlib as mpl
import numpy as np

mpl.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from aerocapture.training import problem as problem_mod  # noqa: E402
from aerocapture.training import train as train_mod  # noqa: E402
from aerocapture.training import trainer as trainer_mod  # noqa: E402
from aerocapture.training.cell_eval import reserved_pool  # noqa: E402
from aerocapture.training.deploy_overrides import load_scaffolding_overrides, overrides_from_params  # noqa: E402
from aerocapture.training.seeds import FINAL_EVAL_SEED_OFFSET  # noqa: E402
from aerocapture.training.toml_utils import set_dot_path, write_toml  # noqa: E402
from aerocapture.training.training_config import build_training_config_from_toml  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "throughput.json"
BUNDLE = REPO / "articles/paper/data/runs"
CLI = REPO / "src/rust/target/release/aerocapture"
HEADLINE_TOML = "configs/training/sweep/mamba_p962.toml"
HEADLINE_RUN = REPO / "training_output/mamba_p962_long"
PAPER_N_POP, PAPER_N_SIMS = 512, 2  # experiments/paper/10b_arch_long_challengers.sh
STAGES = ("scaling", "seams", "grid", "memory", "profile")
# What a timing depends on (docs excluded): a dirty file here makes the stamped commit a lie.
_SOURCES = ("src", "configs", "data", "articles/paper/data/runs", "experiments/throughput/throughput.py")


@dataclass(frozen=True)
class Cell:
    label: str
    toml: str
    bundle: str  # under articles/paper/data/runs/
    scheme: str | None  # classical guidance type (best_params.json routes as its gains); None = NN

    def overrides(self) -> dict[str, object]:
        cell = BUNDLE / self.bundle
        if self.scheme is not None:
            return overrides_from_params(json.loads((cell / "best_params.json").read_text()), self.scheme)
        return {**load_scaffolding_overrides(cell), "data.neural_network": str(cell / "best_model.json")}


CELLS = [
    Cell("FTC", "configs/training/msr_aller_ftc_train.toml", "classical_baselines/ftc", "ftc"),
    Cell("FNPAG", "configs/training/msr_aller_fnpag_train.toml", "classical_baselines/fnpag", "fnpag"),
    Cell("NN dense-515", "configs/training/msr_aller_nn_atan2_best_paper.toml", "headline/dense_p515", None),
    Cell("NN Mamba-962", HEADLINE_TOML, "headline/mamba_p962", None),
]


def _sh(*cmd: str) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()


def _performance_cores() -> int:
    return int(_sh("sysctl", "-n", "hw.perflevel0.physicalcpu") or 0)


def meta() -> dict[str, object]:
    """Machine + commit stamp carried by every stage block (no cross-machine comparisons)."""
    return {
        "cpu": _sh("sysctl", "-n", "machdep.cpu.brand_string") or platform.processor(),
        "cores_performance": _performance_cores(),
        "cores_efficiency": int(_sh("sysctl", "-n", "hw.perflevel1.physicalcpu") or 0),
        "logical_cpus": os.cpu_count(),
        "memory_gib": round(int(_sh("sysctl", "-n", "hw.memsize") or 0) / 2**30),
        "os": f"macOS {platform.mac_ver()[0]}" if sys.platform == "darwin" else platform.platform(),
        "rustc": _sh("rustc", "--version"),
        "python": platform.python_version(),
        "build": "release (lto), f64",
        "commit": _sh("git", "rev-parse", "HEAD"),
        "sources_clean": _sh("git", "status", "--porcelain", "--", *_SOURCES, ":(exclude)*.md") == "",
        "date": time.strftime("%Y-%m-%d"),
    }


def _grid_wall(toml: str, overrides: list[dict[str, object]], seeds: list[int], n_threads: int | None) -> float:
    t0 = time.perf_counter()
    aerocapture_rs.run_grid(toml, overrides, seeds, n_threads=n_threads)
    return time.perf_counter() - t0


def _batch_wall(toml: str, overrides: list[dict[str, object]]) -> float:
    t0 = time.perf_counter()
    aerocapture_rs.run_batch(toml, overrides, n_threads=1)
    return time.perf_counter() - t0


def _median(xs: list[float]) -> float:
    return float(statistics.median(xs))


def _sim_time_col() -> int:
    return int(aerocapture_rs.final_record_indices()["sim_time_s"])


# ── scaling ──────────────────────────────────────────────────────────────────


def thread_counts() -> list[int]:
    n = os.cpu_count() or 1
    return sorted({t for t in (1, 2, 4, 8, _performance_cores(), n) if 1 <= t <= n})


def stage_scaling(args: argparse.Namespace) -> dict[str, object]:
    threads = thread_counts()
    rows = []
    for cell in CELLS:
        ovr = cell.overrides()
        seeds = reserved_pool(Path(cell.toml), FINAL_EVAL_SEED_OFFSET, args.n)
        records = np.asarray(aerocapture_rs.run_grid(cell.toml, [ovr], seeds, n_threads=None))  # warm-up
        walls: dict[int, list[float]] = {t: [] for t in threads}
        for _ in range(args.repeats):
            for t in threads:
                walls[t].append(_grid_wall(cell.toml, [ovr], seeds, t))
        base = args.n / _median(walls[1])
        points = []
        for t in threads:
            rate = args.n / _median(walls[t])
            points.append(
                {
                    "threads": t,
                    "wall_s": [round(w, 4) for w in walls[t]],
                    "sims_per_s": round(rate, 1),
                    "speedup": round(rate / base, 2),
                    "efficiency": round(rate / base / t, 3),
                }
            )
        rows.append(
            {
                "label": cell.label,
                "toml": cell.toml,
                "bundle": cell.bundle,
                "mean_flight_s": round(float(records[0, :, _sim_time_col()].mean()), 1),
                "ms_per_sim_1_thread": round(1000 / base, 4),
                "points": points,
            }
        )
        print(f"  scaling {cell.label:13s} " + "  ".join(f"{p['threads']}t {p['sims_per_s']:>8}" for p in points) + "  sims/s")
    return {"meta": meta(), "seam": "run_grid, n_pop=1", "pool": "final-eval", "n": args.n, "repeats": args.repeats, "thread_counts": threads, "cells": rows}


# ── seams ────────────────────────────────────────────────────────────────────


def _cli_toml(cell: Cell, ovr: dict[str, object], n: int, tmp: Path) -> Path:
    data: dict[str, Any] = {"base": [str(REPO / cell.toml)]}
    for key, value in {**ovr, "simulation.n_sims": n, "data.output_dir": str(tmp), "data.results_suffix": f".throughput_{n}"}.items():
        set_dot_path(data, key, value)
    path = tmp / f"cli_{n}.toml"
    write_toml(data, path)
    return path


def _cli(toml: Path) -> tuple[float, float]:
    """(process wall, the CLI's own simulation-phase wall) at 1 Rayon thread."""
    t0 = time.perf_counter()
    run = subprocess.run([str(CLI), str(toml)], env={**os.environ, "RAYON_NUM_THREADS": "1"}, capture_output=True, text=True, check=True)
    wall = time.perf_counter() - t0
    m = re.search(r"Completed \d+ simulations in ([\d.]+)s", run.stderr)
    return wall, float(m.group(1)) if m else wall


def stage_seams(args: argparse.Namespace) -> dict[str, object]:
    if not CLI.exists():
        raise SystemExit(f"{CLI} missing: build it first (./build.sh)")
    n: int = args.seams_n
    rows = []
    for cell in CELLS:
        ovr = cell.overrides()
        seeds = reserved_pool(Path(cell.toml), FINAL_EVAL_SEED_OFFSET, n)
        batch = [{**ovr, "simulation.n_sims": 1, "monte_carlo.seed": s} for s in seeds]
        grid_w, batch_w, cli_w, cli_sim, cli_one = [], [], [], [], []
        with tempfile.TemporaryDirectory(prefix="throughput_cli_") as tmp:
            cli_n, cli_1 = _cli_toml(cell, ovr, n, Path(tmp)), _cli_toml(cell, ovr, 1, Path(tmp))
            _cli(cli_1)  # warm-up: page in the binary and the tables
            aerocapture_rs.run_grid(cell.toml, [ovr], seeds[:10], n_threads=1)
            for _ in range(args.repeats):
                grid_w.append(_grid_wall(cell.toml, [ovr], seeds, 1))
                batch_w.append(_batch_wall(cell.toml, batch))
                wall, sim = _cli(cli_n)
                cli_w.append(wall)
                cli_sim.append(sim)
                cli_one.append(_cli(cli_1)[0])

        def per_sim(xs: list[float]) -> float:
            return round(1000 * _median(xs) / n, 4)

        row: dict[str, object] = {
            "label": cell.label,
            "run_grid_ms_per_sim": per_sim(grid_w),
            "run_batch_ms_per_sim": per_sim(batch_w),
            "cli_ms_per_sim": per_sim(cli_w),
            "cli_sim_phase_ms_per_sim": per_sim(cli_sim),
            "cli_one_sim_process_s": round(_median(cli_one), 4),
        }
        print(f"  seams   {cell.label:13s} " + "  ".join(f"{k}={v}" for k, v in row.items() if k != "label"))
        walls = {"run_grid": grid_w, "run_batch": batch_w, "cli": cli_w, "cli_sim_phase": cli_sim, "cli_one_sim": cli_one}
        rows.append({**row, "wall_s": {k: [round(w, 4) for w in v] for k, v in walls.items()}})
    return {
        "meta": meta(),
        "threads": 1,
        "n": n,
        "repeats": args.repeats,
        "note": "run_grid and run_batch fly the same final-eval seeds; the CLI flies n draws of the config's own Monte Carlo "
        "(same dispersion model, different scenarios)",
        "cells": rows,
    }


# ── the headline training loop (grid + profile) ──────────────────────────────


@contextmanager
def _patched(obj: object, name: str, value: object) -> Iterator[None]:
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def _train_headline(n_gen: int, n_sims: int, run_loop: Callable[..., dict[str, Any]]) -> None:
    """`train()` resumed from a scratch copy of the headline run's last checkpoint at the
    paper population, with `run_loop` swapped in. Every write lands in the scratch dir:
    save_checkpoint also deploys the NN to `[data] neural_network`, repointed there
    (the TOML's own path is the real sweep cell's model)."""
    ckpts = sorted(HEADLINE_RUN.glob("checkpoint_g*.npz"))
    if not ckpts:
        raise SystemExit(f"{HEADLINE_RUN} has no checkpoint: the grid and profile stages resume the headline run")
    with tempfile.TemporaryDirectory(prefix="throughput_train_") as tmp:
        for f in (ckpts[-1], ckpts[-1].with_suffix(".json")):
            shutil.copy2(f, tmp)
        cfg, _ = build_training_config_from_toml(HEADLINE_TOML)
        cfg.optimizer.n_gen = n_gen
        cfg.optimizer.n_pop = PAPER_N_POP
        cfg.optimizer.training_n_sims = n_sims
        cfg.sim.sim_timeout_secs = 5.0
        cfg.sim.final_file = "output/final.train_nn_temp"
        cfg.sim.exec_dir = "."
        cfg.sim.nn_param_file = str(Path(tmp) / "deployed_best_model.json")
        cfg.save_dir = tmp
        with _patched(train_mod, "run_loop", run_loop):
            train_mod.train(cfg, seed=1, cwd=".", resume_dir=tmp, no_tui=True, verbose=False)


class PhaseClock:
    """Wall time per loop phase, and inside each phase the time spent in the Rust
    seam, in `_run_grid_records` (seam + Python decode/override build) and in
    `compute_cost`."""

    def __init__(self, profiler: cProfile.Profile | None = None) -> None:
        self.phase = "setup"
        self.wall: dict[str, float] = defaultdict(float)
        self.rust: dict[str, float] = defaultdict(float)
        self.grid_records: dict[str, float] = defaultdict(float)
        self.cost: dict[str, float] = defaultdict(float)
        self.generations = 0
        self.logged_gen_s: list[float] = []
        self.loop_s = 0.0
        self._loop_t0 = 0.0
        self.profiler = profiler

    def timed(self, phase: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        def run(*args: Any, **kwargs: Any) -> Any:
            prev, self.phase = self.phase, phase
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                self.wall[phase] += time.perf_counter() - t0
                self.phase = prev

        return run

    def accrue(self, bucket: dict[str, float], fn: Callable[..., Any]) -> Callable[..., Any]:
        def run(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                bucket[self.phase] += time.perf_counter() - t0

        return run

    def start_loop(self) -> None:
        self._loop_t0 = time.perf_counter()
        if self.profiler is not None:
            self.profiler.enable()

    def stop_loop(self) -> None:
        if self.profiler is not None:
            self.profiler.disable()
        self.loop_s = time.perf_counter() - self._loop_t0


class _RustSeam:
    """`aerocapture_rs` with `run_grid` timed into the clock's current phase."""

    def __init__(self, clock: PhaseClock) -> None:
        self.run_grid = clock.accrue(clock.rust, aerocapture_rs.run_grid)

    def __getattr__(self, name: str) -> Any:
        return getattr(aerocapture_rs, name)


class _TimedTrainer:
    """Forwards to the real trainer, timing each loop-contract method as a phase.
    `finalize` (the once-per-run final selection) is skipped: not a generation cost."""

    PHASES = {"re_evaluate": "reevaluation", "advance": "population", "observe": "validation", "emit": "log_display", "maybe_checkpoint": "checkpoint"}

    def __init__(self, inner: Any, clock: PhaseClock) -> None:
        self._inner = inner
        self._clock = clock

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if name == "prologue":
            return self._prologue
        if name == "finalize":
            return self._finalize
        if name == "advance":
            self._clock.generations += 1
        phase = self.PHASES.get(name)
        return self._clock.timed(phase, attr) if phase is not None else attr

    def _prologue(self, *args: Any) -> None:
        self._inner.prologue(*args)
        self._clock.start_loop()

    def _finalize(self, logger: Any) -> dict[str, Any]:
        self._clock.stop_loop()
        logger.close()
        # The loop's own per-generation stamp (advance -> emit), comparable to a real run's JSONL.
        for log in Path(self._inner.save_dir).glob("run_*.jsonl"):
            records = [json.loads(line) for line in log.read_text().splitlines()]
            self._clock.logged_gen_s += [r["gen_elapsed_s"] for r in records if r.get("gen_elapsed_s") is not None]
        return {}


def _instrumented_generations(n_gen: int, n_sims: int, profiler: cProfile.Profile | None) -> PhaseClock:
    clock = PhaseClock(profiler)

    def loop(trainer: Any, **kwargs: Any) -> dict[str, Any]:
        return trainer_mod.run_loop(_TimedTrainer(trainer, clock), **kwargs)

    problem_cls = problem_mod.AerocaptureProblem
    with ExitStack() as stack:
        stack.enter_context(_patched(problem_mod, "_aero_rs", _RustSeam(clock)))
        stack.enter_context(_patched(problem_mod, "compute_cost", clock.accrue(clock.cost, problem_mod.compute_cost)))
        stack.enter_context(_patched(problem_cls, "_run_grid_records", clock.accrue(clock.grid_records, problem_cls._run_grid_records)))
        stack.enter_context(_patched(trainer_mod, "_maybe_curate", clock.timed("curation", trainer_mod._maybe_curate)))
        stack.enter_context(_patched(trainer_mod, "_apply_seed_strategy", clock.timed("seed_draw", trainer_mod._apply_seed_strategy)))
        _train_headline(n_gen, n_sims, loop)
    return clock


# ── grid ─────────────────────────────────────────────────────────────────────


def stage_grid(args: argparse.Namespace) -> dict[str, object]:
    captured: list[Any] = []

    def capture(trainer: Any, **_: Any) -> dict[str, Any]:
        captured.append(trainer)
        return {}

    _train_headline(0, PAPER_N_SIMS, capture)
    trainer = captured[0]
    ks = [1, 2, 4, 8]
    seeds = reserved_pool(Path(HEADLINE_TOML), FINAL_EVAL_SEED_OFFSET, max(ks))
    clock = PhaseClock()
    walls: dict[int, list[float]] = {k: [] for k in ks}
    with _patched(problem_mod, "_aero_rs", _RustSeam(clock)):
        trainer.problem.evaluate_population_per_seed(trainer.X, seeds[:1])  # warm-up
        for _ in range(args.repeats):
            for k in ks:
                before = clock.rust["setup"]
                trainer.problem.evaluate_population_per_seed(trainer.X, seeds[:k])
                walls[k].append(clock.rust["setup"] - before)
    med = np.array([_median(walls[k]) for k in ks])
    slope, intercept = np.polyfit(ks, med, 1)
    n_pop = int(trainer.X.shape[0])
    at_paper = intercept + PAPER_N_SIMS * slope
    print(f"  grid    {n_pop} x k seeds: wall = {intercept:.3f} s + {slope:.3f} s * k; SimData build share at k={PAPER_N_SIMS}: {intercept / at_paper:.0%}")
    return {
        "meta": meta(),
        "seam": "run_grid via AerocaptureProblem (in-memory weights), global Rayon pool",
        "population": f"{HEADLINE_RUN.name} last checkpoint",
        "n_pop": n_pop,
        "n_seeds": ks,
        "repeats": args.repeats,
        "run_grid_wall_s": {str(k): [round(w, 4) for w in walls[k]] for k in ks},
        "fit_intercept_s": round(float(intercept), 4),
        "fit_slope_s_per_seed": round(float(slope), 4),
        "fit_max_abs_residual_s": round(float(np.max(np.abs(med - (intercept + slope * np.array(ks))))), 4),
        "setup_ms_per_individual_wall": round(1000 * float(intercept) / n_pop, 4),
        "sim_ms_per_cell_wall": round(1000 * float(slope) / n_pop, 4),
        "setup_share_at_paper_n_sims": round(float(intercept / at_paper), 3),
    }


# ── memory ───────────────────────────────────────────────────────────────────

_MEMORY_PROBE = """
import json, resource, sys
import aerocapture_rs
toml, ovr, seeds, n_pop = json.loads(sys.argv[1])
aerocapture_rs.run_grid(toml, [ovr], seeds[:1], n_threads=None)  # one-time tables + pool below the baseline
base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
out = aerocapture_rs.run_grid(toml, [ovr] * n_pop, seeds, n_threads=None)
print(json.dumps([base, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, out.nbytes]))
"""


def stage_memory(args: argparse.Namespace) -> dict[str, object]:
    cell = CELLS[-1]
    ovr = cell.overrides()
    unit = 1 if sys.platform == "darwin" else 1024  # ru_maxrss: bytes on macOS, KiB on Linux
    points = []
    for n_pop, n_seeds in [(64, 2), (512, 2), (2048, 2), (1, 1000)]:
        seeds = reserved_pool(Path(cell.toml), FINAL_EVAL_SEED_OFFSET, n_seeds)
        run = subprocess.run([sys.executable, "-c", _MEMORY_PROBE, json.dumps([cell.toml, ovr, seeds, n_pop])], capture_output=True, text=True, check=True)
        base, peak, out_bytes = json.loads(run.stdout.strip().splitlines()[-1])
        points.append(
            {
                "n_pop": n_pop,
                "n_seeds": n_seeds,
                "baseline_rss_mib": round(base * unit / 2**20, 1),
                "peak_rss_added_mib": round((peak - base) * unit / 2**20, 1),
                "output_array_mib": round(out_bytes / 2**20, 2),
            }
        )
        print(f"  memory  {n_pop} x {n_seeds}: +{points[-1]['peak_rss_added_mib']} MiB peak RSS")
    pop = [p for p in points if p["n_seeds"] == 2]
    slope = np.polyfit([p["n_pop"] for p in pop], [p["peak_rss_added_mib"] for p in pop], 1)[0]
    return {
        "meta": meta(),
        "cell": cell.label,
        "note": "individuals load the deployed model from JSON (data.neural_network override); the training path injects the same model in-memory",
        "points": points,
        "mib_per_individual": round(float(slope), 4),
    }


# ── profile ──────────────────────────────────────────────────────────────────


def _anchor(file: str, line: int) -> str:
    path = Path(file)
    if path.is_relative_to(REPO / "src/python"):
        return f"{path.relative_to(REPO)}:{line}"
    parts = path.parts
    if "site-packages" in parts:
        return f"{'/'.join(parts[parts.index('site-packages') + 1 :])}:{line}"
    return f"{file}:{line}"


_Key = tuple[str, int, str]
_KERNEL_LIBS = ("/numpy/", "/scipy/")


def _hotspots(profiler: cProfile.Profile, loop_s: float, k: int = 5) -> dict[str, object]:
    """Top-k functions by self time, the Rust seam excluded (it is the bar's Rust share).
    A builtin (a numpy / scipy kernel) is anchored at the first caller outside those libraries."""
    stats: dict[_Key, tuple[int, int, float, float, dict[_Key, tuple[int, int, float, float]]]] = pstats.Stats(profiler).stats  # type: ignore[attr-defined]

    def where(key: _Key) -> str:
        if key[0] != "~":
            return _anchor(key[0], key[1])
        # Climb the heaviest caller edge out of the numeric kernels to the code that asked.
        site = key
        for _ in range(8):
            callers = stats[site][4]
            if not callers:
                break
            site = max(callers, key=lambda c: callers[c][3])
            if site[0] != "~" and not any(lib in site[0] for lib in _KERNEL_LIBS):
                return f"builtin, via {site[2]} at {_anchor(site[0], site[1])}"
        return "builtin"

    ranked = sorted(((key, v) for key, v in stats.items() if "run_grid" not in key[2]), key=lambda kv: -kv[1][2])
    return {
        "loop_s_under_cprofile": round(loop_s, 3),
        "python_tottime_s": round(sum(v[2] for key, v in ranked if key[0] != "~"), 3),
        "builtin_tottime_s_excl_run_grid": round(sum(v[2] for key, v in ranked if key[0] == "~"), 3),
        "top": [{"function": key[2], "where": where(key), "calls": v[1], "tottime_s": round(v[2], 4), "cumtime_s": round(v[3], 4)} for key, v in ranked[:k]],
    }


def _breakdown(clock: PhaseClock) -> dict[str, float]:
    """Per-generation seconds. Prep = _run_grid_records minus the seam (decode, override
    dicts, weight matrix); pymoo = population-phase wall outside the grid and the cost;
    re-evaluation = the parents re-scored after an adaptive-seed change."""
    g = max(clock.generations, 1)
    prep = sum(clock.grid_records.values()) - sum(clock.rust.values())
    cost = sum(clock.cost.values())
    parts = {
        "rust_population": clock.rust["population"],
        "rust_reevaluation": clock.rust["reevaluation"],
        "rust_validation": clock.rust["validation"],
        "rust_curation": clock.rust["curation"],
        "python_decode_overrides": prep,
        "python_cost": cost,
        "pymoo_operators": clock.wall["population"] - clock.grid_records["population"] - clock.cost["population"],
        "log_display": clock.wall["log_display"],
        "checkpoint": clock.wall["checkpoint"],
    }
    parts["other"] = clock.loop_s - sum(parts.values())
    return {k: round(v / g, 4) for k, v in {"total": clock.loop_s, **parts}.items()}


def stage_profile(args: argparse.Namespace) -> dict[str, object]:
    rows = []
    for n_sims in args.profile_sims:
        clock = _instrumented_generations(args.profile_gens, n_sims, None)
        per_gen = _breakdown(clock)
        rows.append(
            {
                "n_pop": PAPER_N_POP,
                "n_sims": n_sims,
                "generations": clock.generations,
                "per_generation_s": per_gen,
                "logged_gen_elapsed_s_median": round(_median(clock.logged_gen_s), 4),
            }
        )
        rust = sum(v for k, v in per_gen.items() if k.startswith("rust_"))
        print(f"  profile {PAPER_N_POP} x {n_sims}: {per_gen['total']:.3f} s/gen, Rust {rust / per_gen['total']:.0%}")
    profiler = cProfile.Profile()
    clock = _instrumented_generations(args.profile_gens, PAPER_N_SIMS, profiler)
    return {
        "meta": meta(),
        "config": HEADLINE_TOML,
        "algorithm": "ga (the headline optimizer), seed_strategy adaptive, validation + curation on",
        "resumed_from": f"{HEADLINE_RUN.name} last checkpoint",
        "display": "plain (no Rich TUI)",
        "allocations": rows,
        "hotspots": _hotspots(profiler, clock.loop_s),
    }


# ── figures ──────────────────────────────────────────────────────────────────

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # reference categorical order, validated
OTHER = "#b4b3ad"


def _style() -> None:
    plt.rcParams.update(
        {
            "svg.hashsalt": "aerocapture-throughput",
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
    fig.savefig(HERE / name, metadata={"Date": None})
    plt.close(fig)


def plot_scaling(block: dict[str, Any]) -> None:
    threads = block["thread_counts"]
    fig, (ax_s, ax_r) = plt.subplots(1, 2, figsize=(8.4, 3.6), layout="constrained")
    ax_s.plot(threads, threads, color=INK2, lw=1.2, ls=(0, (4, 3)), label="ideal")
    p_cores = block["meta"]["cores_performance"]
    for ax in (ax_s, ax_r):
        if p_cores:
            ax.axvline(p_cores, color=INK2, lw=0.8, ls=":")
    if p_cores:
        ax_s.text(p_cores, 0.5, f" {p_cores} P-cores", color=INK2, fontsize=8, va="bottom")
    for color, cell in zip(SERIES, block["cells"], strict=False):
        xs = [p["threads"] for p in cell["points"]]
        ax_s.plot(xs, [p["speedup"] for p in cell["points"]], color=color, lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5, label=cell["label"])
        ax_r.plot(xs, [p["sims_per_s"] for p in cell["points"]], color=color, lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5, label=cell["label"])
        last = cell["points"][-1]
        ax_r.annotate(f"{last['sims_per_s']:,.0f}", (xs[-1], last["sims_per_s"]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=INK)
    ax_s.set(xlabel="Rayon threads", ylabel="speedup vs 1 thread", title="Parallel speedup", xticks=threads, ylim=(0, max(threads) + 1))
    ax_r.set(xlabel="Rayon threads", ylabel="simulations / s (log)", title=f"Throughput (n = {block['n']} per point)", xticks=threads, yscale="log")
    ax_s.legend(loc="upper left", fontsize=8)
    fig.suptitle(f"run_grid scaling, {block['meta']['cpu']}, commit {block['meta']['commit'][:8]}", color=INK2, fontsize=8, x=0.99, ha="right")
    _save(fig, "fig_scaling.svg")


def plot_profile(block: dict[str, Any]) -> None:
    segments = [
        ("Rust: offspring eval", ["rust_population"]),
        ("Rust: parent re-eval (seed change)", ["rust_reevaluation"]),
        ("Rust: validation + curation", ["rust_validation", "rust_curation"]),
        ("Python at the seam: decode, overrides, cost", ["python_decode_overrides", "python_cost"]),
        ("pymoo operators", ["pymoo_operators"]),
        ("JSONL log, display, checkpoint", ["log_display", "checkpoint"]),
        ("other", ["other"]),
    ]
    colors = [*SERIES, OTHER]
    rows = block["allocations"]
    fig, ax = plt.subplots(figsize=(8.4, 1.2 + 0.55 * len(rows)), layout="constrained")
    labels = [f"n_pop {r['n_pop']} x {r['n_sims']} sims" for r in rows]
    for y, row in enumerate(rows):
        g = row["per_generation_s"]
        left = 0.0
        for (name, keys), color in zip(segments, colors, strict=True):
            width = max(sum(g[k] for k in keys), 0.0)
            ax.barh(y, width, left=left, color=color, edgecolor=SURFACE, linewidth=2, height=0.6, label=name if y == 0 else None)
            if width > 0.06 * g["total"]:
                ax.text(left + width / 2, y, f"{width:.2f}", ha="center", va="center", fontsize=8, color=INK)
            left += width
        ax.text(left, y, f"  {g['total']:.2f} s/gen", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(rows)), labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("wall seconds per generation")
    ax.set_xlim(0, max(r["per_generation_s"]["total"] for r in rows) * 1.18)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3, fontsize=8)
    _save(fig, "fig_generation_profile.svg")


def plot(results: dict[str, Any]) -> None:
    _style()
    if "scaling" in results:
        plot_scaling(results["scaling"])
    if "profile" in results:
        plot_profile(results["profile"])


STAGE_FNS: dict[str, Callable[[argparse.Namespace], dict[str, object]]] = {
    "scaling": stage_scaling,
    "seams": stage_seams,
    "grid": stage_grid,
    "memory": stage_memory,
    "profile": stage_profile,
}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    ap.add_argument("--n", type=int, default=1000, help="sims per scaling point")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seams-n", type=int, default=200, help="sims per seam measurement (1 thread)")
    ap.add_argument("--profile-gens", type=int, default=20)
    ap.add_argument("--profile-sims", type=int, nargs="+", default=[PAPER_N_SIMS, 10], help="training_n_sims per profiled allocation")
    ap.add_argument("--plot-only", action="store_true", help="redraw the figures from throughput.json")
    args = ap.parse_args(argv)
    os.chdir(REPO)  # configs and data paths are repo-relative
    results: dict[str, Any] = json.loads(OUT.read_text()) if OUT.exists() else {}
    if not args.plot_only:
        for stage in args.stages:
            print(f"== {stage}")
            results[stage] = STAGE_FNS[stage](args)
            OUT.write_text(json.dumps(results, indent=2) + "\n")
    plot(results)


if __name__ == "__main__":
    main()
