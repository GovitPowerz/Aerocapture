# QPSO Optimizer — Design

**Date**: 2026-06-10
**Status**: Approved (brainstorming complete, awaiting implementation plan)
**Scope**: New single-algorithm optimizer option `algorithm = "qpso"` (Quantum-behaved Particle Swarm Optimization) in the training pipeline, plus paper-study configs and a batch-4 runner script. Islands stay PSO/GA/DE.

## 1. Motivation

The paper's optimizer comparison (batches 2–3: GA / DE / PSO / CMA-ES / islands on `dense_p515` and `dense_p3998` at @60/@150/@300 evals per generation) needs QPSO as another column. QPSO (Sun, Feng & Xu 2004) is the standard "quantum" PSO variant cited in metaheuristic comparisons: particles have **no velocity**; positions are resampled each generation from a delta-potential-well distribution centered on a per-particle local attractor, with a characteristic length set by the swarm's mean-best position. Fewer hyperparameters than PSO (one annealed coefficient vs w/c1/c2) and heavier-tailed jumps that resist the premature swarm collapse PSO exhibits on this problem.

pymoo has no QPSO, so this is a custom pymoo `Algorithm` subclass. Mirroring pymoo PSO's hook structure makes every downstream consumer — `warm_start_algorithm`, manual `.next()` stepping, checkpoint/resume, seed strategies, validation gate, TUI, report — work untouched.

## 2. Goals and Non-Goals

**Goals:**

1. `[optimizer] algorithm = "qpso"` works everywhere `"pso"` works in the single-algorithm path: fresh start, warm-start seeding, checkpoint/resume, resize-on-resume, all three seed strategies.
2. Canonical, reviewer-defensible formulation: mbest QPSO ("Type 2") with linear contraction–expansion annealing, no non-canonical extras (no `pertube_best`, no weighted mbest).
3. Bound handling consistent with the compared optimizers: same final `repair_random_init` pymoo PSO uses.
4. Paper-ready: `configs/training/paper/opt_qpso.toml` + `optbig_qpso.toml` mirroring the PSO twins, and a `run_paper_experiments4.sh` covering the same budget grid (small-net @300; big-net @60/@150/@300).
5. Pure-Python tests that run in the no-Rust CI job.

**Non-goals:**

1. QPSO as an island type (`island_model.py` is hardcoded to PSO/GA/DE; migration velocity-injection and checkpoint fields are island-type-specific — separate feature if ever wanted).
2. QPSO variants (WQPSO weighted mbest, GAQPSO Gaussian attractor).
3. Persisting QPSO swarm positions across resume (PSO does not persist particles/velocity in single-algo checkpoints either; pop = pbest is the resume contract).
4. Rust changes (none; the optimizer layer is pure Python).

## 3. The Algorithm

Canonical mbest QPSO. Per generation, for particle `i`, dimension `d` (all vectorized in numpy):

```
mbest_d   = mean_i(pbest_{i,d})                              # swarm mean-best
phi       ~ U(0,1) per (i,d)
p_{i,d}   = phi * pbest_{i,d} + (1 - phi) * gbest_d          # local attractor
u         = 1 - U(0,1)  in (0, 1]                            # avoids log(0)
s         = ±1 with probability 0.5 each, per (i,d)
x'_{i,d}  = p_{i,d} + s * alpha * |mbest_d - x_{i,d}| * ln(1/u)
```

`alpha` (contraction–expansion coefficient) anneals linearly from `alpha_start` (default 1.0) to `alpha_end` (default 0.5):

```
progress = clamp((n_iter - 1) / max(1, max_iter - 1), 0, 1)
alpha    = alpha_start + (alpha_end - alpha_start) * progress
```

`max_iter` is passed at construction (`config.n_gen`). Theory bounds convergence at alpha ≲ 1.781 (e^gamma); settings validation caps both alphas at `(0, 2]`.

Out-of-bounds positions are repaired with pymoo's `repair_random_init(Xp, X, xl, xu)` — identical to pymoo PSO's final repair step, so bound handling does not differ across the paper's compared optimizers. The `u in (0, 1]` sampling keeps `ln(1/u)` finite, so no inf coordinates reach the repair.

## 4. Architecture

### 4.1 New module: `src/python/aerocapture/training/qpso.py` (~100 LoC)

`class QPSO(Algorithm)` mirroring `pymoo.algorithms.soo.nonconvex.pso.PSO`'s state conventions:

- `self.pop` is the **pbest** population (what checkpoints save and `_set_optimum` reads).
- `self.particles` is the **current** swarm position population.
- `__init__(pop_size=25, sampling=LHS(), alpha_start=1.0, alpha_end=0.5, max_iter=1000, **kwargs)`; `output=SingleObjectiveOutput()` (PSO's `PSOFuzzyOutput` displays w/c1/c2 fuzzy state that QPSO doesn't have).
- `_initialize_infill()`: `self.initialization.do(...)` — same LHS init as PSO.
- `_initialize_advance(infills)`: `self.particles = self.pop`; `super()._initialize_advance(...)`. No velocity to initialize — this is the hook `warm_start_algorithm` calls, and it works with seeded populations by construction.
- `_infill()`: the update above; reads `gbest` from `self.opt[0].X`; returns `Population.new(X=Xp)`.
- `_advance(infills)`: verbatim PSO's pbest update — `self.particles = infills`, `ImprovementReplacement().do(..., return_indices=True)`, `self.pop[has_improved] = infills[has_improved]`.
- All randomness through `self.random_state` (pymoo seeding convention; deterministic under `seed=`).

### 4.2 `optimizer.py`

- `QPSOSettings` dataclass: `alpha_start: float = 1.0`, `alpha_end: float = 0.5`. `__post_init__` validates both in `(0, 2]`.
- `_VALID_ALGORITHMS = ("ga", "cma_es", "de", "pso", "qpso", "islands")`.
- `OptimizerConfig.qpso: QPSOSettings = field(default_factory=QPSOSettings)`.
- `from_dict`: parse `d["qpso"]` into `QPSOSettings`, and add `"qpso"` to the subsection-exclusion tuple in the `top_level` comprehension (missing this leaks the dict into `cls(**top_level)` → `TypeError`).
- `create_algorithm`: `return QPSO(pop_size=config.n_pop, alpha_start=config.qpso.alpha_start, alpha_end=config.qpso.alpha_end, max_iter=config.n_gen)`.

### 4.3 `train.py` (three one-line touchpoints)

1. `_seed_initial_population`'s algorithm whitelist (`("ga", "de", "pso", "islands")` → add `"qpso"`): warm-start seeding takes the same tile + per-row jitter path as PSO.
2. Settings printout block: `elif opt.algorithm == "qpso": print(alpha_start/alpha_end)`.
3. argparse `--algorithm` help string: add `qpso`.

**Explicitly zero changes** to: checkpoint save/load (pop = pbest, same as PSO; QPSO has no velocity to lose), `warm_start_algorithm`, `resize_population`, seed-strategy helpers, validation gate, logger/TUI/report, `compare_guidance.py` (consumes `best_model.json`, algorithm-agnostic).

**Resume note**: on single-algo resume, `config.optimizer.n_gen` is bumped to the new total (resumed + additional, train.py:1131) *before* `create_algorithm` runs (train.py:1273), while `warm_start_algorithm` restarts `n_iter` at 1. So alpha restarts at `alpha_start` and anneals on the stretched total-gens schedule — it does not resume where the prior run left off, and a short extension stays near `alpha_start`. This is the same family of state reset PSO already has (velocity reinit, adaptive w/c1/c2 reset); documented in the module docstring, not engineered around. Paper runs are single-shot `--from-scratch`, so this is an edge case.

### 4.4 Configs

- `configs/training/common.toml`: new block

  ```toml
  [optimizer.qpso]
  alpha_start = 1.0
  alpha_end = 0.5
  ```

- `configs/training/paper/opt_qpso.toml`: mirror `opt_pso.toml` (base `../sweep/dense_p515.toml`, `[data] neural_network = "training_output/paper_opt_qpso/best_model.json"`, `results_suffix = ".paper_opt_qpso"`, `[optimizer] algorithm = "qpso"`).
- `configs/training/paper/optbig_qpso.toml`: mirror `optbig_pso.toml` (base `../sweep/dense_p3998.toml`, paths `paper_optbig_qpso`).
- `run_paper_experiments4.sh` (new file; batches 2–3 scripts are already-executed records and stay untouched):

  ```
  small net @300: opt_qpso.toml    --n-gen 2000 --n-pop 300 --from-scratch
  big net   @60 : optbig_qpso.toml --n-gen 2000 --n-pop 60  --from-scratch   (default output dir, matches batch 2)
  big net  @150 : optbig_qpso.toml --n-gen 2000 --n-pop 150 --output-dir training_output/paper_optbig_qpso150 --from-scratch
  big net  @300 : optbig_qpso.toml --n-gen 2000 --n-pop 300 --output-dir training_output/paper_optbig_qpso300 --from-scratch
  ```

## 5. Testing

Extend the existing optimizer-config tests in `tests/test_optimizer.py`:

- `"qpso"` accepted by `OptimizerConfig`; unknown values still rejected.
- `from_dict` parses `[optimizer.qpso]` into `QPSOSettings`; defaults apply when absent.
- `QPSOSettings` rejects `alpha_start = 0`, `alpha_end = 2.5` (out of `(0, 2]`).
- `create_algorithm` returns a `QPSO` instance for `algorithm = "qpso"`.

New `tests/test_qpso.py` — pure Python (pymoo + numpy only, no `aerocapture_rs`, runs in the pure-python CI job), on a synthetic sphere problem:

1. **Determinism**: two runs with the same seed produce identical best X/F.
2. **Bounds**: every generation's infill X lies within `[xl, xu]`.
3. **pbest monotonicity**: per-index `pop.F` is non-increasing across generations.
4. **Alpha schedule**: `n_iter = 1` → `alpha_start`; `n_iter = max_iter` → `alpha_end`; `n_iter > max_iter` → clamped at `alpha_end`; `max_iter = 1` does not divide by zero.
5. **Convergence smoke**: 20-D sphere, `pop_size = 20`, 40 generations → best F improves by ≥ 100x over the initial best (generous margin, no flakiness).
6. **Warm-start survival**: `warm_start_algorithm(QPSO(...), problem, seeded_pop)` then one `.next()` — the gen-0 pbest still contains the seeded chromosomes (the invariant `warm_start_algorithm` exists to protect).

## 6. Out of Scope

- `island_model.py` (stays 3-island PSO/GA/DE).
- RL trainers.
- Rust simulator / PyO3 (untouched).
- CMA-ES-style special cases in `train.py` (QPSO follows the default PSO-like path everywhere).
