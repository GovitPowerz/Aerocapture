# Three-Island PSO/GA/DE with Episodic Migration — Design

**Date**: 2026-05-28
**Status**: Approved (brainstorming complete, awaiting implementation plan)
**Scope**: New optimizer mode `algorithm = "islands"` in `train.py` that runs PSO, GA, and DE in parallel with periodic Top-3 / worst-6 migration.

## 1. Motivation

PSO is the empirically strongest single-algorithm trainer for the NN guidance schemes in this codebase (it beats PPO and SAC on the same architecture / problem). Its known failure mode is **premature swarm convergence**: in normalized parameter space, the per-dim diversity collapses to near-zero well before the cost CDF stops improving. When this happens, `gbest` becomes a fixed attractor and the swarm spends thousands of generations refining a local optimum.

GA (with SBX crossover and polynomial mutation) and DE (with differential mutation) maintain diversity through fundamentally different mechanisms than PSO. Running all three in parallel and periodically exchanging top-3 individuals between them gives PSO an injection of fresh, *evaluated*, high-quality search points whenever its swarm has collapsed, while letting GA and DE consume PSO's gradient-style local refinements as free seed material.

This is the textbook heterogeneous-island model (Cantú-Paz 1998, Tomassini 2005), specialized to: shared seed list, shared validation pool, identity-trigger per-island validation, and a Path-A-compatible single-checkpoint output.

## 2. Goals and Non-Goals

**Goals:**

1. Provide a clean A/B against single-island PSO on the same TOML config (same chromosome width, same budget per island), so the paper can quote a single delta number.
2. Reuse existing infrastructure (pymoo `Algorithm` objects, `AerocaptureProblem`, seed curator, validation gate pattern, checkpoint format, Rich TUI, JSONL logger, PDF report, `compare_guidance.py`).
3. Bounded surface area: one new file (`island_model.py`), one edited file (`train.py`), one edited config (`common.toml`).
4. Drop-in deployment: winning island writes the same `best_model.json` + `best_params.json` artifacts the existing single-algorithm path writes — no Rust runtime changes.

**Non-goals:**

1. CMA-ES as a fourth island (chromosome width is large for NN; pymoo CMA-ES already falls back to GA above ~200 dims per existing code).
2. Heterogeneous architectures across islands (all three see the same `architecture` / `input_mask` / `output_param`).
3. Cross-language migration (Rust runtime is identical for all islands; no Rust changes).
4. Asymmetric per-island budgets (each island gets `n_pop = 64`, full strength).
5. Sophisticated migration-impact dashboards beyond a per-event JSONL log and a single overlay panel in the report.

## 3. Architecture

### 3.1 Module Layout

**New file**: `src/python/aerocapture/training/island_model.py` (~250 LoC).

```python
@dataclass
class Island:
    name: str                            # "pso" | "ga" | "de"
    algorithm: pymoo.Algorithm           # PSO / GA / DE instance from create_algorithm()
    problem: AerocaptureProblem          # shared across islands (same instance)
    last_validated_individual: ndarray | None
    best_overall_individual: ndarray | None
    best_overall_cost: float             # val_rms at promotion
    best_val_cost: float                 # for identity-trigger gate
    stagnation_counter: int              # gens since last promotion (TUI display)


@dataclass
class MigrationEvent:
    gen: int
    src_island: str
    dst_island: str
    slot_idx: int
    F_migrant: float
    F_displaced: float
    rng_seed_used: int


class IslandModel:
    islands: list[Island]                # PSO, GA, DE
    seed_curator: SeedCurator | None     # shared (adaptive strategy only)
    validation_seeds: ndarray            # reserved, shared
    final_eval_seeds: ndarray            # reserved, shared
    k_period: int
    k_top: int
    pso_inject_velocity_scale: float
    rng: np.random.Generator
    migration_log: list[MigrationEvent]

    def step(self, gen: int) -> None: ...
    def migrate(self, gen: int) -> list[MigrationEvent]: ...
    def validate_each(self, gen: int) -> None: ...
    def checkpoint(self, path: Path) -> None: ...
    @classmethod
    def from_checkpoint(cls, path: Path, ...) -> "IslandModel": ...
    def final_eval(self) -> dict[str, Any]: ...   # picks the winner
```

Top-level pure functions (easier to unit-test):

```python
def migrate(islands: list[Island], k_top: int, rng) -> list[MigrationEvent]: ...
def inject_into_pso(algorithm: PSO, slot: int, X: ndarray, F: float,
                    velocity_scale: float, rng) -> None: ...
```

**Edited file**: `src/python/aerocapture/training/train.py`.

- Optimizer dispatch gains an `algorithm = "islands"` branch. When selected, instantiates `IslandModel` (which internally creates the 3 pymoo `Algorithm`s via the existing `create_algorithm` factory).
- The outer loop calls `island_model.step(gen)` instead of `algorithm.next()`.
- KeyboardInterrupt block routes to `island_model.checkpoint()`.
- TUI: `display.py` extends to render 3 columns when given an `IslandModel`; falls back to current single-column rendering otherwise. JSONL logger writes one record per gen *per island*, each tagged with `island_name`.

**Edited file**: `configs/training/common.toml` — adds the `[optimizer.islands]` sub-block (see §6).

**Untouched**:

- `AerocaptureProblem` (used as-is, shared instance across islands).
- `evaluate.py` (unchanged — islands call `_run_batch` via the shared problem).
- All Rust code (zero changes — winning island writes the same `best_model.json`).
- `compare_guidance.py` (winning island's artifacts plug in identically).
- `report.py` gains a Part 0 (3-column convergence overlay + migration event timeline); Parts 1–3 are reused, run on the winning island only.
- `seed_curator.py` (shared instance, pools top-K across all 192 individuals).
- All chart functions in `charts.py` (per-island records consumed via the `island_name` field).

### 3.2 Why In-Process (Approach A)

Three integration models were considered:

- **A. In-process trainer** — single `train.py` process drives 3 pymoo Algorithms via a unified outer loop. *Chosen.* Natural extension of the existing pymoo-Algorithm-driven outer loop, reuses all infra, bounded surface area, easy ablation via `enabled = false`.
- **B. Three subprocesses + coordinator** — each island a separate `train.py`, migration via sidecar files. Adds IPC complexity (locking, race conditions) and 3x Python/Rust startup overhead per gen. Only justified if single-island crashes were a real concern (they are not — `sim_timeout` handles runaway sims).
- **C. Single batched evaluation pool** — all 192 individuals in one `run_batch` call. Collides with pymoo's `Evaluator().eval()` inside `algorithm.next()` (the same trap that drove the existing `_run_batch_pyo3` workaround in `problem.py`). Premature optimization; Rayon parallelism is already saturated by per-seed inner batching.

## 4. Per-Generation Sequencing

The ordering inside one outer-loop iteration is load-bearing. Subtle bugs in seed-curator-vs-evaluation ordering and pre-validation-dict semantics have bitten this codebase before (see project memory: `project_resume_cost_incomparability`, `project_pymoo_evaluator_skip`).

```
for gen in range(start_gen, start_gen + n_gen):
    # 1. (Adaptive strategy only) Curator decides whether to refresh seed list.
    #    The curator's `curation_top_k` (default 5) individuals are selected from the
    #    UNION of all 192 individuals across islands — search-space-wide signal, no
    #    algorithm-specific siloing.
    seeds_changed = seed_curator.maybe_curate(islands_top_k_union)
    if seeds_changed:
        problem.update_seeds(new_seeds)
        for island in islands:
            re-evaluate island.algorithm.pop under new seeds   # 3x cost, rare

    # 2. Advance each island one generation. SEQUENTIAL Python calls — Rayon
    #    parallelism happens INSIDE each algorithm.next() via the shared problem's
    #    _run_batch_pyo3 inner loop, which already saturates the CPU per-island.
    #    Three sequential 64-individual batches dominate the per-gen wall time.
    for island in islands:
        island.algorithm.next()   # pymoo: generate offspring, evaluate, survive
    # Invariant: all 192 individuals have F valid under current seed list.

    # 3. Migration step (every K gens, never at gen 0).
    if gen > 0 and gen % k_period == 0:
        events = migrate(islands, k_top=3, rng)
        migration_log.extend(events)

    # 4. Per-island identity-trigger validation gate.
    for island in islands:
        argmin = island.algorithm.pop[argmin_idx(island.algorithm.pop.get("F"))]
        if not np.array_equal(argmin.X, island.last_validated_individual):
            val_rms = validate_on_reserved_seeds(argmin, validation_seeds)
            island.last_validated_individual = argmin.X
            if val_rms < island.best_val_cost:
                island.best_overall_individual = argmin.X
                island.best_overall_cost = val_rms
                island.best_val_cost = val_rms
                island.stagnation_counter = 0
                log "PROMOTED"
            else:
                island.stagnation_counter += 1
                log "REJECTED"
        else:
            island.stagnation_counter += 1

    # 5. Logging + TUI + checkpoint.
    logger.write_gen_records(gen, islands, migration_events_this_gen)
    display.update(gen, islands)
    if gen % checkpoint_interval == 0 or gen == start_gen + n_gen - 1:
        island_model.checkpoint(output_dir / f"checkpoint_g{gen:05d}.npz")

# End-of-training: final eval across islands.
results = island_model.final_eval()
winner = min(results, key=lambda r: r["final_rms"])
write_artifacts(winner)   # best_model.json + best_params.json
```

### 4.1 Critical Ordering Invariants

- **Step 2 → Step 3 is mandatory.** Migration must follow `next()` because F is freshest immediately after `next()` under the current seed list. Migrating before would copy stale parent F.
- **Step 3 → Step 4 is mandatory.** Validation must follow migration so a migrant becoming the new argmin gets validated in the same gen, not deferred.
- **Identity-trigger fires naturally on migration receivers.** A migrant with better F than the previous argmin will displace it; the next iteration's `np.array_equal` check returns False and validation runs. No special-casing needed.

### 4.2 Cost Accounting

Per gen (compared to single-algorithm baseline at `n_pop = 64`):

| Step | Cost (sims per gen) | Cost vs single-island |
|---|---|---|
| Step 2 (3x next()) | `3 * 64 * training_n_sims` | 3.0x |
| Step 4 (validation, avg) | `~3 * 0.2 * validation_n_sims` | 1.5x |
| Seed re-eval on curation (amortized over `seed_pool_interval`) | `(3 * 64 * training_n_sims) / seed_pool_interval` | negligible |
| Migration | 0 | 0 |

**Total: ~3.0–3.1x per-gen MC sim cost vs single-island PSO.** Honest A/B at full per-island strength.

## 5. Migration Mechanics

### 5.1 Pseudocode

```python
def migrate(islands: list[Island], k_top: int, rng) -> list[MigrationEvent]:
    events = []
    # Snapshot top-k from each island BEFORE any in-place replacement.
    emigrants = {}
    for src in islands:
        F = src.algorithm.pop.get("F").flatten()
        top_idx = np.argsort(F)[:k_top]   # stable sort -> lowest idx wins on ties
        emigrants[src.name] = [
            (src.algorithm.pop[i].X.copy(), float(F[i])) for i in top_idx
        ]

    # Apply migrations: each destination receives top-k from each other island.
    for dst in islands:
        incoming = []
        for src in islands:
            if src.name != dst.name:
                for X, F in emigrants[src.name]:
                    incoming.append((X, F, src.name))

        F_dst = dst.algorithm.pop.get("F").flatten()
        n_incoming = len(incoming)  # = 2 * k_top
        worst_slots = np.argsort(F_dst)[-n_incoming:]

        for slot, (X_new, F_new, src_name) in zip(worst_slots, incoming):
            F_displaced = float(F_dst[slot])
            dst.algorithm.pop[slot].X = X_new
            dst.algorithm.pop[slot].F = np.array([F_new])
            if isinstance(dst.algorithm, PSO):
                inject_into_pso(dst.algorithm, slot, X_new, F_new,
                                velocity_scale=0.05, rng=rng)
            events.append(MigrationEvent(
                gen=current_gen, src_island=src_name, dst_island=dst.name,
                slot_idx=int(slot), F_migrant=F_new, F_displaced=F_displaced,
                rng_seed_used=int(rng.bit_generator.state["state"]["state"] & 0xFFFFFFFF),
            ))
    return events
```

### 5.2 PSO-Specific State Injection

When a migrant lands in a PSO slot, pymoo's PSO algorithm reads three per-individual attributes on its next velocity update: `V` (current velocity), `pbest` (personal best position), `pbest_F` (personal best fitness). Naive `V = 0` is a trap: if the swarm has collapsed (the stated pain), `gbest` is at the collapsed point and the migrant is sucked in within 2-3 ticks — defeating the rescue purpose.

```python
def inject_into_pso(algorithm: PSO, slot: int, X: ndarray, F: float,
                    velocity_scale: float, rng) -> None:
    n_params = X.shape[0]
    V = algorithm.pop.get("V")
    pbest = algorithm.pop.get("pbest")
    pbest_F = algorithm.pop.get("pbest_F")

    V[slot] = rng.uniform(-velocity_scale, velocity_scale, size=n_params)
    pbest[slot] = X.copy()           # no inherited history
    pbest_F[slot] = np.array([F])    # current F is the personal best
```

- `velocity_scale = 0.05` is half-width in normalized [0, 1] space, ~5% of cube edge per dim. Small but non-zero — gives the migrant 2-5 ticks of independent exploration before `gbest` pull dominates, enough to evaluate its neighborhood and (if it has good F) become the new `gbest` itself.
- GA and DE destinations need no per-individual state beyond `X` and `F` — clean position swap suffices.

### 5.3 Edge Cases

- **F-tie on slot selection**: `np.argsort` is stable; lowest index wins. Deterministic across runs for fixed RNG.
- **Migrant duplicates an existing PSO position**: rare in continuous normalized space, but if it occurs the random velocity gives the duplicate a different trajectory anyway. No special case.
- **First migration**: fires at `gen = K`, not `gen = 0`. The `gen > 0` guard in step 3 ensures all islands have a full gen of independent evolution before migration begins.
- **Migration disabled (`enabled = false`)**: step 3 skipped entirely. Three islands run as independent trainers. Used as ablation baseline for the paper.

## 6. Configuration

New `[optimizer.islands]` sub-block in `configs/training/common.toml`:

```toml
[optimizer]
algorithm = "islands"        # new value; existing "pso" | "ga" | "de" | "cma_es" still work
n_pop = 64                   # per-island; total = 3 * n_pop = 192
n_gen = 2500
seed_strategy = "adaptive"
training_n_sims = 20
validation_n_sims = 1000

[optimizer.islands]
enabled = true               # set false to ablate: 3 independent islands, no migration
k_period = 25                # gens between migration events
k_top = 3                    # individuals exported per (src -> dst) pair; total per dst = 6
pso_inject_velocity_scale = 0.05   # normalized-space velocity half-width for migrants

# Per-island algorithm tunables inherit the existing sub-blocks unchanged:
[optimizer.pso]              # used only by the PSO island
# ... existing PSOSettings fields ...

[optimizer.ga]               # used only by the GA island
crossover_eta = 15.0
mutation_eta = 20.0

[optimizer.de]               # used only by the DE island
# ... existing DESettings fields ...
```

`OptimizerConfig.from_dict` parses the new `algorithm = "islands"` and validates that the three sub-blocks (`pso`, `ga`, `de`) are present.

## 7. Failure Modes and Resume

### 7.1 Checkpoint Format

Version bump to `v2`. Single `.npz` for atomicity (`np.savez_compressed` + rename via tempfile pattern):

```python
{
    "version": 2,
    "generation": int,
    "base_mc_seed": int,
    "seed_curator_state": dict | None,
    "migration_log": list[MigrationEvent],
    "islands": [
        {
            "name": "pso",
            "algorithm_state": <pymoo serialized state>,
            "last_validated_individual": ndarray | None,
            "best_overall_individual": ndarray | None,
            "best_overall_cost": float,
            "best_val_cost": float,
            "stagnation_counter": int,
        },
        # ... ga, de
    ],
}
```

### 7.2 Resume Semantics

- `_check_resume_chromosome_shape` runs per-island. All three share the same width (same architecture); mismatch on any island → `ValueError` pointing to `--from-scratch`.
- **Cross-gen cost incomparability** (existing memory note) applies per-island: each island's `best_overall_*` is restored verbatim from the checkpoint and **never re-compared against the resumed population's gen-0 cost**. The initial-best-init block in `train.py` stays gated on `best_overall_individual is None` (fresh start only).
- Auto-resume from default output dir works unchanged.
- `--n-gen N` means "N additional generations" on resume (existing semantics).

### 7.3 Ctrl-C Handling

Existing `KeyboardInterrupt` block in `train.py` routes to `island_model.checkpoint()`. Single `.npz` write is atomic (tempfile + rename). No mid-migration corruption window — migration is an in-memory transformation between two checkpoints.

### 7.4 NaN/Inf Handling

Unchanged. Individual sims producing NaN return `+inf` cost via the existing Rust path. A bad individual gets selected against by its own island; the others are unaffected. Per-island NaN does not stop training.

### 7.5 Single-Island Stagnation

Each island has its own `stagnation_counter` for the TUI. A plateaued island continues to receive migrants (potentially rescuing it) or stays stuck (an acceptable, observable outcome). No early-stopping logic — per user preference (see `feedback_no_early_stopping.md`).

### 7.6 Final Evaluation and Winner Selection

End of training:

```python
def final_eval(self) -> list[dict]:
    final_seeds = make_reserved_seeds(base_mc_seed, FINAL_EVAL_SEED_OFFSET, n)
    results = []
    for island in self.islands:
        if island.best_overall_individual is None:
            continue   # never had a validation promotion
        rms = problem.evaluate_individual_per_seed(
            island.best_overall_individual, final_seeds
        )
        results.append({"island": island.name, "rms": rms, ...})
    return results

winner = min(results, key=lambda r: r["rms"])
write_best_model_json(winner, output_dir / "best_model.json")
write_best_params_json(winner, output_dir / "best_params.json")
```

Winning island's artifacts are drop-in compatible with `compare_guidance.py` and the Rust runtime. The PDF report's Part 0 shows the per-island convergence overlay and migration timeline; Parts 1-3 run on the winning island only.

## 8. Testing Strategy

### 8.1 Unit Tests (`@fast`)

In `tests/test_island_model.py`:

1. `test_migrate_determinism` — seeded RNG produces identical migration events on repeat calls with identical inputs.
2. `test_migrate_top_k_selection` — emigrants are the k_top lowest-F individuals per island; ties broken by index.
3. `test_migrate_worst_slot_replacement` — replaced slots are the highest-F slots in the destination.
4. `test_migrate_no_self_migration` — no island appears as both src and dst in any single event.
5. `test_inject_into_pso_writes_state` — `V` / `pbest` / `pbest_F` are written to the correct slot with correct shapes and value ranges.
6. `test_inject_into_pso_velocity_bounded` — injected velocity values are in `[-velocity_scale, +velocity_scale]` per dim.
7. `test_checkpoint_v2_roundtrip` — `checkpoint()` then `from_checkpoint()` preserves all island state, the migration log, and the seed curator state bit-for-bit.
8. `test_resume_preserves_best_overall_per_island` — gen-0 cost of resumed population does NOT overwrite checkpointed `best_overall_*` (regression guard for the cross-gen cost incomparability rule).
9. `test_optimizer_config_islands_parses` — `OptimizerConfig.from_dict` accepts `algorithm = "islands"` and validates sub-block presence.

### 8.2 Integration Test (`@slow`)

In `tests/test_island_model_smoke.py`:

- End-to-end 5-gen smoke with `k_period = 1` (force migration every gen) on a reduced-architecture config (`n_pop = 16`, `training_n_sims = 2`, ~50 MC sims/gen).
- Assertions:
  - Every gen produces exactly 3 JSONL records (one per island, tagged with `island_name`).
  - At least 4 migration events are logged (3 islands × 4 inter-gen periods).
  - `final_eval()` produces a non-None winner.
  - `best_model.json` loads via `aerocapture_rs.nn_forward` and returns a finite output.
  - Checkpoint v2 round-trips: kill at gen 3, resume, verify identical `best_overall_*` per island.
- Runs in the `python-pyo3` CI job.

### 8.3 Ablation Baseline

`enabled = false` → 3 independent island runs with deterministic per-island state. Serves as the "islands without migration" control for the eventual paper figure (`migration on` vs `migration off` head-to-head on identical seed lists).

### 8.4 Regression Surface

- **No new Rust changes** → existing 474 Rust tests unaffected.
- **Existing 787 Python tests** untouched (islands branch fires only under `algorithm = "islands"`).
- **`compare_guidance.py`** consumes winning island's artifacts identically; no scheme registration changes.

## 9. Open Questions and Future Work

These are deliberately out of scope for the first implementation:

- **Adaptive K** based on per-island diversity. Possible follow-up if K = 25 turns out to be problem-dependent.
- **Asymmetric topology** (e.g., DE -> PSO only when PSO diversity drops). Empirical question — the full-mesh baseline must come first.
- **Heterogeneous architectures across islands** (e.g., dense PSO vs GRU PSO). Architecturally orthogonal to this work.
- **Migration ablation panel** in the PDF report beyond a simple timeline. YAGNI for the first paper.
- **A fourth island**. CMA-ES rejected for chromosome-width reasons; future SHADE or jSO are candidates.

## 10. References

- Cantú-Paz, E. (1998). "A Survey of Parallel Genetic Algorithms." *Calculateurs Paralleles*.
- Tomassini, M. (2005). *Spatially Structured Evolutionary Algorithms*. Springer.
- Skolicki, Z., & De Jong, K. (2005). "The Influence of Migration Sizes and Intervals on Island Models." *GECCO 2005*.
- Existing project memory: `project_pymoo_evaluator_skip`, `project_resume_cost_incomparability`, `project_seed_strategy_framework`, `project_cost_function_design`.

## 11. Implementation Sequence (For the Plan)

The implementation plan will sequence work as:

1. `IslandModel` and `Island` scaffolding (no migration yet); `algorithm = "islands"` dispatch in `train.py`; per-island JSONL records.
2. `migrate()` + `inject_into_pso()` + unit tests.
3. Per-island validation gate plumbing; identity-trigger for receivers.
4. Checkpoint v2 format + resume + tests.
5. Adaptive seed curator integration (pooled top-K).
6. Final-eval and winner selection; `best_model.json` / `best_params.json` write path.
7. TUI 3-column extension; PDF report Part 0 (overlay + migration timeline).
8. Integration smoke test (`@slow`, python-pyo3 CI).
9. Warm-start interaction: fan out warm-started chromosome to all 3 islands' initial populations.
10. **Final step**: invoke `smart-commit` skill, taking the whole branch into account (per user planning preference).
