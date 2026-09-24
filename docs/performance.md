# Performance

What the simulator and the training loop cost, where the time goes, and what an accelerator
port would and would not buy. Every number below is from one machine, and each comes from a
block of [`experiments/throughput/throughput.json`](../experiments/throughput/throughput.json)
that records the machine, the commit and `sources_clean: true`:

- **Machine**: Apple M4 Pro, 10 performance + 4 efficiency cores, 48 GiB, macOS 26.6.2
- **Build**: rustc 1.98.1, release profile with LTO, f64 throughout; Python 3.14.7
- **Commits**: `scaling`, `seams`, `grid` and `memory` at `555c0005`; `profile`, `fnpag_step`
  and `benches` at `b805cffd` (2026-09-24). The simulator source is identical at both commits.
  Between them only the driver and the bench harness changed.
- **Regenerate**: `uv run python experiments/throughput/throughput.py`, one command for every
  block and both figures ([`experiments/throughput/README.md`](../experiments/throughput/README.md))

The cells are the paper's deployed operating points from the committed bundle
(`articles/paper/data/runs/`): FTC and FNPAG with their GA-tuned gains, and the dense-515 and
Mamba-962 headline networks. The throughput cells run the default `per_draw` noise regime
(ADR-0006).

**Precision.** Each quoted point is the median of three repeats. Up to 4 threads the repeats
agree within 4%. At 8 threads they spread 2-12%, and at 10 and 14 threads 3-22%
(`(max - min) / median` per point). The machine also drifts between runs. A first full run
(commit `99c8ea84`, same simulator and scaling code, its JSON discarded for a provenance-stamp
bug, so not in the committed file) measured FTC at 14 threads at 12,307 sims/s, against 9,952
here. The cause (thermal or power state, background load) was not isolated. Ratios within one
block are the solid part. Absolute full-thread rates are good to about 20%.

## 1. Thread scaling

One `run_grid` call per point: one deployed cell, n = 1000 seeds of the final-eval pool, three
repeats interleaved across thread counts.

| scheme | mean flight | 1 thread | 10 threads (P-cores) | 14 threads (all) |
|---|---|---|---|---|
| FTC | 478 s | 1,116 sims/s (0.90 ms/sim) | 9,332 (8.4x) | 9,952 (8.9x) |
| FNPAG | 644 s | 11.1 sims/s (89.8 ms/sim) | 101 (9.1x) | 122 (11.0x) |
| NN dense-515 | 780 s | 506 sims/s (1.98 ms/sim) | 3,753 (7.4x) | 4,018 (7.9x) |
| NN Mamba-962 | 767 s | 315 sims/s (3.17 ms/sim) | 2,435 (7.7x) | 2,527 (8.0x) |

![run_grid scaling](../experiments/throughput/fig_scaling.svg)

Scaling is close to linear up to 8 threads, where parallel efficiency is 0.76 (Mamba) to 0.95
(FNPAG). The four efficiency cores add 4-7% for FTC and the two networks, which is inside the
repeat spread at those thread counts, and 20% for FNPAG, which just clears it. FNPAG has the most
work per simulation and scales best. Why the light schemes flatten earlier was not diagnosed.

## 2. The three entry points

Per-sim cost at 1 thread, n = 200 final-eval seeds. The CLI flies 200 draws of the config's own
Monte Carlo instead: the same dispersion model, different scenarios.

| scheme | `run_grid` | `run_batch` | CLI, 200 sims | CLI, simulation phase | CLI, one-sim process |
|---|---|---|---|---|---|
| FTC | 0.891 ms | 0.961 ms | 0.956 ms | 0.910 ms | 8.1 ms |
| FNPAG | 85.7 ms | 87.4 ms | 86.2 ms | 86.2 ms | 64.3 ms |
| NN dense-515 | 1.914 ms | 2.081 ms | 1.994 ms | 1.95 ms | 9.6 ms |
| NN Mamba-962 | 3.138 ms | 3.403 ms | 3.164 ms | 3.12 ms | 10.3 ms |

`run_batch` builds a fresh `SimData` per override: TOML patch, serialize, parse, and for an NN
the model JSON read and parse. That costs 0.07 ms per sim for FTC and 0.27 ms for Mamba, 8% in
both cases. For FNPAG the 2% gap is inside the repeat spread (17.0-17.8 s for the 200 sims).
`run_grid` builds one `SimData` per individual and reuses it across the seed axis (ADR-0004).
Amortized over 200 sims, the CLI costs the same as the seam. Each CLI invocation spends 8.4-9.8
ms outside its own simulation phase (process start, config and table load, CSV output), measured
as the 200-sim run's wall minus the simulation time it prints, repeat by repeat. That is about ten
FTC simulations' worth per process, so process-per-simulation evaluation would be dominated by it.

## 3. Where a training generation goes

The real GA loop (`trainer.run_loop`), resumed from the headline run's generation-20000
checkpoint at the paper allocation (512 individuals x 2 seeds) and at 512 x 10 for contrast.
Settings: GA, adaptive seeds, curation every 2 generations on 1000 sampled seeds, a
1000-seed validation gate. Each run covers 20 generations with the plain display (no Rich TUI);
the once-per-run final selection is excluded. Every loop phase is timed. Inside each phase the
driver separates the time in `aerocapture_rs.run_grid`, in the Python that prepares the call and
in `compute_cost`.

![per-generation profile](../experiments/throughput/fig_generation_profile.svg)

| seconds per generation | 512 x 2 | 512 x 10 |
|---|---|---|
| Rust: offspring evaluation | 0.403 | 1.981 |
| Rust: parent re-evaluation after a seed change | 0.222 | 1.389 |
| Rust: validation gate | 0.288 | 0.350 |
| Rust: seed curation | 0.214 | 0.248 |
| **Rust total** | **1.127 (84%)** | **3.967 (93%)** |
| Python: gene decode, override dicts, weight matrix | 0.002 | 0.003 |
| Python: cost function | 0.033 | 0.117 |
| pymoo operators (selection, SBX, mutation, duplicate elimination, survival) | 0.147 | 0.152 |
| JSONL log and display | 0.024 | 0.025 |
| checkpoint I/O (every 10th generation) | 0.001 | 0.001 |
| other (seed draw, gate and curation bookkeeping) | 0.010 | 0.014 |
| **total** | **1.346** | **4.278** |

Offspring evaluation is 30% of a generation at the paper allocation. The adaptive-seed
methodology's own simulations take 54%: re-scoring the parents on fresh seeds, curating the next
seed set and validating a new argmin. That is the price of the methodology the paper found
load-bearing, not overhead. The Python side is 0.22 s per generation and grows only through the
per-cell cost function (0.31 s at 512 x 10), so its share falls from 16% to 7%.

The loop's own per-generation stamp (`gen_elapsed_s`, from `advance` to `emit`, the number a
campaign's JSONL records) has a median of 1.02 s at 512 x 2 and 2.58 s at 512 x 10. It excludes
the parent re-evaluation and the checkpoint.

Top five by self time under cProfile (the same 20 generations at 512 x 2, `run_grid` excluded):

| # | function | where | self time over 20 gens |
|---|---|---|---|
| 1 | scipy `cdist_euclidean` | pymoo duplicate elimination, `pymoo/core/duplicate.py:66-70` via `pymoo/util/misc.py:155` | 2.04 s |
| 2 | scipy `pdist_euclidean` | `population_diversity`, `src/python/aerocapture/training/metrics.py:33` | 0.47 s |
| 3 | `compute_cost` | `src/python/aerocapture/training/cost.py:98` (72,744 scalar calls) | 0.37 s |
| 4 | `_softplus` | `src/python/aerocapture/training/cost.py:62` (290,976 calls) | 0.27 s |
| 5 | `cross_sbx` | `pymoo/operators/crossover/sbx.py:14` | 0.27 s |

pymoo's GA enables duplicate elimination by default and `optimizer.py` leaves it on. It computes
offspring x offspring and offspring x population distance matrices (512 x 512, 518 dimensions)
on every mating round, about eight calls per generation. That is 0.10 s, 70% of the pymoo share
and 8% of a generation. The diversity metric written to every JSONL record is the whole logging
cost. Turning duplicate elimination off would change which offspring survive whenever an exact
duplicate occurs, so it is a training-behaviour change with its own gate, not a free win.

## 4. Inside `run_grid`: setup vs cells

The Rust side is one opaque call per phase above. This fit splits it. Wall of one `run_grid`
call through `AerocaptureProblem` (in-memory weights, the global Rayon pool) on the converged
512-individual headline population, against k seeds per individual: wall = 0.030 s + 0.184 s x k
(k = 1, 2, 4, 8; worst residual 0.041 s). The intercept is the parallel `SimData` build for 512
individuals; the slope is 512 cells at 0.36 ms of wall each. At the paper's k = 2 the build is
7.5% of the call. The intercept is the size of the residual, so the honest reading is that setup
is under a tenth of the call, not a precise 7.5%. Section 5 splits a simulation further, into
guidance vs navigation and plant.

## 5. Onboard guidance cost per scheme

`src/rust/benches/tick.rs` replays every guidance call of one nominal (undispersed) flight from
its exact pre-call state. It checks the replay against the flight bit for bit before timing
anything, so the time is guidance only: no navigation, pilot or plant. It runs single-threaded
with 20 criterion samples of whole-flight replays. The built-in rows fly the golden test configs
(legacy noise regime) and, for piecewise constant and the NN, the training configs (per_draw),
the NN with the deployed bundle models. The `deployed` rows are the four throughput cells with
every GA parameter routed by the Python deploy rule. The driver writes them and hands them to the
bench.

| scheme | config | guidance calls | guidance per flight | per call |
|---|---|---|---|---|
| FTC | deployed | 488 | 145 µs | 0.30 µs |
| FTC | golden | 534 | 321 µs | 0.60 µs |
| Equilibrium glide | golden | 266 | 40.8 µs | 0.15 µs |
| Energy controller | golden | 428 | 91.1 µs | 0.21 µs |
| PredGuid | golden | 437 | 83.9 µs | 0.19 µs |
| FNPAG | deployed (`prediction_dt` 3.76 s) | 651 | 81.0 ms | 124 µs |
| FNPAG | golden (`prediction_dt` 2.0 s) | 460 | 132 ms | 286 µs |
| Piecewise constant | training config | 526 | 62.3 µs | 0.12 µs |
| NN dense-515 | deployed | 777 | 834 µs | 1.07 µs |
| NN Mamba-962 | deployed | 730 | 1.96 ms | 2.68 µs |

The Mamba forward pass alone (`quant_forward`'s deployed f64 runtime, 100 samples, same block)
takes 2.01 µs, 75% of the deployed Mamba guidance call. The rest builds the 35-candidate input
vector, which includes a per-tick correction-DV prediction on the osculating orbit.

Per-call cost depends on the operating point. Deployed FTC costs half what the golden FTC config
costs per call; why was not diagnosed. For FNPAG the cause is measured. Its predictor
integrates forward at `prediction_dt`, which the GA tuned from the default 2.0 s to 3.76 s. The
`fnpag_step` block flies the deployed cell at both steps (1 thread, n = 50): 89.4 ms/sim at the
tuned step against 163.9 ms at 2.0 s, a 1.83x cost ratio for a 1.88x step ratio.

Combining the deployed per-call cost with the dispersed flights of section 1 (one guidance call
per 1 s tick) gives guidance's share of a simulation. This is an estimate across two blocks: 16%
for FTC, 42% for dense-515 and 65% for Mamba-962. Navigation plus plant integration then comes
to 1.45-1.6 µs per tick for all three. The policy, not the plant, is the larger half of a Mamba
simulation.

## 6. Determinism across thread counts

`tests/test_thread_invariance.py` is a CI gate, unlike everything above. It asserts that
`run_grid` output is byte-identical at `n_threads` = 1, 3, all cores and the global pool, for:

- dispersed equilibrium glide (fixed-step, and adaptive with event location)
- FNPAG
- the deployed Mamba-962
- in-memory dense weights

Each cell owns its `SimState` and RNG streams, `SimData` is shared read-only, and Rayon's indexed
collect places every result by index whatever order the cells ran in. The gate extends
ADR-0004's bit-identity claim to the parallel axis.

## 7. Memory

Peak RSS a `run_grid` call adds, in a fresh process per shape (Mamba-962). The process sits at
30 MiB after import and a one-sim warm-up.

| grid | peak RSS added | output array |
|---|---|---|
| 64 x 2 | 7.0 MiB | 0.05 MiB |
| 512 x 2 (paper allocation) | 15.4 MiB | 0.41 MiB |
| 2048 x 2 | 44.4 MiB | 1.62 MiB |
| 1 x 1000 (validation shape) | 1.9 MiB | 0.40 MiB |

That is about 19 KiB per individual. Memory does not constrain any grid this project runs: a
population 100 times the paper's would fit in about a GiB.

## 8. An accelerator path: feasibility note

**The workload.** A paper-allocation generation flies roughly 3,000 trajectories (offspring,
re-evaluated parents, validation, curation) of about 770 one-second ticks. That is 1.13 s of Rust
wall on 14 threads. Each trajectory is a strictly sequential chain of ticks, so the only
parallelism is across trajectories, about 3,000 wide. Per tick, navigation plus plant cost about
1.5 µs and the Mamba policy 2.7 µs of scalar f64 code.

**What a port would buy.** The DeepMind-style move is a batched plant in JAX. The tick becomes
array code, with `vmap` over trajectories, `lax.scan` over ticks, and the whole rollout one
jitted program. That is the Brax / MJX pattern, not those engines: this plant is a 3-DOF point
mass with aero tables, not a rigid-body system. The ceiling is set by the loop around the
simulator, and it is measured. With a free simulator, the 0.22 s of Python per generation caps
the speed-up at 6.2x at 512 x 2 and 13.8x at 512 x 10, unless the optimizer moves on-device too
(an evosax-style GA). At the paper's allocation the batch is also small for an accelerator: about
3,000 trajectories stepped through 770 dependent ticks is a latency-bound workload. The case for a
port is therefore not a faster paper run. It is allocations the CPU cannot afford: populations and
seed counts ten to a hundred times wider. On the CPU that cost grows linearly, at about 2,500
Mamba trajectories per second. The campaign already found that the ~4,000-parameter cells need
wide populations (`experiments/paper/10_architecture_sweep.sh`). No accelerator was measured; the
Amdahl ceilings are the only quantitative claim here.

**What it would cost.**

- *A second simulator.* Everything in the tick moves to array form:
  - navigation: the bias filter and the 13-state EKF
  - guidance: the seven laws, lateral reversal logic, thermal limiter, command shaping and pilot
  - the plant: Gill RK4
  - the table lookups (atmosphere, aerodynamics, reference trajectory), as gathers

  Discrete logic (phase switches, roll reversals, bounce and pending-crash classification)
  becomes masked `where`, which evaluates both branches. Adaptive DOPRI45 with sub-tick event
  location has no clean lockstep form, but training already runs fixed-step (`[integration]
  mode = "fixed"` in `configs/training/common.toml`). FNPAG's predictor is the worst fit: a
  nested integration whose length varies per trajectory, which would run as a bounded, masked
  loop. The Rust simulator stays as the reference for deploy-side evaluation, so equivalence gates
  must hold the two together, as they already do for the PyTorch mirror of the NN runtime.
- *Bit identity.* XLA reorders floating-point work. On most GPUs f64 runs at a small fraction of
  f32 throughput, and Apple-silicon GPUs have no f64 at all. These gates would all become
  tolerance gates:
  - the six guidance goldens
  - the 22/24 columns matched to the reference implementation
  - the `run_grid` bit-identity gates

  The AMAT cross-check and the published corridor in `docs/validation.md` would need re-running
  against the port. Every number in the paper came from the f64 CPU simulator. A policy trained
  on the port would need a statistical-equivalence study (the same Monte Carlo distributions
  within stated tolerance) before it could be compared with them.
- *Differentiability* would come almost free with JAX. But the objective is discontinuous
  (capture vs crash, event-driven termination), so gradients through rollouts are a research
  question, not a speed-up.

**Cheaper wins first.** Measured above, on the CPU, without touching the physics: duplicate
elimination (8% of a generation), the per-generation diversity metric (2%) and scalar per-cell
cost calls (2.5%, vectorizable). Together about 12% of a paper-allocation generation.

**Verdict.** Not worth building for the paper's allocation. The measured ceiling is about 6x, and
the port gives up the bit identity and physics validation the results rest on. Worth building
only if the research question becomes what a 10-100x wider population or seed budget buys, and
then with the optimizer on-device as well.
