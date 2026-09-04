# Training Strategy: Optimal GA Settings for All Guidance Schemes

## Problem

Current training runs use only ~3% of the available compute budget (2M sims out of ~60M possible in 1 hour). With n_pop=50 and n_gen=100, the GA stagnates on most schemes before meaningful convergence. FTC and piecewise_constant converge well due to strong defaults (FTC) and small search space (piecewise_constant, 11 params), but equilibrium_glide, energy_controller, pred_guid, fnpag, and neural_network all show stagnation.

## Root Cause

The GA explores too little of the search space. For a 24-param scheme at 16-bit encoding, the chromosome is 384 bits. A population of 50 with mutation_rate=0.02 flips ~8 bits per generation -- that's local search in a space of ~10^115 configurations. Stagnation is the expected outcome.

## Approach: Rebalance the Compute Budget

Trade fitness evaluation precision for exploration breadth:

- **Reduce n_sims** (1000 -> 200-300): LHS with 300 samples provides sufficient ranking accuracy for GA selection. This is a training-time override only; standalone TOML stays at 1000.
- **Increase n_pop** (50 -> 40-120 depending on scheme): Better diversity in high-dimensional spaces.
- **Maximize n_gen** within 1-hour budget: 600-3000 generations depending on scheme cost.
- **Higher mutation rate** (0.02 -> 0.03-0.05): More bit flips to escape plateaus.
- **Enable adaptive seeds**: With 600-3000 generations, the seed pool has time to curate hard cases for robustness.
- **Tune CVaR blend**: cost_alpha=0.6, cvar_percentile=15 for moderate worst-case focus without sacrificing mean performance.

## Constraints

- Mission parameters unchanged (vehicle, entry, orbit, constraints)
- Dispersion levels unchanged (medium/low as in common.toml)
- Cost function weights unchanged (1000/1000/1000, dv_threshold=1000)
- Integration mode unchanged (adaptive DOPRI45)
- LHS sampling unchanged
- Training budget: < 1 hour per scheme, 12 CPU cores

## Code Changes

### 1. Add `--mutation-rate` CLI flag to `train.py`

Currently hardcoded at 0.02 in `GAConfig`. Add a CLI argument that overrides `cfg.ga.mutation_rate`. No default change -- existing behavior preserved when flag is absent.

### 2. Add `--train-n-sims` CLI flag to `train.py`

Override `n_sims` during GA evaluation only, without touching the TOML. The TOML's `simulation.n_sims` remains authoritative for standalone runs. Implementation: inject `"simulation.n_sims": N` into the override dict passed to PyO3 `run_mc()` / `run_batch()` calls during training.

The `--final-n-sims` flag (already exists) controls the final re-evaluation at end of training.

### 3. Create `train_all.sh` script

Executable shell script at repo root. Runs all 7 schemes in dependency order (piecewise_constant first, rest after). Supports running a single scheme by name argument.

## Per-Scheme Configurations

### Tier 1: Already Converging (light touch)

#### piecewise_constant (11 params, 176 bits) -- TRAIN FIRST

Produces reference trajectory + corridor boundaries for Tier 2 schemes.

| Setting | Value | Rationale |
|---------|-------|-----------|
| n_pop | 40 | Plenty for 11 dims |
| train-n-sims | 300 | LHS 300 sufficient for ranking |
| n_gen | 3000 | ~35 min wall clock |
| mutation_rate | 0.03 | Mild increase to explore corridor boundaries |
| adaptive_seeds | yes | Robustness |
| cost_alpha | 0.65 | Moderate CVaR weight |
| cvar_percentile | 15 | Worst 15% tail |
| seed_pool_cap | 120 | Adequate for small space |
| stress_interval | 15 | Lower frequency, converges fast |
| stress_probes | 200 | Standard |
| stress_inject | 10 | Conservative injection |
| final_n_sims | 2000 | High-accuracy final eval |

Budget: 40 x 300 = 12k sims/gen, ~0.7s/gen, 3000 gens ~ 35 min.

#### ftc (26 params, 416 bits)

Strong reference defaults mean GA starts near a good basin.

| Setting | Value | Rationale |
|---------|-------|-----------|
| n_pop | 50 | Moderate for 26 dims with good init |
| train-n-sims | 300 | LHS 300 sufficient |
| n_gen | 2500 | ~37 min wall clock |
| mutation_rate | 0.03 | Mild increase |
| adaptive_seeds | yes | Robustness |
| cost_alpha | 0.65 | Moderate CVaR |
| cvar_percentile | 15 | Worst 15% |
| seed_pool_cap | 150 | Standard |
| stress_interval | 10 | Standard |
| stress_probes | 300 | Good coverage |
| stress_inject | 15 | Standard |
| final_n_sims | 2000 | High-accuracy final eval |

Budget: 50 x 300 = 15k sims/gen, ~0.9s/gen, 2500 gens ~ 37 min.

### Tier 2: Stagnation-Prone (aggressive settings)

#### equilibrium_glide (24 params, 384 bits)

7 scheme params interact with 17 shared params -- needs diversity to untangle couplings.

| Setting | Value |
|---------|-------|
| n_pop | 60 |
| train-n-sims | 300 |
| n_gen | 2500 |
| mutation_rate | 0.05 |
| adaptive_seeds | yes |
| cost_alpha | 0.6 |
| cvar_percentile | 15 |
| seed_pool_cap | 150 |
| stress_interval | 10 |
| stress_probes | 300 |
| stress_inject | 15 |
| final_n_sims | 2000 |

Budget: 60 x 300 = 18k sims/gen, ~1.1s/gen, 2500 gens ~ 46 min.

#### energy_controller (20 params, 320 bits)

The `gain` param spans 3 orders of magnitude on log-scale -- GA easily gets stuck in the wrong decade.

Same settings as equilibrium_glide. Budget: ~46 min.

#### pred_guid (20 params, 320 bits)

Also has a log-scale parameter (`pdyn_threshold`). Same profile as energy_controller. Budget: ~46 min.

### Tier 3: Special Cases

#### fnpag (22 params, 352 bits) -- compute-constrained

Forward predictor makes FNPAG ~7.5x slower per sim. Budget is tight.

| Setting | Value | Rationale |
|---------|-------|-----------|
| n_pop | 50 | Reduced to save per-gen cost |
| train-n-sims | 200 | Aggressive reduction for headroom |
| n_gen | 600 | Max that fits in budget |
| mutation_rate | 0.05 | Aggressive to compensate for fewer gens |
| adaptive_seeds | yes | Robustness |
| cost_alpha | 0.6 | Standard |
| cvar_percentile | 15 | Standard |
| seed_pool_cap | 100 | Smaller pool for fewer gens |
| stress_interval | 15 | Reduced frequency to save compute |
| stress_probes | 150 | Reduced for speed |
| stress_inject | 10 | Conservative |
| final_n_sims | 2000 | Same final accuracy |

Budget: 50 x 200 = 10k sims/gen, ~4.5s/gen, 600 gens ~ 45 min. Final eval adds ~5 min.

#### neural_network (1106 params, 17,696 bits)

A bit-flip GA in a 17,696-bit space is fundamentally limited. Smart initialization (Xavier/He) does most of the work; the GA provides local refinement. A different optimizer (CMA-ES, evolution strategies) would be the real fix, but that's out of scope here.

| Setting | Value | Rationale |
|---------|-------|-----------|
| n_pop | 120 | Minimum viable diversity for 1106 dims |
| train-n-sims | 200 | Keep generations fast |
| n_gen | 1500 | ~35 min wall clock |
| mutation_rate | 0.03 | Lower than Tier 2: at 17,696 bits, 0.03 = ~530 flips/gen already |
| adaptive_seeds | yes | Robustness |
| cost_alpha | 0.6 | Standard |
| cvar_percentile | 15 | Standard |
| seed_pool_cap | 100 | Standard |
| stress_interval | 15 | Reduced frequency |
| stress_probes | 200 | Standard |
| stress_inject | 10 | Conservative |
| final_n_sims | 2000 | Same final accuracy |

Budget: 120 x 200 = 24k sims/gen, ~1.4s/gen, 1500 gens ~ 35 min.

## Fair Comparison Protocol

After all schemes are trained, evaluate on identical MC scenarios:

```bash
uv run python -m aerocapture.training.compare_guidance \
    --n-sims 2000 \
    --schemes piecewise_constant ftc equilibrium_glide energy_controller pred_guid fnpag neural_network
```

This uses the same seed and dispersion draws for all schemes, making rankings fair regardless of different training-time settings.

## Training Order

1. **piecewise_constant** (produces ref trajectory + corridor)
2. All others in any order (ftc, equilibrium_glide, energy_controller, pred_guid, fnpag, neural_network)

## Deliverables

1. `train.py` -- two new CLI flags: `--mutation-rate`, `--train-n-sims`
2. `train_all.sh` -- training script at repo root, runnable per-scheme or all-at-once
3. Final comparison via `compare_guidance.py`

## Implementation Plan Final Step

After implementation, invoke the `smart-commit` skill taking the whole git branch into account.
