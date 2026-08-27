# Real-Valued Optimization Framework with pymoo

**Date:** 2026-04-12
**Status:** Design
**Scope:** Replace binary GA with real-valued optimization via pymoo, supporting multiple algorithms

## Motivation

The current GA uses binary-encoded chromosomes (16 bits/param) with roulette wheel selection, uniform crossover, and bit-flip mutation. This has known limitations:

- **Scale-blind mutation:** bit-flip doesn't respect parameter sensitivity -- flipping a high-order bit causes a massive jump while a low-order flip does almost nothing
- **Quantization waste:** 16-bit binary encoding provides ~0.0015% resolution, far more than needed, while consuming chromosome length
- **Single algorithm:** no way to try CMA-ES, DE, or PSO without rewriting the loop
- **Convergence:** some schemes (FNPAG, EqGlide) struggle to converge; better operators could help

**Goals (prioritized):**
1. Faster convergence (A)
2. Higher solution quality ceiling (B)
3. More robust/consistent results across all schemes (C)
4. Pluggable algorithm selection (D)

## Decisions

- **Clean break:** no backward compatibility for checkpoints or TOML config
- **pymoo** as the optimizer framework (real-valued native, algorithm swapping via one config change, active maintenance)
- **Hybrid loop:** pymoo `Algorithm` objects stepped manually via `algorithm.next()`, outer loop handles logging/display/seeds/checkpoints
- **Four algorithms at launch:** GA (SBX + polynomial mutation), CMA-ES, DE, PSO
- **Adaptive seed pool:** checkpoint-based updates every K generations with full population re-evaluation
- **NN included:** real-valued encoding for NN weights (GA/DE only; CMA-ES skipped for high-dim)
- **Bayesian optimization and RL:** flagged as future investigations in IMPROVEMENTS.md
- **Validation:** train piecewise_constant and FTC, compare against current baselines

## Architecture

### 1. Parameter Space & Encoding

`ParamSpec` structure stays unchanged -- `(name, min, max, log_scale)`.

**Internal representation:** all algorithms work on normalized `np.ndarray[float64]` in **[0, 1]**.

- Normalization keeps all algorithms happy (comparable scales, bounded domain)
- pymoo operators (SBX, polynomial mutation) work best on bounded domains
- CMA-ES needs comparable scales across dimensions

**Decoding to physical values** happens at evaluation time only:
- Linear params: `value = p_min + x * (p_max - p_min)`
- Log-scale params: `value = 10^(log_min + x * (log_max - log_min))`

**NN encoding:** same normalized [0, 1] array. The sign-bit trick goes away -- real values are naturally signed. Per-weight bounds are derived from the initialization strategy: for a layer with `fan_in` and `fan_out`, Xavier uniform gives `[-sqrt(6 / (fan_in + fan_out)), +sqrt(6 / (fan_in + fan_out))]` as `[p_min, p_max]`. Each weight gets its own ParamSpec with layer-appropriate bounds, then normalized to [0, 1] internally. A scale multiplier (e.g., 2x-3x) on the Xavier range allows the optimizer to explore beyond the typical initialization region.

**Replaces:** `binary_to_decimal()`, `decode_direct()`, `decode_params_from_chromosome()`, `encode_weights_to_chromosome()`, `encode_params_to_chromosome()`, `perturb_network()`.

**New function:** `decode_normalized(x: np.ndarray, param_specs: list[ParamSpec]) -> dict[str, float]`

### 2. Problem Subclass & Evaluation

`AerocaptureProblem(pymoo.Problem)` bridges pymoo and the simulator.

**Constructor parameters:**
- `param_specs: list[ParamSpec]` -- defines dimensionality and bounds
- `toml_path: str` -- base config for simulations
- `seeds: list[int]` -- current MC seeds (updated externally by seed pool)
- `cost_kwargs: dict` -- constraint limits, penalty weights, DV threshold
- `scheme: str` -- guidance scheme name (for TOML patching routing)
- `sim_timeout: float | None`
- `nn_config: dict | None` -- layer sizes, base weights (NN path only)

**`_evaluate(self, X, out, *args, **kwargs)`:**
1. `X` is shape `(n_pop, n_params)` in [0, 1]
2. Decode each row to physical params via `decode_normalized()`
3. Build TOML override dicts (same prefix routing: `nav.`, `lateral.`, `exit.`, `thermal.`, `shaping.`)
4. For NN: construct weight JSON from decoded values
5. Batch-evaluate all individuals across all current seeds via `aerocapture_rs.run_batch()` or `run_with_draws()`
6. Aggregate costs per individual: RMS across seeds (or alpha-blended mean/CVaR when seed pool is active)
7. Set `out["F"] = costs` (shape `(n_pop, 1)` -- single objective)

**Seed update:** outer loop calls `problem.update_seeds(new_seeds)` at pool checkpoint intervals. Just a setter -- pymoo doesn't know or care.

**Key improvement:** evaluation is fully batched at the population level. Currently `evaluate_chromosome()` handles one individual. The new design pushes the full population through `run_batch()`, letting Rayon parallelize across all sims.

### 3. Algorithm Factory & Configuration

**TOML config structure:**

```toml
[optimizer]
algorithm = "ga"          # "ga", "cma_es", "de", "pso"
n_pop = 60
n_gen = 2500
seed_pool_interval = 50   # re-evaluate population on updated seeds every K gens

[optimizer.ga]
crossover = "sbx"         # SBX crossover
crossover_eta = 15        # distribution index (higher = more conservative)
mutation = "polynomial"   # polynomial mutation
mutation_eta = 20         # distribution index
mutation_prob = null       # null = 1/n_params (pymoo default, adaptive per dimensionality)

[optimizer.cma_es]
sigma0 = 0.3              # initial step size in normalized space
restart_strategy = "ipop"  # increasing population restart

[optimizer.de]
variant = "DE/rand/1/bin"
crossover_prob = 0.7
scaling_factor = 0.5

[optimizer.pso]
w = 0.9                   # inertia
c1 = 2.0                  # cognitive
c2 = 2.0                  # social
```

**Factory function:** `create_algorithm(config: OptimizerConfig, n_params: int) -> pymoo.Algorithm`
- Reads `config.algorithm`, instantiates matching pymoo class with settings from relevant subsection
- For NN (high-dim): if user selects `cma_es`, warn and fall back to `ga`
- `n_pop` shared across algorithms; CMA-ES self-determines population size from dimensionality

**CLI:** `train.py` CLI stays the same. Algorithm choice lives in TOML. The `--mutation-rate` flag becomes algorithm-specific (GA only).

### 4. Hybrid Training Loop

```
1. Load config, build param_specs, create AerocaptureProblem
2. Create Algorithm via create_algorithm()
3. Initialize population:
   - Default/known params encoded to normalized [0,1]
   - Perturbations around defaults for diversity
   - NN: Xavier/He-scaled initial population
   - Oversized pool -> evaluate -> keep best n_pop
4. algorithm.setup(problem, seed=rng_seed)
   - Inject initial population via sampling parameter

5. GENERATION LOOP:
   for gen in range(n_gen):
       algorithm.next()

       pop = algorithm.pop
       costs = pop.get("F")
       best = algorithm.opt

       # Hooks:
       logger.log_generation(gen, costs, best, ...)
       display.update(gen, costs, ...)
       corridor_accumulator.update(...)  # piecewise_constant only

       # Seed pool checkpoint (every K generations):
       if gen % seed_pool_interval == 0 and seed_pool:
           seed_pool.add_seeds(gen)
           seed_pool.score_difficulty(best)
           seed_pool.evict_redundant()
           problem.update_seeds(seed_pool.seeds)
           algorithm.evaluator.eval(problem, pop)  # re-evaluate on new seeds

       # Stress test (existing interval logic)
       # Checkpoint save (existing interval logic)
       # Ctrl+C: save checkpoint and break

6. Post-training:
   - Final MC evaluation on large seed set
   - Export best_params.json / best_model.json
   - Generate PDF report
```

**Checkpoint format:**
- JSON metadata: generation, best_cost, cost_history, seed_pool state, algorithm name, rng_state
- NPZ: `population` (n_pop, n_params) float64 in [0, 1], `costs` (n_pop,), `best_individual` (n_params,)

**Resume strategy:** reconstruct algorithm from config, inject saved population as initial sampling, continue from saved generation count. CMA-ES loses covariance state on resume (re-adapts quickly). GA/DE/PSO resume cleanly.

### 5. Deletion & Replacement Map

**Deleted entirely:**
- `evaluate.py`: `binary_to_decimal()`, `decode_direct()`, `decode_params_from_chromosome()`, `encode_weights_to_chromosome()`, `perturb_network()`
- `population.py`: `encode_weights_to_chromosome()`, `encode_params_to_chromosome()`, binary population generation
- `train.py`: roulette wheel selection, uniform crossover, bit-flip mutation, inner GA loop
- `local_search.py`: bit-flip hill climbing
- `migration.py`: subpopulation migration
- `config.py`: `GAConfig`, `n_bit`, `chrom_len`

**Replaced / refactored:**
- `evaluate.py`: `evaluate_chromosome()` logic moves into `AerocaptureProblem._evaluate()`. TOML patching and cost computation stay.
- `population.py`: initialization rewritten for real-valued arrays
- `train.py`: generation loop replaced by hybrid loop
- `config.py`: `GAConfig` replaced by `OptimizerConfig`

**Unchanged:**
- `param_spaces.py`: `ParamSpec` definitions stay (remove `n_bit` references if any)
- `seed_pool.py`: operates on costs, not chromosomes
- `metrics.py`, `logger.py`, `display.py`: consume costs/params, not chromosomes
- `report.py`, `charts.py`, `corridor.py`: unchanged
- `sensitivity.py`, `parquet_output.py`, `compare_guidance.py`: unchanged

### 6. Testing Strategy

**New tests:**
- Algorithm factory: correct pymoo class instantiated per config, NN high-dim CMA-ES fallback
- `AerocaptureProblem`: evaluation produces correct cost shape, seed update works, TOML override routing
- Normalized encoding/decoding: linear roundtrip, log-scale roundtrip, boundary values
- Checkpoint save/resume: population roundtrip, generation count continuation
- Config parsing: `OptimizerConfig` from TOML, per-algorithm subsections, defaults

**Validation tests:**
- Train piecewise_constant with new framework, compare DV distribution against current baseline
- Train FTC with new framework, same comparison
- Both schemes tested with GA algorithm (direct comparison to old binary GA)

**Deleted tests:**
- All binary encoding/decoding tests
- Bit-flip mutation, uniform crossover, roulette selection tests
- Local search, migration tests

## Future Investigations (IMPROVEMENTS.md additions)

### Bayesian Optimization
- Surrogate-model-based optimization (Gaussian Process or Random Forest as surrogate for expensive MC fitness)
- Promising for low-dimensional guidance schemes (10-26 params)
- Libraries: BoTorch (PyTorch-based) or scikit-optimize
- Challenge: noisy fitness from MC requires noise-aware acquisition functions
- Could complement pymoo as an additional backend

### Reinforcement Learning for NN Training
- Train NN guidance controller as an RL policy instead of optimizing static weights via GA
- Simulator is already step-able (state -> action -> next state)
- Requires wrapping Rust sim as Gym-compatible environment (PyO3 step API)
- Algorithms: PPO, SAC, TD3 (continuous action space for bank angle)
- Separate paradigm from optimizer rework
