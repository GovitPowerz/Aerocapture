# Warm-Start for All NN Architectures (FTC+friends → BPTT → GA fine-tune)

Date: 2026-05-22
Status: design (awaiting user spec review → writing-plans)
Branch: feature/warm_start_full_neural_with_ftc_and_friends (current)

## Background and motivation

Empirically, the user has reached three new training equilibria on `feature/warm_start_full_neural_with_ftc_and_friends`:

1. `magnitude_only` NN guidance trains successfully both with and without the current FTC-only warm-start, with comparable validation RMS.
2. With larger population, plain **GA outperforms PSO** (the trainer used up to now) on every NN scheme tested.
3. A `full_neural` NN (signed-bank output, bypassing exit + lateral + thermal limiter) trains successfully and outperforms every other guidance — but the run is long (hundreds of generations from random Xavier-initialized weights to convergence).

The current warm-start pipeline (`aerocapture.training.warm_start.build_warm_start_chromosome`) was scoped to the `magnitude_only` use case introduced in the parity bundle (`docs/superpowers/specs/2026-05-07-nn-ftc-parity-bundle-design.md`). It is dense-only (raises `NotImplementedError` on any non-dense layer at `_policy_to_flat_weights`), FTC-only (`scheme="ftc"` hardcoded in `_aero_rs.collect_supervised`), and targets the pre-lateral unsigned magnitude (the value produced by FTC's capture-phase predictor-corrector before lateral+thermal+shaping).

Three asymmetries prevent it from accelerating `full_neural` training across the six NN architectures (dense, window, gru, lstm, transformer, mamba):

1. **Wrong target signal for full_neural.** `pre_lateral_magnitude` is unsigned and pre-shaping. The `full_neural` NN replaces the entire FTC+lateral+thermal+shaping chain — its output is the final signed bank command going into the pilot. Cloning unsigned pre-shaping bank teaches the NN magnitude well but leaves the sign essentially random; the first GA generations have to relearn it from scratch.
2. **Dense-only weight extraction.** `_policy_to_flat_weights` walks `policy.layers` assuming each module is a `DenseLayer(nn.Linear)`. Any other layer type raises. Stateful architectures (gru/lstm/mamba/transformer) and the zero-trainable-param window-MLP cannot be warm-started today.
3. **No temporal structure.** `_supervised_pretrain` shuffles steps and trains on IID minibatches. Acceptable for dense; broken for GRU/LSTM/Mamba/Transformer (no BPTT, hidden state never carries) and incoherent for Window-MLP (FIFO buffer is meaningless on shuffled steps).

This spec lands one bundled redesign of the warm-start pipeline. The existing pipeline has one caller (`build_warm_start_chromosome`), so the refactor is atomic — no legacy shim.

## Scope

In scope (one bundled change):

- Rust API: change `collect_supervised`'s capture point (final signed bank command after dispatch) and return shape (per-trajectory grouping with DV + capture flag).
- Python pipeline: multi-supervisor data collection ("best teacher per seed"), chunked truncated BPTT training, per-architecture flat-weights extraction matching Rust canonical order.
- Optimizer interaction: conditional bound widening (`warm_start.bound_multiplier`), per-algorithm initial-population seeding (GA/DE/PSO replicate-and-jitter, CMA-ES seed-mean-and-shrink-sigma), gen-0 validation baseline.
- Config UX: new `[warm_start]` TOML block with documented defaults and per-scheme `params_paths` override map.

Out of scope:

- New optimizer algorithms beyond pymoo's existing GA/CMA-ES/DE/PSO.
- New supervisor schemes beyond the 6 already exposed by Rust `collect_supervised` (ftc, equilibrium_glide, energy_controller, pred_guid, fnpag, piecewise_constant).
- Reward shaping, return normalization, or any RL/PPO-side changes (PPO already uses its own warm-start path via `--data-neural-network` + `load_policy_from_json`; that stays unchanged).
- Auto-tuning of `n_epochs` (per-arch defaults or validation-loss early-stop). Single TOML knob + per-epoch loss logging gives the user visibility; they tune.

## Design decisions

The brainstorming session locked in five decisions. Each is restated here as the canonical design statement.

### Decision 1: Target = signed final commanded bank

The warm-start target is `GuidanceOutput.bank_angle` at the **dispatch.rs return value**, i.e. the bank command the pilot integrates, after thermal limiter, lateral guidance, and command shaper. This is the value the NN's signed output replaces in `full_neural` mode.

For `magnitude_only` mode, the target is derived Python-side via `np.abs(y_signed)` — no second Rust capture path. This preserves the existing `magnitude_only` warm-start behavior (which the user confirms "works perfectly") with a one-line transform.

Rationale: cloning the entire guidance chain the NN replaces is the only way to give `full_neural` a meaningful starting point. Targeting only the inner controller is the bug that makes the current pipeline unusable for `full_neural`.

### Decision 2: Best-teacher-per-seed multi-supervisor

For each supervised seed, run each scheme in `[warm_start] supervisor_schemes` (default: `["ftc", "equilibrium_glide", "energy_controller", "pred_guid", "fnpag"]`) and select the trajectory with the lowest DV among `captured=True` entries. Seeds where no supervisor captures are dropped from the corpus.

`piecewise_constant` is opt-in (available but not in the default list) because it is open-loop and dispersion-agnostic, making it a weak teacher for the dispersed sims the NN must handle.

Rationale: A naive "union of all schemes on all seeds" introduces systematic per-state target contradictions (different schemes command different banks for near-identical states), which the NN learns as a soggy mean. Per-seed best-of selection gives scheme diversity *across* trajectories while keeping per-trajectory targets coherent. The 5× collection cost is a one-time tax (cached).

### Decision 3: Chunked truncated BPTT, uniform across all archs

Sequence training mechanics:

- Each per-seed trajectory is split into chunks of `warm_start.bptt_length` (default 32; must satisfy `bptt_length <= n_seq` for Transformer architectures — validated at config load).
- Chunk c+1 begins with the detached hidden state from chunk c (mirrors the PPO-GRU `ppo_update_bptt` pattern).
- Forward via the existing `V2Policy.evaluate(obs_seq, state_0, dones_seq, raw_seq)` contract, returning the predicted bank parameterization for each step in the chunk.
- Loss is MSE between the predicted parameterization and the target encoding of `y_train`:
  - `output_parameterization = "acos_tanh"`: target = `cos(y_train)`, predicted = `tanh(out[0])`.
  - `output_parameterization = "atan2_signed"`: target = `(sin(y_train), cos(y_train))`, predicted = `(out[0], out[1])` (normalized inside the loss to lie on the unit circle).
- `acos_tanh + full_neural` is rejected at config load (`acos` ∈ [0, π] cannot represent signed bank). This validation already exists for the parity-bundle config and applies unchanged.

For dense and window the chunked-BPTT loop degenerates correctly:
- Dense: no hidden state, chunk boundaries are no-ops, gradient flow is identical to shuffled-step training up to minibatch composition.
- Window: FIFO buffer initializes to zeros at chunk 0 and carries forward across chunks within a trajectory (matches Rust `LayerState::for_layer` behavior at episode start, then standard rollover).

Rationale: One code path for all six architectures. The chunk-boundary discontinuity (state is detached, so gradients don't flow across chunks) is negligible at warm-start — the supervised loss has high SNR, we are cloning a teacher, not solving for tight gradient flow across 500 steps.

### Decision 4: Optimizer interaction (bound widening + per-algo seeding)

**Conditional bound widening.** When `warm_start_from` is set, `nn_param_specs_from_v2` is called with `bound_multiplier = warm_start.bound_multiplier` (default `4.0`, vs the no-warm-start default `2.0`). The widening applies globally to the run, not just to the seeded chromosome — the entire population (jittered copies + any subsequent operators) operates in the wider normalized space. This is correct: warm-start by definition seeds far from random Xavier sampling, so wider bounds give the optimizer room to explore around the teacher.

Per-layer slabs that already carry generous bounds (LSTM `bias_ih` forget-bias slot at `2.0 * bound_multiplier`, Mamba `dt_proj_b` at HiPPO-style inv-softplus centers, Transformer LN gamma at `±0.01 * bound_multiplier` around 1.0) keep their relative widening — only the base multiplier moves.

**Hard clip-rate error.** After normalizing the trained flat weights into [0, 1], if more than 5% of entries clipped to the bounds, raise a hard error pointing the user at `warm_start.bound_multiplier`, `n_epochs`, or `lr`. Silent half-clipped chromosomes are the failure mode this guards against — they superficially succeed but defeat the entire warm-start. The existing 5% warning becomes an error.

**Per-algorithm initial-population seeding** in `train.py`:

- **GA / DE / PSO**: chromosome replicated to `n_pop`; each individual gets `N(0, warm_start.jitter)` (default `0.02`) additive noise in normalized space; values re-clipped to [0, 1] post-jitter.
- **CMA-ES**: chromosome used as initial mean `x0`; initial step size `sigma0 = warm_start.cmaes_sigma0` (default `0.1`, vs the default `0.3`) to keep early samples close to the teacher.

**Gen-0 validation baseline.** Before the first algorithm.next(), run validation MC (existing `validation_n_sims` pool, same seeds the validation gate already uses) on the bare warm-started chromosome and log the result as `val_baseline` in the JSONL log. This catches encoder/extraction bugs (where the chromosome decodes to something nonsensical) and gives the user a quantitative "did warm-start help?" baseline before any optimization.

**Resume guard.** `_check_resume_chromosome_shape` already enforces chromosome-width compatibility. Bound widening changes the per-dimension range but not the chromosome width, so resume is compatible.

### Decision 5: Rust↔Python data contract and orchestration

**Rust `collect_supervised` (PyO3 binding `aerocapture-py/src/lib.rs`):**

- Capture point in `simulation/tick.rs`: change from `guidance_out.pre_lateral_magnitude` to `guidance_out.bank_angle` (the final dispatch output, signed). The accumulator field in `runner.rs::SimState` (`supervised_trace: Vec<(Vec<f64>, f64)>`) keeps its shape; only the f64 payload changes meaning.
- Return shape: change from `(Py<PyArray2>, Py<PyArray1>)` to `Vec<PyDict>` (one dict per seed) with keys `seed: int`, `X: ndarray[T_i, 21]`, `y_signed: ndarray[T_i]`, `dv: float`, `captured: bool`. Trajectory boundaries preserved. `dv` and `captured` are extracted from the per-sim final record (`dv_total_m_s` and `ifinal == 3 && ecc < 1.0 && energy < 0.0` respectively, mirroring `BatchResults` construction in `runner.rs`).
- One caller (`warm_start.py`), atomic breaking change.

**Python orchestration (`aerocapture/training/warm_start.py`):**

```
for scheme in cfg.warm_start.supervisor_schemes:
    params_path = cfg.warm_start.params_paths.get(scheme,
                  Path(f"training_output/{scheme}/best_params.json"))
    overrides = _build_overrides_for_source(json.load(params_path))
    results_by_scheme[scheme] = _aero_rs.collect_supervised(
        toml_path=cfg.sim.toml_config,
        seeds=warm_start_seeds,
        overrides=overrides,
        scheme=scheme,
    )
best_per_seed = _select_best_teacher_per_seed(results_by_scheme)
```

Per-scheme `best_params.json` loading uses the existing `_build_overrides_for_source` helper (prefix routing for `lateral.` / `exit.` / `nav.` / `thermal.` / `shaping.` → TOML dot-paths). FTC's `best_params.json` is the source of truth for routing convention; all five default supervisors share the same prefix layout.

**`_select_best_teacher_per_seed`:**

```
def _select_best_teacher_per_seed(results_by_scheme):
    selected = []
    for seed in sorted({r["seed"] for rs in results_by_scheme.values() for r in rs}):
        candidates = [
            (scheme, r) for scheme, rs in results_by_scheme.items()
            for r in rs if r["seed"] == seed and r["captured"]
        ]
        if not candidates:
            continue  # all supervisors failed this seed; drop
        scheme, r = min(candidates, key=lambda sr: sr[1]["dv"])
        selected.append({"scheme": scheme, **r})
    return selected
```

If the selected corpus has fewer than `max(20, n_warm_seeds // 4)` trajectories, raise a clear error (the supervisor pool is too weak for this mission's dispersions).

**Python per-arch flat-weight extraction (`_policy_to_flat_weights_v2`):**

Walk `policy.layers`, dispatch on layer type via a new `to_flat(self) -> np.ndarray` method on each `rl/layers/<type>.py` module. The flat order **must** mirror Rust's `LayerWeights::to_flat`:

- **Dense** (`dense.py`): `[W row-major, b]`.
- **GRU** (`gru.py`): `[weight_ih row-major, weight_hh row-major, bias_ih, bias_hh]` (3H gate concat order r/z/n, PyTorch `nn.GRUCell` convention).
- **LSTM** (`lstm.py`): `[weight_ih row-major, weight_hh row-major, bias_ih, bias_hh]` (4H gate concat order i/f/g/o, PyTorch `nn.LSTMCell` convention).
- **Window** (`window.py`): empty `np.array([])` — zero trainable params.
- **Transformer** (`transformer.py`): `[w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o, w_ffn1, b_ffn1, w_ffn2, b_ffn2, ln1_gamma, ln1_beta, ln2_gamma, ln2_beta]` (all 2D matrices row-major).
- **Mamba** (`mamba.py`): `[x_proj_w row-major, dt_proj_w row-major, dt_proj_b, a_log row-major, d_skip]`.

The existing dense-only `_policy_to_flat_weights` function is replaced. Each Python layer module's `to_flat` becomes the canonical Python-side mirror of the Rust `LayerWeights::to_flat`. A round-trip equivalence test (Python `to_flat` → `aerocapture_rs.flat_weights_to_json` → Rust load → `aerocapture_rs.nn_forward`) verifies the order on each layer type.

### Decision 6: Config UX

New TOML section `[warm_start]`:

```toml
[warm_start]
supervisor_schemes = ["ftc", "equilibrium_glide", "energy_controller", "pred_guid", "fnpag"]
bptt_length = 32          # must be <= n_seq for Transformer architectures
n_warm_seeds = 200
n_epochs = 10
bound_multiplier = 4.0
jitter = 0.02
cmaes_sigma0 = 0.1

[warm_start.params_paths]   # optional per-scheme path overrides
# ftc = "training_output/ftc_v2/best_params.json"
```

The existing `[guidance.neural_network] warm_start_from` key remains the **primary supervisor for scaffolding**: when `optimize_scaffolding = true`, its `best_params.json` provides the 17-slot scaffolding tail of the warm-started chromosome. If absent, defaults to the first scheme in `supervisor_schemes` (typically `training_output/ftc/best_params.json`).

All new keys are optional with the documented defaults above. **Warm-start activation is unchanged**: it triggers iff `[guidance.neural_network] warm_start_from` is set, exactly as today. The presence or absence of `[warm_start]` only controls supervisor list and tunables — it does not gate warm-start on or off. No legacy fallback path: when warm-start activates, it runs the new multi-supervisor + chunked-BPTT pipeline regardless of whether `[warm_start]` is present (defaults applied).

**Behavior change for existing `magnitude_only` configs**: the target signal moves from `pre_lateral_magnitude` (today) to `abs(final_signed_bank)` (after this spec), which differs whenever the thermal limiter ramps the magnitude down (high heat flux/load fractions). This is a *correctness improvement* — the magnitude_only NN now clones the actual post-limiter command its `.abs()` output replaces — but it is not bit-identical. The equivalence regression target (see Test Plan) is "at least as good", not "bit-identical".

### Decision 7: n_epochs as a single knob with loss visibility

Keep `n_epochs = 10` as the default. Add:

1. Per-epoch MSE logged to `<save_dir>/warm_start_loss.json` as a list of `{"epoch": int, "mean_mse": float, "n_chunks": int}` entries.
2. End-of-training one-line summary printed to stdout: `"[warm_start] supervised MSE: <initial> → <final> over <n_epochs> epochs"`.

No per-arch defaults, no validation-loss early stop. The user inspects the loss curve and tunes `n_epochs` per their experimental loop. Documented rationale in code: too many epochs overfits the teacher's quirks, narrowing the basin GA must explore out of — more epochs is not strictly better.

## Cache key

Extended `_cache_key` includes:

```
{
    "architecture": cfg.network.architecture,
    "input_mask": cfg.network.input_mask,
    "output_parameterization": cfg.network.output_parameterization or "atan2_signed",
    "optimize_scaffolding": bool(cfg.network.optimize_scaffolding),
    "toml_config": str(cfg.sim.toml_config),
    "supervisor_schemes": sorted(cfg.warm_start.supervisor_schemes),
    "supervisor_params": {
        scheme: {"path": str(p), "mtime": p.stat().st_mtime}
        for scheme, p in resolved_supervisor_paths.items()
    },
    "scaffolding_source_path": str(scaffolding_source_path),
    "scaffolding_source_mtime": scaffolding_source_path.stat().st_mtime,
    "n_warm_seeds": cfg.warm_start.n_warm_seeds,
    "n_epochs": cfg.warm_start.n_epochs,
    "bptt_length": cfg.warm_start.bptt_length,
    "bound_multiplier": cfg.warm_start.bound_multiplier,
    "mode": cfg.guidance.neural_network.mode,  # "magnitude_only" or "full_neural"
}
```

Cache hit means **all** keys match. Any change to architecture, input mask, output param, supervisor list, any per-scheme params file (path or mtime), training hyperparameters, or routing mode invalidates the cache. The persisted artifact remains `<save_dir>/warm_start_chromosome.npy` + `<save_dir>/warm_start_cache_key.json`, joined by `<save_dir>/warm_start_loss.json` for per-epoch loss.

## Failure modes

- **Supervisor `best_params.json` missing**: hard `FileNotFoundError` at warm-start start, naming the missing scheme + expected path, before any simulation runs.
- **Single supervisor errors at runtime** (e.g. simulator crash on a specific seed): warn + skip that `(scheme, seed)` pair; proceed with remaining schemes for that seed. If after all schemes the seed has no captures, the seed is dropped by `_select_best_teacher_per_seed`.
- **Zero captures across all supervisors for all seeds**: clear error message advising the user to widen MC dispersions, check their TOML, or pick a different supervisor pool.
- **Supervised corpus too small** (< `max(20, n_warm_seeds // 4)` trajectories): clear error, same advice.
- **Clip rate > 5% after encoding**: hard error pointing at `warm_start.bound_multiplier`, `n_epochs`, `lr`.
- **`bptt_length > n_seq`** for any Transformer layer: rejected at config load.
- **`acos_tanh` + `full_neural`**: rejected at config load (existing validation).

## Test plan

**Equivalence gate (regression target, "at least as good").** Because the `magnitude_only` target signal changes from `pre_lateral_magnitude` to `abs(final_signed_bank)` (post-thermal-limiter), bit-identical convergence is not expected and not desired. Instead: pre-refactor and post-refactor both warm-start the consolidated `msr_aller_nn_train_consolidated.toml` with `supervisor_schemes = ["ftc"]` (matches the old FTC-only behavior), run 20 GA generations under the same fixed seed, and assert that post-refactor best validation RMS is **less than or equal to** pre-refactor + 5% absolute slack. A regression beyond that slack indicates the new target signal materially hurts the magnitude_only basin — flag and investigate before merging.

**Per-architecture smoke tests** (`tests/test_warm_start_per_arch.py`, `@pytest.mark.slow`, python-pyo3 CI):

For each of the 6 architectures (dense, window, gru, lstm, transformer, mamba):
1. Use the corresponding small/reduced training TOML (e.g. matching the existing per-arch smoke configs).
2. Run `build_warm_start_chromosome` end-to-end (2 supervisors, 8 seeds, 1 epoch).
3. Assert `warm_start_chromosome.npy` exists with the expected width matching `nn_param_specs_from_v2(architecture, bound_multiplier=4.0)`.
4. Decode the chromosome via the inverse of `encode_to_normalized`, write to a temporary v2 JSON via `aerocapture_rs.flat_weights_to_json`, and assert `aerocapture_rs.nn_forward` returns a finite output for a sample input.

**`to_flat` round-trip test** (`tests/test_v2_to_flat_roundtrip.py`, fast):

For each layer type, construct a `V2Policy` with that layer, randomize weights, extract via the new `to_flat` method, serialize via `aerocapture_rs.flat_weights_to_json`, load back via `aerocapture_rs.nn_forward`, and verify the Rust-side forward matches the Python-side `V2Policy.evaluate` to machine epsilon (< 1e-10) on a sample input sequence. This is the contract that protects against silent flat-order divergence.

**Multi-supervisor selection** (`tests/test_warm_start_selection.py`, fast):

Mock `_aero_rs.collect_supervised` to return two synthetic schemes with known per-seed DVs (scheme A wins seeds 0–9, scheme B wins seeds 10–19). Assert `_select_best_teacher_per_seed` returns scheme A for seeds 0–9 and scheme B for seeds 10–19. Assert seeds with no captures across both schemes are dropped.

**Per-algorithm seeding** (`tests/test_warm_start_optimizer_seeding.py`, fast):

Construct a synthetic chromosome (e.g. 100-dim, all 0.5). For each algorithm:
- **GA / DE / PSO**: assert initial population is `n_pop` × 100 with row-wise mean ≈ 0.5 and row-wise std ≈ `jitter`.
- **CMA-ES**: assert `x0 == chromosome` to machine epsilon, `sigma0 == warm_start.cmaes_sigma0`.

**Failure-mode tests** (`tests/test_warm_start_failures.py`, fast):

- Missing supervisor `best_params.json` → `FileNotFoundError` with the right scheme name and path.
- All-supervisors-fail-on-seed via mocked `captured=False` → seed dropped, count logged.
- Zero captures everywhere → clear error.
- Clip rate > 5% → hard error.
- `bptt_length=128` + Transformer with `n_seq=64` → config-load `ValueError`.

**Cache invalidation test** (`tests/test_warm_start_cache.py`, fast):

- Touch a supervisor's `best_params.json` (mtime bump) → cache miss → recompute.
- Change `bound_multiplier` → cache miss.
- Re-run with no changes → cache hit, no recomputation (assert `_aero_rs.collect_supervised` not called).

**End-to-end gen-0 validation log** (`tests/test_warm_start_validation_baseline.py`, `@pytest.mark.slow`):

Run a 1-generation training with warm-start on, assert the JSONL log contains a `val_baseline` entry before generation 0 with a finite mean DV.

## Implementation order

The implementation plan (next skill: writing-plans) will sequence these commits:

1. Rust: change `collect_supervised` capture point (pre_lateral_magnitude → final bank_angle), update PyO3 binding return shape to `list[dict]`, regenerate any affected goldens.
2. Python: add `to_flat` method to each `rl/layers/<type>.py` module; add `_policy_to_flat_weights_v2` dispatcher; round-trip test.
3. Python: add `WarmStartConfig` dataclass + `[warm_start]` TOML parser; cache key extension.
4. Python: rewrite `build_warm_start_chromosome` for multi-supervisor + chunked-BPTT; add `_select_best_teacher_per_seed`, `_chunked_bptt_train`, per-arch smoke tests.
5. Python: per-algorithm seeding in `train.py` (`create_algorithm` or initial-pop construction); gen-0 validation baseline log.
6. Python: bound widening dispatch (`nn_param_specs_from_v2` accepts `bound_multiplier` already; thread `warm_start.bound_multiplier` through `train.py`).
7. Wire-up + integration: training configs for each arch updated with `[warm_start]` block; deprecation warning when `[warm_start]` absent on an NN scheme.
8. Final: `smart-commit` over the whole branch.

## References

- `docs/superpowers/specs/2026-05-07-nn-ftc-parity-bundle-design.md` — parity bundle that introduced the original FTC-only warm-start, output parameterizations, optimize_scaffolding.
- `docs/superpowers/specs/2026-04-17-stateful-nn-runtime-infrastructure-design.md` — Phase 0 v2 JSON, LayerWeights trait, NnState plumbing.
- `docs/superpowers/specs/2026-04-18-phase-2a-lstm-mvp-design.md` — LSTM bias slab + activation-aware init pattern; the bound-multiplier widening rationale generalizes from this.
- `src/python/aerocapture/training/warm_start.py` — current pipeline being replaced.
- `src/rust/aerocapture-py/src/lib.rs` — `collect_supervised` PyO3 binding.
- `src/rust/src/simulation/tick.rs` — supervised_trace capture point (line 166).
- `src/python/aerocapture/training/rl/policy.py` — `V2Policy.evaluate` chunked-BPTT contract.
