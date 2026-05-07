# NN-vs-FTC Parity Bundle (3-fix design)

Date: 2026-05-07
Status: design (awaiting user spec review → writing-plans)
Branch: feature/magnitude_only (current)

## Background and motivation

Empirically, no neural-network guidance variant has matched FTC's validation
RMS on MSR aller (best validation RMS, sqrt-transformed cost; lower is
better):

| Scheme                                | Best val RMS | Gen   | Mode             |
| ------------------------------------- | ------------ | ----- | ---------------- |
| FTC (jointly optimized 26 params)     | **11.72**    | 1559  | -                |
| neural_network_gru_pso                | 12.52        | 4261  | magnitude_only   |
| neural_network_gru_pso_magonly        | 14.46        | 1514  | magnitude_only   |
| neural_network (dense, consolidated)  | 14.69        | 704   | magnitude_only   |

The gap is structural, not a wiring bug. The dispatch layer correctly routes
NN-magnitude_only through FTC's exit / lateral / thermal / shaping
scaffolding (verified by `magnitude_only_mode_routes_through_thermal_limiter`
in `dispatch.rs`). The 21-element NN input vector `build_nn_input` even
includes the closed-loop FTC exit-bank teacher signal at index 20.

Three asymmetries explain the gap:

1. **Joint vs frozen scaffolding.** FTC's GA jointly tunes 26 parameters: 9
   capture-phase params plus 17 scaffolding params (`_NAV_PARAMS` +
   `_LATERAL_PARAMS` + `_EXIT_PARAMS` + `_THERMAL_LIMITER_PARAMS` +
   `_SHAPING_PARAMS`, defined in `param_spaces.py`). The NN trainer optimizes
   only the network weights and consumes those 17 scaffolding params
   **frozen at FTC's joint optimum** (copied verbatim from
   `training_output/ftc/best_params.json` into the NN training TOMLs at
   lines 64-85 of `msr_aller_nn_train_consolidated.toml` and equivalents).
   Those values were tuned for FTC's specific bank-command profile, not for
   whatever the NN produces. The NN must drive an actuator pipeline tuned
   for someone else's signal.

2. **Action-space waste in `magnitude_only`.** The NN emits
   `atan2(out[0], out[1]) ∈ (-π, π]`; `dispatch.rs:194-198` then takes
   `signed.abs()`. Half the output range is dead weight — the sign of
   `out[0]` is irrelevant. The optimizer wastes capacity learning that
   degenerate dimension.

3. **No warm-start.** The NN starts from random Xavier-initialized weights;
   FTC starts gen 0 with a closed-form law that already produces sensible
   bank commands. With pop ~ 60-64 on 1266+ params, NN exploration is
   anemic.

This spec lands one bundled fix for all three, gated behind TOML knobs so
existing trained NNs and existing configs are unaffected by default.

## Scope

In scope (all under one spec, three commits in dependency order):
- Fix A: joint scaffolding optimization (NN chromosome gains the 17
  scaffolding params, seeded at FTC's GA optimum + jitter).
- Fix B: replace the wasteful `atan2(out0, out1).abs()` parameterization
  with `acos(tanh(out0))` (single output), gated on `magnitude_only` mode.
- Fix C: behavioural cloning warm-start from FTC, sourced via a new PyO3
  `collect_supervised` helper, supervised pre-train in PyTorch, encoded
  back to PSO chromosome and used as the population seed.

Out of scope:
- RL/PPO training path (`aerocapture/training/rl/train.py`,
  `BatchedSimulation`). PSO empirically beats PPO on this problem. RL can
  pick up the same patterns later if needed.
- Other distillation sources beyond FTC (PiecewiseConstant, FNPAG).
  `collect_supervised` takes a `scheme` arg so future extensions are
  trivial; configs only document FTC for now.
- Architecture changes beyond the last-layer width adjustment that
  `acos_tanh` requires.
- Changes to the cost function, dispersion sampling, validation/final-eval
  seed pools, or any other train-loop infrastructure.

## Backward compatibility

All three fixes are **TOML opt-in, default off**. Existing configs and
existing `best_model.json` files keep training and deploying bit-identically.

| Knob (under `[guidance.neural_network]`) | Default        | Fix |
| ---------------------------------------- | -------------- | --- |
| `optimize_scaffolding`                   | `false`        | A   |
| `output_parameterization`                | `"atan2_signed"` | B   |
| `warm_start_from`                        | unset          | C   |

Resume after flipping any knob fails loudly (chromosome shape mismatch or
architecture mismatch) with a remediation message pointing to
`--from-scratch`.

## Fix A — joint scaffolding optimization

**TOML knob:** `optimize_scaffolding = true`.

**Failure modes (validated at config load):**
- If `optimize_scaffolding = true` and `training_output/ftc/best_params.json`
  is missing → fail with: "joint scaffolding requires FTC training output;
  run `./train_all.sh ftc` first".

**Code changes:**

1. `param_spaces.py`:
   ```python
   _NN_SCAFFOLDING_PARAMS: list[ParamSpec] = [
       *_NAV_PARAMS,
       *_LATERAL_PARAMS,
       *_EXIT_PARAMS,
       *_THERMAL_LIMITER_PARAMS,
       *_SHAPING_PARAMS,
   ]
   ```
   17 specs total; same instances FTC uses, same bounds.

2. `train.py`, NN parameter-spec branch around line 326: when the resolved
   `optimize_scaffolding` flag is true, append `_NN_SCAFFOLDING_PARAMS` to
   the v2 ParamSpec list. Final chromosome layout:
   `[NN weight specs (1266), nav.density_filter_gain, nav.density_gain_max_delta,
   lateral.tau, lateral.threshold, lateral.min_reversal_interval,
   lateral.lateral_activation, lateral.lateral_inhibition, lateral.max_reversals,
   exit.exit_velocity_threshold, exit.exit_pdyn_margin,
   exit.exit_radial_vel_gain, exit.exit_altitude_threshold,
   thermal.heat_flux_activation, thermal.heat_load_activation,
   thermal.heat_flux_ramp_exponent, thermal.heat_load_ramp_exponent,
   shaping.max_bank_acceleration]`.

3. `problem.py::_build_overrides`: the prefix routing already handles
   `lateral.*`, `exit.*`, `nav.*`, `thermal.*`, `shaping.*` (lines 279-289).
   No change.

4. `train.py::build_initial_population_for_v2` (and the v1 path
   `create_nn_initial_population` if `optimize_scaffolding` is allowed in
   v1 configs — see open question below): extend to fill the 17
   scaffolding slots. Read `training_output/ftc/best_params.json`, encode
   each value to its `[0, 1]` slot via the existing
   `encode_to_normalized` helper, then add per-individual
   `N(0, σ_scaffold)` jitter clipped to `[0, 1]`. NN-weight slabs stay
   activation-aware Xavier as today. `σ_scaffold = 0.02` (in the
   normalized [0, 1] space — about 2% of the spec range; FTC's tuned
   values stay within reach but the population is not collapsed).

5. `train.py` end-of-training block: when `optimize_scaffolding`, write
   the optimized scaffolding values to `<save_dir>/best_params.json` —
   same filename and same JSON keys (`lateral.tau`, `exit.exit_pdyn_margin`,
   etc.) as non-NN schemes use. The NN's `best_model.json` (weights) is
   written separately as today. `compare_guidance.py` and report.py
   need a small extension: for an NN scheme whose save_dir contains
   both `best_model.json` AND `best_params.json`, build the override
   dict by routing the scaffolding keys through the existing
   prefix-routing logic (or reuse `_build_overrides` after a small
   refactor that exposes it). Without `optimize_scaffolding` the NN
   save_dir contains only `best_model.json` and the existing deploy
   path is unchanged.

**Chromosome size impact:** 1266 + 17 = 1283 for the consolidated dense
config; 6946 + 17 = 6963 for gru_pso; etc. Negligible PSO/GA overhead.

## Fix B — output parameterization

**TOML knob:** `output_parameterization = "atan2_signed" | "acos_tanh"`,
default `"atan2_signed"`.

**Failure modes (validated at config load):**
- `acos_tanh` requires the NN's last layer to have `output_size = 1` and
  activation `tanh`. Else fail with: "output_parameterization='acos_tanh'
  requires last layer output_size=1 and activation='tanh'; got
  output_size=N, activation=A".
- `acos_tanh` is only legal with `mode = "magnitude_only"`. Else fail
  with: "output_parameterization='acos_tanh' is only meaningful with
  mode='magnitude_only' (it cannot emit signed bank); use 'atan2_signed'
  for full_neural mode".

**Code changes:**

1. New Rust enum in `data/neural.rs`:
   ```rust
   #[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
   #[serde(rename_all = "snake_case")]
   pub enum OutputParam {
       #[default]
       Atan2Signed,
       AcosTanh,
   }
   ```

2. `NeuralNetModel` gains `output_param: OutputParam` field (default
   `Atan2Signed`). Persisted in v2 JSON alongside `input_mask` and
   `ablated_input`. v1 JSON loader stays backward-compatible (always
   defaults to `Atan2Signed`).

3. `gnc/guidance/neural.rs::nn_bank_angle`: dispatch on `nn.output_param`:
   ```rust
   match nn.output_param {
       OutputParam::Atan2Signed => output[0].atan2(output[1]),
       OutputParam::AcosTanh    => output[0].acos(),
   }
   ```
   Last-layer activation `tanh` already constrains `output[0] ∈ [-1, 1]`,
   so `acos` is well-defined. The dispatch's `signed.abs()` for
   `magnitude_only` mode becomes a no-op for `acos_tanh` (output already
   in `[0, π]`) — full output capacity reaches the controller.

4. **TOML/JSON ownership split** (mirrors the existing pattern for
   `[network] architecture`):
   - TOML's `[guidance.neural_network] output_parameterization` is a
     **training-time-only knob**. The trainer reads it and passes the
     value into `write_nn_json` / `flat_weights_to_json`, which embeds
     `output_param` into the v2 JSON model file.
   - The Rust simulator reads `output_param` exclusively from the JSON
     model file via `NeuralNetModel::from_v2_json`. The TOML key is
     ignored at runtime — the deployed model is fully self-describing.
   - `compare_guidance.py` and report.py work unchanged: they load the
     trained `best_model.json`, which carries the parameterization on
     disk.

5. `aerocapture-py/src/lib.rs::flat_weights_to_json`: extend signature to
   accept the parameterization choice (Python path uses this when writing
   the NN JSON during training).

6. Python ParamSpec generator: `nn_param_specs_from_v2` is unchanged when
   `output_parameterization = "atan2_signed"` (last layer 2 outputs); for
   `acos_tanh` the user simply writes `output_size = 1` in the TOML
   architecture entry, so the spec count adjusts naturally. No
   conditional logic in the generator.

**Storage migration:** existing `best_model.json` files load with default
`Atan2Signed` — no on-disk migration needed.

## Fix C — FTC behavioural-cloning warm-start

**TOML knob:** `warm_start_from = "training_output/ftc/best_params.json"`
(absent → no warm-start). The path is to a `best_params.json` for any
unsigned-magnitude scheme whose simulator behaviour we want to clone;
defaults / examples document FTC.

**Failure modes (validated at training start):**
- `warm_start_from` path missing → fail with: "warm_start_from points to
  '<path>' which does not exist".
- `warm_start_from` set but `mode != "magnitude_only"` → fail. The
  cloning targets unsigned bank magnitude, so the trained NN's behaviour
  is only well-defined when consumed as a magnitude. (Full-neural mode
  needs the NN to also emit roll sign, which FTC does not provide.)

**Rust side — new PyO3 helper:**

Add `aerocapture_rs.collect_supervised` in `aerocapture-py/src/lib.rs`:
```python
collect_supervised(
    toml_path: str,
    seeds: list[int],
    overrides: dict[str, object] | None = None,
    scheme: str = "ftc",
    sim_timeout_secs: float | None = None,
) -> tuple[np.ndarray, np.ndarray]
```
- Returns `(X, y)`. `X` shape `(n_total_ticks_across_all_seeds, 21)` of
  f64 — the raw 21-element NN input vector pre-mask, computed from
  `build_nn_input` with `nn = None`-equivalent dummy (no mask, no
  ablation). `y` shape `(n_total_ticks_across_all_seeds,)` of f64,
  `bank_angle_longitudinal` magnitude in radians (post-thermal-limiter,
  pre-lateral-sign, pre-command-shaper — the magnitude the NN must emit
  to fit the same downstream pipeline).
- Internally: runs `simulation::runner::run_for_api_with_overrides` per
  seed with the requested guidance type, hooks into the per-tick
  guidance pipeline to capture the NN input and bank magnitude. Returns
  `(0, 21)` and `(0,)` arrays cleanly when a sim crashes; the caller
  filters by `y.is_finite()`.

The hooked path needs a small adjustment to `simulation/tick.rs` (or a
parallel "trace" runner) to expose per-tick `(nn_input, bank_magnitude)`.
Cleanest implementation: extend `RunOutput` with optional
`supervised_trace: Option<Vec<(Vec<f64>, f64)>>` populated when a flag
is set. The PyO3 wrapper sets the flag, drains the trace into numpy.

**Python side — new module `aerocapture/training/warm_start.py`:**

```python
def build_warm_start_chromosome(
    cfg: TrainingConfig,
    param_specs: list[ParamSpec],
    rng: np.random.Generator,
) -> np.ndarray:
    """Run FTC over the validation seed pool, supervised-train V2Policy
    to mimic FTC's bank, encode to a normalized [0, 1] chromosome that
    fits cfg's NN architecture. Cached to <save_dir>/warm_start_chromosome.npy.
    """
```

Steps:
1. Cache check: if `<save_dir>/warm_start_chromosome.npy` and its sidecar
   JSON cache key (described in step 9 below) both exist and the cache
   key matches the current `cfg`'s architecture / input_mask /
   output_param / source path / source mtime, return the cached
   chromosome directly. Else fall through to steps 2-9.
2. Load `cfg.guidance.neural_network.warm_start_from` (path to a source
   scheme's `best_params.json`). Build the override dict via the
   prefix-routing logic copied from `problem.py::_build_overrides` (or
   reuse it directly via a small refactor that exposes the routing
   function).
3. Pull `n_warm_seeds = 200` deterministic seeds from
   `make_reserved_seeds(base_mc_seed, WARM_START_SEED_OFFSET, 200)`,
   where `WARM_START_SEED_OFFSET = 4_000_000` is a new constant added
   to `evaluate.py` alongside the existing `VALIDATION_SEED_OFFSET = 1M`,
   `FINAL_EVAL_SEED_OFFSET = 2M`, and the RL-training-only
   `RL_TRAINING_SEED_OFFSET = 3M`. The 4M offset gives a stream
   disjoint from all four other reserved pools by construction
   (Welford-bounded collision probability is negligible for
   2^31 / 4 ranges with n ~ 200-1000 draws). This matters: if warm-start
   data overlapped with validation seeds, the validation gate would
   measure performance on seeds the NN was supervised-trained on —
   sample contamination that would silently flatter every reported metric.
4. Call `aerocapture_rs.collect_supervised(toml_path, seeds, overrides,
   scheme="ftc")`. Apply `cfg.network.input_mask` (or default
   `[0..16]`) to `X` to get the masked input the runtime NN actually sees.
5. Build `V2Policy` from the architecture (same dispatch as the RL path,
   `aerocapture/training/rl/policy.py::V2Policy`). For `acos_tanh`
   parameterization, the policy emits 1 output through `tanh`. For
   `atan2_signed`, the policy emits 2 outputs through `asinh` (default)
   or whatever the architecture prescribes.
6. Supervised pre-train with PyTorch Adam:
   - For `acos_tanh`: target `t = cos(y)`, loss `MSE(tanh(out0), t)`.
   - For `atan2_signed`: target `t = (sin(y), cos(y))`, loss
     `MSE((out0_act, out1_act), t)`. (`atan2(sin θ, cos θ) = θ`.)
   - 10 epochs, batch size 256, LR 1e-3, no weight decay, no scheduler.
   Should converge on Mars-aller in seconds.
7. Extract physical weights from `policy.state_dict()` in the canonical
   flat order matching `nn_param_specs_from_v2` (per-layer flat order
   defined by the Rust `LayerWeights` trait, mirrored in
   `aerocapture/training/rl/export.py::export_v2_policy_to_json`).
   The flat-extract path already exists for the RL→PSO bridge; factor
   it into a reusable helper `policy_to_flat_weights(policy, architecture)
   -> np.ndarray[float64]` in `warm_start.py` (or in `rl/export.py`
   alongside the JSON export so the two paths share an inverse).
8. Encode each physical weight to `[0, 1]` via `encode_to_normalized`.
   Concatenate with the scaffolding chromosome (FTC's GA optimum
   encoded, when `optimize_scaffolding` is on; else nothing) — the
   final chromosome shape exactly matches `param_specs` produced by
   `train.py`'s NN branch under the same TOML knobs.
9. Persist to `<save_dir>/warm_start_chromosome.npy` along with a sidecar
   JSON cache key:
   ```json
   {
     "architecture": <list of layer dicts from cfg.network.architecture>,
     "input_mask":   <list[int] from cfg.network.input_mask>,
     "output_param": "atan2_signed" | "acos_tanh",
     "source_path":  <warm_start_from value>,
     "source_mtime": <float, mtime of warm_start_from>,
     "n_warm_seeds": 200,
     "n_epochs":     10
   }
   ```
   Cache hit requires every key to match. Architecture changes,
   re-running FTC, or flipping `output_parameterization` invalidate the
   cache automatically.

**Population seeding:** `train.py::build_initial_population_for_v2` extends:
- If `warm_start_from` is set, build the warm-start chromosome (via
  `build_warm_start_chromosome`).
- Replace the NN-weight slab of every individual with
  `warmed_chromosome[NN-slab] + N(0, σ_warm)` jitter clipped to `[0, 1]`.
- Scaffolding slab still seeded by fix A (FTC optimum + σ_scaffold jitter)
  when `optimize_scaffolding` is on; otherwise `optimize_scaffolding` is
  off and the scaffolding stays in TOML.
- `σ_warm = 0.02` (same magnitude as `σ_scaffold` for consistency; small
  enough that the cloned behaviour survives, large enough for PSO to
  retain diversity).

## Combined behaviour with all three knobs on

A new training config `configs/training/msr_aller_nn_joint_train.toml`
(or similar) with all three knobs flipped produces:
- Chromosome: 1266 NN weights + 17 scaffolding params = 1283 dims, all
  in `[0, 1]`.
- Initial population: NN-weight slabs cloned from FTC behaviour via
  supervised pre-train + small jitter; scaffolding slabs at FTC's GA
  optimum + small jitter.
- Runtime: NN's single tanh output → `acos` → `[0, π]` magnitude → fed
  into FTC's tuned scaffolding (lateral, thermal, exit, shaping) which
  the optimizer is now jointly refining.

PSO begins essentially at "FTC re-encoded as an NN" and refines from
there. Any improvement is a strict win over FTC; any regression
indicates a problem in the training setup (cost function, seed pool,
etc.) rather than the structural disadvantage we are removing.

## Testing

Unit (Python):
- `param_spaces` test: `_NN_SCAFFOLDING_PARAMS` is exactly the 17 specs
  in the documented order (regression guard against future reordering).
- `encode_to_normalized` round trip on FTC's `best_params.json` values
  against `_NN_SCAFFOLDING_PARAMS` — verifies all 17 values land
  inside their [p_min, p_max] ranges.

Unit (Rust):
- `acos_tanh` parameterization end-to-end: zero-weight 21→1 NN,
  bias = 0.5, expect `bank = acos(tanh(0.5))` to within 1e-12.
- `OutputParam` JSON round-trip (v2 save/load).
- Config-load validation rejects:
  - `acos_tanh` + last-layer `output_size = 2`.
  - `acos_tanh` + last-layer activation `asinh`.
  - `acos_tanh` + `mode = "full_neural"`.

Integration (cross-language, `tests/test_v2_rust_python_equivalence.py`):
- New equivalence test: 21→1 NN with `acos_tanh` parameterization, both
  Rust and Python forward → max abs diff < 1e-10 across 100 random
  inputs.

Integration (Python):
- `collect_supervised` smoke test: 1 seed, FTC scheme → returns finite
  `(X, y)` of shape `(n_ticks, 21)` and `(n_ticks,)` with
  `0 <= y <= π` and `np.isfinite(X).all()`.
- Warm-start smoke test (`@pytest.mark.slow`): 4-individual 1-gen PSO
  training run through `train.py` with all three knobs on, asserts:
  - Initial population's NN forward-pass on the supervised set produces
    `MSE < 0.05 rad²` against FTC bank.
  - `<save_dir>/warm_start_chromosome.npy` exists after the run.
  - Resume from this run with the same config succeeds.

Regression:
- All existing guidance golden tests pass unchanged (default knobs are
  off, behaviour is bit-identical).
- One new golden: `acos_tanh` + `optimize_scaffolding=false` +
  `warm_start_from` unset, fixed seed → trajectory matches a recorded
  golden file. Catches accidental changes to the new code paths.

## Open questions

1. v1-vs-v2 architecture support for `optimize_scaffolding`: do we allow
   it on the v1 dense-only path (`layer_sizes` + `activations`) or
   require v2 `[[network.architecture]]`? Cleaner to require v2 so the
   chromosome layout helper is single-sourced. **Default: require v2
   when `optimize_scaffolding = true`; fail loudly otherwise.**
2. Should `collect_supervised` return separate per-seed `(X, y)` tuples
   (so the supervised loader can shuffle by seed) or a flat
   concatenated pair? **Default: flat concatenated; shuffle is on the
   Python side.**
3. Do we also need to expose `n_warm_seeds` as a TOML knob?
   **Default: hard-code at 200; revisit if validation shows it matters.**

## Implementation order (for the writing-plans skill)

1. Fix A first (smallest blast radius, biggest measurable impact):
   - `_NN_SCAFFOLDING_PARAMS` definition.
   - Train.py NN-branch param-spec extension behind the knob.
   - Initial population scaffolding seed from FTC's `best_params.json`.
   - Tests, then commit.
2. Fix B second (depends on A only via the new training config that
   uses both):
   - Rust `OutputParam` enum + JSON v2 round trip + config validation.
   - Python forward path / dispatch already handled — only the
     Rust side branches.
   - Tests, then commit.
3. Fix C last (depends on A's chromosome layout and B's
   parameterization being available so the PyTorch policy can mirror
   the deploy path):
   - Rust `collect_supervised` + tick-trace plumbing.
   - Python `warm_start.py` module.
   - Population-seeding extension in `train.py`.
   - Tests, then commit.
4. Final commit: a new `configs/training/msr_aller_nn_joint_train.toml`
   that flips all three knobs, plus an entry in `train_all.sh` aliases.

## Final step

Per project convention, the implementation plan's final step invokes the
`smart-commit` skill against the whole branch.
