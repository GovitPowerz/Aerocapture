# TOML configuration (`configs/`)

TOML files are the only supported input format, for the CLI (`aerocapture <config.toml>`), the
PyO3 entry points and the training pipeline. Layout:

- `configs/planets/` — planet physical constants (mu, radii, omega, J2/J3/J4)
- `configs/missions/` — shared per-planet base configs (inherit from `planets/`)
- `configs/nominal/` — nominal simulation configs
- `configs/training/` — training configs (one leaf per scheme / cell) + the shared bases
- `configs/test/` — golden test configs (`tests/reference_data/rust_golden/`)

Rust owns the schema: `src/rust/src/config.rs` parses and validates
([src/rust/README.md](../src/rust/README.md)); Python resolves the same `base` chain in
`toml_utils.load_toml_with_bases`, with `aerocapture_rs.load_config` as the parity oracle.
Decisions: ADR-0008 (config keys are declared once), ADR-0006 (per-draw noise is the default
regime), ADR-0003 (the noise-seeding knob). The "Silent Config Wiring" lesson in the root
`CLAUDE.md` is the reason the validation below is strict.

## Base inheritance

A `base` key (string or array of strings) references parent TOML files, resolved relative to the
declaring file. The loader deep-merges bases left-to-right, then overlays the child's own keys.
Mission-level content (entry, vehicle, aero, flight, orbit, incidence, atmosphere paths) lives in
`configs/missions/mars.toml` or `earth.toml`; common training settings (MC dispersions, cost
function, optimizer defaults) in `configs/training/common.toml`; shared NN-training defaults in
`configs/training/nn_common.toml` (`[guidance] type = "neural_network"`) and
`nn_ftc_scaffolding.toml` (the frozen FTC capture/exit scaffolding block shared by
`magnitude_only` and `full_neural` NN configs); shared RL defaults in
`configs/training/rl_common.toml` (`[rl]`/`[rl.reward]`/`[rl.ppo]`/`[rl.sac]`; the GRU/LSTM PPO
configs inherit it and override only per-arch deltas like `bptt_length`/`target_kl`). Each leaf
specifies only its overrides (guidance type, n_sims, results_suffix, architecture, input_mask).

Arrays REPLACE (not merge) under deep-merge, so `[[network.architecture]]` and `input_mask` must
be fully specified per leaf; tables merge key-by-key, so a leaf overrides individual scaffolding
keys (e.g. `magnitude_only` overwrites the tuned exit gains). Both Rust (`resolve_toml_bases()` in
`config.rs`) and Python (`load_toml_with_bases()` in `toml_utils.py`) implement the same rule.
`[success]` is retired: no reader, the root keeps an opaque pass-through for old generated TOMLs.

## Validation is strict

An unknown key in any Rust-owned section (typo, wrong section, stale knob) hard-errors at load
(serde's `unknown field` message, root section names included; the two flatten sections report
their own message, custom MC keys checked regardless of level). The seven Python-only root
sections (`[cost_function]`, `[optimizer]`, `[warm_start]`, `[rl]`, `[corridor]`, `[reference]`,
`[checkpoints]`) pass through Rust and are checked by their Python readers
(`toml_utils.reject_unknown_keys`). Also hard errors at load: an unknown `[monte_carlo.<domain>]
level` (a typo `"of"` for `"off"`) or custom-dispersion key under ANY active level; an unknown
`[navigation] mode`; `guidance.type = "neural_network"` without a model (`[data] neural_network`
or an injected one); mismatched `[incidence]` altitudes/angles lengths; wind `scale_min >
scale_max` (equal bounds are legal and pin the scale); negative/non-finite EKF sensor sigmas; a
non-monotonic atmosphere altitude column; an unrecognized `[integration] mode`; a wrong-sized
embedded or TOML NN normalization block; a decoder / mode / architecture disagreement (below). The
8 per-domain custom-override blocks go through one `resolve_domain` helper in `data/mod.rs` (wind
and density_perturbation stay inline for their unconditional-read / post-validation quirks).

Gates: `src/rust/tests/config_loading.rs::every_committed_config_parses_and_validates` (every
file under `configs/**`), `config_tests.rs::toml_keys_reachable_in_sim_data`, and
`tests/test_config_normalization_blocks.py` for the committed `[network.normalization]` blocks.

## Sections

**`[planet]`** (name, mu, equatorial_radius, polar_radius, omega, j2, j3, j4), inherited from
`configs/planets/*.toml`. J3 and J4 default to 0.0 (J2-only). A new planet is a new preset file,
no Rust change.

**`[mission] phase`**: `"full"` (default, capture + exit with automatic transition),
`"capture_only"` (phase 1 throughout), `"exit_only"` (phase 2 throughout, for testing), or
`"preprogrammed"` (same as full). The transition fires after bounce when velocity drops below
`exit_velocity_threshold`.

**`[simulation]`**: `max_time` (default 3000.0 s) is a hard wall against runaway simulations.

**`[onboard_atmosphere]`**: `mode = "identical"` (truth table), `n_segments = N` (auto-fit N
piecewise exponential segments from truth), or explicit `segments = [...]`. Default: auto-fit with
5 segments. Navigation and guidance query the onboard model; physics propagates on the truth table
with MC dispersions.

**`[integration]`**: `mode = "fixed"` (default, Gill-variant RK4) or `"adaptive"`
(Dormand-Prince 4(5) with error control: `rtol` default 1e-6, `initial_dt` 0.1 s, `min_dt` 1e-6
s, `max_dt` default `periods.integration`). The adaptive integrator sub-steps within each outer
GNC tick; GNC cadences are unchanged.

**`[flight.constraints]`** (mission TOMLs): the authoritative limits `max_heat_flux` (kW/m²),
`max_load_factor` (g), `max_dynamic_pressure` (kPa), `max_heat_load` (kJ/m²), used by the cost
function, report violation rates, chart limit lines, trajectory classification and the
feasibility gate (ADR-0005).

**`[corridor]`** (mission TOMLs): asymmetric restricted corridor bounds
`delta_za_restricted_low`, `delta_za_restricted_high` (km).

**`[data]`**: table paths (`atmosphere`, `wind_table`, `reference_trajectory`, `neural_network`).
The three table-reading training TOMLs (ftc, energy_controller, pred_guid) wire
`data.reference_trajectory` to the mission-optimized `training_output/mars/ref_trajectory.dat`
explicitly; `missions/mars.toml` points at the legacy `data/reference_trajectory/msr_aller.dat`,
which the nominal/test configs and the goldens keep. `train.py::check_ref_trajectory_wiring`
refuses a ref-tracking scheme whose resolved config does not point at the mission's optimized
reference.

**`[navigation]`**: `mode` (`bias` or `ekf`), the density filter `density_filter_gain` and
`density_gain_max_delta` (they affect every guidance scheme via `estimator.rs`; GA-routable with
the `nav.` prefix), EKF sensor sigmas. Navigation-level tunables belong here, never in
`[guidance.ftc]`.

### Monte Carlo

**`[monte_carlo.<domain>]`**: every domain uses level presets (`"off"`, `"low"`, `"medium"`,
`"high"`, `"custom"`), custom keys overriding the preset.
`[monte_carlo.density_perturbation]` configures time-varying (Gauss-Markov / Ornstein-Uhlenbeck)
density noise on top of the static density bias: `level` presets tau/sigma pairs; custom accepts
`tau` (correlation time, s) and `sigma` (steady-state RMS fractional amplitude); absent =
disabled. `[monte_carlo.wind]`: custom accepts `scale_min`, `scale_max` (uniform multiplicative
bounds on wind speed) and `direction_bias_deg`; a config without `level` defaults to `"medium"`.

**`[monte_carlo] sampling`**: `"random"` (default), `"lhs"` (Latin Hypercube, stratified
coverage), or `"sobol"` (Owen-scrambled Sobol via the `sobol_burley` crate, max 65536 samples).
LHS/Sobol only change batch draws (n_sims>1); single-sim runs and curation probing (1 sim per
probe seed) are unaffected.

**`[monte_carlo] noise_seeding`**: how the per-sim stochastic streams (OU density perturbation,
EKF sensor noise) are seeded. `"per_draw"` (the DEFAULT, ADR-0006) derives the stream seed from
an FNV-1a hash of the dispersion draw (`RunState::noise_seed`), so per-seed pools and multi-sim
runs both marginalize over noise realizations (identical draw -> identical stream, distinct draws
-> independent noise). `"legacy"` reproduces the historical `[simulation] random_seed +
env_idx*10_000` behavior, which FREEZES the noise realization across every n_sims=1 config (all
per-seed pools condition on ONE noise path); it exists only to reproduce numbers quoted under that
path: every `configs/test/*.toml` and every evaluation script that re-flies a shared-path cell
(`articles/paper/scripts/*`, `experiments/fnpag_ab/`, `param_sweep --eval`, `quantize`, the probe
drivers) pins it through `deploy_overrides.LEGACY_NOISE_REGIME`, `confirmatory_eval.py` records
it in its JSON, and `report.py` prints the regime it resolved. Unknown values hard-error. The
goldens and the paper's main-body numbers are legacy-regime; `experiments/ou_marginal/` holds the
frozen-vs-marginal quantification and the per-draw retrain campaign (see
[src/python/aerocapture/training/README.md](../src/python/aerocapture/training/README.md)).

### Guidance

**`[guidance] type`** selects one of the seven schemes; the per-scheme block is
`[guidance.<scheme>]` (`ftc`, `equilibrium_glide`, `energy_controller`, `pred_guid`, `fnpag`,
`piecewise_constant`, `neural_network`). A `[guidance.*]` key that drives nothing stays declared
and inert on its Rust `Toml*` struct (6 FTC keys, EC `gain`, star-tracker `attitude_sigma`).

**`[guidance.lateral]`** (unsigned-magnitude schemes: EqGlide, EnergyController, PredGuid, FNPAG,
FTC): `tau` (lookahead horizon, s), `threshold` (projected inclination error, deg),
`min_reversal_interval` (anti-chatter, s), `lateral_activation` / `lateral_inhibition` (MJ/kg
energy window), `max_reversals`. The algorithm projects the inclination error forward by `tau`
seconds using finite-difference rate estimation and reverses only when the projected error
exceeds the threshold. Absent = inactive. GA-optimizable (`lateral.` prefix). NN (`full_neural`)
and PiecewiseConstant bypass lateral guidance (they produce signed bank angles).

**`[guidance.thermal_limiter]`** (unsigned-magnitude schemes): `heat_flux_activation`,
`heat_load_activation` (fraction of max, 0.6-1.0), `heat_flux_ramp_exponent`,
`heat_load_ramp_exponent` (1.0 linear, 2.0 quadratic). Default activation 1.0 = disabled. When
active, smoothly blends the bank toward full lift-up as the thermal quantities approach their
limits. GA-optimizable (`thermal.` prefix).

**`[guidance.command_shaping]`**: `enabled` (default true when the section is present),
`max_bank_acceleration` (deg/s², > 0). Acceleration-limited S-curve rate shaping in the dispatch
layer, using `bank_angle_realized` (pilot feedback) as the baseline each tick, not the previous
command. Absent or `enabled = false`: legacy hard-clamp rate saturation. GA-routable (`shaping.`
prefix; an override that creates the section enables it).

**`[guidance.piecewise_constant]`**: three accepted shapes: `bank_angles = [...]` (canonical,
length sets N), `n_segments = N` + individual `bank_angle_0..N-1 = ...` keys (override-friendly;
the GA writes per-element overrides this way), or neither (10 segments at 65 deg). N resolves by
precedence explicit `n_segments` > array length > highest flat `bank_angle_N` index + 1 > default
10. For the VALUES, flat `bank_angle_N` keys OVERLAY the `bank_angles` array (flat keys > array >
default), so a config can seed via `bank_angles = [...]` and still take GA per-element overrides.
`n_segments` is validated against `bank_angles.len()` and the highest `bank_angle_N` index;
mismatches fail at load. `bank_angles: Vec<f64>` on the Rust side, so any N>=1 is legal; Python
`make_piecewise_constant_specs(N)` (`param_spaces.py`) and `_resolve_piecewise_n_segments(toml)`
(`train.py`) mirror the resolver so the chromosome width matches (the default
`PARAM_SPACES["piecewise_constant"]` stays 10-segment). `reference_only = true` writes the mission
reference without clobbering `corridor_boundaries.npz`.

**`[guidance.fnpag]`**: `replan_period` (default 2.0 s; deliberately not a GA gene, the
wall-time-blind GA would floor it), `prediction_dt` (GA bounds floored at 2.0 s), bank limits,
`exit_velocity_threshold`, `energy_tol` (an apoapsis-radius tolerance in meters; the key name is
historical).

**`[guidance.neural_network]`**:
- `mode = "full_neural"` (default: the NN emits a signed bank and bypasses exit, lateral and
  thermal_limiter) or `"magnitude_only"` (the NN's output is reduced to `.abs()` and fed into the
  unsigned-magnitude pipeline: lateral guidance picks the sign, the thermal limiter applies its
  ramp, the exit law takes over in phase 2 -- the NN replaces only the capture-phase
  predictor-corrector and reuses FTC's tuned `[guidance.lateral]` / `[guidance.thermal_limiter]`
  / `[guidance.ftc]` exit-phase scaffolding). Unknown values error at load.
- `output_parameterization`: `"atan2_signed"` (default, `bank = atan2(out[0], out[1])`, requires
  a 2-output last layer, legal in both modes), `"acos_tanh"` (`bank = acos(tanh(out[0]))`,
  requires `magnitude_only`, a 1-output `tanh` last layer), `"scaled_pi"` (`bank =
  wrap_to_pi(scaled_pi_n * pi * tanh(out[0]))`, knob `scaled_pi_n` default 1.0) and `"delta"`
  (`bank = wrap_to_pi(prev_realized + delta_max * tanh(out[0]))`, knob `delta_max` default 0.35
  rad); the last two require `full_neural` and a 1-output `tanh` last layer. Validated at load by
  `validate_output_parameterization` (`config.rs`) with a model-level guard in
  `data/mod.rs::build_neural_net`, and on the Python side by `NetworkConfig.__post_init__`. The
  knobs are TOML-overridable onto the loaded model and embedded in the deployed `best_model.json`
  (self-describing). A non-fatal advisory prints for combinations other than the two matched
  setups (`magnitude_only` + `acos_tanh`, `full_neural` + `atan2_signed`).
- `scaffolding = "off" | "live" | "full"` (default `"off"`; declared per leaf, NOT in
  `nn_common.toml`; printed at training start): whether the chromosome co-optimizes the actuator
  pipeline. `"live"` appends 3 params (`nav.density_filter_gain`, `nav.density_gain_max_delta`,
  `shaping.max_bank_acceleration`), seeded from the ParamSpec defaults; `"full"` appends the 17
  scaffolding params (nav + lateral + exit + thermal + shaping) seeded at FTC's GA optimum from
  `training_output/ftc/best_params.json`. Requires a v2 `[[network.architecture]]`. Details in
  the training README.
- `reset_state_every_tick` (default false, golden-neutral; eval-only): reconstructs a zeroed
  `NnState` before every guidance tick, making a stateful NN memoryless (the paper's reset
  control); reachable via the PyO3 override dot-path `guidance.neural_network.reset_state_every_tick`.
- `warm_start_from = "<best_params.json>"`: the legacy warm-start trigger (the `[warm_start]`
  block is the current one).

### Network

**`[network]`**: `layer_sizes` + `activations` (v1 dense-only) or `[[network.architecture]]`
(v2 tagged layer list, one entry per layer with `type` = `dense` / `gru` / `lstm` / `window` /
`transformer` / `mamba` / `mamba3` / `cfc` / `slstm` / `mlstm`; the loaded model's `LayerSpec`
list must agree with it). `input_mask`: indices into the 35-element candidate input vector
(validated non-negative and `< NN_FULL_INPUT_SIZE`; absent defaults to `[0..16]`).
`ablated_input`: one input index to zero out for ablation analysis (the model JSON's `ablated_value` field freezes it to a value instead).
`[network.normalization]`: 35 `{transform, scale, center}` entries overriding the model's embedded
block (length-validated). QAT knobs `qat_bits` / `qat_granularity` (`per_channel` /
`per_tensor`) / `qat_tensor_policy` (`all` / `proj_only`, the latter keeping the SSM dynamics
scalars `a_log` / `d_skip` / biases in fp): every candidate's flat weights are rounded via
`quantize_flat_weights_batch` before each `run_grid` evaluation and the deploy writer rounds
`best_model.json` the same way. TOML values override the JSON model file's. The candidate-input
contract itself is in [src/rust/src/data/neural/README.md](../src/rust/src/data/neural/README.md).

### Python-only sections

**`[cost_function]`**: `g_load_weight`, `heat_flux_weight`, `heat_load_weight`, `dv_threshold`
(the softplus-quadratic knee; `common.toml` 1000.0 m/s, code default 500.0), `cost_transform`
(`"linear"` | `"sqrt"` | `"log"` | `"squared"` | `"cubed"`; `common.toml` ships `"cubed"`).
`"log"` (np.log1p) compresses the tail more than `"sqrt"` and preserves the zero-cost identity;
`"squared"` / `"cubed"` amplify tail variance so captures separate more under rank-free
optimizers. PSO argmin is unchanged (monotonic); PPO terminal reward magnitudes are not (the RL
`compute_terminal_cost` uses defaults and ignores the TOML cost kwargs). Read by
`cost.build_cost_kwargs`, the one reader.

**`[optimizer]`**: `algorithm` (`"ga"`, `"cma_es"`, `"de"`, `"pso"`, `"qpso"`, `"islands"`),
`n_pop`, `n_gen`, `seed_strategy` (required: `"fixed"` | `"rotating"` | `"adaptive"`, ADR-0001),
`training_n_sims` (sims per individual; code default 1, `common.toml` 10, the paper's deployed
regime 2), `seed_pool_interval` (adaptive curation fallback interval, default 50),
`validation_n_sims` (default 1000), `curation_sample_size` (default 1000), `curation_top_k`
(default 5), `curation_trim_fraction` (default 0.0, valid [0, 0.5)), `curation_bucket_selection`
(`"random"` | `"min"` | `"max"` | `"middle"`; code default `"random"`, `common.toml` `"max"`),
`grow_fresh_fraction` (fresh-random share when a resumed population grows, default 0.2),
`max_violation_rate` (feasibility ceiling for every promotion site, default 0.0 = strict,
ADR-0005); nested `[optimizer.ga]` (`crossover_eta`, `mutation_eta`), `[optimizer.cma_es]`,
`[optimizer.de]`, `[optimizer.pso]`, `[optimizer.qpso]` (`alpha_start`, `alpha_end`, validated in
(0, 2]), `[optimizer.islands]` (`enabled`, `k_period`, `k_top`, `pso_inject_velocity_scale`).
Defaults live in `configs/training/common.toml`; the CLI flags `--n-gen`, `--n-pop`,
`--algorithm` override when given.

**`[warm_start]`** (its presence enables the multi-supervisor BPTT warm-start):
`supervisor_schemes` (default `["ftc", "equilibrium_glide", "energy_controller", "pred_guid",
"fnpag"]`), `bptt_length` (32), `n_warm_seeds` (200), `n_epochs` (10), `minibatch_size` (128),
`bound_multiplier` (4.0, only applied when warm-start is active; otherwise NN-weight bounds stay
at Xavier × 2), `jitter` (0.02), `cmaes_sigma0` (0.1, applied unconditionally so a resumed CMA-ES
sigma matches), `params_paths` (per-scheme `best_params.json` override dict), `eval_interval` (0 =
disabled; >0 runs MC on the warm-start and validation pools every N epochs and on the last),
`adaptive_bounds` (default true), plus `[warm_start.adam]` (`lr`, `beta1`, `beta2`, `eps`,
`weight_decay`, `amsgrad`, torch defaults). Unknown keys at either level raise (`unknown
[warm_start] keys` / `unknown [warm_start.adam] keys`). What the block drives is described in the
training README.

**`[rl]`**, `[rl.reward]`, `[rl.ppo]`, `[rl.sac]`: see
[src/python/aerocapture/training/rl/README.md](../src/python/aerocapture/training/rl/README.md).

**`[reference] joint_bank = true`** appends a `ref_bank` gene (bounds `bank_low` / `bank_high`,
default [55, 80] deg) for the ref-tracking schemes; see the training README.

**`[checkpoints] keep_last = N`** auto-prunes older `checkpoint_g{NNNNN}.{json,npz}` pairs after
each save (default `null` keeps every checkpoint). The pruner
(`aerocapture.training.cleanup_checkpoints.prune_checkpoints`) only touches `checkpoint_g*.{json,npz}`;
`run_*.jsonl`, `best_model.json` / `best_params.json`, `warm_start_*`, `corridor_boundaries.npz`,
`ref_trajectory.dat`, `final_eval.parquet` and `report.pdf` are preserved.

## Training configs by scheme

| scheme | leaf config |
|---|---|
| `piecewise_constant` | `msr_aller_piecewise_constant_train.toml` (train first: produces the corridor; `msr_aller_pc_ref_train.toml` writes the reference only) |
| `ftc` | `msr_aller_ftc_train.toml` (`msr_aller_ftc_joint_ref_train.toml` for the joint reference) |
| `equilibrium_glide` | `msr_aller_eqglide_train.toml` |
| `energy_controller` | `msr_aller_energy_controller_train.toml` |
| `pred_guid` | `msr_aller_pred_guid_train.toml` |
| `fnpag` | `msr_aller_fnpag_train.toml` |
| `neural_network` (v1 dense) | `msr_aller_nn_train_consolidated.toml`; islands: `msr_aller_islands_train.toml` |
| `neural_network_joint` | `msr_aller_nn_joint_train.toml` (`magnitude_only` + `scaffolding = "full"` + `acos_tanh` + FTC warm-start) |
| `neural_network_scaledpi_pso` / `_delta_pso` | `msr_aller_nn_scaledpi_train.toml` / `msr_aller_nn_delta_train.toml` |
| `neural_network_gru_pso` / `_magonly` | `msr_aller_gru_pso_train.toml` / `msr_aller_gru_pso_magonly_train.toml` |
| `neural_network_lstm_pso` / `window_pso` / `transformer_pso` / `mamba_pso` | `msr_aller_{lstm,window,transformer,mamba}_pso_train.toml` |
| `neural_network_gru_ppo` / `lstm_ppo` / `rl` / `atan2_rl` | `msr_aller_gru_ppo_train.toml` / `msr_aller_lstm_ppo_train.toml` / `msr_aller_rl_train.toml` / `msr_aller_nn_atan2_ppo_train.toml` |
| paper cells | `training/paper/` (optimizer studies, `rl/`), `training/sweep/`, `training/quant/`, `training/ou_marginal/`, `training/mamba3_962/` |

`msr_aller_nn_atan2_train.toml` is the paper's atan2 environment (17-input calibrated mask +
`[network] normalization`, `scaffolding = "live"`) that the sweep, probe and ou_marginal configs
base-inherit.
