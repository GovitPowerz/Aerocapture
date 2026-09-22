# ADR-0008: A config key is declared once; the mirror denies the rest and two gates prove the relay

**Status:** accepted · **Date:** 2026-09-19 (issue #105)

## Context

The config module is two-stage by design: a serde mirror of the file (`config.rs`, the
`Toml*` structs) and the runtime types the sim consumes (`data/mod.rs` builders into
`SimData`). #76 gave it one no-IO validation pass and named builders. The relay between the
stages had no rule: a key changed name on the way through (`reset_state_every_tick` ->
`nn_reset_state_every_tick`), three relay styles coexisted in one builder file, and nothing
under `src/rust` used `deny_unknown_fields`, so a key in the wrong section, a typo in one of
the dotted override strings Python passes, or a `Toml*` field nobody relays vanished without a
message. A committed training config carried two keys Rust never read for five months
(`configs/training/msr_aller_rl_train.toml`, one of them a mission type the parser rejects).
The "all configs parse" gate covered 56 of 183 files, and because every mirror field was `pub`
in a `pub` module, rustc's dead-code lint never saw a field no builder read.

## Decision

- A `Toml*` field is its TOML key. serde enforces this; an unavoidable rename is declared by
  `#[serde(rename)]` on the mirror, as `type` is (`TomlMission`, `TomlGuidance`).
- A new runtime relay keeps the key: `[guidance.neural_network]` lands as
  `GuidanceParams::neural_network: NeuralNetworkParams { mode, reset_state_every_tick }`, so
  `data.guidance.neural_network.reset_state_every_tick` greps end to end. Runtime names that
  predate this rule (`eq_glide`, `energy_ctrl`, `sim_phase`, `capsule`, `dispersion_config`,
  `atmosphere_onboard`, `entry.initial_bank`, the constraint / orbit / period / pilot tables
  hoisted onto `SimData`, the FTC and navigation gains flattened onto `GuidanceParams`) stay;
  the reachability table below is the key-to-runtime map for them.
- Every serde type that reads a TOML section is `#[serde(deny_unknown_fields)]`: the root
  `TomlConfig` (its seven Python-only sections `[optimizer]`, `[cost_function]`,
  `[warm_start]`, `[rl]`, `[checkpoints]`, `[reference]`, `[corridor]` are declared as
  pass-through `toml::Table`s, so a misspelled section name is rejected while the sections
  themselves stay Python-owned), every section struct, `PlanetConfig`, the `TomlLayerSpec` tag
  enum and the `NormSpec` entries. The two structs with a `#[serde(flatten)]` map, which serde
  cannot deny, validate their extra keys explicitly and regardless of level:
  `TomlPiecewiseConstantParams::resolve_bank_angles_deg` accepts only `bank_angle_N`, and
  `TomlMcDomain.custom` goes through `take_custom` even when the domain is `off`.
- A Python-only key inside a Rust-owned section (`scaffolding`, `warm_start_from`,
  `layer_sizes`, `activations`, `qat_*`, `reference_only`) is declared on the mirror with a
  `// Python-only` comment naming its reader, so the Rust struct is the single list of what
  may appear in the section.
- Section-struct fields are `pub(crate)`, so `cargo clippy --workspace --all-targets
  -D warnings` fails on a declared field no builder reads; the Python-only fields carry
  `#[allow(dead_code)]`. The four fields read outside the crate (`TomlData.atmosphere`,
  `TomlData.wind_table`, `TomlMonteCarlo.seed`, `TomlMonteCarlo.noise_seeding`) and the root
  stay `pub`.
- Two gates: `tests/config_loading.rs::every_committed_config_parses_and_validates` walks
  `configs/**` recursively (a leaf is a file whose resolved root carries both `[mission]` and
  `[guidance]`) plus `experiments/trainer_seam_gate/`, parses each leaf, checks it is
  consolidated and runs the no-IO `validate` pass, so a key no struct declares fails CI with
  serde's `unknown field` message and the file; `planet_presets_parse` covers the planet
  presets no leaf inherits. `config_tests.rs::toml_keys_reachable_in_sim_data` patches one key
  per row with a value that differs from the fixture and asserts it is observable in `SimData`
  (or `SimInput` where the runner reads it), catching the declared-and-relayed-to-the-wrong-place
  case neither serde nor the lint sees.

## Consequences

- Numbers do not move: the six guidance goldens, `test_pyo3_matches_subprocess`,
  `test_run_grid.py` and `test_noise_seeding.py` are unchanged; no default value changed. Two
  latent defaults were fixed on the way: an absent `[simulation]` or `[guidance.piecewise_constant]`
  table now yields the per-key defaults (1 sim / 3000 s, -6 / 5 MJ/kg) instead of
  `derive(Default)`'s zeros; and `[onboard_atmosphere] mode` accepts only `"identical"`, alone.
- A typo'd override dot path surfaces as serde's `unknown field` from `SimInput::from_toml`
  and as `ValueError` from `validate_config`
  (`tests/test_pyo3.py::test_unknown_key_in_rust_section_raises_value_error`); the four
  `run_grid` rejections of ADR-0004 still fire first for their keys.
- `[guidance.<other scheme>]` tables on a config stay legal: they are declared `Option` fields
  of `TomlGuidance`; deny only rejects keys no struct declares. Python writers
  (`toml_utils.write_toml`, `evaluate.write_guidance_toml`, the GA `bank_angle_N` per-element
  overrides) target exactly the flatten sections.
- A new `Toml*` field is either relayed (and gets a reachability row) or `// Python-only` with
  `#[allow(dead_code)]`; a new Python-only root section is one `Option<toml::Table>` line on
  `TomlConfig`. A relayed field nothing reads is invisible to both gates, so the runtime structs
  carry no `#[allow(dead_code)]`: the lint is that gate (#128, option c, 2026-09-22 -- the four
  dead GA genes left `PARAM_SPACES`, their runtime fields are gone, the TOML keys stay declared
  and inert on the `Toml*` struct with a per-field allow naming this decision; `[success]` was
  deleted outright).
- The model JSON loaders (`NnJsonFile`, `NnJsonFileV2`, the JSON `LayerSpec`) deny unknown keys
  the same way since #128 (the one legacy key, `output_interpretation`, is declared and ignored),
  and a `[[network.architecture]]` block must equal the loaded model's architecture. The
  Python-owned `[cost_function]` / `[corridor]` / `[reference]` sections are read once each and
  reject unknown keys (`toml_utils.reject_unknown_keys`).
