# Three-way `scaffolding` knob for NN training

**Date:** 2026-05-29
**Branch:** `feature/mutualize-nn-training-configs`
**Status:** Design approved, pending implementation plan

## Problem

The NN training pipeline has a single boolean `[guidance.neural_network] optimize_scaffolding`.
When `true`, the PSO/GA chromosome is extended with the full 17-element
`_NN_SCAFFOLDING_PARAMS` pack (`nav` + `lateral` + `exit` + `thermal_limiter` +
`shaping`), seeded from FTC's GA optimum
(`training_output/ftc/best_params.json`). When `false`, none of those params are
optimized.

Two consequences fall out of the all-or-nothing boolean:

1. **Inert dimensions in `full_neural` mode.** In `full_neural`, the NN emits a
   signed bank and bypasses the exit / lateral / thermal_limiter modules. Of the
   17 scaffolding params, only 3 are actually live in `full_neural`:
   `nav.density_filter_gain` and `nav.density_gain_max_delta` (they feed the NN's
   observation vector via navigation) and `shaping.max_bank_acceleration` (it
   shapes the NN's output in dispatch). The other ~14 are bypassed, so PSO wastes
   search dimensions on parameters that have zero effect on cost.

2. **Hard FTC dependency.** `optimize_scaffolding = true` calls
   `build_scaffolding_initial_slab("training_output/ftc/best_params.json", …)`,
   which raises `FileNotFoundError` if FTC has not been trained. This is correct
   for `magnitude_only` (where the lateral/exit/thermal params are live and worth
   seeding from FTC), but inappropriate for `full_neural`, where the 3 live params
   have perfectly good standalone defaults and need no FTC seeding.

Putting `optimize_scaffolding = true` in the shared `nn_common.toml` base
(introduced earlier on this branch) made both problems worse: it silently flipped
the flag on for the `full_neural` `gru_pso` / `lstm_pso` configs, dragging in the
FTC dependency and the inert dims.

## Goal

Let the 3 live params (`nav` ×2 + `shaping`) be optimized **without** the FTC
dependency, and make the scaffolding choice explicit and visible at training
start.

## Design

### Config surface

Replace the boolean with a three-valued string under
`[guidance.neural_network]`:

```toml
[guidance.neural_network]
scaffolding = "off" | "live" | "full"   # default "off"
```

| Value    | Params appended to chromosome                                              | Seeding                                  | FTC dependency |
|----------|----------------------------------------------------------------------------|------------------------------------------|----------------|
| `"off"`  | none (NN weights only)                                                     | —                                        | no             |
| `"live"` | 3: `nav.density_filter_gain`, `nav.density_gain_max_delta`, `shaping.max_bank_acceleration` | each `ParamSpec.default` + population jitter | **no**         |
| `"full"` | 17 (`_NN_SCAFFOLDING_PARAMS`, unchanged order)                             | FTC optimum via `build_scaffolding_initial_slab` | yes            |

Rationale for the `live` pack: in `full_neural` exactly these 3 params affect the
simulated cost. They have sane standalone defaults
(`nav.density_filter_gain=0.8`, `nav.density_gain_max_delta=0.1`,
`shaping.max_bank_acceleration=5.0`), so they can be optimized from defaults with
no FTC seeding. `full` remains the right choice for `magnitude_only`, where the
lateral / exit / thermal params are live and benefit from FTC seeding.

`scaffolding` is declared **per-leaf config, not in `nn_common.toml`.** The shared
base carrying `optimize_scaffolding = true` is what coupled the choice to all
inheritors; the fix is to keep `nn_common.toml` minimal (`type = "neural_network"`
only) and have each leaf state its own scaffolding choice. This makes the choice
visible in the config actually being run.

Config-load validation:
- Unknown string value → `ValueError` listing the three accepted values.
- `scaffolding != "off"` with a v1 `layer_sizes`/`activations` arch → the same
  "requires v2 `[[network.architecture]]`" error the boolean raises today.

### Single resolver replaces scattered `17 if … else 0`

`param_spaces.py`:

```python
_NN_LIVE_PARAMS: list[ParamSpec] = [*_NAV_PARAMS, *_SHAPING_PARAMS]   # 3 params

def active_scaffolding_specs(scaffolding: str) -> list[ParamSpec]:
    return {
        "off": [],
        "live": _NN_LIVE_PARAMS,
        "full": _NN_SCAFFOLDING_PARAMS,
    }[scaffolding]
```

`_NN_SCAFFOLDING_PARAMS` (17) is kept unchanged for `full`.

Every site that currently computes `n_scaff = 17 if optimize_scaffolding else 0`
and imports `_NN_SCAFFOLDING_PARAMS` is rewritten to:

```python
pack = active_scaffolding_specs(cfg.network.scaffolding)
n_scaff = len(pack)
```

Affected sites (from the touch-point survey):
- `train.py`: param-spec assembly, initial-population assembly, warm-start tail
  handling, best-individual scaffolding extraction (×3 occurrences across the
  single-algo / islands / resume paths). All iterate `pack` instead of the fixed
  17-element list; the hardcoded `17` is removed.
- `problem.py`: chromosome-decode tail cap (`opt_scaff` boolean → `len(pack)`).
- `warm_start.py`: FTC-seed path runs only for `full`; cache key carries
  `"scaffolding": str` (replacing `"optimize_scaffolding": bool`).
- `config.py`: `NetworkConfig.optimize_scaffolding: bool` → `scaffolding: str = "off"`.
- `report.py` / `compare_guidance.py`: load `best_params.json` scaffolding
  overrides whenever `scaffolding != "off"`; they apply whatever keys are present,
  so a `live` deploy (3 keys) works without special-casing.

### Seeding split

- `full`: unchanged — `build_scaffolding_initial_slab(ftc_path, pack, n_pop, rng, jitter)`
  reads the FTC JSON and seeds at FTC optimum + jitter.
- `live`: a new sibling builds the `(n_pop, n_scaff)` normalized slab from each
  spec's `default` (encoded to normalized) + per-individual jitter, with **no file
  read**. Same shape contract as `build_scaffolding_initial_slab`, so the existing
  `build_initial_population_for_v2` tail-fill path
  (`normalized[:, n_weight_specs:] = slab`) is reused verbatim. The FTC-seeding
  code path is never reached for `live`.

This can be implemented either as a separate `build_default_scaffolding_slab(...)`
or by extending the existing helper to accept a params-dict source (FTC JSON for
`full`, `{spec.name: spec.default}` for `live`). Either is acceptable; the
separate helper keeps the FTC error message localized to `full`.

### Startup visibility

At training start, for NN schemes, print one line mirroring the existing
piecewise startup print:

```
scaffolding optimization: LIVE — 3 params (nav density filter ×2, command shaping); no FTC dependency
scaffolding optimization: FULL — 17 params, seeded from training_output/ftc/best_params.json
scaffolding optimization: OFF — NN weights only
```

### Resume / cache behavior

Switching `scaffolding` between runs changes the chromosome width
(`n_weights + len(pack)`), so the existing `_check_resume_chromosome_shape`
(single-algo) and `IslandModel.from_checkpoint` width guard already catch a
mid-run flip and raise pointing at `--from-scratch`. The warm-start cache key
includes `scaffolding`, so a flip invalidates the cached chromosome.

### Config migration (hard cut)

`optimize_scaffolding` is removed entirely (no deprecated alias).

- `nn_common.toml`: drop the `[guidance.neural_network] optimize_scaffolding`
  entry; the file goes back to just `[guidance] type = "neural_network"`.
- `msr_aller_gru_pso_train.toml`, `msr_aller_lstm_pso_train.toml`
  (`full_neural`): add `scaffolding = "live"`.
- `msr_aller_gru_pso_magonly_train.toml`, `msr_aller_nn_joint_train.toml`
  (`magnitude_only`): add `scaffolding = "full"`.
- Any other config that set `optimize_scaffolding = true` is migrated to the
  equivalent string.

### No Rust changes

Scaffolding optimization is purely chromosome composition on the Python training
side. The Rust runtime only ever receives `nav.*` / `shaping.*` / `lateral.*` /
`exit.*` / `thermal.*` as TOML dot-path overrides at evaluation time. `config.rs`
and the rest of the Rust crate are untouched.

## Testing

- `param_spaces`: `active_scaffolding_specs` returns `[]` / 3 / 17 for off / live /
  full; `_NN_LIVE_PARAMS` names and order are `nav.density_filter_gain`,
  `nav.density_gain_max_delta`, `shaping.max_bank_acceleration`.
- `config`: `scaffolding` parses to the field; unknown value raises; v1-arch +
  non-off raises.
- `live` seeding: slab is built from defaults (no FTC file present on disk), shape
  `(n_pop, 3)`, centered on encoded defaults; running a `live` config does **not**
  touch `training_output/ftc/best_params.json`.
- `full` regression: existing FTC-seed tests
  (`test_warm_start_scaffolding_seed`, `test_nn_optimize_scaffolding_specs`,
  `test_nn_scaffolding_params`) updated to the string key, still assert 17 params
  and FTC seeding.
- Width / resume: flipping `scaffolding` changes chromosome width and trips the
  resume shape guard (update `test_island_model` resume test to the string key).
- Cache key: `test_warm_start_pipeline` asserts `"scaffolding"` is in the cache
  key with the expected value (replacing the `optimize_scaffolding` assertion).

## Out of scope

- No change to the contents of the `full` 17-param pack or its FTC seeding.
- No change to the Rust runtime or guidance dispatch.
- No deprecated-alias support for the old boolean.
