# TOML Base Inheritance Design

## Problem

All 20 TOML configs across `configs/training/`, `configs/test/`, and `configs/nominal/` duplicate the same mission-specific content (entry conditions, vehicle, aerodynamics, flight constraints, orbit targets, success tolerances, incidence profile, atmosphere/reference trajectory paths). The 18 Mars configs are nearly identical in these sections. Training configs additionally share Monte Carlo dispersion profiles and cost function settings.

This means a single change to Mars entry conditions requires editing 18 files.

## Solution

A `base` key in TOML files that references shared base configs. The config loader resolves and deep-merges bases before deserialization, so the entire downstream pipeline sees the same flat `TomlConfig` it always has.

## Base Key Semantics

### Declaration

Top-level `base` field — a single string or array of strings:

```toml
base = ["../missions/mars.toml", "common.toml"]

[guidance]
type = "ftc"
# ... scheme-specific overrides only
```

### Path Resolution

Paths are resolved **relative to the file that declares `base`**, not relative to CWD. This makes configs self-contained and portable.

### Merge Order

1. Load base files left-to-right, deep-merging each into an accumulator
2. Overlay the declaring file's own keys on top
3. Strip the `base` key before passing to serde deserialization

### Deep-Merge Rules

- **Tables** merge recursively — keys from the overlay are inserted into or replace the base
- **Non-table values** (scalars, arrays) are fully replaced by the overlay — no array concatenation
- A key present in the overlay always wins over the base

### Recursive Bases

Base files can themselves declare `base`, resolved relative to their own location. A `HashSet<PathBuf>` of canonical paths is threaded through the recursion for cycle detection. In practice, only 1 level of nesting is expected (scheme config -> mission base).

### Error Handling

- Base file not found -> `ParseError` with the full resolution path and the declaring file
- Cycle detected -> `ParseError` listing the chain of files
- Invalid TOML in a base -> `ParseError` with the base file path

## File Layout

```
configs/
  missions/
    mars.toml              # [mission], [entry], [vehicle], [aerodynamics], [flight],
                           # [success], [incidence], [data] (atmosphere + ref traj paths)
    earth.toml             # Same sections, Earth values
  training/
    common.toml            # [monte_carlo], [cost_function], [simulation] defaults
    msr_aller_ftc.toml     # base = ["../missions/mars.toml", "common.toml"]
                           # + [guidance], [data].results_suffix
    msr_aller_eqglide.toml # same pattern
    msr_aller_nn.toml      # + [network], [data].neural_network
    ...
  test/
    test_ftc_golden.toml   # base = ["../missions/mars.toml"]
                           # + [guidance], [simulation], [monte_carlo], [data].results_suffix
    ...
  nominal/
    msr_aller_ftc.toml     # base = ["../missions/mars.toml"]
                           # + [guidance], [simulation]
    ...
```

### Content of `configs/missions/mars.toml`

All sections that are identical across the 18 Mars configs:

- `[mission]` — type = "aerocapture", planet = "mars"
- `[entry]` — altitude, longitude, latitude, velocity, FPA, azimuth, initial_time, initial_bank_angle, initial_aoa
- `[vehicle]` — mass, reference_area, cq, max_bank_rate, periods, pilot
- `[aerodynamics]` — equilibrium_aoa, points table
- `[flight]` — wind, constraints, final_conditions, target_orbit, parking_orbit
- `[success]` — tolerances
- `[incidence]` — altitudes, angles
- `[data]` — atmosphere path, reference_trajectory path

Note: `[data].output_dir` defaults to `"output"` via serde and is the same everywhere, so it does NOT need to be in the mission base. `[data].results_suffix` is per-config and always stays in the child. `[data].neural_network` is NN-only and stays in the NN child config.

### Content of `configs/missions/earth.toml`

Same sections as `mars.toml` but with Earth values. There is only 1 Earth config today (`esr_aller_ftc_nominal.toml`), but extracting the base still makes the pattern consistent and prepares for future Earth configs.

### Content of `configs/training/common.toml`

Shared across all 6 training configs:

- `[simulation]` — random_seed = 0.6866
- `[monte_carlo]` — seed + all 8 dispersion domain subsections (initial_state, atmosphere, aerodynamics, navigation, mass, vehicle, pilot, nav_filter) with their preset levels
- `[cost_function]` — g_load_limit, heat_flux_limit, g_load_weight, heat_flux_weight

Note: `n_sims` is NOT in common.toml because the NN config uses 50 while others use 10.

## Rust Implementation

### New: `resolve_toml_bases()`

In `config.rs`, a function that takes a `toml::Value` tree and the file's path, resolves any `base` key, and returns a merged tree:

```rust
fn resolve_toml_bases(
    root: toml::Value,
    file_path: &Path,
    visited: &mut HashSet<PathBuf>,
) -> Result<toml::Value, ParseError>
```

~40 lines: check for `base` key, iterate base paths, read each, recurse, deep-merge, strip `base`.

### New: `deep_merge()`

```rust
fn deep_merge(base: &mut toml::Value, overlay: toml::Value)
```

~15 lines: if both are tables, recursively merge keys; otherwise replace base with overlay.

### New: `SimInput::from_toml_file()`

```rust
pub fn from_toml_file(path: &Path) -> Result<(Self, TomlConfig), ParseError>
```

Reads the file, calls `resolve_toml_bases()`, then delegates to existing `from_toml()`.

### Changed: `main.rs`

Replace `fs::read_to_string` + `from_toml()` with `from_toml_file()`. ~3 lines changed.

### Unchanged

`SimInput::from_toml(&str)` stays as-is for backwards compatibility — tests and PyO3 can still pass pre-merged TOML strings.

All serde structs (`TomlConfig`, `TomlVehicle`, `TomlEntry`, etc.) are unchanged. The `base` key is stripped before deserialization so serde never sees it.

## PyO3 Implementation

### Core change: `resolve_toml_bases()` is in the core crate

`resolve_toml_bases()` and `deep_merge()` live in the core `aerocapture` crate's `config.rs`, so both the CLI binary and the PyO3 crate can call them.

### `aerocapture-py/src/config.rs`

`load_and_override()` currently receives TOML content as `&str`. It will change to receive a file path (`&Path`), read the file, call `resolve_toml_bases()` to get a merged `toml::Value` tree, then apply overrides on the merged tree. Overrides are applied AFTER base resolution, on the fully merged TOML tree.

### `aerocapture-py/src/lib.rs`

`run()`, `run_mc()`, `run_batch()` currently read the file to a string and pass it to `load_and_override()`. They will change to pass the file path instead. Since `load_and_override()` now handles file reading + base resolution internally, the callers simplify slightly.

`load_config()` currently parses the TOML string directly via `toml::from_str()`. It will change to call `resolve_toml_bases()` before converting to a Python dict, so that inherited values from base files are visible.

### `aerocapture-py/src/batch.rs`

`run_batch()` parses the base TOML once, then clones and patches per-run. It will change to accept the file path, call `resolve_toml_bases()` on the initial parse, then proceed as before (clone merged tree, apply overrides, serialize, run). The base resolution happens once, not per-run.

## Python-Only Sections in TOML

`[cost_function]` and `[network]` are read only by the Python training pipeline, not by Rust serde. `TomlConfig` does not declare these fields, but since it doesn't use `deny_unknown_fields`, serde silently ignores them. This is intentional — these sections pass through Rust parsing without issue.

## Python Implementation

### New: `load_toml_with_bases(path)`

A small utility function (~20 lines) that mirrors the Rust merge logic in Python:

```python
def load_toml_with_bases(path: Path) -> dict:
    """Load a TOML file, recursively resolving 'base' references."""
```

Location: `src/python/aerocapture/training/toml_utils.py` (new module) or added to an existing utils module.

Uses `tomllib.load()` + recursive dict merge. Same path resolution and cycle detection semantics as Rust.

### Changed call sites

Several places in the training pipeline read TOMLs directly via `tomllib.load()` to extract config values (`cost_function`, `monte_carlo.seed`, `network.layer_sizes`, etc.). These must switch to `load_toml_with_bases()` so they see values inherited from base files:

- `train.py` — reads cost_function, monte_carlo.seed, network config
- `compare_guidance.py` — reads base TOML to patch guidance type
- `final_report.py` — reads target inclination, patches for final eval
- `evaluate.py` — `write_guidance_toml()`, `patch_toml_mc_seed()`

Specific call sites that need changing (9 total):

- `train.py` line 248: reads cost_function, monte_carlo.seed (1 call)
- `train.py` line 677: reads network config, neural_network path (1 call)
- `compare_guidance.py` line 44: reads base TOML to patch guidance type (1 call)
- `compare_guidance.py` line 211: reads cost_function config (1 call)
- `final_report.py` line 50: reads target inclination (1 call)
- `final_report.py` line 65: reads base for patching n_sims/seed (1 call)
- `final_report.py` line 389: reads network config (1 call)
- `evaluate.py` line 354: `write_guidance_toml()` reads base TOML (1 call)
- `evaluate.py` line 489: `patch_toml_mc_seed()` reads base TOML (1 call)

Note: all 9 read config files that will gain `base` keys after migration. None read non-config TOMLs (checkpoint metadata, params files, etc.).

## Config Migration

All 20 existing TOML configs are rewritten to use `base`. Each config shrinks ~70% as mission content moves to the shared base files.

No backwards compatibility shims — configs are in-repo and migrated atomically.

## Testing Strategy

### Rust unit tests (~50 lines)

New tests in `config.rs` `#[cfg(test)]` module:

- `test_resolve_single_base` — one base file, overlay wins
- `test_resolve_multiple_bases` — left-to-right merge order
- `test_recursive_base` — base file that itself has a base
- `test_cycle_detection` — A -> B -> A returns error
- `test_missing_base_error` — clear error with file path
- `test_deep_merge_table_recursion` — nested tables merge correctly
- `test_deep_merge_scalar_replacement` — overlay scalar replaces base
- `test_deep_merge_array_replacement` — overlay array replaces base (no concatenation)
- `test_base_stripped_before_deser` — `base` key not present after resolution

### Rust integration tests (~50 lines)

In `tests/` directory:

- **Equivalence test:** For each migrated config, load via `from_toml_file()` (with bases) and compare the resulting `SimInput` + `SimData` against loading the original flat config via `from_toml()`. All fields must match exactly.

### Existing golden regressions (unchanged)

The 9 test configs in `configs/test/` have golden reference data. After migration, `cargo test` must pass unchanged — this is the ultimate correctness gate.

### Python tests (~50 lines)

- `test_load_toml_with_bases_single` — basic inheritance
- `test_load_toml_with_bases_multiple` — merge order
- `test_load_toml_with_bases_cycle` — error raised
- `test_load_toml_with_bases_missing` — error raised
- `test_load_toml_with_bases_deep_merge` — nested dict merge

### CI

No changes needed — existing CI runs `cargo test` + `pytest`.

## Scope Summary

| Component | Change | Size |
|-----------|--------|------|
| `config.rs` (Rust core) | `resolve_toml_bases()` + `deep_merge()` + `from_toml_file()` | ~60 lines |
| `main.rs` (Rust) | Switch to `from_toml_file()` | ~3 lines |
| `aerocapture-py/src/config.rs` | `load_and_override()` takes `&Path` instead of `&str` | ~10 lines |
| `aerocapture-py/src/lib.rs` | Pass path to `load_and_override()`; `load_config()` resolves bases | ~15 lines |
| `aerocapture-py/src/batch.rs` | Resolve bases on initial parse before clone+override loop | ~5 lines |
| Python `toml_utils.py` | `load_toml_with_bases()` | ~20 lines |
| Python training pipeline | Replace `tomllib.load()` with `load_toml_with_bases()` | 9 call sites |
| `configs/missions/mars.toml` | New shared Mars mission config | ~80 lines |
| `configs/missions/earth.toml` | New shared Earth mission config | ~80 lines |
| `configs/training/common.toml` | New shared training defaults | ~30 lines |
| 20 existing TOML configs | Rewritten with `base` + overrides only | each shrinks ~70% |
| Rust tests | Unit + integration equivalence | ~100 lines |
| Python tests | `load_toml_with_bases` unit tests | ~50 lines |

## What Doesn't Change

- `SimInput`, `TomlConfig`, all serde structs
- Simulation runner, guidance code, GA pipeline logic
- CI configuration
- CLI interface (still `./aerocapture config.toml`)
