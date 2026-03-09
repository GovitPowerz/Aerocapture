# Harden & Clean Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the Rust simulator + Python tools fully self-sufficient — no dependency on `old_codebase/` — with comprehensive test coverage.

**Architecture:** Gradual removal approach. Build golden reference outputs first (safety net), expand domain test coverage, convert all configs to consolidated TOML (inline data), remove suffix mode and `old_codebase/` dependencies, then do a full variable rename as final polish. CI update is a quick independent task.

**Tech Stack:** Rust (cargo test, clippy), Python (pytest, ruff, mypy, uv), GitHub Actions

---

## Discovery: What Changed From The Design

During exploration, we found that **Phase 4 (remove legacy `.in` format) is already done** for Rust — there is no stdin parsing. The real legacy dependency is:

1. **Suffix mode** in TOML configs — `[data.files]` sections that reference `old_codebase/donnees/` files via `SimData::load()` (`data/mod.rs:169`)
2. **Consolidated mode** — `SimData::from_toml()` (`data/mod.rs:232`) has inline data but still reads 2 external files (atmosphere, reference_trajectory)
3. **Python training paths** — hardcoded `old_codebase/exec`, `old_codebase/donnees/`, `old_codebase/sorties/` in `config.py`, `train.py`, `compare_guidance.py`

Also found a bug: `evaluate.py:221` uses invalid `except A, B:` syntax (should be `except (A, B):`).

---

## Task 1: Generate Rust Golden Reference Outputs

**Purpose:** Capture current Rust simulator outputs as the new "source of truth", replacing Fortran reference data.

**Files:**
- Create: `tests/reference_data/rust_golden/` (directory)
- Create: `tests/generate_golden.sh` (script to regenerate)
- Modify: `tests/test_regression.py` — point at rust golden data

### Step 1: Identify which configs produce golden outputs

Current E2E reference scenarios (from `test_regression.py`):
- `test_ref_orig.toml` → `ref_orig/` (reference trajectory, bank=0.1°)
- `test_high_bank_orig.toml` → `high_bank_orig/` (high bank angle)
- `test_guided_orig.toml` → `guided_orig/` (FTC guided, bank=64.77°)

All three use **suffix mode** and depend on `old_codebase/donnees/`. We need consolidated equivalents.

### Step 2: Create consolidated TOML configs for golden scenarios

For each of the 3 test scenarios, create a consolidated TOML that embeds all data inline (no `old_codebase/` dependency). Use the existing `msr_aller_ftc_consolidated.toml` as template.

**Files to create:**
- `configs/test_ref_consolidated.toml`
- `configs/test_high_bank_consolidated.toml`
- `configs/test_guided_consolidated.toml`

Each must produce **identical output** to the suffix-mode version (validated by comparing photo/final files).

### Step 3: Run Rust simulator and capture golden outputs

```bash
cd /Users/govit/Git/Govit/Aerocapture
mkdir -p tests/reference_data/rust_golden/{ref,high_bank,guided}

# For each config, run simulator and copy outputs
./src/rust/target/release/aerocapture configs/test_ref_consolidated.toml
cp <output_dir>/photo.* tests/reference_data/rust_golden/ref/
cp <output_dir>/final.* tests/reference_data/rust_golden/ref/

# Repeat for high_bank and guided
```

### Step 4: Create `tests/generate_golden.sh`

A script that regenerates golden outputs from consolidated configs. This makes it easy to refresh after intentional changes.

```bash
#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BINARY="$REPO_ROOT/src/rust/target/release/aerocapture"

# Build if needed
cargo build --release --manifest-path "$REPO_ROOT/src/rust/Cargo.toml"

for scenario in ref high_bank guided; do
    echo "Generating golden output for $scenario..."
    "$BINARY" "$REPO_ROOT/configs/test_${scenario}_consolidated.toml"
    # Copy outputs to golden directory
done
```

### Step 5: Update `test_regression.py` to use rust golden data

Change `TEST_CASES` to point at `rust_golden/` instead of Fortran reference data. Keep the same comparison logic (auto-detect CSV vs Fortran format).

### Step 6: Run tests to verify

```bash
uv run pytest tests/test_regression.py -v
```

Expected: All 3 regression tests pass against rust golden data.

### Step 7: Commit

```bash
git add tests/reference_data/rust_golden/ tests/generate_golden.sh configs/test_*_consolidated.toml tests/test_regression.py
git commit -m "test: add Rust golden reference outputs for regression testing"
```

---

## Task 2: Domain Test Coverage — Guidance Schemes

**Purpose:** Ensure every guidance scheme has at least one integration test that validates output correctness (not just "it runs without crashing").

**Files:**
- Create: `src/rust/tests/guidance_regression.rs` — per-scheme output validation
- Modify: `src/rust/tests/e2e.rs` — keep existing smoke tests

### Step 1: Audit current coverage

Current E2E tests only check **exit code == 0** for non-FTC schemes. No output validation for:
- EquilibriumGlide (`msr_aller_eqglide_train.toml`)
- EnergyController (`msr_aller_energy_controller_train.toml`)
- PredGuid (`msr_aller_pred_guid_train.toml`)
- FNPAG (`msr_aller_fnpag_train.toml`)
- NeuralNetwork (`msr_aller_nn_train_consolidated.toml`)

### Step 2: Generate golden outputs for each guidance scheme

Run each training config (single sim) and capture the final.* output as golden reference.

```bash
mkdir -p tests/reference_data/rust_golden/{eqglide,energy_ctrl,pred_guid,fnpag,neural}
```

For each scheme, run and capture `final.*` output (photo is optional since training configs may use stats_only mode).

### Step 3: Write `guidance_regression.rs`

Create parameterized integration tests (using `rstest`) that:
1. Run the simulator with each scheme's config
2. Parse the final.* output
3. Compare against golden reference with tight tolerance

```rust
#[rstest]
#[case("msr_aller_eqglide_train.toml", "eqglide")]
#[case("msr_aller_energy_controller_train.toml", "energy_ctrl")]
#[case("msr_aller_pred_guid_train.toml", "pred_guid")]
#[case("msr_aller_fnpag_train.toml", "fnpag")]
fn guidance_output_matches_golden(#[case] config: &str, #[case] golden_dir: &str) {
    // Run sim, compare final.* to golden
}
```

### Step 4: Run tests

```bash
cd src/rust && cargo test --release guidance_regression
```

### Step 5: Commit

```bash
git commit -m "test: add per-guidance-scheme regression tests with golden outputs"
```

---

## Task 3: Domain Test Coverage — Edge Cases

**Purpose:** Test bounce detection, phase transitions, MC determinism with consolidated configs.

**Files:**
- Modify: `tests/test_mc_domain.py` — ensure no `old_codebase/` dependency
- Create: `src/rust/tests/edge_cases.rs` — Rust-side edge case tests

### Step 1: Audit MC domain tests for `old_codebase/` paths

Check `test_mc_domain.py` — it creates temp TOML configs. Verify it uses consolidated mode (no suffix dependency on `old_codebase/donnees/`).

### Step 2: Add edge case tests

Rust integration tests for:
- **Single sim with dispersions off** (baseline)
- **MC with 3 sims** (same seed = deterministic, different seed = different)
- **ESR variant** if ESR consolidated config exists (multi-planet coverage)

### Step 3: Run all tests

```bash
cargo test --release && uv run pytest tests -v
```

### Step 4: Commit

```bash
git commit -m "test: add edge case tests for MC determinism and multi-config"
```

---

## Task 4: Convert Remaining Suffix-Mode Configs to Consolidated

**Purpose:** Eliminate all TOML configs that depend on `old_codebase/donnees/` files.

**Files:**
- Modify: All suffix-mode configs in `configs/`
- May create: New data files under `data/` (for atmosphere tables, reference trajectories)

### Step 1: Identify suffix-mode configs

From exploration, these configs use `[data.files]` (suffix mode):
- `msr_aller_ftc_nominal.toml`
- `msr_aller_ftc_mc100.toml`
- `msr_aller_ftc_train.toml`
- `msr_aller_nn_train.toml`
- `msr_aller_nn_mc100.toml`
- `msr_aller_reference.toml`
- `test_guided_orig.toml`
- `test_high_bank_orig.toml`
- `test_ref_orig.toml`
- `esr_aller_ftc_nominal.toml`
- `esr_retour_ftc_nominal.toml`

### Step 2: Discuss with user which mission variants to keep

**Decision needed:** Which of these are worth converting vs. deleting?
- MSR aller (Mars Sample Return outbound) — definitely keep
- ESR aller/retour — keep? Archive?
- MC100 lottery-based configs — replace with domain-based MC?

### Step 3: Convert each kept config to consolidated format

For each config:
1. Read the referenced `old_codebase/donnees/` files
2. Embed the data inline in the TOML `[vehicle]`, `[entry]`, `[aerodynamics]`, etc. sections
3. Move atmosphere table and reference trajectory to `data/` directory
4. Validate: run both old and new config, diff outputs

### Step 4: Move external data files to `data/`

Create `data/` at repo root for files that can't be inlined (atmosphere tables, large reference trajectories):
```
data/
  atmosphere/mars.dat
  atmosphere/earth.dat
  reference_trajectory/msr_aller.dat
  reference_trajectory/esr_aller.dat
```

### Step 5: Update consolidated configs to point at `data/` instead of `old_codebase/donnees/`

### Step 6: Run full test suite

```bash
cargo test --release && uv run pytest tests -v
```

### Step 7: Commit

```bash
git commit -m "feat: convert all configs to consolidated TOML with data/ directory"
```

---

## Task 5: Remove Suffix Mode From Rust Code

**Purpose:** Simplify the Rust codebase by removing the dual loading path.

**Files:**
- Modify: `src/rust/src/config.rs` — remove `DataSuffixes`, `[data.files]` parsing
- Modify: `src/rust/src/data/mod.rs` — remove `SimData::load()`, keep only `SimData::from_toml()`
- Modify: `src/rust/src/main.rs` — remove `is_consolidated` branch
- Delete: All per-category file parsers that are only used by suffix mode (if any)
- Modify: `src/rust/tests/config_loading.rs` — update tests

### Step 1: Write failing test

Add a test that consolidated mode is the only path:

```rust
#[test]
fn all_configs_are_consolidated() {
    for entry in std::fs::read_dir("../../configs").unwrap() {
        let path = entry.unwrap().path();
        let content = std::fs::read_to_string(&path).unwrap();
        let (_, toml_config) = SimInput::from_toml(&content).unwrap();
        assert!(SimInput::is_consolidated(&toml_config),
            "{} is not consolidated", path.display());
    }
}
```

### Step 2: Verify it passes (all configs already consolidated from Task 4)

### Step 3: Remove suffix mode code

- Delete `DataSuffixes` struct and `data_path()` method from `config.rs`
- Delete `SimData::load()` from `data/mod.rs`
- Delete `is_consolidated()` check from `main.rs` — always use `from_toml()`
- Remove `[data.files]` TOML struct (`TomlDataFiles`)

### Step 4: Run clippy and fix dead code warnings

```bash
cd src/rust && cargo clippy --all-targets -- -D warnings
```

### Step 5: Run full test suite

```bash
cargo test --release
```

### Step 6: Commit

```bash
git commit -m "refactor: remove suffix mode, consolidated TOML is the only data loading path"
```

---

## Task 6: Update Python Paths and Fix Bug

**Purpose:** Remove all `old_codebase/` hardcoded paths from Python code.

**Files:**
- Modify: `src/python/aerocapture/training/config.py:75` — `exec_dir`
- Modify: `src/python/aerocapture/training/train.py:387-394` — default paths
- Modify: `src/python/aerocapture/training/compare_guidance.py:63,90,119` — NN path, output dir
- Modify: `src/python/aerocapture/training/evaluate.py:221` — fix `except` syntax bug

### Step 1: Fix the bug in evaluate.py

```python
# Line 221: Change
except subprocess.TimeoutExpired, FileNotFoundError:
# To
except (subprocess.TimeoutExpired, FileNotFoundError):
```

### Step 2: Update default paths in config.py

Change `exec_dir`, `nn_param_file`, `final_file` defaults to reference `data/` and repo-root-relative paths instead of `old_codebase/`.

### Step 3: Update train.py and compare_guidance.py

Replace hardcoded `old_codebase/` paths with paths derived from the TOML config's `output_dir` and `base_dir`.

### Step 4: Run Python lints and tests

```bash
uv run ruff check src/python tests && uv run mypy src/python && uv run pytest tests -v
```

### Step 5: Commit

```bash
git commit -m "fix: remove old_codebase paths from Python, fix except syntax bug"
```

---

## Task 7: Remove `old_codebase/`

**Purpose:** Delete the Fortran source, Makefiles, and donnees directory.

**Files:**
- Delete: `old_codebase/` (entire directory)
- Modify: `CLAUDE.md` — remove Fortran-specific sections (build commands, Makefile tips)
- Modify: `.gitignore` — remove old_codebase entries if any

### Step 1: Verify no code references `old_codebase/`

```bash
rg "old_codebase" --type rust --type python --type toml
```

Should return zero matches (all cleaned up in Tasks 4-6).

### Step 2: Run full test suite one last time with `old_codebase/` still present

```bash
cargo test --release && uv run pytest tests -v
```

All green = safety net confirmed.

### Step 3: Delete `old_codebase/`

```bash
rm -rf old_codebase/
```

### Step 4: Run tests again without `old_codebase/`

```bash
cargo test --release && uv run pytest tests -v
```

All green = no hidden dependencies.

### Step 5: Update CLAUDE.md

Remove:
- Fortran build commands section
- `old_codebase/` references in Architecture
- Fortran common block notes (move to design doc if historically valuable)
- Input file format variants section (`.in` is gone)

Keep:
- Fortran bug lessons (they explain *why* certain Rust code looks the way it does)

### Step 6: Commit

```bash
git add -A
git commit -m "chore: remove old_codebase/ — Rust simulator is fully self-sufficient"
```

---

## Task 8: CI Update

**Purpose:** Run CI only on PRs to `main` + manual trigger.

**Files:**
- Modify: `.github/workflows/ci.yml:3-7`

### Step 1: Update CI triggers

```yaml
# Before
on:
  push:
    branches: [main, "feature/**"]
  pull_request:
    branches: [main]

# After
on:
  pull_request:
    branches: [main]
  workflow_dispatch:
```

### Step 2: Commit

```bash
git add .github/workflows/ci.yml
git commit -m "ci: trigger only on PRs to main and manual dispatch"
```

---

## Task 9: Full Variable Rename (Final Polish)

**Purpose:** Rename all French/Fortran-legacy variable names to clear English.

**Note:** This is intentionally last — maximal test coverage + stable codebase = safe to rename.

**Files:**
- All files under `src/rust/src/`

### Step 1: Inventory legacy names

Create a rename mapping by module. Focus on:
- Public struct fields (visible in tests and configs)
- Function names
- Module-level constants
- Internal variables in hot paths (guidance, navigation, integration)

### Step 2: Rename module by module

For each Rust module:
1. Create rename mapping (old → new)
2. Apply renames
3. Run `cargo test --release`
4. Run `cargo clippy`
5. Commit per module (small, reviewable diffs)

### Step 3: Update Python code if any Rust-facing names changed

If output column names or config field names changed, update Python parsers accordingly.

### Step 4: Final full test suite run

```bash
./check_all.sh && uv run pytest tests -v
```

---

## Dependency Graph

```
Task 1 (golden refs) ──┐
                        ├── Task 4 (convert configs) ── Task 5 (remove suffix mode) ──┐
Task 2 (guidance tests) ┘                                                              ├── Task 7 (rm old_codebase)
Task 3 (edge cases) ────────────────────── Task 6 (Python paths) ─────────────────────┘
                                                                                        ├── Task 9 (rename)
Task 8 (CI update) ── independent, can be done anytime ────────────────────────────────┘
```

**Parallelizable:** Tasks 1+2+3 can run concurrently. Task 8 is fully independent.
