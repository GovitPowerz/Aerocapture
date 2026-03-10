# Directory Restructure Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean up the repository layout — delete dead files, split configs into subcategories, rename `save_net/` to `training_output/`, purge tracked caches from git.

**Architecture:** Pure file moves, deletes, and path reference updates. No logic changes. The Rust simulator and Python tools are unchanged — only paths that reference moved files need updating.

**Tech Stack:** git, bash, Rust (test path updates), Python (training module path updates)

**Spec:** `docs/superpowers/specs/2026-03-10-directory-restructure-design.md`

---

## Chunk 1: Delete dead files and purge caches

### Task 1: Delete dead files

**Files:**
- Delete: `animation_frames/` (directory)
- Delete: `scripts/` (directory)
- Delete: `corridor_nn_trained.png`
- Delete: `guidance_comparison.png`
- Delete: `guidance_convergence.png`
- Delete: `mc_comparison_ftc_vs_nn.png`

- [ ] **Step 1: Delete the files**

```bash
rm -rf animation_frames/ scripts/
rm -f corridor_nn_trained.png guidance_comparison.png guidance_convergence.png mc_comparison_ftc_vs_nn.png
```

- [ ] **Step 2: Verify they're gone**

```bash
ls animation_frames/ scripts/ corridor_nn_trained.png guidance_comparison.png guidance_convergence.png mc_comparison_ftc_vs_nn.png 2>&1
```

Expected: all "No such file or directory"

- [ ] **Step 3: Commit**

```bash
git add -A animation_frames/ scripts/ corridor_nn_trained.png guidance_comparison.png guidance_convergence.png mc_comparison_ftc_vs_nn.png
git commit -m "chore: delete orphaned files (animation_frames, scripts, loose PNGs)"
```

### Task 2: Purge tracked caches and update .gitignore

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Add missing entries to `.gitignore`**

Add these lines to the `# === Python ===` section of `.gitignore`:

```
.mypy_cache/
.pytest_cache/
.ruff_cache/
```

- [ ] **Step 2: Remove cached files from git tracking**

```bash
git rm -r --cached .mypy_cache/ .pytest_cache/ .ruff_cache/ 2>/dev/null || true
git rm -r --cached src/python/aerocapture/__pycache__/ tests/__pycache__/ src/python/aerocapture/io/__pycache__/ src/python/aerocapture/training/__pycache__/ 2>/dev/null || true
git ls-files -z '*.DS_Store' | xargs -0 git rm --cached 2>/dev/null || true
```

- [ ] **Step 3: Verify no caches are tracked**

```bash
git status
```

Expected: `.mypy_cache/`, `.pytest_cache/`, `.ruff_cache/`, `__pycache__/`, `.DS_Store` files show as deleted from index.

- [ ] **Step 4: Commit**

```bash
git add .gitignore
git commit -m "chore: purge tracked caches and add missing .gitignore entries"
```

---

## Chunk 2: Split configs/ into subcategories

### Task 3: Move config files into subdirectories

**Files:**
- Move: all 20 files in `configs/` into `configs/nominal/`, `configs/training/`, `configs/test/`

- [ ] **Step 1: Create subdirectories and move files**

```bash
mkdir -p configs/nominal configs/training configs/test

# Nominal
mv configs/esr_aller_ftc_nominal.toml configs/nominal/
mv configs/msr_aller_ftc_consolidated.toml configs/nominal/
mv configs/msr_aller_ftc_mc_domain.toml configs/nominal/
mv configs/msr_aller_ftc_nominal.toml configs/nominal/
mv configs/msr_aller_reference.toml configs/nominal/

# Training
mv configs/msr_aller_energy_controller_train.toml configs/training/
mv configs/msr_aller_eqglide_train.toml configs/training/
mv configs/msr_aller_fnpag_train.toml configs/training/
mv configs/msr_aller_ftc_train.toml configs/training/
mv configs/msr_aller_nn_train_consolidated.toml configs/training/
mv configs/msr_aller_pred_guid_train.toml configs/training/

# Test
mv configs/test_energy_ctrl_golden.toml configs/test/
mv configs/test_eqglide_golden.toml configs/test/
mv configs/test_fnpag_golden.toml configs/test/
mv configs/test_ftc_golden.toml configs/test/
mv configs/test_guided_orig.toml configs/test/
mv configs/test_high_bank_orig.toml configs/test/
mv configs/test_neural_golden.toml configs/test/
mv configs/test_pred_guid_golden.toml configs/test/
mv configs/test_ref_orig.toml configs/test/
```

- [ ] **Step 2: Verify no TOML files remain at `configs/` root**

```bash
ls configs/*.toml 2>&1
```

Expected: "No such file or directory" (all moved to subdirectories)

- [ ] **Step 3: Commit the file moves**

```bash
git add configs/
git commit -m "chore: split configs/ into nominal/, training/, test/ subdirectories"
```

### Task 4: Update Rust test config paths

**Files:**
- Modify: `src/rust/tests/common/mod.rs:12` (doc comment)
- Modify: `src/rust/tests/e2e.rs` (lines 41, 59, 72, 77, 82, 87, 92, 96, 116, 133, 147)
- Modify: `src/rust/tests/guidance_regression.rs` (lines 46-65, all 6 config names)
- Modify: `src/rust/tests/edge_cases.rs` (lines 41, 71)
- Modify: `src/rust/tests/config_loading.rs` (lines 7, 17, 27, 36-53, 57-73)

The `config_path()` helper in `common/mod.rs` stays generic (joins `configs/` + name). All callers include the subdirectory prefix.

- [ ] **Step 4: Update `common/mod.rs` doc comment**

Change line 12 from:
```rust
/// Get path to a TOML config in configs/.
```
to:
```rust
/// Get path to a TOML config in configs/<subdir>/.
```

- [ ] **Step 5: Update `e2e.rs` config paths**

All `run_sim()` and `run_guidance_config()` calls need prefixes:

| Old | New |
|-----|-----|
| `"msr_aller_reference.toml"` | `"nominal/msr_aller_reference.toml"` |
| `"msr_aller_ftc_consolidated.toml"` | `"nominal/msr_aller_ftc_consolidated.toml"` |
| `"msr_aller_eqglide_train.toml"` | `"training/msr_aller_eqglide_train.toml"` |
| `"msr_aller_energy_controller_train.toml"` | `"training/msr_aller_energy_controller_train.toml"` |
| `"msr_aller_pred_guid_train.toml"` | `"training/msr_aller_pred_guid_train.toml"` |
| `"msr_aller_fnpag_train.toml"` | `"training/msr_aller_fnpag_train.toml"` |
| `"msr_aller_ftc_train.toml"` | `"training/msr_aller_ftc_train.toml"` |
| `"msr_aller_ftc_mc_domain.toml"` | `"nominal/msr_aller_ftc_mc_domain.toml"` |

Note: `"msr_aller_ftc_mc_domain.toml"` appears 3 times — in `mc_domain_completes` (line 116), and twice in `mc_deterministic_same_seed` (lines 133, 147). All three must be updated.

Also update the assertion message on line 99 from `"Config file missing: configs/{}"` to `"Config file missing: configs/{}"` (no change needed — it already interpolates the full name).

- [ ] **Step 6: Update `guidance_regression.rs` config paths**

All 6 `#[case]` config names need `test/` prefix:

| Old | New |
|-----|-----|
| `"test_eqglide_golden.toml"` | `"test/test_eqglide_golden.toml"` |
| `"test_energy_ctrl_golden.toml"` | `"test/test_energy_ctrl_golden.toml"` |
| `"test_pred_guid_golden.toml"` | `"test/test_pred_guid_golden.toml"` |
| `"test_fnpag_golden.toml"` | `"test/test_fnpag_golden.toml"` |
| `"test_ftc_golden.toml"` | `"test/test_ftc_golden.toml"` |
| `"test_neural_golden.toml"` | `"test/test_neural_golden.toml"` |

- [ ] **Step 7: Update `edge_cases.rs` config paths**

| Old | New |
|-----|-----|
| `"msr_aller_ftc_consolidated.toml"` | `"nominal/msr_aller_ftc_consolidated.toml"` |
| `"esr_aller_ftc_nominal.toml"` | `"nominal/esr_aller_ftc_nominal.toml"` |

- [ ] **Step 8: Update `config_loading.rs`**

Update the 3 direct `config_path()` calls:

| Old | New |
|-----|-----|
| `config_path("msr_aller_ftc_consolidated.toml")` | `config_path("nominal/msr_aller_ftc_consolidated.toml")` |
| `config_path("msr_aller_reference.toml")` | `config_path("nominal/msr_aller_reference.toml")` |
| `config_path("msr_aller_ftc_mc_domain.toml")` | `config_path("nominal/msr_aller_ftc_mc_domain.toml")` |

Replace `parse_all_available_configs` (lines 35-54) with a recursive version:

```rust
#[test]
fn parse_all_available_configs() {
    let configs_dir = common::repo_root().join("configs");
    let mut count = 0;
    for subdir in ["nominal", "training", "test"] {
        let dir = configs_dir.join(subdir);
        for entry in std::fs::read_dir(&dir).expect("read configs subdir") {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|e| e == "toml") {
                let content = std::fs::read_to_string(&path).expect("read config");
                let result = SimInput::from_toml(&content);
                assert!(
                    result.is_ok(),
                    "Failed to parse {}: {:?}",
                    path.display(),
                    result.err()
                );
                count += 1;
            }
        }
    }
    assert!(count >= 10, "Expected at least 10 configs, found {}", count);
}
```

Replace `all_configs_are_consolidated` (lines 56-73) similarly:

```rust
#[test]
fn all_configs_are_consolidated() {
    let configs_dir = common::repo_root().join("configs");
    for subdir in ["nominal", "training", "test"] {
        let dir = configs_dir.join(subdir);
        for entry in std::fs::read_dir(&dir).expect("read configs subdir") {
            let path = entry.unwrap().path();
            if path.extension().is_none_or(|e| e != "toml") {
                continue;
            }
            let content = std::fs::read_to_string(&path).expect("read config");
            let toml_config: TomlConfig =
                toml::from_str(&content).unwrap_or_else(|e| panic!("{}: {}", path.display(), e));
            assert!(
                toml_config.vehicle.is_some(),
                "{} is not consolidated (missing [vehicle] section)",
                path.display()
            );
        }
    }
}
```

- [ ] **Step 9: Run Rust tests to verify**

```bash
cd src/rust && cargo test --release 2>&1 | tail -20
```

Expected: all tests pass

- [ ] **Step 10: Commit Rust test updates**

```bash
git add src/rust/tests/
git commit -m "fix: update Rust test config paths for configs/ subdirectories"
```

### Task 5: Update Python test config paths

**Files:**
- Modify: `tests/test_regression.py:22-25`
- Modify: `tests/generate_golden.sh:37-39`

- [ ] **Step 11: Update `test_regression.py` paths**

Change lines 22-25:

```python
GOLDEN_CASES = [
    ("ref", "configs/test/test_ref_orig.toml", "ref", "test_ref_orig"),
    ("high_bank", "configs/test/test_high_bank_orig.toml", "high_bank", "test_high_bank_orig"),
    ("guided", "configs/test/test_guided_orig.toml", "guided", "test_guided_orig"),
]
```

- [ ] **Step 12: Update `generate_golden.sh` paths**

Change lines 37-39:

```bash
run_and_copy "configs/test/test_ref_orig.toml"       "ref"       "test_ref_orig"
run_and_copy "configs/test/test_high_bank_orig.toml"  "high_bank"  "test_high_bank_orig"
run_and_copy "configs/test/test_guided_orig.toml"     "guided"     "test_guided_orig"
```

- [ ] **Step 13: Run Python tests to verify**

```bash
uv run pytest tests/test_regression.py -v 2>&1 | tail -20
```

Expected: all tests pass

- [ ] **Step 14: Commit Python test updates**

```bash
git add tests/test_regression.py tests/generate_golden.sh
git commit -m "fix: update Python test config paths for configs/ subdirectories"
```

---

## Chunk 3: Rename save_net/ and update documentation

### Task 6: Rename `save_net/` to `training_output/`

**Files:**
- Modify: `.gitignore`
- Modify: `src/python/aerocapture/training/config.py:86`
- Modify: `src/python/aerocapture/training/train.py:402`
- Modify: `src/python/aerocapture/training/compare_guidance.py:8,193`
- Modify: `src/python/aerocapture/training/plot_comparison.py:8,142`

- [ ] **Step 1: Update `.gitignore`**

Change `save_net/` to `training_output/`.

- [ ] **Step 2: Update `config.py`**

Change line 86 from:
```python
    save_dir: str = "save_net"
```
to:
```python
    save_dir: str = "training_output"
```

- [ ] **Step 3: Update `train.py`**

Change line 402 from:
```python
    cfg.save_dir = f"save_net/{cfg.guidance_type}"
```
to:
```python
    cfg.save_dir = f"training_output/{cfg.guidance_type}"
```

- [ ] **Step 4: Update `compare_guidance.py`**

Change line 8 from:
```
        --base-toml configs/msr_aller_eqglide_train.toml \
```
to:
```
        --base-toml configs/training/msr_aller_eqglide_train.toml \
```

Change line 193 from:
```python
    parser.add_argument("--params-dir", type=str, default="save_net", help="Directory with optimized params")
```
to:
```python
    parser.add_argument("--params-dir", type=str, default="training_output", help="Directory with optimized params")
```

- [ ] **Step 5: Update `plot_comparison.py`**

Change line 8 from:
```
        --results save_net/comparison_results.json \
```
to:
```
        --results training_output/comparison_results.json \
```

Change line 142 from:
```python
    parser.add_argument("--results", type=str, default="save_net/comparison_results.json")
```
to:
```python
    parser.add_argument("--results", type=str, default="training_output/comparison_results.json")
```

- [ ] **Step 6: Rename the directory on disk**

```bash
mv save_net training_output
```

- [ ] **Step 7: Run Python lint to verify**

```bash
uv run ruff check src/python/ && uv run mypy src/python/
```

Expected: no errors

- [ ] **Step 8: Commit**

```bash
git add .gitignore src/python/aerocapture/training/config.py src/python/aerocapture/training/train.py src/python/aerocapture/training/compare_guidance.py src/python/aerocapture/training/plot_comparison.py
git commit -m "refactor: rename save_net/ to training_output/"
```

### Task 7: Update documentation

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`
- Modify: `docs/plans/*.md` (best-effort)

- [ ] **Step 9: Update `README.md`**

Line 15 — Quick Start config path:
```
./target/release/aerocapture ../../configs/nominal/msr_aller_ftc_nominal.toml
```

Lines 20-28 — Project Structure:
```
src/
  rust/                    Rust simulator (validated reimplementation)
  python/                  Python analysis package (parsing, plotting, training)
configs/
  nominal/                 Nominal simulation configurations
  training/                GA training configurations (per guidance scheme)
  test/                    Golden test configurations (regression tests)
data/
  atmosphere/              Atmosphere density tables (Mars, Earth)
  reference_trajectory/    Reference trajectories for guided schemes
tests/                     Test framework and golden reference data
```

Lines 62-63 — GA Optimization example paths:
```bash
    --toml configs/training/msr_aller_eqglide_train.toml \
```
```bash
    --base-toml configs/training/msr_aller_eqglide_train.toml \
```

- [ ] **Step 10: Update `CLAUDE.md`**

Update all `configs/` path references to include subdirectory. Key sections:
- Build & Development Commands: `configs/test_ref_orig.toml` → `configs/test/test_ref_orig.toml`
- Input Configuration: describe the subdirectory layout
- GA Training & Comparison: update all `configs/msr_aller_*` paths to `configs/training/msr_aller_*`
- Guidance schemes training configs list: update all 6 paths
- `save_net/` references → `training_output/`

- [ ] **Step 11: Best-effort update `docs/plans/*.md`**

Search for `configs/` and `save_net` references in plan docs and update where found. These are historical docs so best-effort is fine.

- [ ] **Step 12: Run full test suite**

```bash
cd src/rust && cargo test --release 2>&1 | tail -5
cd ../.. && uv run pytest tests -v 2>&1 | tail -10
```

Expected: all Rust and Python tests pass

- [ ] **Step 13: Commit documentation updates**

```bash
git add README.md CLAUDE.md docs/
git commit -m "docs: update all path references for directory restructure"
```
