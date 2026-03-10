# Directory Restructure Design

**Date:** 2026-03-10
**Goal:** Clean up the repository layout for tidiness — no functional changes.

## Changes

### 1. Delete dead files

| Path | Reason |
|------|--------|
| `animation_frames/` | Orphaned directory with a single PNG |
| `scripts/` | Single orphaned plotting script |
| `corridor_nn_trained.png` | Loose root PNG (already gitignored) |
| `guidance_comparison.png` | Loose root PNG (already gitignored) |
| `guidance_convergence.png` | Loose root PNG (already gitignored) |
| `mc_comparison_ftc_vs_nn.png` | Loose root PNG (already gitignored) |

### 2. Split `configs/` into subcategories

```
configs/
├── nominal/
│   ├── esr_aller_ftc_nominal.toml
│   ├── msr_aller_ftc_consolidated.toml
│   ├── msr_aller_ftc_mc_domain.toml
│   ├── msr_aller_ftc_nominal.toml
│   └── msr_aller_reference.toml
├── training/
│   ├── msr_aller_energy_controller_train.toml
│   ├── msr_aller_eqglide_train.toml
│   ├── msr_aller_fnpag_train.toml
│   ├── msr_aller_ftc_train.toml
│   ├── msr_aller_nn_train_consolidated.toml
│   └── msr_aller_pred_guid_train.toml
└── test/
    ├── test_energy_ctrl_golden.toml
    ├── test_eqglide_golden.toml
    ├── test_fnpag_golden.toml
    ├── test_ftc_golden.toml
    ├── test_guided_orig.toml
    ├── test_high_bank_orig.toml
    ├── test_neural_golden.toml
    ├── test_pred_guid_golden.toml
    └── test_ref_orig.toml
```

#### Migration strategy for `config_path()`

Keep `config_path()` in `common/mod.rs` generic — it joins `configs/` + whatever string is passed. Update all call sites to include the subdirectory prefix (e.g. `config_path("test/test_ref_orig.toml")`). This keeps the helper layout-agnostic since callers already span test, nominal, and training configs.

#### Files that reference config paths (must update)

- `src/rust/tests/common/mod.rs` — `config_path()` helper itself is unchanged; update its doc comment.
- `src/rust/tests/e2e.rs` — config path string references (test configs).
- `src/rust/tests/guidance_regression.rs` — 6 test config names need `test/` prefix.
- `src/rust/tests/edge_cases.rs` — 2 nominal config names need `nominal/` prefix.
- `src/rust/tests/config_loading.rs` — 3 direct `config_path()` calls need `nominal/` prefix. **Additionally**, `parse_all_available_configs` and `all_configs_are_consolidated` use non-recursive `read_dir("configs/")` — must switch to recursive directory walking or glob, since `.toml` files are now one level deeper.
- `tests/test_regression.py` — hardcoded `"configs/test_*.toml"` paths.
- `tests/generate_golden.sh` — hardcoded `"configs/test_*.toml"` paths.
- `src/python/aerocapture/training/compare_guidance.py` — docstring example path.
- `README.md` — Quick Start code block references `configs/msr_aller_ftc_nominal.toml`, must become `configs/nominal/...`.
- `CLAUDE.md` — documentation references throughout.
- `docs/plans/*.md` — documentation references (best-effort update).

### 3. Rename `save_net/` to `training_output/`

Legacy name from when only the neural network was trained. Now holds GA checkpoints and optimized params for all 6 guidance schemes.

#### Files that reference `save_net` (must update)

- `.gitignore` — `save_net/` -> `training_output/`
- `src/python/aerocapture/training/config.py:86` — default `save_dir = "save_net"`
- `src/python/aerocapture/training/train.py:402` — `save_net/{cfg.guidance_type}`
- `src/python/aerocapture/training/compare_guidance.py:193` — default `--params-dir`
- `src/python/aerocapture/training/plot_comparison.py:8` — docstring example path.
- `src/python/aerocapture/training/plot_comparison.py:142` — default `--results` path.
- `CLAUDE.md` — documentation references.

#### Local rename

`save_net/` directory itself gets renamed to `training_output/` on disk. It's gitignored so no tracked content moves, but the rename avoids confusion.

### 4. Purge tracked caches from git

Remove from git tracking (but keep locally via `.gitignore`):

- `.mypy_cache/` — add to `.gitignore`, `git rm -r --cached`
- `.pytest_cache/` — add to `.gitignore`, `git rm -r --cached`
- `.ruff_cache/` — add to `.gitignore`, `git rm -r --cached`
- `__pycache__/` directories — already in `.gitignore` but some tracked; `git rm -r --cached`
- `.DS_Store` files — already in `.gitignore` but some tracked; `git rm --cached`

### 5. No changes

- `src/python/aerocapture/` package structure
- `src/rust/` crate structure
- `tests/` at root for Python, `src/rust/tests/` for Rust
- Shell scripts at root (`setup_env.sh`, `lint_code.sh`, `check_all.sh`, `upgrade_dependencies.sh`)
- `data/` at root
- `output/` (gitignored)

## Result

```
aerocapture/
├── .github/workflows/
├── configs/
│   ├── nominal/
│   ├── training/
│   └── test/
├── data/
│   ├── atmosphere/
│   ├── neural_network/
│   └── reference_trajectory/
├── docs/
├── output/                    (gitignored)
├── training_output/           (gitignored)
├── src/
│   ├── python/aerocapture/
│   │   ├── io/
│   │   ├── plotting/
│   │   └── training/
│   └── rust/
│       ├── src/
│       └── tests/
├── tests/
│   └── reference_data/
├── check_all.sh
├── lint_code.sh
├── setup_env.sh
├── upgrade_dependencies.sh
├── pyproject.toml
├── CLAUDE.md
├── README.md
├── TODO.md
└── IMPROVEMENTS.md
```

## Risk

Low. All changes are file moves, deletes, and path updates. No logic changes. Tests must pass after path updates to confirm nothing was missed.
