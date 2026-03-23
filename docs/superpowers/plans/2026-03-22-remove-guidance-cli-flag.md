# Remove `--guidance` CLI Flag Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the redundant `--guidance` CLI flag from `train.py`, reading the guidance type from the TOML config instead, and making the TOML path a required positional argument.

**Architecture:** The `__main__` block of `train.py` is rewritten to accept a positional `toml` argument, load the TOML with base inheritance, extract and validate `guidance.type`, then configure `TrainingConfig` from it. The `train()` function and all downstream code are unchanged.

**Tech Stack:** Python argparse, tomllib (via `toml_utils.load_toml_with_bases`)

**Spec:** `docs/superpowers/specs/2026-03-20-remove-guidance-cli-flag-design.md`

---

### Task 1: Rewrite CLI argument parsing in `train.py`

**Files:**
- Modify: `src/python/aerocapture/training/train.py:726-866`

- [ ] **Step 1: Replace `--toml`, `--guidance`, `--cwd` arguments with positional `toml`**

In the `if __name__ == "__main__"` block, replace lines 731-745 with:

```python
parser = argparse.ArgumentParser(description="Train guidance parameters via GA")
parser.add_argument("toml", type=str, help="TOML training config path (must contain [guidance] type)")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--n-gen", type=int, default=100, help="Number of generations (additional when resuming)")
parser.add_argument("--n-pop", type=int, default=20)
parser.add_argument("--resume", type=str, default=None, help="Checkpoint directory to resume from (auto-detected if omitted and checkpoint exists)")
parser.add_argument("-fs", "--from-scratch", action="store_true", help="Wipe existing training output and start fresh (deletes checkpoints, logs, reports)")
parser.add_argument("--no-tui", action="store_true", help="Disable Rich TUI (use plain-text output)")
```

Note: `--cwd` removed entirely. `--guidance` and `--toml` replaced by positional `toml`.

- [ ] **Step 2: Read guidance type from TOML and validate**

Replace lines 757-794 (the config setup, TOML loading, non-TOML path, and non-NN guard) with:

```python
cfg = TrainingConfig()
cfg.ga.n_gen = args.n_gen
cfg.ga.n_pop = args.n_pop
cfg.ga.n_runs = 1
cfg.ga.rotate_seeds = args.rotate_seeds
cfg.ga.adaptive_seeds = args.adaptive_seeds
cfg.ga.seed_pool_cap = args.seed_pool_cap
cfg.ga.cost_alpha = args.cost_alpha
cfg.ga.cvar_percentile = args.cvar_percentile

# Load TOML and extract guidance type
from aerocapture.training.toml_utils import load_toml_with_bases

_toml_data = load_toml_with_bases(Path(args.toml))
guidance_type = _toml_data.get("guidance", {}).get("type")
if guidance_type is None:
    print("ERROR: TOML config must contain [guidance] type = '<scheme>'")
    print("  Valid schemes: neural_network, equilibrium_glide, energy_controller, pred_guid, fnpag, ftc, piecewise_constant")
    raise SystemExit(1)

from aerocapture.training.param_spaces import PARAM_SPACES

_valid_types = set(PARAM_SPACES.keys()) | {"neural_network"}
if guidance_type not in _valid_types:
    print(f"ERROR: Unknown guidance type '{guidance_type}' in TOML")
    print(f"  Valid schemes: {', '.join(sorted(_valid_types))}")
    raise SystemExit(1)

cfg.guidance_type = guidance_type
cfg.sim.toml_config = args.toml
cfg.sim.executable = "src/rust/target/release/aerocapture"
cfg.sim.nn_param_file = _toml_data.get("data", {}).get("neural_network", "data/neural_network/nn_model.json")
# Override NN architecture from TOML [network] section if present
_net = _toml_data.get("network", {})
if "layer_sizes" in _net:
    cfg.network.layer_sizes = _net["layer_sizes"]
if "activations" in _net:
    cfg.network.activations = _net["activations"]
cfg.sim.final_file = "output/final.train_nn_temp"
cfg.sim.exec_dir = "."
cwd = "."
```

Key points:
- `guidance_type` extracted from TOML early, before `save_dir` and auto-resume
- Validated against `PARAM_SPACES` keys + `"neural_network"`
- `cwd` hardcoded to `"."`
- Non-TOML path and guard removed entirely

- [ ] **Step 3: Clean up conditional guards and dead code in the rest of `__main__`**

Since TOML is now always present and `cwd` is always `"."`, simplify:

1. Line ~824: `mission_name = Path(args.toml).stem if args.toml else "unknown"` → `mission_name = Path(args.toml).stem`
2. Line ~826: Remove `if args.toml:` guard around the base-detection block — it always runs. Dedent the body.
3. Line ~847: Update the error message:
   ```python
   print("  uv run python -m aerocapture.training.train configs/training/msr_aller_piecewise_constant_train.toml")
   ```
4. Line ~853: `if cfg.guidance_type == "piecewise_constant" and args.toml:` → `if cfg.guidance_type == "piecewise_constant":`
5. Line ~854-856: Reuse `_toml_data` instead of loading the TOML again:
   ```python
   _pc_toml = _toml_data  # Already loaded above
   ```
6. Line ~902: `Path(cwd or ".") / cfg.sim.toml_config` → `Path(cwd) / cfg.sim.toml_config`
7. Line ~978: `Path(cwd or ".") / args.toml` → `Path(cwd) / args.toml`
8. Line ~1021: `Path(cwd or ".") / cfg.sim.toml_config` → `Path(cwd) / cfg.sim.toml_config`

- [ ] **Step 4: Run existing tests to verify nothing breaks**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/ -x -q`
Expected: All ~276 tests pass (no test exercises the CLI argparse layer)

- [ ] **Step 5: Run linter**

Run: `cd /Users/govit/Git/Govit/Aerocapture && ./lint_code.sh`
Expected: Clean (no ruff/mypy errors)

- [ ] **Step 6: Commit**

```bash
git add src/python/aerocapture/training/train.py
git commit -m "refactor(train): replace --guidance/--toml with positional toml arg

Read guidance type from TOML [guidance].type instead of redundant CLI flag.
Remove unused non-TOML code path and --cwd flag."
```

---

### Task 2: Update documentation

**Files:**
- Modify: `CLAUDE.md:137,155-178`
- Modify: `README.md:64-78`

- [ ] **Step 1: Update CLAUDE.md train.py description line**

Line 137 — update the CLI signature:
```
  - `train.py` — Main GA loop with checkpoint save/resume (`<config.toml> [--no-tui] [--rotate-seeds | --adaptive-seeds] [--seed-pool-cap N] [--cost-alpha F] [--cvar-percentile P] [--skip-final-report] [--final-n-sims N]`). ...
```

- [ ] **Step 2: Update CLAUDE.md training command examples**

Lines 155-178 — remove `--guidance <scheme>` and `--toml` from all examples:

```bash
# ── Optimize a guidance scheme (with Rich TUI) ──
uv run python -m aerocapture.training.train \
    configs/training/msr_aller_eqglide_train.toml \
    --n-gen 50 --n-pop 20

# ── Disable TUI (e.g. in CI or when piping output) ──
uv run python -m aerocapture.training.train \
    configs/training/msr_aller_eqglide_train.toml \
    --n-gen 50 --n-pop 20 --no-tui

# ── Adaptive seed pool (curates MC seeds by difficulty) ──
uv run python -m aerocapture.training.train \
    configs/training/msr_aller_eqglide_train.toml \
    --n-gen 50 --n-pop 20 --adaptive-seeds

# ── Resume training (auto-detects checkpoint; --n-gen means "N additional") ──
uv run python -m aerocapture.training.train \
    configs/training/msr_aller_eqglide_train.toml \
    --n-gen 50
```

- [ ] **Step 3: Update README.md training command examples**

Lines 64-78 — update the full example and the abbreviated `...` examples:
```bash
# Optimize any guidance scheme (Rich TUI with sparklines and ETA)
uv run python -m aerocapture.training.train \
    configs/training/msr_aller_eqglide_train.toml \
    --n-gen 50 --n-pop 20

# Disable TUI (CI / piped output)
uv run python -m aerocapture.training.train <config.toml> --no-tui

# Rotate MC dispersion seeds each generation (prevents overfitting)
uv run python -m aerocapture.training.train <config.toml> --rotate-seeds

# Adaptive seed pool (curates seeds by difficulty, CVaR-blended fitness)
uv run python -m aerocapture.training.train <config.toml> --adaptive-seeds
```

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: update CLI examples for positional toml argument"
```

---

### Task 3: Final smart-commit

- [ ] **Step 1: Invoke the `smart-commit` skill**

Take the whole git branch into account. Review all changes on the branch, ensure CLAUDE.md and README.md are in sync with the code, then commit.
