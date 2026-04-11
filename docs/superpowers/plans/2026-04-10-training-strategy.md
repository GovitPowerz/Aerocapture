# Training Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two CLI flags to `train.py` (`--mutation-rate`, `--train-n-sims`) and create a `train_all.sh` script with optimized per-scheme training commands.

**Architecture:** Two small additions to the training CLI plumbing, plus a shell script. `--mutation-rate` overrides `GAConfig.mutation_rate`. `--train-n-sims` stores on `SimConfig` and gets injected into override dicts / patched TOMLs during GA evaluation. No changes to the Rust simulator or TOML configs.

**Tech Stack:** Python (argparse, dataclass), Bash

---

### File Map

- Modify: `src/python/aerocapture/training/config.py` -- add `train_n_sims` field to `SimConfig`
- Modify: `src/python/aerocapture/training/train.py` -- add two CLI flags, wire them through
- Modify: `src/python/aerocapture/training/evaluate.py` -- inject `train_n_sims` into TOML patching
- Test: `tests/test_training_config.py` -- test new config fields
- Test: `tests/test_train_cli.py` -- test CLI arg parsing
- Create: `train_all.sh` -- training script

---

### Task 1: Add `train_n_sims` to SimConfig

**Files:**
- Modify: `src/python/aerocapture/training/config.py:82-91`
- Test: `tests/test_training_config.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_training_config.py`:

```python
from aerocapture.training.config import SimConfig


def test_sim_config_train_n_sims_default_none() -> None:
    sc = SimConfig()
    assert sc.train_n_sims is None


def test_sim_config_train_n_sims_override() -> None:
    sc = SimConfig(train_n_sims=300)
    assert sc.train_n_sims == 300
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_training_config.py::test_sim_config_train_n_sims_default_none tests/test_training_config.py::test_sim_config_train_n_sims_override -v`
Expected: FAIL with `TypeError: SimConfig.__init__() got an unexpected keyword argument 'train_n_sims'`

- [ ] **Step 3: Add `train_n_sims` field to SimConfig**

In `src/python/aerocapture/training/config.py`, add to the `SimConfig` dataclass after line 91:

```python
    train_n_sims: int | None = None  # override n_sims during GA training; None = use TOML value
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_training_config.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/config.py tests/test_training_config.py
git commit -m "feat: add train_n_sims field to SimConfig"
```

---

### Task 2: Inject `train_n_sims` into evaluate.py TOML patching

**Files:**
- Modify: `src/python/aerocapture/training/evaluate.py:337-400` (`write_guidance_toml`)
- Modify: `src/python/aerocapture/training/evaluate.py:544-611` (`evaluate_chromosome`)

The non-adaptive path calls `evaluate_chromosome` -> `write_guidance_toml` which writes a patched TOML file. We need to inject `simulation.n_sims` into that patched TOML when `config.sim.train_n_sims` is set.

- [ ] **Step 1: Patch `write_guidance_toml` to accept optional `n_sims` override**

In `src/python/aerocapture/training/evaluate.py`, modify `write_guidance_toml` signature and body. Add `n_sims_override: int | None = None` parameter:

```python
def write_guidance_toml(
    base_toml_path: str | Path,
    guidance_type: str,
    params: dict[str, float],
    output_path: str | Path | None = None,
    mc_seed: int | None = None,
    n_sims_override: int | None = None,
) -> Path:
```

Add after the existing mc_seed injection block (around line 410, after `toml_data["monte_carlo"]["seed"] = mc_seed`), before `_write_toml`:

```python
    if n_sims_override is not None:
        toml_data.setdefault("simulation", {})["n_sims"] = n_sims_override
```

- [ ] **Step 2: Patch `evaluate_chromosome` to pass `train_n_sims` through**

In `evaluate_chromosome`, for the non-NN path (line 596), pass the override:

```python
        patched_toml = write_guidance_toml(base_toml, config.guidance_type, params, mc_seed=mc_seed, n_sims_override=config.sim.train_n_sims)
```

For the NN path with `mc_seed` (line 579), inject `train_n_sims` into the patched TOML too. Modify `patch_toml_mc_seed` to also accept `n_sims_override`:

```python
def patch_toml_mc_seed(base_toml_path: str | Path, mc_seed: int, n_sims_override: int | None = None) -> Path:
```

Add inside `patch_toml_mc_seed`, after the mc_seed injection:

```python
    if n_sims_override is not None:
        toml_data.setdefault("simulation", {})["n_sims"] = n_sims_override
```

Then in `evaluate_chromosome` NN path (line 579):

```python
            patched_toml = patch_toml_mc_seed(Path(cwd) / config.sim.toml_config, mc_seed, n_sims_override=config.sim.train_n_sims)
```

For the NN path without mc_seed (line 588), we need to handle `train_n_sims` too. When `mc_seed is None` but `train_n_sims` is set, pass it as an override to `run_simulation`. Add an overrides dict:

```python
        else:
            nn_overrides: dict[str, object] | None = None
            if config.sim.train_n_sims is not None:
                nn_overrides = {"simulation.n_sims": config.sim.train_n_sims}
            final = run_simulation(config, cwd=cwd, overrides=nn_overrides)
```

- [ ] **Step 3: Run existing tests to verify no regressions**

Run: `uv run pytest tests/ -x -q`
Expected: all PASS (no behavior change when `train_n_sims` is None)

- [ ] **Step 4: Commit**

```bash
git add src/python/aerocapture/training/evaluate.py
git commit -m "feat: inject train_n_sims override into TOML patching"
```

---

### Task 3: Add CLI flags to train.py

**Files:**
- Modify: `src/python/aerocapture/training/train.py:860-893`
- Test: `tests/test_train_cli.py` (create)

- [ ] **Step 1: Write failing test for CLI arg parsing**

Create `tests/test_train_cli.py`:

```python
"""Tests for train.py CLI argument parsing."""

from __future__ import annotations

import argparse


def _build_parser() -> argparse.ArgumentParser:
    """Replicate the train.py argument parser for testing without running main."""
    parser = argparse.ArgumentParser()
    parser.add_argument("toml", type=str)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-gen", type=int, default=100)
    parser.add_argument("--n-pop", type=int, default=50)
    parser.add_argument("--mutation-rate", type=float, default=None)
    parser.add_argument("--train-n-sims", type=int, default=None)
    parser.add_argument("--final-n-sims", type=int, default=1000)
    return parser


def test_mutation_rate_default_none() -> None:
    parser = _build_parser()
    args = parser.parse_args(["dummy.toml"])
    assert args.mutation_rate is None


def test_mutation_rate_override() -> None:
    parser = _build_parser()
    args = parser.parse_args(["dummy.toml", "--mutation-rate", "0.05"])
    assert args.mutation_rate == 0.05


def test_train_n_sims_default_none() -> None:
    parser = _build_parser()
    args = parser.parse_args(["dummy.toml"])
    assert args.train_n_sims is None


def test_train_n_sims_override() -> None:
    parser = _build_parser()
    args = parser.parse_args(["dummy.toml", "--train-n-sims", "300"])
    assert args.train_n_sims == 300
```

- [ ] **Step 2: Run tests to verify they pass** (these test a local parser copy, so they pass immediately)

Run: `uv run pytest tests/test_train_cli.py -v`
Expected: all PASS

- [ ] **Step 3: Add CLI flags to train.py**

In `src/python/aerocapture/training/train.py`, after line 878 (`--sim-timeout`), add:

```python
    parser.add_argument("--mutation-rate", type=float, default=None, help="Override mutation rate (default: 0.02 from GAConfig)")
    parser.add_argument("--train-n-sims", type=int, default=None, help="Override n_sims during GA training evaluations (default: use TOML value)")
```

- [ ] **Step 4: Wire the flags to config objects**

After line 892 (`cfg.ga.stress_inject = args.stress_inject`), add:

```python
    if args.mutation_rate is not None:
        cfg.ga.mutation_rate = args.mutation_rate
    cfg.sim.train_n_sims = args.train_n_sims
```

- [ ] **Step 5: Inject `train_n_sims` into the adaptive-seed batch evaluator override dicts**

In the `_make_batch_eval` factory (around line 496 and 520), the override dicts already set `"simulation.n_sims": 1` because adaptive mode evaluates one seed at a time. This is correct and should NOT be changed -- adaptive mode always uses n_sims=1 per seed by design.

However, `train_n_sims` should still be respected when NOT using adaptive seeds. The non-adaptive path calls `evaluate_chromosome` which we already patched in Task 2. No additional changes needed here.

- [ ] **Step 6: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add src/python/aerocapture/training/train.py tests/test_train_cli.py
git commit -m "feat: add --mutation-rate and --train-n-sims CLI flags to train.py"
```

---

### Task 4: Create train_all.sh

**Files:**
- Create: `train_all.sh`

- [ ] **Step 1: Create the script**

Create `train_all.sh` at repo root:

```bash
#!/usr/bin/env bash
set -euo pipefail

# Train all guidance schemes with optimized GA settings.
# Usage:
#   ./train_all.sh                    # train all schemes in order
#   ./train_all.sh eqglide            # train a single scheme
#   ./train_all.sh ftc fnpag          # train specific schemes
#
# Piecewise constant must run first (produces ref trajectory + corridor).
# All others can run in any order.

TRAIN="uv run python -m aerocapture.training.train"

train_piecewise_constant() {
    echo "=== piecewise_constant (11 params, ~35 min) ==="
    $TRAIN configs/training/msr_aller_piecewise_constant_train.toml \
        --n-gen 3000 --n-pop 40 --train-n-sims 300 \
        --mutation-rate 0.03 \
        --adaptive-seeds --cost-alpha 0.65 --cvar-percentile 15 \
        --seed-pool-cap 120 --stress-interval 15 --stress-probes 200 --stress-inject 10 \
        --final-n-sims 2000 --from-scratch
}

train_ftc() {
    echo "=== ftc (26 params, ~37 min) ==="
    $TRAIN configs/training/msr_aller_ftc_train.toml \
        --n-gen 2500 --n-pop 50 --train-n-sims 300 \
        --mutation-rate 0.03 \
        --adaptive-seeds --cost-alpha 0.65 --cvar-percentile 15 \
        --seed-pool-cap 150 --stress-interval 10 --stress-probes 300 --stress-inject 15 \
        --final-n-sims 2000 --from-scratch
}

train_eqglide() {
    echo "=== equilibrium_glide (24 params, ~46 min) ==="
    $TRAIN configs/training/msr_aller_eqglide_train.toml \
        --n-gen 2500 --n-pop 60 --train-n-sims 300 \
        --mutation-rate 0.05 \
        --adaptive-seeds --cost-alpha 0.6 --cvar-percentile 15 \
        --seed-pool-cap 150 --stress-interval 10 --stress-probes 300 --stress-inject 15 \
        --final-n-sims 2000 --from-scratch
}

train_energy_controller() {
    echo "=== energy_controller (20 params, ~46 min) ==="
    $TRAIN configs/training/msr_aller_energy_controller_train.toml \
        --n-gen 2500 --n-pop 60 --train-n-sims 300 \
        --mutation-rate 0.05 \
        --adaptive-seeds --cost-alpha 0.6 --cvar-percentile 15 \
        --seed-pool-cap 150 --stress-interval 10 --stress-probes 300 --stress-inject 15 \
        --final-n-sims 2000 --from-scratch
}

train_pred_guid() {
    echo "=== pred_guid (20 params, ~46 min) ==="
    $TRAIN configs/training/msr_aller_pred_guid_train.toml \
        --n-gen 2500 --n-pop 60 --train-n-sims 300 \
        --mutation-rate 0.05 \
        --adaptive-seeds --cost-alpha 0.6 --cvar-percentile 15 \
        --seed-pool-cap 150 --stress-interval 10 --stress-probes 300 --stress-inject 15 \
        --final-n-sims 2000 --from-scratch
}

train_fnpag() {
    echo "=== fnpag (22 params, ~50 min) ==="
    $TRAIN configs/training/msr_aller_fnpag_train.toml \
        --n-gen 600 --n-pop 50 --train-n-sims 200 \
        --mutation-rate 0.05 \
        --adaptive-seeds --cost-alpha 0.6 --cvar-percentile 15 \
        --seed-pool-cap 100 --stress-interval 15 --stress-probes 150 --stress-inject 10 \
        --final-n-sims 2000 --from-scratch
}

train_neural_network() {
    echo "=== neural_network (1106 params, ~35 min) ==="
    $TRAIN configs/training/msr_aller_nn_train_consolidated.toml \
        --n-gen 1500 --n-pop 120 --train-n-sims 200 \
        --mutation-rate 0.03 \
        --adaptive-seeds --cost-alpha 0.6 --cvar-percentile 15 \
        --seed-pool-cap 100 --stress-interval 15 --stress-probes 200 --stress-inject 10 \
        --final-n-sims 2000 --from-scratch
}

train_all() {
    train_piecewise_constant
    echo ""
    train_ftc
    echo ""
    train_eqglide
    echo ""
    train_energy_controller
    echo ""
    train_pred_guid
    echo ""
    train_fnpag
    echo ""
    train_neural_network
}

# Dispatch: no args = all, otherwise run named schemes
if [ $# -eq 0 ]; then
    train_all
else
    for scheme in "$@"; do
        case "$scheme" in
            piecewise_constant|piecewise|pc)  train_piecewise_constant ;;
            ftc)                               train_ftc ;;
            eqglide|equilibrium_glide|eq)      train_eqglide ;;
            energy_controller|energy|ec)       train_energy_controller ;;
            pred_guid|predguid|pg)             train_pred_guid ;;
            fnpag)                             train_fnpag ;;
            neural_network|nn)                 train_neural_network ;;
            all)                               train_all ;;
            *)
                echo "Unknown scheme: $scheme"
                echo "Valid: piecewise_constant ftc eqglide energy_controller pred_guid fnpag neural_network all"
                exit 1
                ;;
        esac
    done
fi
```

- [ ] **Step 2: Make executable**

Run: `chmod +x train_all.sh`

- [ ] **Step 3: Verify script syntax**

Run: `bash -n train_all.sh`
Expected: no output (syntax OK)

- [ ] **Step 4: Commit**

```bash
git add train_all.sh
git commit -m "feat: add train_all.sh with optimized per-scheme GA settings"
```

---

### Task 5: Lint and verify

**Files:** All modified files

- [ ] **Step 1: Run ruff**

Run: `uv run ruff check src/python/aerocapture/training/config.py src/python/aerocapture/training/train.py src/python/aerocapture/training/evaluate.py tests/test_training_config.py tests/test_train_cli.py`
Expected: no errors

- [ ] **Step 2: Run mypy**

Run: `uv run mypy src/python/aerocapture/training/config.py src/python/aerocapture/training/evaluate.py`
Expected: no errors

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: all PASS

- [ ] **Step 4: Commit any lint fixes if needed**

---

### Task 6: Smart commit

Invoke the `smart-commit` skill, taking the whole git branch into account.
