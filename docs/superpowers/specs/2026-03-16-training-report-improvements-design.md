# Training Report Improvements

## Problem

Two issues with the current training report system:

1. **Resume overwrites generation numbers.** When resuming training with `--n-gen 100`, the loop counter resets to 1-100 instead of continuing at 101-200. The JSONL log contains duplicate generation numbers, and the report shows only one set of 100 data points instead of the full 200.

2. **Reports ignore dispersion/seed data.** The JSONL already logs `mc_seed` (rotate/adaptive seeds) and `pool_metrics` (adaptive seeds), but the report's 6 panels don't use any of this data.

## Root Cause (Resume Bug)

In `train.py` line 436: `for gen in range(gen_start, config.ga.n_gen)`.

- First run: `--n-gen 100` → checkpoint saves `generation: 100`
- Resume: `--n-gen 100` → `start_gen=100`, `config.ga.n_gen=100` → `range(100, 100)` is empty — no new training happens at all

## Design

### Part 1: Resume continuation fix (`train.py`)

**Change:** After loading a checkpoint (around line 307), **before** `create_display()` (line 333), offset the total generation count:

```python
if resumed is not None:
    config.ga.n_gen += start_gen
```

Placement is critical: the offset must happen before `create_display()` so the TUI progress bar receives the correct total.

This makes `--n-gen` mean "N additional generations" when resuming. The loop `range(start_gen, start_gen + original_n_gen)` produces gens 101-200 with unique numbers.

**Affected behaviors:**
- TUI progress bar: receives `n_generations = start_gen + n_gen`, shows "Gen 150/200" — correct. The denominator intentionally shows cumulative total generations, not the `--n-gen` argument.
- Verbose print: `Gen {gen+1}/{config.ga.n_gen}` → "Gen 101/200" — correct (same cumulative total convention)
- Checkpoint: saves at `(gen+1) % interval == 0` → checkpoints at 110, 120, etc. — correct
- `mc_seed`: `base_mc_seed + gen` naturally continues the sequence (gen=100 → seed offset 100) — correct
- Cost history: checkpoint's `cost_history` already has gens 1-100, new gens append — correct
- No resume path: `start_gen=0`, no offset, behavior unchanged
- Argparse help text: update `--n-gen` to `"Number of generations (additional when resuming)"`

### Part 2: New report panels (`report.py`)

Add two conditional panels that only render when the relevant JSONL fields exist:

#### 2a. Seed Pool Evolution (adaptive seeds only)

Shown when `pool_metrics` field is present in the JSONL data.

- **Left y-axis:** `pool_size` (line chart) — shows how the seed pool grows over training
- **Right y-axis:** `difficulty_min` and `difficulty_max` (shaded band) — shows how the difficulty range evolves
- **X-axis:** Generation number
- **`make_subplots` spec:** `{"secondary_y": True}`

#### 2b. MC Seed Trace (rotate seeds only)

Shown when `mc_seed` field is present in the JSONL data. Note: adaptive seeds don't log a single `mc_seed` per generation (the seed pool evaluates per-seed), so this panel only appears for `--rotate-seeds` mode.

- **Line/scatter plot** of `mc_seed` value vs generation
- Useful to confirm seeds are changing and to correlate seed switches with cost jumps in the convergence panel
- **`make_subplots` spec:** `{}` (default, single y-axis)

### Part 3: Resume indicators (all panels)

Detect resume points from **JSONL file boundaries**. Each `TrainingLogger` session creates a new JSONL file with a timestamp in the filename (e.g., `run_000_20260316T123622.jsonl`). When `load_run_data()` reads multiple JSONL files, it can track which file each record came from. The last generation of file N and the first generation of file N+1 mark a resume boundary.

This is deterministic and immune to timing noise (unlike timestamp-gap heuristics which would false-positive on slow generations or false-negative on quick resumes).

- **Vertical dashed line** drawn at each detected resume point on all panels (existing and new)
- Subtle annotation label: "resumed" at the top of the line
- Color: gray, semi-transparent — informational, not distracting

**Implementation detail:** `load_run_data()` currently returns `list[dict]`. Extend it to also return a `list[int]` of resume generation numbers (the first generation of each file after the first file). The plotting functions receive this list and add `vline` shapes. Both `generate_single_report()` and `generate_comparison_report()` call `load_run_data()` and need updating to accept the new return type. Resume markers should appear in both report types.

### Part 4: Dynamic grid layout (`report.py`)

The current report uses a fixed 3x2 grid of 6 subplots. Change to dynamic:

1. Always include the 6 base panels (convergence, diversity, capture rate, cost distribution, parameter evolution, summary)
2. Count how many conditional panels are needed (0, 1, or 2 based on JSONL fields)
3. Build `make_subplots` grid with `6 + N` panels, arranged in rows of 2
4. Build `specs` array dynamically: base panels keep their current specs (e.g., `{"secondary_y": True}` for diversity), seed pool panel gets `{"secondary_y": True}`, all others get `{}`

This keeps the report clean for fixed-seed runs (same 6 panels as today) while adding relevant panels when dispersion data exists.

### Part 5: Summary panel update

Update the existing summary text panel to include:
- Number of resume points detected (if any)
- Total generations across all sessions

## Backward Compatibility

No JSONL format changes required. Existing logs remain fully compatible. Reports generated from old JSONL files (without seed fields) show the same 6 panels as before. The deduplication logic in `load_run_data()` is retained as a safety net for legacy logs with duplicate generation numbers, and also handles partial-overlap at resume boundaries (e.g., checkpoint saved at gen 95, gens 96-100 exist in both the old and new JSONL files).

## Files Changed

| File | Change |
|------|--------|
| `src/python/aerocapture/training/train.py` | Offset `config.ga.n_gen += start_gen` after checkpoint load, before `create_display()` |
| `src/python/aerocapture/training/report.py` | Add seed pool panel, MC seed trace panel, resume vertical lines, dynamic grid, updated summary |

## Testing

- **Unit test:** Verify generation numbering is continuous after resume (mock checkpoint with `generation: 100`, run with `n_gen=50`, assert loop covers gens 100-149)
- **Unit test:** Verify `--n-gen` without resume still means total generations (no offset)
- **Report tests:** Verify conditional panels appear/disappear based on JSONL field presence
- **Report tests:** Verify resume detection from JSONL file boundaries
- **Report tests:** Verify resume markers appear on all panels at correct generation
- **Integration:** Run a short training (5 gens), resume (5 more), generate report, verify 10 data points with a resume marker at gen 5
