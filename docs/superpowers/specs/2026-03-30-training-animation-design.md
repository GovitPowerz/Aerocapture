# Training Animation Design

**Date:** 2026-03-30
**Status:** Approved

## Goal

Add a standalone CLI script that generates a GIF animation showing how entry corridors and trajectories evolve during GA training, by replaying checkpoints.

## CLI Interface

```
python -m aerocapture.training.animate <training_output_dir> \
    --toml <config.toml> \
    --n-sims 100 \
    --fps 4 \
    --output animation.gif \
    --every N
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `training_output_dir` | yes | — | Path to scheme output (e.g., `training_output/piecewise_constant/`) |
| `--toml` | yes | — | Training TOML config (needed to re-run sims via PyO3) |
| `--n-sims` | no | 100 | MC simulations per frame |
| `--fps` | no | 4 | Frames per second in output GIF |
| `--output` | no | `<training_output_dir>/animation.gif` | Output GIF path |
| `--every N` | no | 1 | Use every Nth checkpoint (1 = all) |

## Frame Layout

2x2 grid with a title bar:

```
Title: "Generation {gen} | Best cost: {cost:.1f} | Capture: {rate:.0%}"

┌──────────────────────┬──────────────────────┐
│  Corridor (E vs pdyn)│  Inclination (E vs i)│
│  - 4-layer envelope  │  - trajectory spaghetti│
│  - trajectory spaghetti│  - captured envelope │
│  - 3-class coloring  │  - 3-class coloring  │
├──────────────────────┼──────────────────────┤
│  Bank angle (E vs φ) │  Cost CDF            │
│  - trajectory spaghetti│  - histogram + ECDF │
│  - captured envelope │  - best/mean/median  │
│  - 3-class coloring  │    annotations       │
└──────────────────────┴──────────────────────┘
```

### Visual style

- Colors: blue/orange/red from `charts.py` (`COLOR_CAPTURE`, `COLOR_CONSTRAINED`, `COLOR_HYPERBOLIC`)
- Corridor envelopes: from checkpoint's `CorridorAccumulator` state (cumulative up to that generation), same zone fills as `chart_corridor_pdyn`
- Trajectory spaghetti: from MC re-evaluation of that checkpoint's best chromosome, classified via `classify_trajectories`
- Cost CDF: from the checkpoint's population `costs` array (full population)
- Axis ranges: fixed across all frames (computed from final checkpoint on first pass) to prevent jumping

## Data Flow

```
1. Discovery
   - Glob training_output_dir for checkpoint_r*_g*.json
   - Sort by generation number
   - Apply --every N filter

2. Axis range pre-computation
   - Load final checkpoint, run its MC eval
   - Extract global energy/pdyn/inclination/bank/cost ranges
   - All frames use these fixed limits

3. Per-frame pipeline (for each checkpoint):
   a. Load checkpoint .json + .npz
   b. Extract best chromosome + population costs
   c. Reconstruct CorridorAccumulator from checkpoint arrays
   d. Decode best chromosome -> TOML overrides (via param_spaces + evaluate)
   e. Run MC eval via aerocapture_rs.run_mc(toml, overrides, include_trajectories=True)
   f. Classify trajectories via charts.classify_trajectories()
   g. Render 2x2 figure

4. Composition
   - Collect all figures
   - Write GIF via matplotlib PillowWriter at --fps
   - Close figures after each frame to avoid memory bloat
```

### Requirements

- `aerocapture_rs` (PyO3) is required — no subprocess fallback (need `include_trajectories=True`)
- Script errors early with a clear message if PyO3 is unavailable
- Rich progress bar for feedback during long runs

## Module Structure

Single new file: `src/python/aerocapture/training/animate.py`

```python
# Public API
def generate_animation(
    training_dir: Path,
    toml_path: Path,
    n_sims: int = 100,
    fps: int = 4,
    output: Path | None = None,
    every: int = 1,
) -> Path:
    """Main entry point. Returns path to generated GIF."""

# Internal helpers
def _discover_checkpoints(training_dir: Path, every: int) -> list[dict]
def _compute_axis_ranges(final_checkpoint, toml_path, n_sims) -> dict
def _render_frame(checkpoint, toml_path, n_sims, axis_ranges) -> Figure
def _render_corridor_panel(ax, trajectories, traj_class, corridor_data, axis_ranges)
def _render_inclination_panel(ax, trajectories, traj_class, axis_ranges)
def _render_bank_panel(ax, trajectories, traj_class, axis_ranges)
def _render_cost_panel(ax, population_costs, best_cost)

# CLI
def main():  # argparse, calls generate_animation()

if __name__ == "__main__":
    main()
```

### Reuse from existing code

- `charts.py`: `classify_trajectories`, `_draw_spaghetti`, `COLOR_*` constants, seaborn theme
- `corridor.py`: `CorridorAccumulator.from_checkpoint()`
- `train.py`: `load_checkpoint()` for checkpoint parsing
- `evaluate.py`: chromosome decoding (scheme-specific `decode_fn` + `param_spaces`)

## Deletions

- `src/python/aerocapture/plotting/plot_corridor_animation.py` — obsolete, removed entirely

## Dependencies

No new dependencies. Uses matplotlib's `PillowWriter` (Pillow already in deps) and Rich (already in deps).
