# Final Evaluation Report Design

## Problem

The current end-of-training flow in `train.py` prints a few mean values (apoapsis error, periapsis error, delta-V) for captured trajectories and stops. There is no statistical distribution view, no breakdown of the 3 orbital correction burns (dv1, dv2, dv3), no inclination error analysis, and the re-evaluation uses the same small MC sample size as training (typically 20-100 sims).

## Goal

After GA training completes, run a large Monte Carlo re-evaluation (default 1000 sims) on the best solution using a distinct seed, then generate a self-contained Plotly HTML report with proper statistical distributions of all key metrics.

## Approach

New module `src/python/aerocapture/training/final_report.py` — separate from the existing `report.py` (which covers training convergence). Clean separation: training report shows *how you got there*, final report shows *how good the result is*.

## Data Source

The Rust simulator writes a final conditions CSV with 40 columns per simulation. `_parse_final_to_legacy_array()` in `evaluate.py` maps these into a 53-column legacy array (col 0 = sim_number, then legacy indices offset by +1). Key columns in the legacy array:

| Array Column | CSV Column | Metric |
|---|---|---|
| 8 | `energy_mj_kg` | Orbital energy (< 0 = captured) |
| 10 | `eccentricity` | Orbit shape (< 1 = captured) |
| 11 | `inclination_deg` | Final orbital inclination |
| 28 | `sim_time_s` | Flight duration |
| 30 | `periapsis_err_km` | Periapsis altitude error vs target |
| 31 | `apoapsis_err_km` | Apoapsis altitude error vs target |
| 5 | `flight_path_deg` | Entry flight path angle |
| 4 | `velocity_m_s` | Entry velocity |
| 38 | `dv1_m_s` | Out-of-plane correction (inclination) |
| 39 | `dv2_m_s` | In-plane correction (SMA/ecc) |
| 40 | `dv3_m_s` | Out-of-plane correction (RAAN) |
| 42 | `dv_total_m_s` | Total delta-V |

Note: dv1/dv2/dv3 legacy indices are 37/38/39, but the legacy array adds +1 for the sim_number column, so they land at array columns 38/39/40.

Inclination error is computed as `inclination_deg - target_inclination`, where `target_inclination` comes from `[flight.target_orbit] inclination` in the TOML config.

Capture filter: `(eccentricity < 1.0) & (energy < 0)`. Non-captured trajectories are excluded from distribution plots but counted in the summary. If capture rate is 0%, distribution panels show empty plots with a "No captured trajectories" annotation and the summary table shows the capture rate with NaN percentiles.

## Report Layout (8 panels)

Self-contained Plotly HTML, 4x2 subplot grid:

1. **Total Delta-V distribution** — histogram + overlaid CDF with percentile lines (p5/p25/p50/p75/p95)
2. **Individual correction burns** — dv1, dv2, dv3 as 3 overlaid semi-transparent histograms
3. **Apoapsis error distribution** — histogram + CDF (km)
4. **Periapsis error distribution** — histogram + CDF (km)
5. **Inclination error distribution** — histogram + CDF (deg)
6. **Entry conditions scatter** — entry FPA vs entry velocity, colored by outcome (captured/hyperbolic), marker size scaled by delta-V
7. **Delta-V vs orbital error scatter** — dv_total (m/s) vs combined orbital error sqrt(apo_err^2 + peri_err^2) (km), to reveal clustering
8. **Summary statistics table** — `go.Table` with capture rate, percentiles (p5/p25/p50/p75/p95), mean, std for all metrics

Color scheme consistent with existing `report.py`.

## Module Structure

### Functions

```python
def run_final_evaluation(
    cfg: TrainingConfig,
    n_sims: int = 1000,
    seed: int | None = None,
    cwd: Path | None = None,
) -> np.ndarray | None:
    """Run large-MC re-evaluation of best solution.

    Patches the TOML config to override n_sims and mc_seed, then runs
    the simulator. Returns final conditions array (n_sims, 53) in
    legacy format, or None if the simulation fails.

    The seed parameter is the re-evaluation seed. If None, the caller
    is responsible for providing a distinct seed (e.g. training_seed + 9999).
    """

def generate_final_report(
    final_array: np.ndarray,
    scheme: str,
    target_inclination: float,
    output_path: Path,
) -> Path:
    """Generate self-contained Plotly HTML report.

    Returns path to generated HTML file.
    Handles 0% capture rate gracefully (empty distribution panels with annotation).
    """
```

Separation of concerns: `run_final_evaluation` handles sim execution (no plotting), `generate_final_report` handles visualization (no sim execution).

`run_final_evaluation` patches `n_sims` and `mc_seed` in the TOML config before running the simulator. This extends the existing TOML patching in `evaluate.py` (which already patches guidance params and MC seed).

### CLI (`__main__` block)

```bash
uv run python -m aerocapture.training.final_report \
    training_output/equilibrium_glide/ \
    --toml configs/training/msr_aller_eqglide_train.toml \
    --n-sims 1000 \
    --seed 42
```

Positional arg: scheme output directory (contains `best_params.json` or `best_model.json`).
Loads best params, patches TOML, runs sim, generates report.

### Integration in `train.py`

After the GA loop and best-params save:

1. Call `run_final_evaluation(cfg, n_sims=1000)` with distinct seed
2. Call `generate_final_report(final_array, scheme, target_incl, output_dir / "final_report.html")`
3. Print path to generated report

New CLI flags on `train.py`:
- `--skip-final-report` — skip re-evaluation and report generation
- `--final-n-sims N` — override default 1000 sims for re-evaluation

## Re-evaluation Seed Strategy

The training seed is passed as `--seed` to `train.py` (not stored in `TrainingConfig`). For the integrated path, `train.py` computes `training_seed + 9999` and passes it explicitly to `run_final_evaluation(seed=...)`. For the standalone CLI, the user provides `--seed` directly.

This ensures:
- Deterministic (reproducible re-evaluation)
- Distinct from any training seed (training uses base seed, seed+1, seed+2, ... for runs)
- Overridable via `--seed` on both CLIs

## Output

Report written to `training_output/<scheme>/final_report.html`.

## Testing

- **Unit tests**: `generate_final_report` with synthetic final arrays (known distributions), verify HTML file is produced and contains expected plot titles
- **Unit tests**: `run_final_evaluation` config override logic (n_sims, seed patching)
- **Integration test**: standalone CLI produces HTML file (requires Rust binary; skip if not built)

## Files Modified

| File | Change |
|---|---|
| `src/python/aerocapture/training/final_report.py` | New module (report generation + CLI) |
| `src/python/aerocapture/training/train.py` | Add final report call after GA loop; add `--skip-final-report` and `--final-n-sims` flags |
| `tests/test_final_report.py` | New test file |

## Finalization

After all implementation steps are complete and tests pass, use the `smart-commit` skill to commit everything. The smart-commit should take into account all commits linked to this plan (i.e. squash/summarize the full feature branch work into coherent documentation updates).
