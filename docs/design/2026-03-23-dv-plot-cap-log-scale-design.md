# Cap DV at 5000 m/s + Log Scale in Final Report

**Date:** 2026-03-23
**Scope:** `src/python/aerocapture/training/final_report.py`

## Problem

The final evaluation report's DV distribution plots auto-scale to accommodate virtual DV penalties (10,000–20,000 m/s for non-captured trajectories), making the captured trajectory distribution (100–2,600 m/s) unreadable. The performance table also shows these inflated values.

## Decision

Clip all DV values at 5,000 m/s and use log scale on DV axes. Include all trajectories (captured + non-captured) in DV plots and table, clipped uniformly.

## Design

### Constants

```python
DV_CAP = 5000.0  # m/s — clip virtual DV penalties for plot readability
DV_FLOOR = 0.1   # m/s — epsilon floor to avoid log(0) on log-scale axes
```

Defined at module level alongside existing column-index constants.

### Change 1: Include all trajectories, clipped (line ~254)

Currently only captured trajectories are used for DV plots. Change to use all trajectories with clipping:

```python
dv_total = np.clip(final_array[:, _COL_DV_TOTAL], DV_FLOOR, DV_CAP)
dv1 = np.clip(final_array[:, _COL_DV1], DV_FLOOR, DV_CAP)
dv2 = np.clip(final_array[:, _COL_DV2], DV_FLOOR, DV_CAP)
dv3 = np.clip(final_array[:, _COL_DV3], DV_FLOOR, DV_CAP)
```

These four arrays move **outside** the `if n_captured > 0` block so they are computed for all trajectories. The histogram and individual burns plots use these clipped arrays regardless of capture status.

The orbital error metrics (apo/peri/inclination histograms) remain captured-only — they are only meaningful for captured trajectories.

The `DV_FLOOR` (0.1 m/s) prevents `log(0)` issues on log-scale axes. Individual burns (`dv1`, `dv2`, `dv3`) can legitimately be 0.0 when a correction is negligible; the floor is physically insignificant.

### Change 2: Log scale on DV x-axes

After the DV histogram calls, set x-axis to log scale:

```python
fig.update_xaxes(type="log", row=1, col=1)  # Total DV histogram
fig.update_xaxes(type="log", row=1, col=2)  # Individual burns histogram
```

Also apply log scale to the DV-vs-error scatter y-axis (row 3, col 2):

```python
fig.update_yaxes(type="log", row=3, col=2)
```

### Change 3: DV-vs-error scatter uses clipped DV (line ~288)

The scatter at row 3, col 2 currently uses `cap[:, _COL_DV_TOTAL]` uncapped. Change to:

```python
y=np.clip(cap[:, _COL_DV_TOTAL], DV_FLOOR, DV_CAP),
```

This keeps it captured-only (orbital error is meaningless for non-captured) but clips for visual consistency with the histogram.

### Change 4: Performance table uses clipped DV (line ~474)

Change the "Correction cost ΔV" row in the performance table to use all trajectories, clipped:

```python
"Correction cost ΔV (m/s)": np.clip(final_array[:, _COL_DV_TOTAL], DV_FLOOR, DV_CAP),
```

This replaces `cap[:, _COL_DV_TOTAL]` (captured-only, uncapped).

### Change 5: Dispersion grid uses clipped DV

In `_build_dispersion_grid()` (line ~808), the y-axis for all 24 dispersion correlation subplots uses `final_array[captured, _COL_DV_TOTAL]` uncapped. Change to:

```python
cap_dv = np.clip(final_array[captured, _COL_DV_TOTAL], DV_FLOOR, DV_CAP)
```

This prevents outlier barely-captured trajectories from compressing the useful correlation range.

### What stays the same

- Orbital error metrics (apo/peri/inclination) — captured-only, unchanged
- Exit conditions scatter marker sizing — uses `dv_all / 20` with its own clip(3, 15); marker size range shifts slightly with capped data but remains visually appropriate
- Convergence report (`report.py`) — already uses log scale
- Corridor PNG — `dv_captured` is used for colormap scaling of MC spaghetti lines, not as an axis; capping would subtly change the color distribution but is not worth the complexity
- Rust DV computation — unchanged

## Testing

- Existing final report tests pass (clipping doesn't break shape/type contracts)
- Add a test with synthetic data where some trajectories have DV > 5000, verifying the clipped values appear in the plot data
- Add a test with DV=0 values, verifying the `DV_FLOOR` epsilon prevents log-scale rendering issues
- Optionally inspect Plotly figure JSON to confirm `xaxis.type == "log"` on DV panels
