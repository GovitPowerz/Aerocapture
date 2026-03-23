# Cap DV at 5000 m/s + Log Scale in Final Report

**Date:** 2026-03-23
**Scope:** `src/python/aerocapture/training/final_report.py`

## Problem

The final evaluation report's DV distribution plots auto-scale to accommodate virtual DV penalties (10,000–20,000 m/s for non-captured trajectories), making the captured trajectory distribution (100–2,600 m/s) unreadable. The performance table also shows these inflated values.

## Decision

Clip all DV values at 5,000 m/s and use log scale on DV axes. Include all trajectories (captured + non-captured) in DV plots and table, clipped uniformly.

## Design

### Constant

```python
DV_CAP = 5000.0  # m/s — clip virtual DV penalties for plot readability
```

Defined at module level alongside existing column-index constants.

### Change 1: Include all trajectories, clipped (line ~254)

Currently only captured trajectories are used for DV plots. Change to use all trajectories with clipping:

```python
dv_total = np.clip(final_array[:, _COL_DV_TOTAL], 0, DV_CAP)
dv1 = np.clip(final_array[:, _COL_DV1], 0, DV_CAP)
dv2 = np.clip(final_array[:, _COL_DV2], 0, DV_CAP)
dv3 = np.clip(final_array[:, _COL_DV3], 0, DV_CAP)
```

These four arrays move **outside** the `if n_captured > 0` block so they are computed for all trajectories. The histogram and individual burns plots use these clipped arrays regardless of capture status.

The orbital error metrics (apo/peri/inclination histograms, DV-vs-error scatter) remain captured-only — they are only meaningful for captured trajectories.

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

### Change 3: Performance table uses clipped DV (line ~474)

Change the "Correction cost ΔV" row in the performance table to use all trajectories, clipped:

```python
"Correction cost ΔV (m/s)": np.clip(final_array[:, _COL_DV_TOTAL], 0, DV_CAP),
```

This replaces `cap[:, _COL_DV_TOTAL]` (captured-only, uncapped).

### What stays the same

- Orbital error metrics (apo/peri/inclination) — captured-only, unchanged
- Exit conditions scatter marker sizing — already uses all trajectories
- Convergence report (`report.py`) — already uses log scale
- Corridor PNG — unchanged
- Rust DV computation — unchanged

## Testing

- Existing final report tests should still pass (clipping doesn't break shape/type contracts)
- Visual verification: open a generated report and confirm log scale + 5000 m/s cap produce a readable distribution
