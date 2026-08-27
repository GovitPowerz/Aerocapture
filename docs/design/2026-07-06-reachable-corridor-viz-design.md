# Reachable-corridor visualization

**Date:** 2026-07-06
**Status:** design (awaiting review)
**Author:** Grégory Gelly (with Claude)

## Goal

Replace `fig_corridor`'s current four-envelope corridor (built from the piecewise-constant GA
`CorridorAccumulator` cache) with a **first-principles reachable corridor** traced by a large,
dispersed, randomized piecewise-constant Monte-Carlo run in the (orbital energy, dynamic pressure)
plane:

- **Upper boundary** (crash-side edge): the `p99.5` of dynamic pressure per energy bin among all
  trajectories that **capture** (`ifinal == 3 & eccentricity < 1`) -- the deepest-diving captures.
- **Lower boundary** (escape-side edge): the `p0.5` of dynamic pressure per energy bin among
  trajectories that capture into a **usefully-bound orbit** (`apoapsis_alt_km < 5000`) -- the
  shallowest useful captures.

The band between is the operational reachable corridor. The figure overlays the deployed Mamba
policy's dispersed ensemble and undispersed nominal, showing the flown trajectories nest inside it.

## Decisions (from the brainstorm)

- **Environment dispersions ON** (the paper's controlled MC regime) + randomized bank -> the corridor
  is the operational reachable envelope under uncertainty, consistent with how the policy is evaluated.
- **Boundary = high percentile + light smoothing** (p99.5 / p0.5), robust to outliers, not absolute
  max/min.
- **Overlay:** shaded band + deployed Mamba ensemble spaghetti + undispersed nominal.
- **Bank sampling:** N piecewise segments, each uniform in [0deg, 180deg] (positive / in-plane, no
  roll reversals; 0 = full lift-up, 180 = full lift-down). **N is a CLI parameter** (default 10).
- **Scale:** **n_sims is a CLI parameter** (default 300,000; scale to 1M if the boundaries look
  under-populated). Batched so trajectory data never all lives in RAM.

## Architecture -- collector-vs-figure split (unchanged pattern)

The collector reads `training_output/` + configs and does the heavy MC; the committed `corridor.npz`
+ the figure script are the durable artifacts. Two files change.

### 1. `articles/paper/scripts/collect_corridor.py` (rewrite)

**CLI:** `--n-sims 300000 --n-segments 10 --apoapsis-max-km 5000 --n-energy-bins 200
--chunk-size 20000 --upper-pct 99.5 --lower-pct 0.5 --smooth-sigma 2.5 --ensemble-sims 200`

**Base config:** the piecewise-constant training TOML
(`configs/training/msr_aller_piecewise_constant_train.toml`), which inherits the mission +
the paper's controlled dispersion regime and pins `guidance.type = "piecewise_constant"`. Per-sim
overrides supply the random bank and the dispersion seed; `simulation.n_sims = 1` per override.

**Randomized-bank sampling (reproducible):**
```
rng = np.random.default_rng(CORRIDOR_BANK_SEED)          # fixed -> reproducible corridor
banks = rng.uniform(0.0, 180.0, size=(chunk_n, N))       # per-sim, per-segment
override_j = {"simulation.n_sims": 1,
              "monte_carlo.seed": CORRIDOR_SEED_OFFSET + global_index_j,   # env dispersion draw
              "guidance.piecewise_constant.n_segments": N,
              **{f"guidance.piecewise_constant.bank_angle_{i}": banks[j, i] for i in range(N)}}
```
`CORRIDOR_SEED_OFFSET` is a dedicated offset (e.g. `10_000_000`) disjoint from the reserved pools.

**Batched streaming histogram (memory-bounded percentile):** two 2-D count histograms of shape
`(n_energy_bins, n_pdyn_buckets)` (energy over [-6, +5] MJ/kg; pdyn over [0, PDYN_MAX ~= 2.6 kPa],
`n_pdyn_buckets ~= 400`). For each chunk:
```
batch = aerocapture_rs.run_batch(base_toml, overrides_chunk, include_trajectories=True, sim_timeout_secs=5)
recs, trajs = batch.final_records, batch.trajectories
cap = (recs[:, IFINAL] == 3) & (recs[:, ECC] < 1)
apo = recs[:, APOAPSIS_ALT_KM]
for j where cap[j]:
    E, P = traj_j[:, _TC_ENERGY], traj_j[:, _TC_PDYN]
    ei = digitize(E, energy_edges); pi = digitize(clip(P, 0, PDYN_MAX), pdyn_edges)
    np.add.at(hist_upper, (ei, pi), 1)
    if apo[j] < apoapsis_max_km: np.add.at(hist_lower, (ei, pi), 1)
del batch, trajs   # free the chunk
```
After all chunks: per energy bin, read the pdyn value at cumulative fraction `upper_pct/100`
(upper, from `hist_upper`) and `lower_pct/100` (lower, from `hist_lower`) via the cumulative
histogram. Interpolate NaN gaps (empty bins) over energy, then `gaussian_filter1d(sigma=smooth_sigma)`
-- the same fill+smooth pattern as `corridor.py::CorridorAccumulator.to_npz`.

**Overlay data (reuse current logic):** run the deployed Mamba ensemble (`--ensemble-sims`, dispersed,
`include_trajectories`) pinned to the committed-bundle `best_model.json` + scaffolding, subsample to
~200 trajectories x every-3rd point; plus the undispersed nominal via `reference.nominal_flight_overrides`.

**Output `articles/paper/data/corridor.npz`:** `energy_bins, upper_pdyn, lower_pdyn,
nominal_energy, nominal_pdyn, ens_energy (object array), ens_pdyn (object array)`, plus scalar
metadata `n_sims, n_segments, apoapsis_max_km, upper_pct, lower_pct` for provenance in the caption.

A run-summary print reports capture rate and how many energy bins are populated (a sanity gate: if
many bins are empty at the tails, bump `--n-sims`).

### 2. `articles/paper/scripts/fig_corridor.py` (rewrite)

- `fill_between(energy_bins, lower_pdyn, upper_pdyn, color=C["mamba"], alpha=0.15)` -- the corridor band.
- Plot `upper_pdyn` and `lower_pdyn` as boundary lines (thin, `C["mamba"]`).
- Deployed Mamba ensemble spaghetti inside (subsampled, low alpha, `C["mamba"]`), like now.
- Undispersed nominal line (black, on top).
- `axvline(0)` E=0 divider + "bound (E<0)" / "hyperbolic (E>0)" labels.
- figlib STIX-serif style; drop the four-zone fills entirely.

## Data flow

```
collect_corridor.py  (piecewise MC + deployed-policy ensemble, PyO3 batched)
        v  writes (committed)
articles/paper/data/corridor.npz
        v  read by
articles/paper/scripts/fig_corridor.py  ->  fig_corridor.svg  ->  paper.typ (Figure 1)
```

## Caption update (paper.typ)

Rewrite the Figure-1 caption to describe the new construction: "The shaded band is the reachable
capture corridor, traced by a 300k-run dispersed Monte-Carlo of randomized piecewise-constant bank
profiles: the upper edge is the p99.5 dynamic pressure of capturing trajectories (the crash-side
limit), the lower edge the p0.5 of trajectories capturing below a 5000 km apoapsis (the escape-side
limit). The deployed Mamba ensemble and its nominal fly well inside it."

## Non-goals / YAGNI

- Not touching the training `CorridorAccumulator` / `corridor_boundaries.npz` (still used by the
  training-report corridor panels and the appendix). Only the paper's `fig_corridor` changes.
- No new Rust code -- per-sim bank + n_segments go through existing TOML dot-path overrides.
- No inclination/bank corridor variants here (this is the (energy, pdyn) figure only).

## Cost & risks

- **Compute:** 300k piecewise sims with dispersions, batched. Piecewise is a fast scheme; estimate
  ~3-5 min. Scaling to 1M is ~10-15 min. One-time, committed.
- **Memory:** bounded by the two `(200 x 400)` histograms + one 20k-sim chunk of trajectories
  (~0.7 GB/chunk), freed each iteration -- never the full 300k in RAM.
- **Capture yield:** random banks under dispersions -> many crash/escape, but even a 10-30% capture
  rate over 300k gives 30k-90k capturing trajectories, ample for stable p99.5/p0.5. The summary
  print flags thin tails.
- **Override contract:** per-sim `guidance.piecewise_constant.n_segments` + `bank_angle_i` overrides
  must be honored by `run_batch`. Verify on a 100-sim smoke run before the full sweep (a wrong
  override path would silently fly the config's default bank).
- **pdyn clip:** points above `PDYN_MAX` are clipped into the top bucket; these are deep divers that
  crash (not captures), so they do not affect the capture-only histograms. Choose `PDYN_MAX` from a
  quick max-pdyn probe so p99.5 is never clipped.

## Testing / verification

- 100-sim smoke: confirm the bank overrides take effect (trajectories differ from the default-bank
  run; range of peak pdyn spans shallow-to-deep).
- After the full run: `capture rate` printed, `>= 95%` of energy bins in the operating range
  populated; visually inspect that upper > lower everywhere and both are smooth.
- Compile the paper; the new Figure 1 renders with the band + ensemble + nominal, boundaries
  monotone-ish and smooth, no empty-bin gaps in the operating energy range.
