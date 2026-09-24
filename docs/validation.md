# Physics validation

How far the simulator's physics can be trusted, and on what evidence. "Validation" here means
physics validation, not the reserved validation seed pool of the training loop.

| Evidence | What it establishes | Gate |
|---|---|---|
| Legacy reference match | the Rust port reproduces the reference implementation | the six guidance goldens |
| Vacuum conservation | integrator and gravity conserve the two-body invariants | `tests/test_vacuum_conservation.py` |
| AMAT cross-check | dynamics, density lookup, loads and heating agree with an independent public tool on our vehicle | `tests/test_external_validation.py` (slow) |
| Published result | the simulator reproduces a published Mars aerocapture entry corridor | `tests/test_external_validation.py` (slow) |

Nothing in the simulator changed for the last two; no golden moved.

## 1. Legacy reference

The Rust simulator was validated against a reference implementation across all 725 timesteps of a
guided FTC trajectory: 22 of 24 photo output columns are bit-identical; the remaining 2 differ only
at the first timestep, where the reference read uninitialized variables. This proves the port is
faithful to that code. It does not prove the code is right, and an outside reader cannot inspect
the reference.

## 2. Vacuum two-body conservation

With a near-zero atmosphere, zonal harmonics and rotation zeroed and a spherical planet, the flown
trajectory is pure two-body motion. Over 371 samples the fixed-step Gill RK4 integrator conserves
specific energy to about 2e-14 and specific angular momentum to about 3e-15 (relative); the gate
is 1e-11.

## 3. Cross-check against AMAT

### The oracle

AMAT 2.4.0 (Aerocapture Mission Analysis Tool, Girija et al., JOSS 6(67), 3710, 2021; source
commit `e6f4d5adb51abdc8b25fea01e718c8f416852863`), an open-source Python tool for aerocapture
conceptual design. It ran once, in a scratch venv, through
`experiments/external_validation/amat_oracle.py`; its outputs and the inputs it flew are frozen in
`experiments/external_validation/amat_frozen.json`. AMAT is not a dependency: the gate re-flies our
side against the JSON and first asserts that the Rust-resolved configs still state the inputs AMAT
flew. What AMAT does and where its conventions differ from ours is in
[docs/research/2026-09-23-amat-capabilities.md](research/2026-09-23-amat-capabilities.md).

### What is matched, what differs

`configs/validation/amat_matched.toml` is the MSR mission at a constant bank held from t = 0:
1089 kg, 14.7 m2, Ca/Cn at the -27.5 deg trim (beta = 60.71 kg/m2, L/D = 0.3312), entry at 130 km,
5687.16 m/s, -10.8125 deg, azimuth 38.04 deg on the equator.

- Identical in both tools: mu, equatorial radius, rotation rate and J2; the density table
  `data/atmosphere/mars.dat`, exported to AMAT's column format and interpolated linearly as the Rust
  lookup does (AMAT's cubic default would not match); exit at 130.988 km and crash at the surface;
  beta and L/D.
- Switched off on our side: oblateness (polar radius set to the equatorial one) and J4, which AMAT
  lacks, and J3, which AMAT has but the issue's J2-only model excludes. Winds and every dispersion
  are already off in the mission base.
- Conventions mapped: AMAT's heading runs from east toward north (`90 deg - azimuth`); its positive
  bank turns left and ours right, so AMAT flies `delta = -bank`; AMAT's accelerations are divided by
  our G0 = 9.81 instead of its 9.80665.
- Remaining differences, each named in the tables: the integrator (our fixed-step Gill RK4 at 1 s
  against AMAT's `solve_ivp` RK45 at rtol = atol = 1e-10), peak sampling (1 s ticks against 0.1 s
  output), AMAT's regularized heading equation, and the heating law in the rows labelled
  Sutton-Graves.

Cases: `lift_up` (0 deg, captures with an 11,032 km apoapsis), `mid_bank` (60 deg, inside the
capture band with a 1159 km apoapsis; the 500 km target sits near 63.3 deg and the band ends between
66 and 67 deg), `lift_down` (180 deg, surface impact at 188 s). Then the lift-modulation entry corridor (entry
flight-path angle interval) at this entry state for the mission's 500.13 km target apoapsis: overshoot bound at full lift-down,
undershoot bound at full lift-up, AMAT's own `findOverShootLimit2` / `findUnderShootLimit2` against
our bisection.

### Tolerances

Each tolerance is the budget of the named difference in the Explanation column, not a fit to the
observed gap. They were fixed after a first comparison run, which surfaced the bank sign convention
(a 29% apoapsis gap on `mid_bank` that vanished once mapped). Margins run from just over 1x (the `lift_up`
flight time, 0.93 s of a 1 s budget) and about 2x (the Sutton-Graves rows) to over 1000x; about half the rows sit more than 10x inside their
tolerance. The author adopted them on 2026-09-23, together with the 60 deg mid case, the nose
radius matched at entry speed and the entry corridor as the corridor comparable.

| Quantity | Tolerance | Budget |
|---|---|---|
| Exit inertial energy | 1e-3 relative | AMAT's heading regularization (2.2e-4 on `mid_bank`) plus our 1 s RK4 step (5e-6) |
| Apoapsis altitude | 5e-3 relative | the energy gap amplified by d(apoapsis)/d(energy), about 6x at 1159 km |
| Periapsis altitude of the pass | 20 m | our value is the first 1 s tick past periapsis |
| Peak heat flux, peak load | 1e-3 relative | a 1 s tick misses a peak of width tau by at most (0.5 s / tau)^2 / 2 |
| Heat load | 1e-3 relative | our integral stops at the terminating tick, AMAT's at its last sample |
| Flight time | 1 s | fixed-step termination is quantized to the tick |
| Sutton-Graves rows | 2% relative | `(v0 / v)^0.05` across each flight's speed range |
| Corridor bounds | 0.01 deg | AMAT's bisection resolves 1e-4 deg, ours 1e-6 deg |

### Results

| Case | Quantity | Ours | AMAT | Ours - AMAT (rel) | Tolerance | Within | Explanation |
|---|---|---|---|---|---|---|---|
| lift_up | Outcome | exit | exit | | | yes | |
| mid_bank | Outcome | exit | exit | | | yes | |
| lift_down | Outcome | crash | crash | | | yes | |
| lift_up | Exit inertial energy (MJ/kg) | -2.39958 | -2.39959 | 6.58e-06 (-2.7e-06) | 0.001 rel | yes | AMAT's cos(gamma) + 0.01 heading regularization (diagnosed below); at full lift also our 1 s RK4 step (5e-6) |
| lift_up | Apoapsis altitude (km) | 11031.5 | 11031.5 | 0.0476386 (+4.3e-06) | 0.005 rel | yes | the exit-energy gap, amplified by d(apoapsis)/d(energy) |
| lift_up | Periapsis altitude of the pass (km) | 47.806 | 47.8051 | 8.88e-04 | 0.02 | yes | ours is the 1 s tick where the flight-path angle turns positive |
| lift_up | Peak heat flux, our law (kW/m2) | 163.341 | 163.347 | -0.00687557 (-4.2e-05) | 0.001 rel | yes | our peak is sampled at 1 s ticks, AMAT's at 0.1 s |
| lift_up | Peak aerodynamic load (g) | 2.01775 | 2.01778 | -2.47e-05 (-1.2e-05) | 0.001 rel | yes | our peak is sampled at 1 s ticks, AMAT's at 0.1 s |
| lift_up | Heat load, our law (MJ/m2) | 16.9368 | 16.9365 | 3.42e-04 (+2.0e-05) | 0.001 rel | yes | our integral stops at the terminating tick, AMAT's trapezoid at its last sample |
| lift_up | Atmospheric flight time (s) | 333 | 332.067 | 0.933132 | 1 | yes | fixed-step termination is quantized to 1 s: the record carries the first tick past the boundary |
| lift_up | Peak heat flux, AMAT's Sutton-Graves (kW/m2) | 163.341 | 163.941 | -0.600096 (-3.7e-03) | 0.02 rel | yes | Sutton-Graves v^3 vs our v^3.05, nose radius matched at entry speed: (v0/v)^0.05 |
| lift_up | Heat load, AMAT's Sutton-Graves (MJ/m2) | 16.9368 | 17.0352 | -0.0984282 (-5.8e-03) | 0.02 rel | yes | Sutton-Graves v^3 vs our v^3.05, nose radius matched at entry speed: (v0/v)^0.05 |
| mid_bank | Exit inertial energy (MJ/kg) | -5.394 | -5.39517 | 0.00117614 (-2.2e-04) | 0.001 rel | yes | AMAT's cos(gamma) + 0.01 heading regularization (diagnosed below); at full lift also our 1 s RK4 step (5e-6) |
| mid_bank | Apoapsis altitude (km) | 1159.35 | 1157.69 | 1.65873 (+1.4e-03) | 0.005 rel | yes | the exit-energy gap, amplified by d(apoapsis)/d(energy) |
| mid_bank | Periapsis altitude of the pass (km) | 44.6106 | 44.6097 | 9.10e-04 | 0.02 | yes | ours is the 1 s tick where the flight-path angle turns positive |
| mid_bank | Peak heat flux, our law (kW/m2) | 174.163 | 174.185 | -0.0213059 (-1.2e-04) | 0.001 rel | yes | our peak is sampled at 1 s ticks, AMAT's at 0.1 s |
| mid_bank | Peak aerodynamic load (g) | 2.59786 | 2.5979 | -4.45e-05 (-1.7e-05) | 0.001 rel | yes | our peak is sampled at 1 s ticks, AMAT's at 0.1 s |
| mid_bank | Heat load, our law (MJ/m2) | 19.8792 | 19.8797 | -4.86e-04 (-2.4e-05) | 0.001 rel | yes | our integral stops at the terminating tick, AMAT's trapezoid at its last sample |
| mid_bank | Atmospheric flight time (s) | 473 | 472.51 | 0.490397 | 1 | yes | fixed-step termination is quantized to 1 s: the record carries the first tick past the boundary |
| mid_bank | Peak heat flux, AMAT's Sutton-Graves (kW/m2) | 174.163 | 174.965 | -0.801794 (-4.6e-03) | 0.02 rel | yes | Sutton-Graves v^3 vs our v^3.05, nose radius matched at entry speed: (v0/v)^0.05 |
| mid_bank | Heat load, AMAT's Sutton-Graves (MJ/m2) | 19.8792 | 20.0635 | -0.184315 (-9.2e-03) | 0.02 rel | yes | Sutton-Graves v^3 vs our v^3.05, nose radius matched at entry speed: (v0/v)^0.05 |
| lift_down | Peak heat flux, our law (kW/m2) | 228.78 | 228.798 | -0.0184907 (-8.1e-05) | 0.001 rel | yes | our peak is sampled at 1 s ticks, AMAT's at 0.1 s |
| lift_down | Peak aerodynamic load (g) | 11.6156 | 11.6165 | -8.88e-04 (-7.6e-05) | 0.001 rel | yes | our peak is sampled at 1 s ticks, AMAT's at 0.1 s |
| lift_down | Heat load, our law (MJ/m2) | 15.3216 | 15.3206 | 0.00102155 (+6.7e-05) | 0.001 rel | yes | our integral stops at the terminating tick, AMAT's trapezoid at its last sample |
| lift_down | Atmospheric flight time (s) | 189 | 188.279 | 0.720982 | 1 | yes | fixed-step termination is quantized to 1 s: the record carries the first tick past the boundary |
| lift_down | Peak heat flux, AMAT's Sutton-Graves (kW/m2) | 228.78 | 230.871 | -2.09136 (-9.1e-03) | 0.02 rel | yes | Sutton-Graves v^3 vs our v^3.05, nose radius matched at entry speed: (v0/v)^0.05 |
| lift_down | Heat load, AMAT's Sutton-Graves (MJ/m2) | 15.3216 | 15.4823 | -0.160744 (-1.0e-02) | 0.02 rel | yes | Sutton-Graves v^3 vs our v^3.05, nose radius matched at entry speed: (v0/v)^0.05 |

### The banked residual is AMAT's heading regularization

The largest gap above sampling level is `mid_bank`'s exit energy (-2.2e-4) and apoapsis (+0.14%).
AMAT divides every `1 / cos(gamma)` of its heading equation by `cos(gamma) + 1e-2`, which only
matters when the lift has a lateral component. The oracle also flies `mid_bank` with those three
factors made exact (`exact_heading_eom2`, every other term AMAT's own), and our side flies it with
DOPRI45 at rtol 1e-12 with the exit located by event:

| Case | Quantity | Ours, DOPRI45 rtol 1e-12 | AMAT, exact heading | Ours - AMAT (rel) | Tolerance | Within | Explanation |
|---|---|---|---|---|---|---|---|
| mid_bank | Exit inertial energy (MJ/kg) | -5.39399 | -5.39399 | 4.21e-07 (-7.8e-08) | 1e-06 rel | yes | converged integrators, identical models |
| mid_bank | Apoapsis altitude (km) | 1159.35 | 1159.35 | 5.22e-04 (+4.5e-07) | 1e-05 rel | yes | converged integrators, identical models |
| mid_bank | Atmospheric flight time (s) | 472.429 | 472.429 | -3.21e-05 | 0.01 | yes | exit located by event (ours) and by interpolation between 0.1 s samples (AMAT) |

With the regularization removed and both integrators converged, the two tools agree to 8e-8 in
energy, 0.5 m in apoapsis and 3e-5 s in exit time. Our default 1 s RK4 is itself within 5 m of the
converged apoapsis on this case (1159.350 against 1159.355 km).

### Entry corridor

| Corridor | Bound (deg) | Ours | AMAT | Ours - AMAT (rel) | Tolerance | Within | Explanation |
|---|---|---|---|---|---|---|---|
| lift modulation | overshoot_efpa_deg | -9.46627 | -9.46622 | -5.76e-05 | 0.01 | yes | AMAT's bisection resolves 1e-4 deg, ours 1e-6 deg; full lift up/down has no lateral force |
| lift modulation | undershoot_efpa_deg | -11.9179 | -11.9178 | -6.90e-05 | 0.01 | yes | AMAT's bisection resolves 1e-4 deg, ours 1e-6 deg; full lift up/down has no lateral force |
| lift modulation | width_deg | 2.45161 | 2.4516 | 1.14e-05 | 0.01 | yes | AMAT's bisection resolves 1e-4 deg, ours 1e-6 deg; full lift up/down has no lateral force |

`training_output/mars/corridor_boundaries.npz`, suggested in #112 as a second comparable, is an
envelope in energy-dynamic pressure space built from flown trajectories, not an entry flight-path
angle interval; the corridor above is the quantity AMAT computes, flown directly by both tools.

### AMAT's own heating law

AMAT's Mars heating is convective Sutton-Graves, `1.8980e-8 sqrt(rho / RN) v^3` W/cm2; ours is
`cq sqrt(rho) v^3.05` W/m2 with no nose radius. With RN = 2.234 m the two coincide at the entry
speed; below it Sutton-Graves reads higher by `(v0 / v)^0.05`, 0.4 to 1.0% on these flights (the
Sutton-Graves rows above). Our `cq` is a vehicle input, so this row checks consistency with the
standard correlation for a plausible nose radius, not the value of `cq`.

### Findings

- AMAT `computeAccelerationLoad` computes `sqrt(a_s^2 + a_n^2 + a_w*2)`, a typo for `a_w**2`: the
  reported peak load is right at bank 0/180, under-reads at a positive AMAT bank (2.536 g against
  2.605 g on our vehicle at +60 deg) and is NaN at a negative one. The table uses the norm of AMAT's
  own `a_s`, `a_n`, `a_w`.
- AMAT's heading-equation regularization, above.
- AMAT's CD(Mach) path calls `np.float`, removed in numpy 1.24; not exercised here (constant CD).

## 4. A published result reproduced

Girija, "Aerocapture Design Reference Missions for Solar System Exploration: from Venus to
Neptune", arXiv:2308.10384 (2023), Table 1, Mars smallsat row: a drag-modulation smallsat (37 kg,
1.767 m2, beta1 = 20 kg/m2, beta2/beta1 = 7.5) entering at 120 km and 5.358 km/s for a 2000 km
apoapsis has the corridor [-9.86, -8.78] deg, 1.09 deg wide. The numbers come from AMAT's notebook
`docs/source/mdpi-aerospace-notebooks/smallsat-mission-concepts/section-3-4-mars-smallsat-nominal-aerocapture-trajectory.ipynb`,
which fixes the full entry state (longitude 88.15 deg, latitude -0.65 deg, heading 8.5458 deg from
east).

Each bound is one ballistic flight: the overshoot bound keeps the drag skirt (beta1) throughout,
the undershoot bound jettisons it at entry (beta2). Our simulator flies both through
`configs/validation/mars_smallsat_drm.toml` (`cn = 0`, so L/D = 0; beta2 by scaling the reference
area) with its full Mars model: oblate planet, J2 to J4, our MarsGram 3.8 table.

Tolerance: 0.1 deg on each bound and on the width. The published case uses AMAT's Mars-GRAM mean
table; ours is a different table, and AMAT alone moves the overshoot bound by 0.063 deg when its
table and constants are swapped for ours.

| Bound (deg) | Published | AMAT, notebook settings | AMAT, our Mars | Ours | Ours - published | Tolerance | Within |
|---|---|---|---|---|---|---|---|
| overshoot_efpa_deg | -8.78 | -8.7811 | -8.7181 | -8.7183 | +0.0617 | 0.1 | yes |
| undershoot_efpa_deg | -9.86 | -9.8682 | -9.8507 | -9.8514 | +0.0086 | 0.1 | yes |
| width_deg | 1.09 | 1.0871 | 1.1326 | 1.1331 | +0.0431 | 0.1 | yes |


- AMAT 2.4.0 with the notebook's settings reproduces the published bounds to the two decimals
  quoted (the paper truncates -9.868 to -9.86).
- AMAT with our Mars (our table, our constants, spherical) and our simulator agree to 7e-4 deg, the
  residual being the oblateness and J4 that only our side flies.
- So all of our gap to the published corridor (+0.009 and +0.062 deg on the bounds, +0.043 deg on
  the width) is the atmosphere model, not the dynamics.

The published number was itself produced with AMAT by AMAT's author, so this section is not a
second independent tool: it shows that our simulator, flown with its own planet and atmosphere,
lands on a published Mars corridor within the spread the atmosphere model explains.

## Reproduce

```bash
uv run pytest tests/test_external_validation.py -q
```

```bash
uv run python -m aerocapture.physics_crosscheck
```

The second prints the tables of sections 3 and 4; the gate fails if they differ from this page, if a
validation config no longer states the inputs AMAT flew, or if the oracle script changed after its
outputs were frozen. Regenerating the frozen AMAT outputs (scratch
venv, AMAT 2.4.0, an AMAT git checkout for `atmdata/`) is described at the top of
`experiments/external_validation/amat_oracle.py`.
