# AMAT as an external oracle: capabilities, conventions, pitfalls (2026-09-23)

Research note for issue #112 (independent physics validation). Sources: the AMAT 2.4.0 wheel from
PyPI (read line by line), the AMAT repository at commit `e6f4d5adb51abdc8b25fea01e718c8f416852863`
(<https://github.com/athulpg007/AMAT>, examples and `atmdata/`), the JOSS paper (Girija et al., JOSS
6(67), 3710, 2021, doi:10.21105/joss.03710), and two open-access papers whose Mars numbers AMAT
produced: Girija, "Aerocapture Design Reference Missions for Solar System Exploration: from Venus to
Neptune", arXiv:2308.10384 (2023), and Girija, "A Low Cost Mars Aerocapture Technology
Demonstrator", arXiv:2307.11378 (2023). The outcome is in [docs/validation.md](../validation.md).

## What it is

- Aerocapture Mission Analysis Tool, Purdue (Girija, Saikia, Longuski) with JPL (Cutts). MIT
  licence. `pip install AMAT` (2.4.0 installs cleanly on Python 3.12 with numpy 2.5 / scipy 1.18).
  Pure Python: `planet`, `vehicle` (7.8k lines, all trajectory work), `approach`, `interplanetary`,
  `launcher`, `orbiter`, `telecom`, `visibility`.
- The wheel ships no atmosphere data. `atmdata/<Planet>/` (Mars-GRAM mean table
  `mars-gram-avg.dat`, GRAM Monte Carlo profiles `LAT*.txt`) lives only in the git repository.
- The JOSS paper positions it as a low-to-mid fidelity conceptual design tool (its words), for
  corridor widths, feasibility charts and Monte Carlo trade studies.

## Dynamics

- 3-DOF point mass over a rotating spherical planet in Vinh's planet-relative variables (r,
  longitude, latitude, v, heading, flight-path angle, downrange), non-dimensionalized by the planet
  radius `RP` and `sqrt(GM/RP)`. Centrifugal and Coriolis terms included.
- Gravity: point mass + J2 + J3 (`grbar`, `gphibar`). No J4, no oblate surface: altitude is
  `r - RP` everywhere.
- Heading is measured from east toward north (`dtheta/dt = v cos(gamma) cos(psi) / (r cos(phi))`).
  Our azimuth is from north toward east, so `psi_AMAT = 90 deg - azimuth`.
- Bank: lift components `L cos(delta)` in the vertical plane and `L sin(delta)` lateral, with the
  lateral term entering `dpsi/dt` positively. With AMAT's counter-clockwise heading a positive bank
  turns left; ours turns right. Full lift up/down (0 / 180 deg) are unaffected; any other bank maps
  as `delta_AMAT = -bank_ours`.
- The heading equation is regularized: its three `1 / cos(gamma)` factors (the lateral-force term
  in `EOM`/`EOM2`, `cfpsibar`, `copsibar`) are `1 / (cos(gamma) + 1e-2)`. At a 60 deg bank on our MSR case
  this alone moves exit energy by 2.2e-4 and apoapsis by 0.14% (measured with the terms removed;
  [docs/validation.md](../validation.md)).
- Aerodynamics: constant CD and CL from `beta`, `L/D` and area (`CD = m / (beta A)`,
  `CL = (L/D) CD`) unless a user CD(Mach) function is set. That path calls `np.float`, removed in
  numpy 1.24: with `userDefinedCDMach=True`, 2.4.0 raises on current numpy.
- Integrators: `propogateEntry` (scipy `odeint`) and the `...2` variants (`solve_ivp` RK45 with a
  terminal event at gamma = -88 deg), `rtol = atol = tol` on the non-dimensional state.

## Atmosphere input

`Planet.loadAtmosphereModel(datfile, heightCol, tempCol, presCol, densCol, intType='cubic',
heightInKmFlag=False)`: a whitespace table read with `np.loadtxt`; `scipy.interp1d` of each column
with `intType` in {linear, quadratic, cubic}. Temperature and pressure feed only the speed of sound
(Mach, stagnation temperature and pressure). Density is 0 above `h_thres`, `rho0` below the
surface. Our tables (`data/atmosphere/*.dat`, altitude and density only, Fortran `D` exponents,
linear interpolation in `atmosphere.rs`) export to it with placeholder T/p columns and
`intType='linear'`, which makes the two density models identical inside the table.

## Termination and classification

`classifyTrajectory` truncates the propagated arrays: exit at the first sample above `h_skip`,
crash at the first sample below `h_trap`. Mars defaults are `h_skip = h_thres = 120 km`,
`h_trap = 10 km`; our MSR case enters at 130 km and exits at 130.988 km, so both are overridden.
`hitsTargetApoapsis2` treats a terminal altitude below `h_low` (50 km at Mars) as undershoot.

## Outputs

Per trajectory: time histories of altitude, speed, flight-path angle, heading, latitude,
longitude, downrange; `acc_net_g`, `acc_drag_g`; dynamic and stagnation pressure; convective,
radiative and total stagnation-point heat rate (W/cm2) and cumulative heat load (J/cm2).
Terminal apoapsis/periapsis (`compute_ApoapsisAltitudeKm`, inertial velocity via `OMEGA x r`),
periapsis-raise delta-v. Corridor searches: lift modulation `findOverShootLimit2` (bank 180) /
`findUnderShootLimit2` (bank 0) / `computeTCW2`; drag modulation `findOverShootLimitD2` (beta1) /
`findUnderShootLimitD2` (beta2 = ratio x beta1). Guided flight: equilibrium glide + exit phase for
lift modulation, single-event jettison with a predictor for drag modulation. Monte Carlo over GRAM
perturbation profiles.

Two output caveats:

- `computeAccelerationLoad` computes `sqrt(a_s^2 + a_n^2 + a_w*2)` (a typo for `a_w**2`) and
  divides by 9.80665. At bank 0/180 `a_w` is zero and the number is right; at a positive AMAT bank
  it under-reports (2.536 g against 2.605 g at 60 deg on our case); at a negative bank the square
  root goes negative and the peak is NaN. Our cross-check recomputes the norm from AMAT's own
  `a_s`, `a_n`, `a_w`.
- Mars heating is convective Sutton-Graves only, `1.8980e-8 sqrt(rho / RN) v^3` W/cm2, radiative
  set to 0. Ours is `cq sqrt(rho) v^3.05` W/m2 with no nose radius. Our `cq = 8.242e-5` equals
  Sutton-Graves at the MSR entry speed (5687 m/s) for RN = 2.234 m; below that speed the two laws
  drift apart as `(v / v0)^0.05` (0.4-1% over our flights).

## Published Mars cases AMAT reproduces

| Case | Where in AMAT | Control | Vehicle | Entry | Published result |
|---|---|---|---|---|---|
| Girija 2023 DRM, Mars smallsat | `docs/source/mdpi-aerospace-notebooks/smallsat-mission-concepts/section-3-4-*` | drag modulation | 37 kg, beta1 20 kg/m2, ratio 7.5 | 120 km, 5.3581 km/s, heading 8.5458 deg, lon 88.15, lat -0.65 | corridor [-9.86, -8.78] deg, 1.09 deg wide, 2000 km apoapsis (arXiv:2308.10384, Table 1) |
| Girija 2023 Mars technology demonstrator | (paper only) | drag modulation | 30 kg, beta 11.3 / 62.9 | 120 km, 5.6053 km/s, polar | corridor [-8.99, -7.93] deg (arXiv:2307.11378) |
| Werner & Braun 2019 smallsat flight test | `examples/example-51..53` | drag modulation | 25.97 kg, beta1 66.4, ratio 4.72 | 150 km, 5.74 km/s | corridor about 0.70 deg (JSR 56(6), paywalled; the number is the notebook's) |
| Viking 1/2, Pathfinder, DS-2, Beagle 2, MER, MSL, ExoMars EDL | `examples/example-20..54` | ballistic / lifting entry | | | reconstructions, no aerocapture |

Lift-modulation Mars corridors appear only as charts (TCW against entry speed, JSR comparative
study notebook `17-comparative-studies`), not as quotable numbers.

## Overlap with our vehicle class

Our MSR orbiter is lift-modulated: beta = 60.7 kg/m2 (m / (S Cx)), L/D = 0.331 at the -27.5 deg
trim, 5.687 km/s at 130 km. No published AMAT Mars case shares that class; every quotable Mars
aerocapture number is a drag-modulation smallsat (L/D = 0). Consequences for the cross-check:

1. The matched cases (same vehicle, planet, table) use AMAT as a dynamics oracle on our vehicle,
   including its native lift-modulation corridor search.
2. The published reproduction flies the DRM smallsat through our simulator. Each drag-modulation
   bound is a single ballistic flight at constant beta, which our constant-bank mode flies exactly
   with `cn = 0`. The DRM case is preferred over the demonstrator because its full entry state and
   AMAT settings are in the notebook, and over Werner & Braun because its number is in an
   open-access table rather than a paywalled paper.
