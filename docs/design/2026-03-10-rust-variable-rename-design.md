# Design: Full Sweep Rust Variable Rename

**Date**: 2026-03-10
**Status**: Complete
**Scope**: Replace all French/Fortran legacy variable names with explicit English names across the entire Rust codebase, including tests. Strip all Fortran-origin comments.

## Decisions

- **Scope**: Full sweep — struct fields, locals, test variables, comments
- **Fortran comments**: Strip entirely (git history is the archaeologist's tool)
- **Approach**: Bottom-up, file-by-file (leaf modules first, core structs last)
- **Verification**: `cargo build` after each file, `cargo test` after each tier

## Naming Conventions

| Domain term | Old patterns | New convention |
|---|---|---|
| Atmospheric density | `ro`, `rho`, `coefro` | `density`, `density_gain` |
| Bank angle | `git`, `gite` | `bank_angle` |
| Velocity | `vit`, `vites` | `velocity` |
| Energy | `enr`, `enrj` | `energy` |
| Dynamic pressure | `pdyn` | `dynamic_pressure` |
| Damping | `amor` | `damping` |
| Pulsation/frequency | `puls` | `frequency` |
| Mass | `xmasse` | `mass` |
| Reference area | `srefer`, `sref` | `reference_area` |
| Number/count | `nb` prefix | `n_` prefix |
| Integer flags | `i` prefix (`ibounc`, `iphase`) | descriptive (`bounce_flag`, `guidance_phase`) |
| Fortran `x` prefix | `xinccr`, `xinmax` | drop prefix (`inclination_error`, `inclination_max`) |
| RK4 internals | `xk`, `qk`, `ix` | `step_increment`, `accumulator`, `gill_toggle` |

## Rename Tiers (Bottom-Up)

### Tier 1 — Leaf modules (no downstream dependents)
- `integration/rk4.rs` — 3 renames (`xk`→`step_increment`, `qk`→`accumulator`, `ix`→`gill_toggle`)
- `physics/` modules — any remaining legacy locals
- `gnc/navigation/coordinates.rs` — locals only

### Tier 2 — Mid-level modules
- `data/guidance_params.rs` — strip Fortran mapping comments (fields already English)
- `gnc/control/pilot.rs` — check for legacy names
- `gnc/guidance/equilibrium_glide.rs` — locals
- `gnc/guidance/energy_controller.rs` — locals
- `gnc/guidance/predguid.rs` — locals
- `gnc/guidance/fnpag.rs` — locals

### Tier 3 — Core GNC structs (highest fan-out)
- `gnc/guidance/ftc.rs` — FtcState (~14 field renames), FtcOutput, all locals (~20+)
- `gnc/navigation/estimator.rs` — NavigationState (~6 fields), NavigationOutput (~12 fields), locals

### Tier 4 — Wiring & consumers
- `data/mod.rs` (SimData) — cascading field renames
- `simulation/runner.rs` — largest consumer, update all references
- `simulation/init.rs` — initialization code
- `simulation/output.rs` — output formatting

### Tier 5 — Tests
- All inline `#[cfg(test)]` modules (updated in-tier with each file)
- `src/rust/tests/` integration tests — update struct field accesses and local names
- Strip all Fortran-reference comments in tests

## Not Changing
- File names (already English)
- Module structure
- Function signatures already in English
- Standard math/control-theory notation (`tau`, `zeta`, `omega` in pilot.rs)
- Python code (separate concern)

## Full Rename Mapping

### FtcState fields
| Old | New |
|-----|-----|
| `gitcom` | `bank_angle_commanded` |
| `gitpre` | `bank_angle_previous` |
| `gpilpr` | `pilot_bank_angle_previous` |
| `alfcom` | `aoa_commanded` |
| `sgngit` | `roll_sign` |
| `somgit` | `cumulative_bank_change` |
| `nbroll` | `n_reversals` |
| `indrvr` | `reversal_active` |
| `rolway` | `roll_way` |
| `trevrs` | `reversal_duration` |
| `iprepr` | `securization_counters` |
| `iguida` | `guidance_active` |
| `vitref` | `reference_velocity` |
| `vitgit` | `bank_rate` |

### NavigationState fields
| Old | New |
|-----|-----|
| `coefro` | `density_gain` |
| `vitpre` | `previous_radial_velocity` |
| `ibounc` | `bounce_flag` |
| `iphase` | `guidance_phase` |
| `tcaptr` | `capture_time` |

### FtcOutput fields
| Old | New |
|-----|-----|
| `gitcom` | `bank_angle_commanded` |
| `alfcom` | `aoa_commanded` |
| `vitgit` | `bank_rate` |
| `ilongi` | `longitudinal_active` |
| `isatur` | `rate_saturated` |
| `indrol` | `roll_reversal_active` |

### NavigationOutput fields
| Old | New |
|-----|-----|
| `positn` | `position_estimated` |
| `vitesn` | `velocity_estimated` |
| `acceln` | `acceleration_estimated` |
| `coefan` | `aero_coefficients` |
| `roguid` | `density_guidance` |
| `roexit` | `density_exit` |
| `pdynan` | `dynamic_pressure_estimated` |
| `energn` | `energy_estimated` |
| `ecartn` | `orbital_errors` |
| `vitref` | `reference_velocity` |
| `icrash` | `crash_flag` |
| `indext` | `phase_transition_flag` |

### FTC local variables (ftc.rs)
| Old | New |
|-----|-----|
| `sgnpre` | `previous_roll_sign` |
| `enrjlt` | `energy` |
| `gitlon` | `bank_angle_longitudinal` |
| `vitrel` | `velocity_relative` |
| `vitrad` | `velocity_radial` |
| `pdyneq` | `dynamic_pressure_equilibrium` |
| `cmunom` | `cos_bank_nominal` |
| `prenom` | `dynamic_pressure_nominal` |
| `hdtnom` | `altitude_rate_nominal` |
| `cosmuc` | `cos_bank_commanded` |
| `inumer` | `table_index` |
| `coefpd_a` | `pressure_coeff_a` |
| `amorft` | `damping_capture` |
| `pulsft` | `frequency_capture` |
| `srefer` | `reference_area` |
| `xmasse` | `mass` |
| `gaindh` | `gain_altitude_rate` |
| `gainpd` | `gain_dynamic_pressure` |
| `xinccr` | `inclination_error` |
| `xinmax` | `inclination_max` |
| `coridx` | `corridor_slope` |
| `coridy` | `corridor_intercept` |
| `dgitcm` | `bank_angle_change` |
| `vgitmx` | `max_bank_rate` |
| `tguida` | `guidance_period` |

### Navigation local variables (estimator.rs)
| Old | New |
|-----|-----|
| `roesti` | `density_estimated` |
| `vitrel` | `velocity_relative` |
| `vitrad` | `velocity_radial` |
| `dvitrd` | `delta_radial_velocity` |
| `acdram` | `drag_acceleration_measured` |

### Runner local variables (runner.rs)
| Old | New |
|-----|-----|
| `romver` | `density_estimate` |
| `xsauve` | `final_record` |
| `xenerg` | `energy` |
| `vitrad` | `velocity_radial` |
| `altitr` | `altitude` |
| `xlatit` | `latitude` |
| `enerjr` | `energy` |
| `isimul` | `sim_index` |
| `somgit` | `cumulative_bank_change` |
| `gitref` | `reference_bank_angle` |
| `degrad` | `DEG_TO_RAD` |

### RK4 variables (rk4.rs)
| Old | New |
|-----|-----|
| `xk` | `step_increment` |
| `qk` | `accumulator` |
| `ix` | `gill_toggle` |
