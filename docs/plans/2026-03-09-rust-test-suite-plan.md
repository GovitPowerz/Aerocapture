# Rust Test Suite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a standalone Rust test suite (unit + integration + E2E) so the simulator can be tested, refactored, and extended without depending on the legacy Fortran codebase.

**Architecture:** Three-tier testing pyramid. Unit tests use analytical ground truth (inline `#[cfg(test)]`). Integration tests use snapshots from the validated simulator (`src/rust/tests/`). E2E tests run full simulations from TOML configs and assert on physical invariants.

**Tech Stack:** Rust `#[test]`, `approx` (float comparison), `rstest` (parameterized tests), `serde_json` (snapshot I/O).

---

## Task 1: Add dev-dependencies

**Files:**
- Modify: `src/rust/Cargo.toml`

**Step 1: Add approx and rstest**

```toml
[dev-dependencies]
approx = "0.5"
rstest = "0.25"
```

**Step 2: Verify it compiles**

Run: `cd src/rust && cargo test --quiet 2>&1 | tail -5`
Expected: existing 14 tests pass, no compilation errors

**Step 3: Commit**

```bash
git add src/rust/Cargo.toml src/rust/Cargo.lock
git commit -m "chore: add approx and rstest dev-dependencies for test suite"
```

---

## Task 2: Unit tests — physics/gravity.rs

**Files:**
- Modify: `src/rust/src/physics/gravity.rs`

**Step 1: Write tests**

Append to `gravity.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn spherical_gravity_at_surface() {
        // With J2=0, gravity should be mu/r^2
        // Moon has negligible J2 (4.458e-6), use it as near-spherical
        let r = Planet::Moon.equatorial_radius();
        let (gravtl, gravtr) = gravity(r, 0.0, &Planet::Moon);
        let expected = Planet::Moon.mu() / (r * r);
        assert_relative_eq!(gravtr, expected, epsilon = 0.1); // J2 adds ~0.001%
        assert_relative_eq!(gravtl, 0.0, epsilon = 1e-6); // zero at equator
    }

    #[test]
    fn j2_lateral_zero_at_equator() {
        // At equator (lat=0), sin(lat)=0, so gravtl must be exactly 0
        let r = Planet::Mars.equatorial_radius() + 100e3;
        let (gravtl, _) = gravity(r, 0.0, &Planet::Mars);
        assert_eq!(gravtl, 0.0);
    }

    #[test]
    fn j2_lateral_zero_at_pole() {
        // At pole (lat=pi/2), cos(lat)=0, so gravtl must be 0
        let r = Planet::Mars.equatorial_radius() + 100e3;
        let lat = std::f64::consts::FRAC_PI_2;
        let (gravtl, _) = gravity(r, lat, &Planet::Mars);
        assert_relative_eq!(gravtl, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn j2_radial_stronger_at_pole() {
        // J2 makes gravity stronger at pole than equator (oblate planet)
        // gravtr_pole > gravtr_equator for same radius
        let r = Planet::Mars.equatorial_radius() + 200e3;
        let (_, gravtr_eq) = gravity(r, 0.0, &Planet::Mars);
        let (_, gravtr_pole) = gravity(r, std::f64::consts::FRAC_PI_2, &Planet::Mars);
        assert!(gravtr_pole > gravtr_eq,
            "pole gravity {gravtr_pole} should exceed equator {gravtr_eq}");
    }

    #[test]
    fn gravity_decreases_with_altitude() {
        let r_low = Planet::Mars.equatorial_radius() + 50e3;
        let r_high = Planet::Mars.equatorial_radius() + 300e3;
        let (_, gravtr_low) = gravity(r_low, 0.3, &Planet::Mars);
        let (_, gravtr_high) = gravity(r_high, 0.3, &Planet::Mars);
        assert!(gravtr_low > gravtr_high);
    }

    #[test]
    fn mars_surface_gravity_ballpark() {
        // Mars surface gravity ~3.72 m/s^2
        let r = Planet::Mars.equatorial_radius();
        let (_, gravtr) = gravity(r, 0.0, &Planet::Mars);
        assert_relative_eq!(gravtr, 3.72, epsilon = 0.05);
    }

    #[test]
    fn earth_surface_gravity_ballpark() {
        // Earth surface gravity ~9.81 m/s^2
        let r = Planet::Earth.equatorial_radius();
        let (_, gravtr) = gravity(r, 0.0, &Planet::Earth);
        assert_relative_eq!(gravtr, 9.81, epsilon = 0.05);
    }

    #[test]
    fn j2_lateral_symmetry() {
        // gravtl should be antisymmetric: gravtl(lat) = -gravtl(-lat)
        let r = Planet::Mars.equatorial_radius() + 150e3;
        let lat = 0.7; // ~40 deg
        let (gravtl_pos, _) = gravity(r, lat, &Planet::Mars);
        let (gravtl_neg, _) = gravity(r, -lat, &Planet::Mars);
        assert_relative_eq!(gravtl_pos, -gravtl_neg, epsilon = 1e-10);
    }

    #[test]
    fn j2_lateral_max_at_45_deg() {
        // sin(lat)*cos(lat) is maximized at lat=pi/4
        let r = Planet::Mars.equatorial_radius() + 100e3;
        let (gravtl_45, _) = gravity(r, std::f64::consts::FRAC_PI_4, &Planet::Mars);
        let (gravtl_30, _) = gravity(r, 0.5, &Planet::Mars); // ~28.6 deg
        let (gravtl_60, _) = gravity(r, 1.0, &Planet::Mars); // ~57.3 deg
        assert!(gravtl_45.abs() > gravtl_30.abs());
        assert!(gravtl_45.abs() > gravtl_60.abs());
    }
}
```

**Step 2: Run tests**

Run: `cd src/rust && cargo test physics::gravity --quiet`
Expected: all 9 tests pass

**Step 3: Commit**

```bash
git add src/rust/src/physics/gravity.rs
git commit -m "test: add unit tests for J2 gravity model"
```

---

## Task 3: Unit tests — physics/atmosphere.rs and data/atmosphere.rs

**Files:**
- Modify: `src/rust/src/physics/atmosphere.rs`

Tests exercise both the `density()` wrapper and `AtmosphereModel::density_at()`.

**Step 1: Write tests**

Append to `physics/atmosphere.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn make_test_atm() -> AtmosphereModel {
        // Simple 3-point table
        AtmosphereModel {
            n_points: 3,
            altitudes: vec![0.0, 50_000.0, 100_000.0],
            densities: vec![0.02, 0.001, 0.00005],
            ref_density: 0.02,
            scale_factor: 1.0 / 10_000.0, // H = 10 km
            ref_altitude: 0.0,
            gas_constant: 1.3,
            density_profile: crate::data::atmosphere::DensityProfile {
                altitudes: vec![],
                max_dispersion: vec![],
                slopes: vec![],
                intercepts: vec![],
            },
        }
    }

    #[test]
    fn exact_table_hit() {
        let atm = make_test_atm();
        let rho = atm.density_at(50_000.0);
        assert_relative_eq!(rho, 0.001, epsilon = 1e-15);
    }

    #[test]
    fn interpolation_midpoint() {
        let atm = make_test_atm();
        // Midpoint between 0 and 50km: linear interp of 0.02 and 0.001
        let rho = atm.density_at(25_000.0);
        let expected = 0.02 + 0.5 * (0.001 - 0.02);
        assert_relative_eq!(rho, expected, epsilon = 1e-15);
    }

    #[test]
    fn below_table_clamps() {
        let atm = make_test_atm();
        let rho = atm.density_at(-1000.0);
        assert_relative_eq!(rho, 0.02, epsilon = 1e-15);
    }

    #[test]
    fn above_table_uses_exponential() {
        let atm = make_test_atm();
        let alt = 120_000.0;
        let rho = atm.density_at(alt);
        let expected = atm.ref_density * (-atm.scale_factor * (alt - atm.ref_altitude)).exp();
        assert_relative_eq!(rho, expected, epsilon = 1e-15);
    }

    #[test]
    fn density_bias_positive() {
        let atm = make_test_atm();
        let rho_nominal = density(&atm, 50_000.0, 0.0);
        let rho_biased = density(&atm, 50_000.0, 0.1);
        assert_relative_eq!(rho_biased, rho_nominal * 1.1, epsilon = 1e-15);
    }

    #[test]
    fn density_bias_zero_is_nominal() {
        let atm = make_test_atm();
        let rho = density(&atm, 50_000.0, 0.0);
        assert_relative_eq!(rho, atm.density_at(50_000.0), epsilon = 1e-15);
    }

    #[test]
    fn density_bias_negative() {
        let atm = make_test_atm();
        let rho = density(&atm, 50_000.0, -0.2);
        let expected = atm.density_at(50_000.0) * 0.8;
        assert_relative_eq!(rho, expected, epsilon = 1e-15);
    }
}
```

**Step 2: Run tests**

Run: `cd src/rust && cargo test physics::atmosphere --quiet`
Expected: 7 tests pass

**Step 3: Commit**

```bash
git add src/rust/src/physics/atmosphere.rs
git commit -m "test: add unit tests for atmosphere density model"
```

---

## Task 4: Unit tests — physics/aerodynamics.rs

**Files:**
- Modify: `src/rust/src/physics/aerodynamics.rs`

**Step 1: Write tests**

Append to `aerodynamics.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use crate::data::TimePeriods;

    fn make_test_aero() -> AeroTables {
        // Two-point table: Cx linear from 1.0 to 2.0, Cz from -0.2 to -0.4
        // over AoA range [0, 1 rad]
        AeroTables {
            equilibrium_aoa: 0.5,
            n_points: 2,
            incidence: vec![0.0, 1.0],
            cx: vec![1.0, 2.0],
            cz: vec![-0.2, -0.4],
            nominal_cx: 1.5,
            nominal_cz: -0.3,
            nominal_finesse: -0.3 / 1.5,
            ballistic_coeff: 0.0,
        }
    }

    fn make_test_capsule() -> Capsule {
        Capsule {
            mass: 1000.0,
            reference_area: 10.0,
            cq: 1e-4,
            max_bank_rate: 0.2,
            periods: TimePeriods::default(),
        }
    }

    #[test]
    fn zero_velocity_zero_forces() {
        let aero = make_test_aero();
        let cap = make_test_capsule();
        let f = aero_forces(&aero, &cap, 0.01, 0.0, 0.5, 0.0, 0.0);
        assert_eq!(f.drag, 0.0);
        assert_eq!(f.lift, 0.0);
        assert_eq!(f.heat_flux, 0.0);
    }

    #[test]
    fn drag_is_cx_q_s() {
        let aero = make_test_aero();
        let cap = make_test_capsule();
        let rho = 0.01;
        let v = 5000.0;
        let alpha = 0.0; // Cx = 1.0 at this AoA
        let f = aero_forces(&aero, &cap, rho, v, alpha, 0.0, 0.0);
        let q = 0.5 * rho * v * v;
        let expected_drag = q * cap.reference_area * 1.0;
        assert_relative_eq!(f.drag, expected_drag, epsilon = 1e-10);
    }

    #[test]
    fn lift_is_cz_q_s() {
        let aero = make_test_aero();
        let cap = make_test_capsule();
        let rho = 0.01;
        let v = 5000.0;
        let alpha = 0.0; // Cz = -0.2
        let f = aero_forces(&aero, &cap, rho, v, alpha, 0.0, 0.0);
        let q = 0.5 * rho * v * v;
        let expected_lift = q * cap.reference_area * (-0.2);
        assert_relative_eq!(f.lift, expected_lift, epsilon = 1e-10);
    }

    #[test]
    fn cx_bias_scales_drag() {
        let aero = make_test_aero();
        let cap = make_test_capsule();
        let f_nom = aero_forces(&aero, &cap, 0.01, 5000.0, 0.5, 0.0, 0.0);
        let f_biased = aero_forces(&aero, &cap, 0.01, 5000.0, 0.5, 0.1, 0.0);
        assert_relative_eq!(f_biased.drag, f_nom.drag * 1.1, epsilon = 1e-10);
    }

    #[test]
    fn cz_bias_scales_lift() {
        let aero = make_test_aero();
        let cap = make_test_capsule();
        let f_nom = aero_forces(&aero, &cap, 0.01, 5000.0, 0.5, 0.0, 0.0);
        let f_biased = aero_forces(&aero, &cap, 0.01, 5000.0, 0.5, 0.0, -0.15);
        assert_relative_eq!(f_biased.lift, f_nom.lift * 0.85, epsilon = 1e-10);
    }

    #[test]
    fn heat_flux_formula() {
        let aero = make_test_aero();
        let cap = make_test_capsule();
        let rho = 0.01;
        let v = 5000.0;
        let f = aero_forces(&aero, &cap, rho, v, 0.5, 0.0, 0.0);
        let expected = cap.cq * rho.sqrt() * v.powi(3);
        assert_relative_eq!(f.heat_flux, expected, epsilon = 1e-10);
    }

    #[test]
    fn interpolation_at_boundary() {
        let aero = make_test_aero();
        let cap = make_test_capsule();
        // Alpha beyond table range should clamp
        let f_below = aero_forces(&aero, &cap, 0.01, 5000.0, -0.5, 0.0, 0.0);
        let f_at_zero = aero_forces(&aero, &cap, 0.01, 5000.0, 0.0, 0.0, 0.0);
        assert_relative_eq!(f_below.drag, f_at_zero.drag, epsilon = 1e-10);
    }
}
```

**Step 2: Run tests**

Run: `cd src/rust && cargo test physics::aerodynamics --quiet`
Expected: 7 tests pass

**Step 3: Commit**

```bash
git add src/rust/src/physics/aerodynamics.rs
git commit -m "test: add unit tests for aerodynamic force computation"
```

---

## Task 5: Unit tests — gnc/navigation/coordinates.rs

**Files:**
- Modify: `src/rust/src/gnc/navigation/coordinates.rs`

**Step 1: Write tests**

Append to `coordinates.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    // --- Vector math ---

    #[test]
    fn cross_product_orthogonal() {
        // x cross y = z
        let result = cross(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]);
        assert_eq!(result, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn cross_product_anticommutative() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let ab = cross(&a, &b);
        let ba = cross(&b, &a);
        for i in 0..3 {
            assert_relative_eq!(ab[i], -ba[i], epsilon = 1e-15);
        }
    }

    #[test]
    fn dot_product() {
        assert_relative_eq!(dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
    }

    #[test]
    fn norm_unit_vectors() {
        assert_relative_eq!(norm(&[1.0, 0.0, 0.0]), 1.0);
        assert_relative_eq!(norm(&[0.0, 0.0, 0.0]), 0.0);
        assert_relative_eq!(norm(&[3.0, 4.0, 0.0]), 5.0);
    }

    // --- Position conversions ---

    #[test]
    fn position_to_cartesian_at_origin() {
        // r along x-axis: lon=0, lat=0
        let r = 1e6;
        let pos = position_to_cartesian(r, 0.0, 0.0);
        assert_relative_eq!(pos[0], r, epsilon = 1e-5);
        assert_relative_eq!(pos[1], 0.0, epsilon = 1e-5);
        assert_relative_eq!(pos[2], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn position_to_cartesian_at_pole() {
        // lat=pi/2 → along z
        let r = 1e6;
        let pos = position_to_cartesian(r, 0.0, PI / 2.0);
        assert_relative_eq!(pos[0], 0.0, epsilon = 1e-5);
        assert_relative_eq!(pos[1], 0.0, epsilon = 1e-5);
        assert_relative_eq!(pos[2], r, epsilon = 1e-5);
    }

    #[test]
    fn position_roundtrip_norm() {
        // |position_to_cartesian(r, lon, lat)| = r for any lon, lat
        let r = 3.5e6;
        let pos = position_to_cartesian(r, 0.7, 0.3);
        assert_relative_eq!(norm(&pos), r, epsilon = 1e-5);
    }

    // --- Geodetic conversions ---

    #[test]
    fn geodetic_spherical_planet() {
        // Moon is spherical (req == rpol), so geodetic = geocentric
        let r = Planet::Moon.equatorial_radius() + 100e3;
        let lat = 0.5;
        let (alt, lat_geo) = geodetic_from_spherical(r, 0.0, lat, &Planet::Moon);
        assert_relative_eq!(alt, 100e3, epsilon = 100.0); // ~100m tolerance
        assert_relative_eq!(lat_geo, lat, epsilon = 1e-6);
    }

    #[test]
    fn geodetic_at_equator() {
        // At equator on oblate planet, geodetic lat = geocentric lat = 0
        let r = Planet::Mars.equatorial_radius() + 130e3;
        let (alt, lat_geo) = geodetic_from_spherical(r, 0.0, 0.0, &Planet::Mars);
        assert_relative_eq!(lat_geo, 0.0, epsilon = 1e-10);
        assert_relative_eq!(alt, 130e3, epsilon = 500.0);
    }

    #[test]
    fn geodetic_at_pole() {
        // At pole, altitude = r - rpol
        let r = Planet::Mars.polar_radius() + 130e3;
        let (alt, lat_geo) = geodetic_from_spherical(r, 0.0, PI / 2.0, &Planet::Mars);
        assert_relative_eq!(alt, 130e3, epsilon = 500.0);
        assert_relative_eq!(lat_geo, PI / 2.0, epsilon = 1e-6);
    }

    // --- Rotation matrix ---

    #[test]
    fn rotation_matrix_is_orthogonal() {
        let m = local_to_geocentric_matrix(0.5, 0.3);
        // M * M^T should be identity
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += m[i][k] * m[j][k];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(sum, expected, epsilon = 1e-12);
            }
        }
    }

    // --- Total energy ---

    #[test]
    fn circular_orbit_energy() {
        // Circular orbit: E = -mu/(2r)
        // V_circular_relative needs to account for planet rotation
        let planet = &Planet::Mars;
        let r = planet.equatorial_radius() + 300e3;
        let v_circ_abs = (planet.mu() / r).sqrt();
        // At equator heading east, V_rel = V_abs - omega*r
        let v_rel = v_circ_abs - planet.omega() * r;
        let energy = total_energy(r, 0.0, 0.0, v_rel, 0.0, PI / 2.0, planet);
        let expected = -planet.mu() / (2.0 * r);
        assert_relative_eq!(energy, expected, epsilon = 1e3); // ~1 kJ/kg tolerance
    }

    #[test]
    fn hyperbolic_energy_positive() {
        // Entry at 5687 m/s (Mars aerocapture) should have positive energy
        let planet = &Planet::Mars;
        let r = planet.equatorial_radius() + 130e3;
        let energy = total_energy(r, 0.0, 0.0, 5687.0, -0.19, 0.66, planet);
        assert!(energy > 0.0, "hyperbolic entry should have positive energy: {energy}");
    }

    // --- Absolute velocity ---

    #[test]
    fn absolute_velocity_includes_rotation() {
        // At equator heading east, V_abs > V_rel (planet rotation adds)
        let planet = &Planet::Mars;
        let r = planet.equatorial_radius() + 130e3;
        let v_rel = 5000.0;
        let (_, v_abs_vec) = to_absolute_cartesian(r, 0.0, 0.0, v_rel, 0.0, PI / 2.0, planet);
        let v_abs = norm(&v_abs_vec);
        assert!(v_abs > v_rel, "absolute velocity should exceed relative");
    }
}
```

**Step 2: Run tests**

Run: `cd src/rust && cargo test gnc::navigation::coordinates --quiet`
Expected: all tests pass

**Step 3: Commit**

```bash
git add src/rust/src/gnc/navigation/coordinates.rs
git commit -m "test: add unit tests for coordinate transforms and energy"
```

---

## Task 6: Unit tests — integration/rk4.rs

**Files:**
- Modify: `src/rust/src/integration/rk4.rs`

**Step 1: Write tests**

Append to `rk4.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Integrate dy/dx = x from x=0 to x=1, y(0) = 0.
    /// Exact solution: y(1) = 0.5
    #[test]
    fn gill_rk4_linear_ode() {
        let dt = 0.01;
        let n = 1;
        let steps = 100;
        let mut state = [0.0]; // y
        let mut qk = [0.0];
        let mut ix: i32 = 0;

        for step in 0..steps {
            let x = step as f64 * dt;
            ix = 0;
            for k in 1..=4 {
                // Derivative at current state: dy/dx = x + offset for RK substep
                let x_sub = match k {
                    1 => x,
                    2 | 3 => x + dt / 2.0,
                    4 => x + dt,
                    _ => x,
                };
                let derivs = [x_sub];
                rk4_increment(dt, &derivs, k, n, &mut ix, &mut qk, &mut state);
            }
        }

        assert_relative_eq!(state[0], 0.5, epsilon = 1e-4);
    }

    /// Integrate dy/dx = y from x=0 to x=1, y(0) = 1.
    /// Exact solution: y(1) = e ≈ 2.71828
    #[test]
    fn gill_rk4_exponential_ode() {
        let dt = 0.01;
        let n = 1;
        let steps = 100;
        let mut state = [1.0]; // y(0) = 1
        let mut qk = [0.0];
        let mut ix: i32 = 0;

        for _ in 0..steps {
            ix = 0;
            let y_start = state[0];
            for k in 1..=4 {
                let derivs = [state[0]]; // dy/dx = y (evaluate at current substep state)
                rk4_increment(dt, &derivs, k, n, &mut ix, &mut qk, &mut state);
            }
            let _ = y_start; // suppress unused warning
        }

        assert_relative_eq!(state[0], std::f64::consts::E, epsilon = 1e-6);
    }

    /// Integrate 2D system: dx/dt = v, dv/dt = -x (simple harmonic oscillator)
    /// x(0) = 1, v(0) = 0 → x(t) = cos(t), v(t) = -sin(t)
    #[test]
    fn gill_rk4_harmonic_oscillator() {
        let dt = 0.01;
        let n = 2;
        let steps = (2.0 * std::f64::consts::PI / dt) as usize; // one full period
        let mut state = [1.0, 0.0]; // x=1, v=0
        let mut qk = [0.0, 0.0];
        let mut ix: i32 = 0;

        for _ in 0..steps {
            ix = 0;
            for k in 1..=4 {
                let derivs = [state[1], -state[0]]; // dx/dt = v, dv/dt = -x
                rk4_increment(dt, &derivs, k, n, &mut ix, &mut qk, &mut state);
            }
        }

        // After one period, should return to (1, 0)
        assert_relative_eq!(state[0], 1.0, epsilon = 1e-5);
        assert_relative_eq!(state[1], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn gill_rk4_preserves_constant() {
        // dy/dx = 0 → y stays constant
        let dt = 1.0;
        let n = 1;
        let mut state = [42.0];
        let mut qk = [0.0];
        let mut ix: i32 = 0;

        for _ in 0..10 {
            ix = 0;
            for k in 1..=4 {
                rk4_increment(dt, &[0.0], k, n, &mut ix, &mut qk, &mut state);
            }
        }

        assert_eq!(state[0], 42.0);
    }
}
```

**Step 2: Run tests**

Run: `cd src/rust && cargo test integration::rk4 --quiet`
Expected: all 4 tests pass

**Step 3: Commit**

```bash
git add src/rust/src/integration/rk4.rs
git commit -m "test: add unit tests for Gill's RK4 integrator"
```

---

## Task 7: Unit tests — integration/sequencer.rs

**Files:**
- Modify: `src/rust/src/integration/sequencer.rs`

**Step 1: Write tests**

Append to `sequencer.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_call_always_fires() {
        let mut seq = SequencerState::new();
        let periods = TimePeriods {
            navigation: 1.0,
            guidance: 2.0,
            pilot: 0.5,
            prediction: 5.0,
            integration: 1.0,
            photo: 10.0,
        };
        let flags = seq.update(0.0, &periods);
        assert!(flags.nav);
        assert!(flags.guid);
        assert!(flags.pilot);
        assert!(flags.pred);
        assert!(flags.photo);
    }

    #[test]
    fn respects_cadence() {
        let mut seq = SequencerState::new();
        let periods = TimePeriods {
            navigation: 1.0,
            guidance: 2.0,
            pilot: 0.5,
            prediction: 5.0,
            integration: 1.0,
            photo: 10.0,
        };
        seq.update(0.0, &periods); // t=0, all fire

        let flags = seq.update(0.5, &periods);
        assert!(!flags.nav);   // nav period=1.0, only 0.5 elapsed
        assert!(!flags.guid);  // guidance period=2.0
        assert!(flags.pilot);  // pilot period=0.5, exactly elapsed
        assert!(!flags.pred);  // prediction period=5.0
        assert!(!flags.photo); // photo period=10.0
    }

    #[test]
    fn fires_at_period() {
        let mut seq = SequencerState::new();
        let periods = TimePeriods {
            navigation: 1.0,
            guidance: 2.0,
            pilot: 0.5,
            prediction: 5.0,
            integration: 1.0,
            photo: 10.0,
        };
        seq.update(0.0, &periods);

        let flags = seq.update(1.0, &periods);
        assert!(flags.nav);    // 1.0 >= 1.0
        assert!(!flags.guid);  // 1.0 < 2.0
        assert!(flags.pilot);  // 1.0 >= 0.5
    }

    #[test]
    fn zero_period_always_fires() {
        assert!(should_execute(0.0, 0.0, 0.0));
        assert!(should_execute(100.0, 99.999, 0.0));
    }

    #[test]
    fn tolerance_handling() {
        // should_execute uses 1e-10 tolerance
        assert!(should_execute(1.0, 0.0, 1.0 + 1e-11));
    }
}
```

**Step 2: Run tests**

Run: `cd src/rust && cargo test integration::sequencer --quiet`
Expected: all 5 tests pass

**Step 3: Commit**

```bash
git add src/rust/src/integration/sequencer.rs
git commit -m "test: add unit tests for module cadence sequencer"
```

---

## Task 8: Unit tests — orbit/elements.rs

**Files:**
- Modify: `src/rust/src/orbit/elements.rs`

**Step 1: Write tests**

Append to `elements.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn circular_equatorial_orbit() {
        // Circular equatorial orbit at 300 km altitude
        let planet = &Planet::Mars;
        let r = planet.equatorial_radius() + 300e3;
        let v_circ_abs = (planet.mu() / r).sqrt();
        // At equator heading east: V_rel = V_abs - omega*r
        let v_rel = v_circ_abs - planet.omega() * r;

        let elems = from_spherical(r, 0.0, 0.0, v_rel, 0.0, PI / 2.0, planet);

        assert_relative_eq!(elems.semi_major_axis, r, epsilon = 1e3);
        assert_relative_eq!(elems.eccentricity, 0.0, epsilon = 0.01);
        assert_relative_eq!(elems.inclination, 0.0, epsilon = 0.01);
    }

    #[test]
    fn hyperbolic_orbit_has_negative_sma() {
        // Mars entry at 5687 m/s → hyperbolic
        let planet = &Planet::Mars;
        let r = planet.equatorial_radius() + 130e3;
        let elems = from_spherical(r, 0.0, 0.0, 5687.0, -0.19, 0.66, planet);
        assert!(elems.semi_major_axis < 0.0, "hyperbolic SMA should be negative");
        assert!(elems.eccentricity > 1.0, "hyperbolic eccentricity > 1");
    }

    #[test]
    fn periapsis_below_apoapsis() {
        let planet = &Planet::Mars;
        let r = planet.equatorial_radius() + 300e3;
        let elems = from_spherical(r, 0.5, 0.3, 3500.0, -0.05, 0.66, planet);
        if elems.eccentricity < 1.0 {
            assert!(elems.periapsis_alt < elems.apoapsis_alt,
                "periapsis {} should be below apoapsis {}",
                elems.periapsis_alt, elems.apoapsis_alt);
        }
    }

    #[test]
    fn inclination_from_azimuth() {
        // Pure east (psi = pi/2) at equator → near-zero inclination
        let planet = &Planet::Mars;
        let r = planet.equatorial_radius() + 300e3;
        let v = 3500.0;
        let elems_east = from_spherical(r, 0.0, 0.0, v, 0.0, PI / 2.0, planet);

        // Pure north (psi = 0) at equator → ~90 deg inclination
        let elems_north = from_spherical(r, 0.0, 0.0, v, 0.0, 0.0, planet);

        assert!(elems_east.inclination < 0.1, "eastward → low inclination");
        assert!(elems_north.inclination > 1.4, "northward → high inclination");
    }

    #[test]
    fn sma_matches_vis_viva() {
        // a = -mu / (2E), E = v_abs^2/2 - mu/r
        let planet = &Planet::Mars;
        let r = planet.equatorial_radius() + 200e3;
        let v = 4000.0;
        let gamma = -0.1;
        let psi = 0.66;

        let elems = from_spherical(r, 0.3, 0.2, v, gamma, psi, planet);
        let energy = crate::gnc::navigation::coordinates::total_energy(
            r, 0.3, 0.2, v, gamma, psi, planet
        );

        // For non-parabolic orbits
        if energy.abs() > 1e-3 {
            let expected_sma = -planet.mu() / (2.0 * energy);
            assert_relative_eq!(elems.semi_major_axis, expected_sma, epsilon = 1.0);
        }
    }
}
```

**Step 2: Run tests**

Run: `cd src/rust && cargo test orbit::elements --quiet`
Expected: all 5 tests pass

**Step 3: Commit**

```bash
git add src/rust/src/orbit/elements.rs
git commit -m "test: add unit tests for orbital element computation"
```

---

## Task 9: Unit tests — orbit/maneuver.rs

**Files:**
- Modify: `src/rust/src/orbit/maneuver.rs`

**Step 1: Write tests**

Append to `maneuver.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn test_orbit() -> OrbitalElements {
        OrbitalElements {
            semi_major_axis: 3.7e6,
            eccentricity: 0.067,
            inclination: 0.87, // ~50 deg
            raan: -0.13,
            arg_periapsis: 1.5,
            true_anomaly: 0.5,
            periapsis_alt: 11e3,    // 11 km
            apoapsis_alt: 500e3,    // 500 km
        }
    }

    fn test_target() -> OrbitalTarget {
        OrbitalTarget {
            apoapsis: 500e3,
            periapsis: 11e3,
            semi_major_axis: 3.65e6,
            eccentricity: 0.067,
            inclination: 0.873, // ~50 deg
            raan: -0.13,
        }
    }

    fn test_parking() -> ParkingOrbit {
        ParkingOrbit {
            apoapsis: 500e3,
            periapsis: 500e3,
        }
    }

    #[test]
    fn non_exit_returns_penalty() {
        let dv = compute_deltav(&test_orbit(), 1, &test_target(), &test_parking(), &Planet::Mars);
        assert_eq!(dv.dv1, 1e30);
        assert_eq!(dv.total, 1e30);
    }

    #[test]
    fn exit_returns_finite_cost() {
        let dv = compute_deltav(&test_orbit(), 3, &test_target(), &test_parking(), &Planet::Mars);
        assert!(dv.total.is_finite(), "total dv should be finite: {}", dv.total);
        assert!(dv.total > 0.0, "total dv should be positive");
        assert!(dv.total < 5000.0, "total dv should be reasonable (< 5 km/s): {}", dv.total);
    }

    #[test]
    fn total_is_sum_of_abs() {
        let dv = compute_deltav(&test_orbit(), 3, &test_target(), &test_parking(), &Planet::Mars);
        assert_relative_eq!(dv.total, dv.dv1.abs() + dv.dv2.abs() + dv.dv3.abs(), epsilon = 1e-10);
    }

    #[test]
    fn optimal_has_zero_dv3() {
        let dv = compute_deltav_optimal(&test_target(), &test_parking(), &Planet::Mars);
        assert_eq!(dv.dv3, 0.0);
        assert!(dv.total.is_finite());
    }

    #[test]
    fn zero_inclination_error_small_dv3() {
        let orbit = OrbitalElements {
            inclination: 0.873, // same as target
            ..test_orbit()
        };
        let target = test_target();
        let dv = compute_deltav(&orbit, 3, &target, &test_parking(), &Planet::Mars);
        assert!(dv.dv3.abs() < 1.0, "near-zero inclination error → tiny dv3: {}", dv.dv3);
    }
}
```

**Step 2: Run tests**

Run: `cd src/rust && cargo test orbit::maneuver --quiet`
Expected: all 5 tests pass

**Step 3: Commit**

```bash
git add src/rust/src/orbit/maneuver.rs
git commit -m "test: add unit tests for delta-V maneuver cost"
```

---

## Task 10: Unit tests — gnc/control/pilot.rs and attitude.rs

**Files:**
- Modify: `src/rust/src/gnc/control/pilot.rs`
- Modify: `src/rust/src/gnc/control/attitude.rs`

**Step 1: Write pilot tests**

Append to `pilot.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn zero_biases() -> PilotBiases {
        PilotBiases::default()
    }

    #[test]
    fn perfect_pilot_tracks_immediately() {
        let model = PilotModel { pilot_type: PilotType::Perfect, time_constant: 1.0, damping: 0.7, frequency: 1.0 };
        let state = PilotState { bank_angle: 0.5, bank_rate: 0.0 };
        let result = apply_pilot(&model, 1.0, &state, 0.1, 0.5, &zero_biases());
        assert_eq!(result.bank_angle, 1.0);
        assert_eq!(result.bank_rate, 0.0);
    }

    #[test]
    fn first_order_moves_toward_command() {
        let model = PilotModel { pilot_type: PilotType::FirstOrder, time_constant: 1.0, damping: 0.7, frequency: 1.0 };
        let state = PilotState { bank_angle: 0.0, bank_rate: 0.0 };
        let result = apply_pilot(&model, 1.0, &state, 0.1, 10.0, &zero_biases());
        // After dt=0.1 with tau=1.0: rate = (1.0 - 0.0)/1.0 = 1.0
        // new angle = 0.0 + 1.0 * 0.1 = 0.1
        assert_relative_eq!(result.bank_angle, 0.1, epsilon = 1e-10);
    }

    #[test]
    fn first_order_rate_clamped() {
        let model = PilotModel { pilot_type: PilotType::FirstOrder, time_constant: 0.1, damping: 0.7, frequency: 1.0 };
        let state = PilotState { bank_angle: 0.0, bank_rate: 0.0 };
        let max_rate = 0.5;
        let result = apply_pilot(&model, 10.0, &state, 0.1, max_rate, &zero_biases());
        // Unclamped rate = 10.0/0.1 = 100, but max_rate = 0.5
        assert_relative_eq!(result.bank_angle, 0.05, epsilon = 1e-10); // 0.5 * 0.1
        assert_relative_eq!(result.bank_rate, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn second_order_at_rest_accelerates() {
        let model = PilotModel { pilot_type: PilotType::SecondOrder, time_constant: 1.0, damping: 0.7, frequency: 2.0 };
        let state = PilotState { bank_angle: 0.0, bank_rate: 0.0 };
        let result = apply_pilot(&model, 1.0, &state, 0.1, 10.0, &zero_biases());
        // error = 0 - 1 = -1, accel = -2*0.7*2*0 - 4*(-1) = 4
        // rate = 0 + 4*0.1 = 0.4
        // angle = 0 + 0.4*0.1 = 0.04
        assert_relative_eq!(result.bank_rate, 0.4, epsilon = 1e-10);
        assert_relative_eq!(result.bank_angle, 0.04, epsilon = 1e-10);
    }

    #[test]
    fn first_order_bias_slows_response() {
        let model = PilotModel { pilot_type: PilotType::FirstOrder, time_constant: 1.0, damping: 0.7, frequency: 1.0 };
        let state = PilotState { bank_angle: 0.0, bank_rate: 0.0 };
        let biases = PilotBiases { tau: 1.0, damping: 0.0, frequency: 0.0 }; // double tau
        let result = apply_pilot(&model, 1.0, &state, 0.1, 10.0, &biases);
        // tau_biased = 1.0 * (1+1) = 2.0, rate = 1.0/2.0 = 0.5
        assert_relative_eq!(result.bank_angle, 0.05, epsilon = 1e-10);
    }
}
```

**Step 2: Write attitude tests**

Append to `attitude.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn within_limit_reaches_target() {
        let result = rate_limited_bank(0.0, 0.01, 0.5, 0.1);
        assert_eq!(result, 0.01); // 0.01 < 0.5*0.1 = 0.05
    }

    #[test]
    fn exceeds_limit_is_clamped() {
        let result = rate_limited_bank(0.0, 1.0, 0.5, 0.1);
        assert_relative_eq!(result, 0.05, epsilon = 1e-15); // max_change = 0.5*0.1
    }

    #[test]
    fn negative_direction() {
        let result = rate_limited_bank(1.0, 0.0, 0.5, 0.1);
        assert_relative_eq!(result, 0.95, epsilon = 1e-15); // 1.0 - 0.05
    }

    #[test]
    fn zero_error_stays_put() {
        let result = rate_limited_bank(0.5, 0.5, 0.5, 0.1);
        assert_eq!(result, 0.5);
    }
}
```

**Step 3: Run tests**

Run: `cd src/rust && cargo test gnc::control --quiet`
Expected: all 9 tests pass

**Step 4: Commit**

```bash
git add src/rust/src/gnc/control/pilot.rs src/rust/src/gnc/control/attitude.rs
git commit -m "test: add unit tests for pilot dynamics and rate limiter"
```

---

## Task 11: Unit tests — data module TOML parsing

**Files:**
- Modify: `src/rust/src/data/mod.rs`

**Step 1: Write tests**

Append to `data/mod.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_data_file_skips_comments() {
        let dir = std::env::temp_dir().join("aerocapture_test_parse");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_data.dat");
        std::fs::write(&path, "# Header comment\n  Another header\n1.0 2.0 3.0\n4.0D+01 5.0 6.0\n").unwrap();

        let rows = parse_data_file(path.to_str().unwrap()).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(rows[1], vec![40.0, 5.0, 6.0]); // D-notation converted

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn parse_data_file_handles_d_notation() {
        let dir = std::env::temp_dir().join("aerocapture_test_dnotation");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_d.dat");
        std::fs::write(&path, "1.23D+04\n-5.67d-03\n").unwrap();

        let rows = parse_data_file(path.to_str().unwrap()).unwrap();
        assert_eq!(rows.len(), 2);
        assert!((rows[0][0] - 12300.0).abs() < 1e-10);
        assert!((rows[1][0] - (-0.00567)).abs() < 1e-10);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn parse_data_file_empty_lines_skipped() {
        let dir = std::env::temp_dir().join("aerocapture_test_empty");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_empty.dat");
        std::fs::write(&path, "\n\n1.0\n\n2.0\n\n").unwrap();

        let rows = parse_data_file(path.to_str().unwrap()).unwrap();
        assert_eq!(rows.len(), 2);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn spherical_state_default_is_zero() {
        let s = SphericalState::default();
        assert_eq!(s.altitude, 0.0);
        assert_eq!(s.velocity, 0.0);
    }
}
```

**Step 2: Run tests**

Run: `cd src/rust && cargo test data::tests --quiet`
Expected: all 4 tests pass

**Step 3: Commit**

```bash
git add src/rust/src/data/mod.rs
git commit -m "test: add unit tests for data file parsing"
```

---

## Task 12: Integration tests — test helpers and config loading

**Files:**
- Create: `src/rust/tests/common/mod.rs`
- Create: `src/rust/tests/config_loading.rs`

**Step 1: Create test helpers**

```rust
// src/rust/tests/common/mod.rs
use std::path::PathBuf;

/// Get absolute path to repo root (2 levels up from src/rust/)
pub fn repo_root() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    PathBuf::from(manifest).join("../..").canonicalize().unwrap()
}

/// Get path to a TOML config in configs/
pub fn config_path(name: &str) -> String {
    repo_root().join("configs").join(name).to_str().unwrap().to_string()
}
```

**Step 2: Create config loading test**

```rust
// src/rust/tests/config_loading.rs
mod common;

use aerocapture::config::SimInput;

#[test]
fn load_ftc_consolidated_toml() {
    let path = common::config_path("msr_aller_ftc_consolidated.toml");

    // This exercises the full TOML parsing + SimData construction pipeline
    let config = SimInput::from_toml(&path).expect("should parse TOML config");

    assert_eq!(config.planet, aerocapture::config::Planet::Mars);
    assert_eq!(config.n_sims, 1);
    assert!(!config.reference_trajectory);
}

#[test]
fn load_reference_toml() {
    let path = common::config_path("msr_aller_reference.toml");
    let config = SimInput::from_toml(&path).expect("should parse reference TOML config");

    assert!(config.reference_trajectory);
    assert_eq!(config.n_sims, 1);
}
```

**Step 3: Run tests**

Run: `cd src/rust && cargo test --test config_loading --quiet`
Expected: both tests pass

**Step 4: Commit**

```bash
git add src/rust/tests/
git commit -m "test: add integration test helpers and config loading tests"
```

---

## Task 13: E2E tests — reference trajectory

**Files:**
- Create: `src/rust/tests/e2e_reference.rs`

**Step 1: Write E2E reference trajectory test**

```rust
// src/rust/tests/e2e_reference.rs
//! End-to-end test: reference trajectory (constant bank angle, no guidance)
mod common;

use aerocapture::config::SimInput;
use aerocapture::data::SimData;
use std::fs;

#[test]
fn reference_trajectory_completes_and_produces_output() {
    let config_path = common::config_path("msr_aller_reference.toml");
    let config = SimInput::from_toml(&config_path).expect("parse config");

    // Override output to temp dir
    let tmp = tempfile::tempdir().unwrap_or_else(|_| {
        let p = std::env::temp_dir().join("aerocapture_e2e_ref");
        fs::create_dir_all(&p).ok();
        tempfile::TempDir::new_in(&p).unwrap()
    });

    let data = SimData::from_toml(
        &config.toml_config.as_ref().unwrap(),
        &config,
    ).expect("load sim data");

    // Run simulation
    let result = aerocapture::simulation::runner::run(&config, &data);
    assert!(result.is_ok(), "simulation should complete without error: {:?}", result.err());
}
```

> **Note to implementer:** The exact API for `run()` and how to override output dir may need adjustment based on the actual function signatures. Check `runner.rs` and `config.rs` for the correct invocation pattern. The test may need to set `config.output_dir` to a temp path, or the output writing may need to be configurable. Adapt accordingly.

**Step 2: Run test**

Run: `cd src/rust && cargo test --test e2e_reference --quiet`
Expected: test passes, simulation completes

**Step 3: Commit**

```bash
git add src/rust/tests/e2e_reference.rs
git commit -m "test: add E2E reference trajectory test"
```

---

## Task 14: E2E tests — guided trajectories per scheme

**Files:**
- Create: `src/rust/tests/e2e_guidance.rs`

**Step 1: Write parameterized guidance E2E tests**

```rust
// src/rust/tests/e2e_guidance.rs
//! End-to-end tests: one test per guidance scheme
mod common;

use rstest::rstest;

#[rstest]
#[case("msr_aller_ftc_consolidated.toml", "ftc")]
#[case("msr_aller_eqglide_train.toml", "equilibrium_glide")]
#[case("msr_aller_energy_controller_train.toml", "energy_controller")]
#[case("msr_aller_pred_guid_train.toml", "pred_guid")]
#[case("msr_aller_fnpag_train.toml", "fnpag")]
fn guidance_scheme_completes(#[case] config_file: &str, #[case] scheme_name: &str) {
    let config_path = common::config_path(config_file);
    let config = aerocapture::config::SimInput::from_toml(&config_path)
        .unwrap_or_else(|e| panic!("{scheme_name}: failed to parse config: {e}"));

    let data = aerocapture::data::SimData::from_toml(
        config.toml_config.as_ref().unwrap(),
        &config,
    ).unwrap_or_else(|e| panic!("{scheme_name}: failed to load data: {e}"));

    let result = aerocapture::simulation::runner::run(&config, &data);
    assert!(result.is_ok(), "{scheme_name} should complete: {:?}", result.err());
}
```

> **Note to implementer:** Some training configs may require `save_net/<scheme>/best_params.json` to exist. If a scheme's optimized params aren't available, the test may need to use a nominal config or skip that case. Check which configs work out of the box. Also verify output doesn't clobber existing files — may need temp dir override.

**Step 2: Run tests**

Run: `cd src/rust && cargo test --test e2e_guidance --quiet`
Expected: all schemes that have valid configs pass

**Step 3: Commit**

```bash
git add src/rust/tests/e2e_guidance.rs
git commit -m "test: add parameterized E2E tests for all guidance schemes"
```

---

## Task 15: E2E tests — Monte Carlo

**Files:**
- Create: `src/rust/tests/e2e_monte_carlo.rs`

**Step 1: Write MC E2E tests**

```rust
// src/rust/tests/e2e_monte_carlo.rs
//! End-to-end Monte Carlo tests
mod common;

#[test]
fn mc_produces_correct_number_of_results() {
    let config_path = common::config_path("msr_aller_ftc_mc_domain.toml");
    let config = aerocapture::config::SimInput::from_toml(&config_path)
        .unwrap_or_else(|e| panic!("failed to parse MC config: {e}"));

    let data = aerocapture::data::SimData::from_toml(
        config.toml_config.as_ref().unwrap(),
        &config,
    ).expect("load MC sim data");

    assert!(config.n_sims > 1, "MC config should have n_sims > 1");

    let result = aerocapture::simulation::runner::run(&config, &data);
    assert!(result.is_ok(), "MC run should complete: {:?}", result.err());

    // Verify output file has correct number of lines
    // (implementation depends on how output is written — adapt path accordingly)
}

#[test]
fn mc_deterministic_with_same_seed() {
    // Run twice with same config → should produce identical output
    let config_path = common::config_path("msr_aller_ftc_mc_domain.toml");

    for _ in 0..2 {
        let config = aerocapture::config::SimInput::from_toml(&config_path).unwrap();
        let data = aerocapture::data::SimData::from_toml(
            config.toml_config.as_ref().unwrap(),
            &config,
        ).unwrap();
        let result = aerocapture::simulation::runner::run(&config, &data);
        assert!(result.is_ok());
    }
    // TODO: Compare outputs once output capture is available
}
```

> **Note to implementer:** These tests need output capture/comparison. The simplest approach is to check that the `final.*` output file has the expected number of lines. You may need to override the output directory to a temp dir and read back the results.

**Step 2: Run tests**

Run: `cd src/rust && cargo test --test e2e_monte_carlo --quiet`
Expected: tests pass

**Step 3: Commit**

```bash
git add src/rust/tests/e2e_monte_carlo.rs
git commit -m "test: add E2E Monte Carlo tests"
```

---

## Task 16: Run full test suite and verify

**Step 1: Run all Rust tests**

Run: `cd src/rust && cargo test --quiet`
Expected: all tests pass (existing 14 + ~60 new)

**Step 2: Run cargo clippy**

Run: `cd src/rust && cargo clippy --all-targets --all-features --quiet -- -D warnings`
Expected: no warnings

**Step 3: Run cargo fmt check**

Run: `cd src/rust && cargo fmt --all --check`
Expected: no formatting issues

**Step 4: Fix any issues found**

If clippy or fmt report issues, fix them and re-run.

**Step 5: Commit any fixes**

```bash
git add -u src/rust/
git commit -m "fix: resolve clippy/fmt issues in test suite"
```

---

## Task 17: Update check_all.sh and documentation

**Files:**
- Verify: `check_all.sh` (should already pick up new tests via `cargo test`)
- Update: `CLAUDE.md` testing section if needed

**Step 1: Verify check_all.sh runs everything**

Run: `./check_all.sh`
Expected: all tests pass, fmt clean, clippy clean

**Step 2: Update CLAUDE.md testing section**

Add note about new test infrastructure:

```markdown
## Testing
- **Rust unit tests**: `cd src/rust && cargo test` — physics, GNC, integration, orbit modules
- **Rust E2E tests**: included in `cargo test` — full sim runs from TOML configs
- **Python tests**: `pytest tests` — parser tests, analysis tools
- Dev-dependencies: `approx` (float comparison), `rstest` (parameterized)
```

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update testing documentation for new Rust test suite"
```

---

## Future Tasks (not in this plan)

These are explicitly deferred for later:

1. **Snapshot-based integration tests** — requires building a snapshot generation harness that dumps intermediate GNC state to JSON. Do this when you start refactoring the GNC chain.

2. **Navigation estimator unit tests** — complex setup (needs full SimData). Add when refactoring navigation module.

3. **Guidance scheme unit tests** (FTC, EqGlide, etc.) — need NavigationOutput fixtures. Add incrementally as you modify each scheme.

4. **Retire Python regression tests** — only after Rust E2E tests cover all scenarios currently in `test_regression.py` and `test_mc_domain.py`.

5. **CI/CD pipeline** — GitHub Actions workflow running `check_all.sh` + `pytest tests`.
