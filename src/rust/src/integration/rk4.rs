//! Runge-Kutta 4th order integrator (Gill's variant).
//!
//! Matches Fortran rkutta.f exactly.
//!
//! This is called 4 times per integration step (k=1..4).
//! Between calls, the caller recomputes derivatives using the updated state.
//! The `qk` and `ix` variables persist across the 4 calls.

/// Perform one increment of the Gill RK4 method.
///
/// - `dt`: integration time step
/// - `derivs`: state derivatives at current point
/// - `k`: RK4 increment (1-based: 1, 2, 3, or 4)
/// - `n`: number of state components
/// - `ix`: internal variable (modified in place, initially 0)
/// - `qk`: internal storage (modified in place)
/// - `state`: state vector (modified in place)
pub fn rk4_increment(
    dt: f64,
    derivs: &[f64],
    k: usize,
    n: usize,
    ix: &mut i32,
    qk: &mut [f64],
    state: &mut [f64],
) {
    let a = std::f64::consts::SQRT_2;
    let ix_f = *ix as f64;

    // xk = dt * derivs
    match k {
        1 => {
            for i in 0..n {
                let xk = dt * derivs[i];
                state[i] += 0.5 * xk;
                qk[i] = xk;
            }
            *ix = 1;
        }
        2 | 3 => {
            for i in 0..n {
                let xk = dt * derivs[i];
                state[i] += (1.0 - ix_f / a) * (xk - qk[i]);
                qk[i] = qk[i] * (-2.0 + 3.0 * ix_f / a) + xk * (2.0 - ix_f * a);
            }
            *ix = -1;
        }
        4 => {
            for i in 0..n {
                let xk = dt * derivs[i];
                state[i] += (xk - 2.0 * qk[i]) / 6.0;
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: run one full RK4 step with a derivative function that takes (x, state) -> derivs.
    /// `x` is the independent variable (time), `dt` is the step size.
    fn rk4_step(x: f64, dt: f64, state: &mut [f64], ix: &mut i32, qk: &mut [f64], deriv_fn: impl Fn(f64, &[f64]) -> Vec<f64>) {
        let n = state.len();
        // k=1: derivs at x
        let derivs = deriv_fn(x, state);
        rk4_increment(dt, &derivs, 1, n, ix, qk, state);
        // k=2: derivs at x + dt/2
        let derivs = deriv_fn(x + dt / 2.0, state);
        rk4_increment(dt, &derivs, 2, n, ix, qk, state);
        // k=3: derivs at x + dt/2
        let derivs = deriv_fn(x + dt / 2.0, state);
        rk4_increment(dt, &derivs, 3, n, ix, qk, state);
        // k=4: derivs at x + dt
        let derivs = deriv_fn(x + dt, state);
        rk4_increment(dt, &derivs, 4, n, ix, qk, state);
    }

    #[test]
    fn gill_rk4_linear_ode() {
        // dy/dx = x, y(0) = 0 => y(1) = 0.5
        let dt = 0.01;
        let n_steps = 100;
        let mut state = vec![0.0]; // y
        let mut qk = vec![0.0];
        let mut ix: i32;

        for step in 0..n_steps {
            let x = step as f64 * dt;
            ix = 0;
            rk4_step(x, dt, &mut state, &mut ix, &mut qk, |t, _| vec![t]);
        }

        assert!((state[0] - 0.5).abs() < 1e-10, "Expected 0.5, got {}", state[0]);
    }

    #[test]
    fn gill_rk4_exponential_ode() {
        // dy/dx = y, y(0) = 1 => y(1) = e
        let dt = 0.01;
        let n_steps = 100;
        let mut state = vec![1.0];
        let mut qk = vec![0.0];
        let mut ix: i32;

        for step in 0..n_steps {
            let x = step as f64 * dt;
            ix = 0;
            rk4_step(x, dt, &mut state, &mut ix, &mut qk, |_, s| vec![s[0]]);
        }

        let expected = std::f64::consts::E;
        assert!(
            (state[0] - expected).abs() < 1e-8,
            "Expected e ≈ {}, got {}",
            expected,
            state[0]
        );
    }

    #[test]
    fn gill_rk4_harmonic_oscillator() {
        // dx/dt = v, dv/dt = -x, x(0)=1, v(0)=0
        // After t = 2*pi, should return to (1, 0)
        let dt = 0.01;
        let period = 2.0 * std::f64::consts::PI;
        let n_steps = (period / dt).round() as usize;
        let mut state = vec![1.0, 0.0]; // [x, v]
        let mut qk = vec![0.0, 0.0];
        let mut ix: i32;

        for step in 0..n_steps {
            let t = step as f64 * dt;
            ix = 0;
            rk4_step(t, dt, &mut state, &mut ix, &mut qk, |_, s| vec![s[1], -s[0]]);
        }

        // Tolerance accounts for both integrator error and period discretization
        // (628 steps * 0.01 = 6.28 != 2*pi = 6.28318...)
        assert!(
            (state[0] - 1.0).abs() < 5e-3,
            "Expected x ≈ 1.0, got {}",
            state[0]
        );
        assert!(
            state[1].abs() < 5e-3,
            "Expected v ≈ 0.0, got {}",
            state[1]
        );
    }

    #[test]
    fn gill_rk4_preserves_constant() {
        // dy/dx = 0, y(0) = 42 => y stays 42
        let dt = 0.1;
        let mut state = vec![42.0];
        let mut qk = vec![0.0];
        let mut ix: i32;

        for step in 0..10 {
            let x = step as f64 * dt;
            ix = 0;
            rk4_step(x, dt, &mut state, &mut ix, &mut qk, |_, _| vec![0.0]);
        }

        assert!(
            (state[0] - 42.0).abs() < 1e-15,
            "Expected 42.0, got {}",
            state[0]
        );
    }
}
