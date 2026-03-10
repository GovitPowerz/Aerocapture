//! Runge-Kutta 4th order integrator (Gill's variant).
//!
//! This is called 4 times per integration step (k=1..4).
//! Between calls, the caller recomputes derivatives using the updated state.
//! The `accumulator` and `gill_toggle` variables persist across the 4 calls.

/// Perform one increment of the Gill RK4 method.
///
/// - `dt`: integration time step
/// - `derivs`: state derivatives at current point
/// - `k`: RK4 increment (1-based: 1, 2, 3, or 4)
/// - `n`: number of state components
/// - `gill_toggle`: Gill's variant toggle (-1 or +1, modified in place, initially 0)
/// - `accumulator`: internal RK4 storage (modified in place)
/// - `state`: state vector (modified in place)
pub fn rk4_increment(
    dt: f64,
    derivs: &[f64],
    k: usize,
    n: usize,
    gill_toggle: &mut i32,
    accumulator: &mut [f64],
    state: &mut [f64],
) {
    let a = std::f64::consts::SQRT_2;
    let gill_toggle_f = *gill_toggle as f64;

    match k {
        1 => {
            for i in 0..n {
                let step_increment = dt * derivs[i];
                state[i] += 0.5 * step_increment;
                accumulator[i] = step_increment;
            }
            *gill_toggle = 1;
        }
        2 | 3 => {
            for i in 0..n {
                let step_increment = dt * derivs[i];
                state[i] += (1.0 - gill_toggle_f / a) * (step_increment - accumulator[i]);
                accumulator[i] = accumulator[i] * (-2.0 + 3.0 * gill_toggle_f / a) + step_increment * (2.0 - gill_toggle_f * a);
            }
            *gill_toggle = -1;
        }
        4 => {
            for i in 0..n {
                let step_increment = dt * derivs[i];
                state[i] += (step_increment - 2.0 * accumulator[i]) / 6.0;
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
    fn rk4_step(
        x: f64,
        dt: f64,
        state: &mut [f64],
        gill_toggle: &mut i32,
        accumulator: &mut [f64],
        deriv_fn: impl Fn(f64, &[f64]) -> Vec<f64>,
    ) {
        let n = state.len();
        // k=1: derivs at x
        let derivs = deriv_fn(x, state);
        rk4_increment(dt, &derivs, 1, n, gill_toggle, accumulator, state);
        // k=2: derivs at x + dt/2
        let derivs = deriv_fn(x + dt / 2.0, state);
        rk4_increment(dt, &derivs, 2, n, gill_toggle, accumulator, state);
        // k=3: derivs at x + dt/2
        let derivs = deriv_fn(x + dt / 2.0, state);
        rk4_increment(dt, &derivs, 3, n, gill_toggle, accumulator, state);
        // k=4: derivs at x + dt
        let derivs = deriv_fn(x + dt, state);
        rk4_increment(dt, &derivs, 4, n, gill_toggle, accumulator, state);
    }

    #[test]
    fn gill_rk4_linear_ode() {
        // dy/dx = x, y(0) = 0 => y(1) = 0.5
        let dt = 0.01;
        let n_steps = 100;
        let mut state = vec![0.0]; // y
        let mut accumulator = vec![0.0];
        let mut gill_toggle: i32;

        for step in 0..n_steps {
            let x = step as f64 * dt;
            gill_toggle = 0;
            rk4_step(x, dt, &mut state, &mut gill_toggle, &mut accumulator, |t, _| vec![t]);
        }

        assert!(
            (state[0] - 0.5).abs() < 1e-10,
            "Expected 0.5, got {}",
            state[0]
        );
    }

    #[test]
    fn gill_rk4_exponential_ode() {
        // dy/dx = y, y(0) = 1 => y(1) = e
        let dt = 0.01;
        let n_steps = 100;
        let mut state = vec![1.0];
        let mut accumulator = vec![0.0];
        let mut gill_toggle: i32;

        for step in 0..n_steps {
            let x = step as f64 * dt;
            gill_toggle = 0;
            rk4_step(x, dt, &mut state, &mut gill_toggle, &mut accumulator, |_, s| vec![s[0]]);
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
        let mut accumulator = vec![0.0, 0.0];
        let mut gill_toggle: i32;

        for step in 0..n_steps {
            let t = step as f64 * dt;
            gill_toggle = 0;
            rk4_step(t, dt, &mut state, &mut gill_toggle, &mut accumulator, |_, s| {
                vec![s[1], -s[0]]
            });
        }

        // Tolerance accounts for both integrator error and period discretization
        // (628 steps * 0.01 = 6.28 != 2*pi = 6.28318...)
        assert!(
            (state[0] - 1.0).abs() < 5e-3,
            "Expected x ≈ 1.0, got {}",
            state[0]
        );
        assert!(state[1].abs() < 5e-3, "Expected v ≈ 0.0, got {}", state[1]);
    }

    #[test]
    fn gill_rk4_preserves_constant() {
        // dy/dx = 0, y(0) = 42 => y stays 42
        let dt = 0.1;
        let mut state = vec![42.0];
        let mut accumulator = vec![0.0];
        let mut gill_toggle: i32;

        for step in 0..10 {
            let x = step as f64 * dt;
            gill_toggle = 0;
            rk4_step(x, dt, &mut state, &mut gill_toggle, &mut accumulator, |_, _| vec![0.0]);
        }

        assert!(
            (state[0] - 42.0).abs() < 1e-15,
            "Expected 42.0, got {}",
            state[0]
        );
    }
}
