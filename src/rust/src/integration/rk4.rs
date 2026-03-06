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
