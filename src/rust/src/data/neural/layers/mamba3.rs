//! Mamba-3 ablation layer: euler|trapezoidal x real|complex. PSO-only spike.
//!
//! Extends the Phase 4a `MambaLayer` (selective SSM core) with two orthogonal,
//! opt-in recurrence modes from Mamba-3 (arXiv 2603.15569):
//!   - `trapezoidal`: exponential-trapezoidal discretization (second-order),
//!     a strict generalization of the deployed ZOH euler (lambda -> 1 == euler).
//!   - `complex`: complex-diagonal (rotational) state, the RoPE-on-B/C equivalent.
//!
//! Real-mode recurrence reuses `helpers::expm1_over_x` (with `f64::exp_m1`) so
//! `real`+`euler` is BIT-identical to `MambaLayer`. Complex arithmetic is explicit
//! (re, im) to preserve cross-language bit-identity with the Python mirror.
//! See docs/superpowers/specs/2026-07-07-mamba3-ablation-design.md.

/// Complex `(exp(z) - 1) / z` with Taylor fallback for |z| < 1e-8.
/// z = zr + i·zi. Returns (re, im). Explicit real arithmetic for
/// cross-language bit-identity with the Python mirror.
pub(super) fn expm1_over_x_complex(zr: f64, zi: f64) -> (f64, f64) {
    let mag = (zr * zr + zi * zi).sqrt();
    if mag < 1e-8 {
        // Taylor 1 + z/2 + z^2/6; z^2 = (zr^2 - zi^2) + i(2 zr zi)
        let z2r = zr * zr - zi * zi;
        let z2i = 2.0 * zr * zi;
        (1.0 + 0.5 * zr + z2r / 6.0, 0.5 * zi + z2i / 6.0)
    } else {
        // exp(z) = e^zr (cos zi + i sin zi)
        let er = zr.exp();
        let ez_r = er * zi.cos();
        let ez_i = er * zi.sin();
        let num_r = ez_r - 1.0;
        let num_i = ez_i;
        // (num) / (zr + i zi) = num·conj(z) / |z|^2
        let denom = zr * zr + zi * zi;
        (
            (num_r * zr + num_i * zi) / denom,
            (num_i * zr - num_r * zi) / denom,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complex_reduces_to_real_on_real_axis() {
        // On the real axis (zi=0), Re matches the real (exp(z)-1)/z form; Im ~ 0.
        for zr in [-2.0, -0.5, 0.3, 1.5] {
            let (re, im) = expm1_over_x_complex(zr, 0.0);
            let expected = (zr.exp() - 1.0) / zr;
            assert!((re - expected).abs() < 1e-12, "zr={zr}");
            assert!(im.abs() < 1e-15, "zr={zr}");
        }
    }

    #[test]
    fn complex_taylor_branch_finite_at_zero() {
        let (re, im) = expm1_over_x_complex(0.0, 0.0);
        assert!((re - 1.0).abs() < 1e-15);
        assert!(im.abs() < 1e-15);
    }
}
