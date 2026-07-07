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

use super::super::LayerWeights;
#[allow(unused_imports)]
use super::helpers::expm1_over_x; // real path reuses this (bit-identity anchor); consumed in Task 3

/// Mamba-3 ablation layer. `trapezoidal` / `complex` are orthogonal opt-in flags.
/// `x_proj` shape is fixed `(dt_rank + 2*d_state, input_size)` in all modes (B, C real).
#[derive(Debug, Clone)]
pub struct Mamba3Layer {
    pub input_size: usize,
    pub d_state: usize,
    pub dt_rank: usize,
    pub trapezoidal: bool,
    pub complex: bool,
    pub x_proj_w: nalgebra::DMatrix<f64>, // (dt_rank + 2*d_state, input_size)
    pub dt_proj_w: nalgebra::DMatrix<f64>, // (input_size, dt_rank)
    pub dt_proj_b: nalgebra::DVector<f64>, // (input_size,)
    pub a_log: nalgebra::DMatrix<f64>,    // (input_size, d_state)
    pub a_imag: Option<nalgebra::DMatrix<f64>>, // (input_size, d_state) iff complex
    pub lambda_logit: Option<nalgebra::DVector<f64>>, // (input_size,) iff trapezoidal
    pub d_skip: nalgebra::DVector<f64>,   // (input_size,)
}

impl LayerWeights for Mamba3Layer {
    fn n_params(&self) -> usize {
        let base = self.input_size * (3 * self.d_state + 2 * self.dt_rank + 2);
        base + if self.complex {
            self.input_size * self.d_state
        } else {
            0
        } + if self.trapezoidal {
            self.input_size
        } else {
            0
        }
    }

    fn to_flat(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.n_params());
        let push_mat = |out: &mut Vec<f64>, m: &nalgebra::DMatrix<f64>| {
            for i in 0..m.nrows() {
                for j in 0..m.ncols() {
                    out.push(m[(i, j)]);
                }
            }
        };
        push_mat(&mut out, &self.x_proj_w);
        push_mat(&mut out, &self.dt_proj_w);
        out.extend(self.dt_proj_b.iter().copied());
        push_mat(&mut out, &self.a_log);
        if let Some(ai) = &self.a_imag {
            push_mat(&mut out, ai);
        }
        if let Some(ll) = &self.lambda_logit {
            out.extend(ll.iter().copied());
        }
        out.extend(self.d_skip.iter().copied());
        out
    }

    #[allow(clippy::wrong_self_convention)]
    fn from_flat(&mut self, flat: &[f64]) -> usize {
        let mut c = 0;
        let xr = self.dt_rank + 2 * self.d_state;
        self.x_proj_w = nalgebra::DMatrix::from_row_slice(
            xr,
            self.input_size,
            &flat[c..c + xr * self.input_size],
        );
        c += xr * self.input_size;
        self.dt_proj_w = nalgebra::DMatrix::from_row_slice(
            self.input_size,
            self.dt_rank,
            &flat[c..c + self.input_size * self.dt_rank],
        );
        c += self.input_size * self.dt_rank;
        self.dt_proj_b = nalgebra::DVector::from_row_slice(&flat[c..c + self.input_size]);
        c += self.input_size;
        self.a_log = nalgebra::DMatrix::from_row_slice(
            self.input_size,
            self.d_state,
            &flat[c..c + self.input_size * self.d_state],
        );
        c += self.input_size * self.d_state;
        if self.complex {
            self.a_imag = Some(nalgebra::DMatrix::from_row_slice(
                self.input_size,
                self.d_state,
                &flat[c..c + self.input_size * self.d_state],
            ));
            c += self.input_size * self.d_state;
        } else {
            self.a_imag = None;
        }
        if self.trapezoidal {
            self.lambda_logit =
                Some(nalgebra::DVector::from_row_slice(&flat[c..c + self.input_size]));
            c += self.input_size;
        } else {
            self.lambda_logit = None;
        }
        self.d_skip = nalgebra::DVector::from_row_slice(&flat[c..c + self.input_size]);
        c += self.input_size;
        c
    }
}

impl Mamba3Layer {
    /// Zero-weight constructor for the given shape + flags (weights filled by `from_flat`).
    pub fn zeros(
        input_size: usize,
        d_state: usize,
        dt_rank: usize,
        trapezoidal: bool,
        complex: bool,
    ) -> Self {
        Self {
            input_size,
            d_state,
            dt_rank,
            trapezoidal,
            complex,
            x_proj_w: nalgebra::DMatrix::zeros(dt_rank + 2 * d_state, input_size),
            dt_proj_w: nalgebra::DMatrix::zeros(input_size, dt_rank),
            dt_proj_b: nalgebra::DVector::zeros(input_size),
            a_log: nalgebra::DMatrix::zeros(input_size, d_state),
            a_imag: if complex {
                Some(nalgebra::DMatrix::zeros(input_size, d_state))
            } else {
                None
            },
            lambda_logit: if trapezoidal {
                Some(nalgebra::DVector::zeros(input_size))
            } else {
                None
            },
            d_skip: nalgebra::DVector::zeros(input_size),
        }
    }
}

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

    #[test]
    fn flat_roundtrip_all_flag_combos() {
        for &(trap, cplx) in &[(false, false), (true, false), (false, true), (true, true)] {
            let mut m = Mamba3Layer::zeros(4, 3, 2, trap, cplx);
            let n = m.n_params();
            let slab: Vec<f64> = (0..n).map(|i| 0.01 * (i as f64 + 1.0)).collect();
            let consumed = m.from_flat(&slab);
            assert_eq!(consumed, n, "trap={trap} cplx={cplx}");
            assert_eq!(m.to_flat(), slab, "trap={trap} cplx={cplx}");
        }
    }
}
