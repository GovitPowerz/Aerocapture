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
//! See docs/design/2026-07-07-mamba3-ablation-design.md.

use super::helpers::expm1_over_x; // real path reuses expm1_over_x (bit-identity anchor)
use super::mamba::{MambaScratch, a_from_log, a_is_current};
use super::tensor::tensor_table;

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
    // Private with `lambda_logit`: the only write path is a load, which
    // rebuilds the derived `a` / `lam`.
    a_log: nalgebra::DMatrix<f64>,              // (input_size, d_state)
    pub a_imag: Option<nalgebra::DMatrix<f64>>, // (input_size, d_state) iff complex
    lambda_logit: Option<nalgebra::DVector<f64>>, // (input_size,) iff trapezoidal
    pub d_skip: nalgebra::DVector<f64>,         // (input_size,)

    // Derived, rebuilt by `rebuild_derived` on every load; not in the flat
    // chromosome or the JSON.
    a: nalgebra::DMatrix<f64>,   // -exp(a_log)
    lam: nalgebra::DVector<f64>, // sigmoid(lambda_logit); 1.0 in euler mode
}

// Flat order: x_proj_w, dt_proj_w, dt_proj_b, a_log, [a_imag iff complex],
// [lambda_logit iff trapezoidal], d_skip. The two `Option` fields are present
// exactly when `zeros` built them, so the flags fix the table at construction.
tensor_table!(
    Mamba3Layer {
        x_proj_w,
        dt_proj_w,
        dt_proj_b,
        a_log,
        a_imag,
        lambda_logit,
        d_skip
    },
    post_load = rebuild_derived
);

fn sigmoid(l: f64) -> f64 {
    1.0 / (1.0 + (-l).exp())
}

/// Trapezoidal weight `lambda = sigmoid(lambda_logit)`; 1.0 (euler) without one.
fn lam_from_logit(
    lambda_logit: Option<&nalgebra::DVector<f64>>,
    input_size: usize,
) -> nalgebra::DVector<f64> {
    lambda_logit.map_or_else(
        || nalgebra::DVector::from_element(input_size, 1.0),
        |ll| ll.map(sigmoid),
    )
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
        let a_log = nalgebra::DMatrix::zeros(input_size, d_state);
        let lambda_logit = trapezoidal.then(|| nalgebra::DVector::zeros(input_size));
        Self {
            input_size,
            d_state,
            dt_rank,
            trapezoidal,
            complex,
            x_proj_w: nalgebra::DMatrix::zeros(dt_rank + 2 * d_state, input_size),
            dt_proj_w: nalgebra::DMatrix::zeros(input_size, dt_rank),
            dt_proj_b: nalgebra::DVector::zeros(input_size),
            a: a_from_log(&a_log),
            a_log,
            a_imag: if complex {
                Some(nalgebra::DMatrix::zeros(input_size, d_state))
            } else {
                None
            },
            lam: lam_from_logit(lambda_logit.as_ref(), input_size),
            lambda_logit,
            d_skip: nalgebra::DVector::zeros(input_size),
        }
    }

    pub fn lambda_logit(&self) -> Option<&nalgebra::DVector<f64>> {
        self.lambda_logit.as_ref()
    }

    /// Recompute the derived `a` / `lam`. Both load paths (JSON and PSO
    /// chromosome) reach it through the table's `post_load` hook.
    fn rebuild_derived(&mut self) {
        self.a = a_from_log(&self.a_log);
        self.lam = lam_from_logit(self.lambda_logit.as_ref(), self.input_size);
    }

    fn derived_is_current(&self) -> bool {
        a_is_current(&self.a, &self.a_log)
            && self.lam.len() == self.input_size
            && match &self.lambda_logit {
                Some(ll) => self
                    .lam
                    .iter()
                    .zip(ll.iter())
                    .all(|(lam, l)| lam.to_bits() == sigmoid(*l).to_bits()),
                None => self.lam.iter().all(|lam| *lam == 1.0),
            }
    }

    /// Single-tick forward. Mutates state in place, returns `y` (length input_size)
    /// borrowed from `scratch`. Allocates nothing.
    ///
    /// State: `h_re`/`h_im` are `(input_size, d_state)` (h_im unused in real mode);
    /// `x_prev` is `(input_size,)`, `b_prev` is `(d_state,)` (both unused in euler mode);
    /// `scratch` is `MambaScratch::new(input_size, d_state, dt_rank)`.
    ///
    /// Real mode reuses `expm1_over_x` (with `exp_m1`) so `real`+`euler` is bit-identical
    /// to `MambaLayer::forward`. Complex mode uses `expm1_over_x_complex`. Trapezoidal adds
    /// the `(1-lambda)` cross term on the previous (B, x); `lambda -> 1` recovers euler.
    pub fn forward<'s>(
        &self,
        x: &[f64],
        h_re: &mut nalgebra::DMatrix<f64>,
        h_im: &mut nalgebra::DMatrix<f64>,
        x_prev: &mut nalgebra::DVector<f64>,
        b_prev: &mut nalgebra::DVector<f64>,
        scratch: &'s mut MambaScratch,
    ) -> &'s [f64] {
        debug_assert_eq!(x.len(), self.input_size);
        debug_assert!(
            self.derived_is_current(),
            "Mamba3Layer: derived A / lambda is stale (a_log or lambda_logit written without a rebuild)"
        );

        scratch.project(x, &self.x_proj_w, &self.dt_proj_w, &self.dt_proj_b);
        let (delta, b_vec, c_vec, y) = scratch.split(self.d_state);

        for d in 0..self.input_size {
            let dd = delta[d];
            let xd = x[d];
            let lam = self.lam[d];
            let xp = x_prev[d];
            let mut acc = 0.0;
            for n in 0..self.d_state {
                let za_r = dd * self.a[(d, n)];
                if self.complex {
                    let ai = self.a_imag.as_ref().unwrap()[(d, n)];
                    let za_i = dd * ai;
                    let r = za_r.exp();
                    let (alpha_r, alpha_i) = (r * za_i.cos(), r * za_i.sin());
                    let (ex_r, ex_i) = expm1_over_x_complex(za_r, za_i);
                    // current-input drive: b_bar = delta * B[n] * expm1_over_x_complex(za)
                    let bb_r = dd * b_vec[n] * ex_r;
                    let bb_i = dd * b_vec[n] * ex_i;
                    let hr = h_re[(d, n)];
                    let hi = h_im[(d, n)];
                    // h = alpha*h + lambda*b_bar*x  (+ (1-lambda)*delta*alpha*B_prev*x_prev)
                    let mut nr = alpha_r * hr - alpha_i * hi + lam * bb_r * xd;
                    let mut ni = alpha_r * hi + alpha_i * hr + lam * bb_i * xd;
                    if self.trapezoidal {
                        let cross = (1.0 - lam) * dd * b_prev[n] * xp;
                        nr += alpha_r * cross;
                        ni += alpha_i * cross;
                    }
                    h_re[(d, n)] = nr;
                    h_im[(d, n)] = ni;
                    acc += nr * c_vec[n]; // readout reads Re(h)
                } else {
                    let alpha = za_r.exp();
                    let bb = dd * b_vec[n] * expm1_over_x(za_r);
                    let mut nr = alpha * h_re[(d, n)] + lam * bb * xd;
                    if self.trapezoidal {
                        nr += (1.0 - lam) * dd * alpha * b_prev[n] * xp;
                    }
                    h_re[(d, n)] = nr;
                    acc += nr * c_vec[n];
                }
            }
            y[d] = acc + self.d_skip[d] * xd;
        }
        if self.trapezoidal {
            x_prev.copy_from_slice(x);
            b_prev.copy_from_slice(b_vec);
        }
        y
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
    use crate::data::neural::LayerWeights;

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

    // real+euler Mamba3 shares MambaLayer's exact flat layout; load the same slab into both.
    fn mamba_ref(
        input_size: usize,
        d_state: usize,
        dt_rank: usize,
    ) -> (super::super::MambaLayer, Mamba3Layer) {
        let mut m3 = Mamba3Layer::zeros(input_size, d_state, dt_rank, false, false);
        let n = m3.n_params();
        let slab: Vec<f64> = (0..n).map(|i| 0.05 * ((i % 7) as f64 - 3.0)).collect();
        m3.from_flat(&slab);
        let mut m = super::super::MambaLayer::zeros(input_size, d_state, dt_rank);
        m.from_flat(&slab);
        (m, m3)
    }

    #[test]
    fn real_euler_bit_identical_to_mamba() {
        let (m, m3) = mamba_ref(4, 3, 2);
        let mut h = nalgebra::DMatrix::zeros(4, 3);
        let mut hr = nalgebra::DMatrix::zeros(4, 3);
        let mut hi = nalgebra::DMatrix::zeros(4, 3);
        let mut xp = nalgebra::DVector::zeros(4);
        let mut bp = nalgebra::DVector::zeros(3);
        let (mut s, mut s3) = (MambaScratch::new(4, 3, 2), MambaScratch::new(4, 3, 2));
        for t in 0..20 {
            let x: Vec<f64> = (0..4)
                .map(|d| 0.1 * (d as f64 + 1.0) * (t as f64 + 1.0).sin())
                .collect();
            let ym = m.forward(&x, &mut h, &mut s);
            let y3 = m3.forward(&x, &mut hr, &mut hi, &mut xp, &mut bp, &mut s3);
            for d in 0..4 {
                assert_eq!(ym[d], y3[d], "t={t} d={d}"); // BIT-identical
            }
        }
    }

    #[test]
    fn trapezoidal_reduces_to_euler_at_high_lambda() {
        let mut euler = Mamba3Layer::zeros(4, 3, 2, false, false);
        let n_e = euler.n_params();
        let slab: Vec<f64> = (0..n_e).map(|i| 0.05 * ((i % 5) as f64 - 2.0)).collect();
        euler.from_flat(&slab);
        // trapezoidal layout = euler + lambda_logit(input_size) inserted before d_skip.
        let mut trap = Mamba3Layer::zeros(4, 3, 2, true, false);
        let split = n_e - 4; // everything up to (but not incl.) d_skip
        let mut tslab = slab[..split].to_vec();
        tslab.extend(std::iter::repeat_n(30.0, 4)); // lambda_logit -> sigmoid ~ 1
        tslab.extend(&slab[split..]); // d_skip
        trap.from_flat(&tslab);
        let mut he = nalgebra::DMatrix::zeros(4, 3);
        let (mut hr, mut hi) = (
            nalgebra::DMatrix::zeros(4, 3),
            nalgebra::DMatrix::zeros(4, 3),
        );
        let mut xp = nalgebra::DVector::zeros(4);
        let mut bp = nalgebra::DVector::zeros(3);
        let mut he0 = nalgebra::DMatrix::zeros(4, 3);
        let (mut z1, mut z2) = (nalgebra::DVector::zeros(4), nalgebra::DVector::zeros(3));
        let (mut se, mut st) = (MambaScratch::new(4, 3, 2), MambaScratch::new(4, 3, 2));
        for t in 0..15 {
            let x: Vec<f64> = (0..4)
                .map(|d| 0.2 * (d as f64 - 1.0) * (t as f64).cos())
                .collect();
            let ye = euler.forward(&x, &mut he, &mut he0, &mut z1, &mut z2, &mut se);
            let yt = trap.forward(&x, &mut hr, &mut hi, &mut xp, &mut bp, &mut st);
            for d in 0..4 {
                assert!(
                    (ye[d] - yt[d]).abs() < 1e-12,
                    "t={t} d={d} {} vs {}",
                    ye[d],
                    yt[d]
                );
            }
        }
    }

    #[test]
    fn complex_warmup_deterministic() {
        let mut m = Mamba3Layer::zeros(3, 4, 1, false, true);
        let n = m.n_params();
        let slab: Vec<f64> = (0..n).map(|i| 0.03 * ((i % 9) as f64 - 4.0)).collect();
        m.from_flat(&slab);
        let run = || {
            let (mut hr, mut hi) = (
                nalgebra::DMatrix::zeros(3, 4),
                nalgebra::DMatrix::zeros(3, 4),
            );
            let (mut xp, mut bp) = (nalgebra::DVector::zeros(3), nalgebra::DVector::zeros(4));
            let mut s = MambaScratch::new(3, 4, 1);
            let mut last = vec![];
            for t in 0..10 {
                let x: Vec<f64> = (0..3).map(|d| 0.15 * (d as f64 + t as f64).sin()).collect();
                last = m
                    .forward(&x, &mut hr, &mut hi, &mut xp, &mut bp, &mut s)
                    .to_vec();
            }
            last
        };
        assert_eq!(run(), run());
        assert!(run().iter().all(|v| v.is_finite()));
    }

    fn fly_trapezoidal(m: &Mamba3Layer) -> Vec<f64> {
        let (mut hr, mut hi) = (
            nalgebra::DMatrix::zeros(3, 2),
            nalgebra::DMatrix::zeros(3, 2),
        );
        let (mut xp, mut bp) = (nalgebra::DVector::zeros(3), nalgebra::DVector::zeros(2));
        let mut s = MambaScratch::new(3, 2, 1);
        let mut y = Vec::new();
        for t in 0..5 {
            let x = [0.4, -0.2 * t as f64, 0.9];
            y = m
                .forward(&x, &mut hr, &mut hi, &mut xp, &mut bp, &mut s)
                .to_vec();
        }
        y
    }

    fn loaded_trapezoidal(scale: f64) -> Mamba3Layer {
        let mut m = Mamba3Layer::zeros(3, 2, 1, true, false);
        let flat: Vec<f64> = (0..m.n_params())
            .map(|i| scale * (0.07 * (i % 11) as f64 - 0.3))
            .collect();
        m.from_flat(&flat);
        m
    }

    #[test]
    fn reload_rebuilds_derived_a_and_lam() {
        let fresh = loaded_trapezoidal(1.0);
        assert_eq!(fresh.a, a_from_log(&fresh.a_log));
        assert_eq!(fresh.lam, fresh.lambda_logit.as_ref().unwrap().map(sigmoid));

        let mut reloaded = loaded_trapezoidal(-2.0);
        assert_ne!(fly_trapezoidal(&reloaded), fly_trapezoidal(&fresh));
        reloaded.from_flat(&fresh.to_flat());
        assert_eq!(fly_trapezoidal(&reloaded), fly_trapezoidal(&fresh));
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "is stale")]
    fn forward_refuses_a_stale_a() {
        let mut m = loaded_trapezoidal(1.0);
        m.a_log[(1, 1)] = -0.5; // in-module write that skips the rebuild
        fly_trapezoidal(&m);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "is stale")]
    fn forward_refuses_a_stale_lambda() {
        let mut m = loaded_trapezoidal(1.0);
        m.lambda_logit.as_mut().unwrap()[0] = 2.0;
        fly_trapezoidal(&m);
    }
}
