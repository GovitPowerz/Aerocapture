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
use super::helpers::{expm1_over_x, softplus}; // real path reuses expm1_over_x (bit-identity anchor)

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

    /// Single-tick forward. Mutates state in place, returns `y` (length input_size).
    ///
    /// State: `h_re`/`h_im` are `(input_size, d_state)` (h_im unused in real mode);
    /// `x_prev` is `(input_size,)`, `b_prev` is `(d_state,)` (both unused in euler mode).
    ///
    /// Real mode reuses `expm1_over_x` (with `exp_m1`) so `real`+`euler` is bit-identical
    /// to `MambaLayer::forward`. Complex mode uses `expm1_over_x_complex`. Trapezoidal adds
    /// the `(1-lambda)` cross term on the previous (B, x); `lambda -> 1` recovers euler.
    pub fn forward(
        &self,
        x: &[f64],
        h_re: &mut nalgebra::DMatrix<f64>,
        h_im: &mut nalgebra::DMatrix<f64>,
        x_prev: &mut nalgebra::DVector<f64>,
        b_prev: &mut nalgebra::DVector<f64>,
    ) -> Vec<f64> {
        debug_assert_eq!(x.len(), self.input_size);

        let x_vec = nalgebra::DVector::from_row_slice(x);
        let proj = &self.x_proj_w * &x_vec;
        let dt_pre: Vec<f64> = (0..self.dt_rank).map(|i| proj[i]).collect();
        let b_vec: Vec<f64> = (0..self.d_state).map(|i| proj[self.dt_rank + i]).collect();
        let c_vec: Vec<f64> = (0..self.d_state)
            .map(|i| proj[self.dt_rank + self.d_state + i])
            .collect();

        let dt_pre_v = nalgebra::DVector::from_row_slice(&dt_pre);
        let dt_lifted = &self.dt_proj_w * &dt_pre_v + &self.dt_proj_b;
        let delta: Vec<f64> = (0..self.input_size).map(|i| softplus(dt_lifted[i])).collect();

        let mut y = vec![0.0_f64; self.input_size];
        for d in 0..self.input_size {
            let dd = delta[d];
            let xd = x[d];
            let lam = self
                .lambda_logit
                .as_ref()
                .map_or(1.0, |ll| 1.0 / (1.0 + (-ll[d]).exp()));
            let xp = x_prev[d];
            let mut acc = 0.0;
            for n in 0..self.d_state {
                let ar = -self.a_log[(d, n)].exp();
                let za_r = dd * ar;
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
            *x_prev = x_vec;
            *b_prev = nalgebra::DVector::from_row_slice(&b_vec);
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
        let mut m = super::super::MambaLayer {
            input_size,
            d_state,
            dt_rank,
            x_proj_w: nalgebra::DMatrix::zeros(dt_rank + 2 * d_state, input_size),
            dt_proj_w: nalgebra::DMatrix::zeros(input_size, dt_rank),
            dt_proj_b: nalgebra::DVector::zeros(input_size),
            a_log: nalgebra::DMatrix::zeros(input_size, d_state),
            d_skip: nalgebra::DVector::zeros(input_size),
        };
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
        for t in 0..20 {
            let x: Vec<f64> = (0..4)
                .map(|d| 0.1 * (d as f64 + 1.0) * (t as f64 + 1.0).sin())
                .collect();
            let ym = m.forward(&x, &mut h);
            let y3 = m3.forward(&x, &mut hr, &mut hi, &mut xp, &mut bp);
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
        let (mut hr, mut hi) = (nalgebra::DMatrix::zeros(4, 3), nalgebra::DMatrix::zeros(4, 3));
        let mut xp = nalgebra::DVector::zeros(4);
        let mut bp = nalgebra::DVector::zeros(3);
        let mut he0 = nalgebra::DMatrix::zeros(4, 3);
        let (mut z1, mut z2) = (nalgebra::DVector::zeros(4), nalgebra::DVector::zeros(3));
        for t in 0..15 {
            let x: Vec<f64> = (0..4)
                .map(|d| 0.2 * (d as f64 - 1.0) * (t as f64).cos())
                .collect();
            let ye = euler.forward(&x, &mut he, &mut he0, &mut z1, &mut z2);
            let yt = trap.forward(&x, &mut hr, &mut hi, &mut xp, &mut bp);
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
            let (mut hr, mut hi) = (nalgebra::DMatrix::zeros(3, 4), nalgebra::DMatrix::zeros(3, 4));
            let (mut xp, mut bp) = (nalgebra::DVector::zeros(3), nalgebra::DVector::zeros(4));
            let mut last = vec![];
            for t in 0..10 {
                let x: Vec<f64> = (0..3).map(|d| 0.15 * (d as f64 + t as f64).sin()).collect();
                last = m.forward(&x, &mut hr, &mut hi, &mut xp, &mut bp);
            }
            last
        };
        assert_eq!(run(), run());
        assert!(run().iter().all(|v| v.is_finite()));
    }
}
