//! Mamba / Mamba-3 forward: bit-identity against the pre-#238 forward and a
//! zero-allocation gate.
//!
//! `reference` is the forward as it stood before #238 (weight-only
//! transcendentals evaluated per tick, nine allocations per call), copied
//! verbatim onto tensors parsed from the flat slab. The layers must reproduce
//! it bit for bit, state included, in every Mamba-3 flag combination.
//!
//! The allocation gate counts through a thread-local counting allocator, so it
//! is exact under the parallel test harness.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use aerocapture::data::neural::{LayerWeights, Mamba3Layer, MambaLayer, MambaScratch};
use nalgebra::{DMatrix, DVector};
use rand::{RngExt, SeedableRng};

struct Counting;

thread_local! {
    static ALLOCS: Cell<u64> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn allocs() -> u64 {
    ALLOCS.with(Cell::get)
}

#[test]
fn counting_allocator_counts() {
    let before = allocs();
    std::hint::black_box(Vec::<u8>::with_capacity(8));
    assert_eq!(allocs() - before, 1);
}

mod reference {
    use super::{DMatrix, DVector};

    pub struct Weights {
        pub input_size: usize,
        pub d_state: usize,
        pub dt_rank: usize,
        pub trapezoidal: bool,
        pub complex: bool,
        pub x_proj_w: DMatrix<f64>,
        pub dt_proj_w: DMatrix<f64>,
        pub dt_proj_b: DVector<f64>,
        pub a_log: DMatrix<f64>,
        pub a_imag: Option<DMatrix<f64>>,
        pub lambda_logit: Option<DVector<f64>>,
        pub d_skip: DVector<f64>,
    }

    impl Weights {
        /// Flat order: x_proj_w, dt_proj_w, dt_proj_b, a_log, [a_imag],
        /// [lambda_logit], d_skip; matrices row-major.
        pub fn from_flat(
            flat: &[f64],
            input_size: usize,
            d_state: usize,
            dt_rank: usize,
            trapezoidal: bool,
            complex: bool,
        ) -> Self {
            let mut at = 0;
            let mut take = |n: usize| {
                at += n;
                &flat[at - n..at]
            };
            let rows = dt_rank + 2 * d_state;
            let x_proj_w = DMatrix::from_row_slice(rows, input_size, take(rows * input_size));
            let dt_proj_w =
                DMatrix::from_row_slice(input_size, dt_rank, take(input_size * dt_rank));
            let dt_proj_b = DVector::from_row_slice(take(input_size));
            let a_log = DMatrix::from_row_slice(input_size, d_state, take(input_size * d_state));
            let a_imag = complex
                .then(|| DMatrix::from_row_slice(input_size, d_state, take(input_size * d_state)));
            let lambda_logit = trapezoidal.then(|| DVector::from_row_slice(take(input_size)));
            let d_skip = DVector::from_row_slice(take(input_size));
            assert_eq!(at, flat.len());
            Self {
                input_size,
                d_state,
                dt_rank,
                trapezoidal,
                complex,
                x_proj_w,
                dt_proj_w,
                dt_proj_b,
                a_log,
                a_imag,
                lambda_logit,
                d_skip,
            }
        }
    }

    fn softplus(x: f64) -> f64 {
        let a = x.abs();
        x.max(0.0) + (-a).exp().ln_1p()
    }

    fn expm1_over_x(z: f64) -> f64 {
        if z.abs() < 1e-8 {
            1.0 + z * 0.5 + z * z / 6.0
        } else {
            z.exp_m1() / z
        }
    }

    fn expm1_over_x_complex(zr: f64, zi: f64) -> (f64, f64) {
        let mag = (zr * zr + zi * zi).sqrt();
        if mag < 1e-8 {
            let z2r = zr * zr - zi * zi;
            let z2i = 2.0 * zr * zi;
            (1.0 + 0.5 * zr + z2r / 6.0, 0.5 * zi + z2i / 6.0)
        } else {
            let er = zr.exp();
            let ez_r = er * zi.cos();
            let ez_i = er * zi.sin();
            let num_r = ez_r - 1.0;
            let num_i = ez_i;
            let denom = zr * zr + zi * zi;
            (
                (num_r * zr + num_i * zi) / denom,
                (num_i * zr - num_r * zi) / denom,
            )
        }
    }

    /// `MambaLayer::forward` before #238.
    pub fn mamba(w: &Weights, x: &[f64], h: &mut DMatrix<f64>) -> Vec<f64> {
        let x_vec = DVector::from_row_slice(x);

        let proj = &w.x_proj_w * &x_vec;
        let dt_pre: Vec<f64> = (0..w.dt_rank).map(|i| proj[i]).collect();
        let b_vec: Vec<f64> = (0..w.d_state).map(|i| proj[w.dt_rank + i]).collect();
        let c_vec: Vec<f64> = (0..w.d_state)
            .map(|i| proj[w.dt_rank + w.d_state + i])
            .collect();

        let dt_pre_v = DVector::from_row_slice(&dt_pre);
        let dt_lifted = &w.dt_proj_w * &dt_pre_v + &w.dt_proj_b;
        let delta: Vec<f64> = (0..w.input_size).map(|i| softplus(dt_lifted[i])).collect();

        let mut y = vec![0.0_f64; w.input_size];
        for d in 0..w.input_size {
            let delta_d = delta[d];
            let x_d = x[d];
            let mut acc = 0.0;
            for n in 0..w.d_state {
                let a_dn = -w.a_log[(d, n)].exp();
                let za = delta_d * a_dn;
                let a_bar = za.exp();
                let b_bar = delta_d * b_vec[n] * expm1_over_x(za);
                h[(d, n)] = a_bar * h[(d, n)] + b_bar * x_d;
                acc += h[(d, n)] * c_vec[n];
            }
            y[d] = acc + w.d_skip[d] * x_d;
        }
        y
    }

    /// `Mamba3Layer::forward` before #238.
    pub fn mamba3(
        w: &Weights,
        x: &[f64],
        h_re: &mut DMatrix<f64>,
        h_im: &mut DMatrix<f64>,
        x_prev: &mut DVector<f64>,
        b_prev: &mut DVector<f64>,
    ) -> Vec<f64> {
        let x_vec = DVector::from_row_slice(x);
        let proj = &w.x_proj_w * &x_vec;
        let dt_pre: Vec<f64> = (0..w.dt_rank).map(|i| proj[i]).collect();
        let b_vec: Vec<f64> = (0..w.d_state).map(|i| proj[w.dt_rank + i]).collect();
        let c_vec: Vec<f64> = (0..w.d_state)
            .map(|i| proj[w.dt_rank + w.d_state + i])
            .collect();

        let dt_pre_v = DVector::from_row_slice(&dt_pre);
        let dt_lifted = &w.dt_proj_w * &dt_pre_v + &w.dt_proj_b;
        let delta: Vec<f64> = (0..w.input_size).map(|i| softplus(dt_lifted[i])).collect();

        let mut y = vec![0.0_f64; w.input_size];
        for d in 0..w.input_size {
            let dd = delta[d];
            let xd = x[d];
            let lam = w
                .lambda_logit
                .as_ref()
                .map_or(1.0, |ll| 1.0 / (1.0 + (-ll[d]).exp()));
            let xp = x_prev[d];
            let mut acc = 0.0;
            for n in 0..w.d_state {
                let ar = -w.a_log[(d, n)].exp();
                let za_r = dd * ar;
                if w.complex {
                    let ai = w.a_imag.as_ref().unwrap()[(d, n)];
                    let za_i = dd * ai;
                    let r = za_r.exp();
                    let (alpha_r, alpha_i) = (r * za_i.cos(), r * za_i.sin());
                    let (ex_r, ex_i) = expm1_over_x_complex(za_r, za_i);
                    let bb_r = dd * b_vec[n] * ex_r;
                    let bb_i = dd * b_vec[n] * ex_i;
                    let hr = h_re[(d, n)];
                    let hi = h_im[(d, n)];
                    let mut nr = alpha_r * hr - alpha_i * hi + lam * bb_r * xd;
                    let mut ni = alpha_r * hi + alpha_i * hr + lam * bb_i * xd;
                    if w.trapezoidal {
                        let cross = (1.0 - lam) * dd * b_prev[n] * xp;
                        nr += alpha_r * cross;
                        ni += alpha_i * cross;
                    }
                    h_re[(d, n)] = nr;
                    h_im[(d, n)] = ni;
                    acc += nr * c_vec[n];
                } else {
                    let alpha = za_r.exp();
                    let bb = dd * b_vec[n] * expm1_over_x(za_r);
                    let mut nr = alpha * h_re[(d, n)] + lam * bb * xd;
                    if w.trapezoidal {
                        nr += (1.0 - lam) * dd * alpha * b_prev[n] * xp;
                    }
                    h_re[(d, n)] = nr;
                    acc += nr * c_vec[n];
                }
            }
            y[d] = acc + w.d_skip[d] * xd;
        }
        if w.trapezoidal {
            *x_prev = x_vec;
            *b_prev = DVector::from_row_slice(&b_vec);
        }
        y
    }
}

// (input_size, d_state, dt_rank); the first is the deployed Mamba-962 layer.
const SHAPES: [(usize, usize, usize); 4] = [(16, 12, 1), (4, 3, 2), (5, 2, 3), (1, 1, 1)];
const FLAGS: [(bool, bool); 4] = [(false, false), (true, false), (false, true), (true, true)];
const TICKS: usize = 60;

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn random(rng: &mut rand::rngs::StdRng, n: usize, span: f64) -> Vec<f64> {
    (0..n).map(|_| rng.random_range(-span..span)).collect()
}

#[test]
fn mamba_forward_bit_identical_to_reference() {
    for (k, &(i, n, r)) in SHAPES.iter().enumerate() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(238 + k as u64);
        let mut layer = MambaLayer::zeros(i, n, r);
        let flat = random(&mut rng, layer.n_params(), 2.0);
        layer.from_flat(&flat);
        let w = reference::Weights::from_flat(&flat, i, n, r, false, false);

        let (mut h, mut h_ref) = (DMatrix::zeros(i, n), DMatrix::zeros(i, n));
        let mut scratch = MambaScratch::new(i, n, r);
        for t in 0..TICKS {
            let x = random(&mut rng, i, 3.0);
            let want = reference::mamba(&w, &x, &mut h_ref);
            let got = layer.forward(&x, &mut h, &mut scratch);
            assert_eq!(bits(got), bits(&want), "shape {k} tick {t}: y");
            assert_eq!(
                bits(h.as_slice()),
                bits(h_ref.as_slice()),
                "shape {k} tick {t}: h"
            );
        }
    }
}

#[test]
fn mamba3_forward_bit_identical_to_reference() {
    for (k, &(i, n, r)) in SHAPES.iter().enumerate() {
        for &(trap, cplx) in &FLAGS {
            let tag = format!("shape {k} trap={trap} cplx={cplx}");
            let mut rng = rand::rngs::StdRng::seed_from_u64(1238 + k as u64);
            let mut layer = Mamba3Layer::zeros(i, n, r, trap, cplx);
            let flat = random(&mut rng, layer.n_params(), 2.0);
            layer.from_flat(&flat);
            let w = reference::Weights::from_flat(&flat, i, n, r, trap, cplx);

            let (mut h_re, mut h_im) = (DMatrix::zeros(i, n), DMatrix::zeros(i, n));
            let (mut x_prev, mut b_prev) = (DVector::zeros(i), DVector::zeros(n));
            let (mut r_re, mut r_im) = (DMatrix::zeros(i, n), DMatrix::zeros(i, n));
            let (mut r_xp, mut r_bp) = (DVector::zeros(i), DVector::zeros(n));
            let mut scratch = MambaScratch::new(i, n, r);
            for t in 0..TICKS {
                let x = random(&mut rng, i, 3.0);
                let want = reference::mamba3(&w, &x, &mut r_re, &mut r_im, &mut r_xp, &mut r_bp);
                let got = layer.forward(
                    &x,
                    &mut h_re,
                    &mut h_im,
                    &mut x_prev,
                    &mut b_prev,
                    &mut scratch,
                );
                assert_eq!(bits(got), bits(&want), "{tag} tick {t}: y");
                assert_eq!(
                    bits(h_re.as_slice()),
                    bits(r_re.as_slice()),
                    "{tag} tick {t}: h_re"
                );
                assert_eq!(
                    bits(h_im.as_slice()),
                    bits(r_im.as_slice()),
                    "{tag} tick {t}: h_im"
                );
                assert_eq!(
                    bits(x_prev.as_slice()),
                    bits(r_xp.as_slice()),
                    "{tag} tick {t}: x_prev"
                );
                assert_eq!(
                    bits(b_prev.as_slice()),
                    bits(r_bp.as_slice()),
                    "{tag} tick {t}: b_prev"
                );
            }
        }
    }
}

#[test]
fn mamba_forward_does_not_allocate() {
    let (i, n, r) = SHAPES[0];
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let mut layer = MambaLayer::zeros(i, n, r);
    let flat = random(&mut rng, layer.n_params(), 2.0);
    layer.from_flat(&flat);
    let mut h = DMatrix::zeros(i, n);
    let mut scratch = MambaScratch::new(i, n, r);
    let x = random(&mut rng, i, 3.0);

    let before = allocs();
    for _ in 0..TICKS {
        layer.forward(&x, &mut h, &mut scratch);
    }
    assert_eq!(allocs() - before, 0);
}

#[test]
fn mamba3_forward_does_not_allocate() {
    let (i, n, r) = SHAPES[0];
    for &(trap, cplx) in &FLAGS {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let mut layer = Mamba3Layer::zeros(i, n, r, trap, cplx);
        let flat = random(&mut rng, layer.n_params(), 2.0);
        layer.from_flat(&flat);
        let (mut h_re, mut h_im) = (DMatrix::zeros(i, n), DMatrix::zeros(i, n));
        let (mut x_prev, mut b_prev) = (DVector::zeros(i), DVector::zeros(n));
        let mut scratch = MambaScratch::new(i, n, r);
        let x = random(&mut rng, i, 3.0);

        let before = allocs();
        for _ in 0..TICKS {
            layer.forward(
                &x,
                &mut h_re,
                &mut h_im,
                &mut x_prev,
                &mut b_prev,
                &mut scratch,
            );
        }
        assert_eq!(allocs() - before, 0, "trap={trap} cplx={cplx}");
    }
}
