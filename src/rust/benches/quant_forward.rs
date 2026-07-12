//! Deployment-benefit microbenchmark for the Mamba-962 guidance head.
//!
//! Kernels: f64 (deployed `NeuralNetModel::forward`), hand-rolled f64/f32
//! references, weight-only-int8 + dynamic-int8-activation (w8a8), and packed
//! int4 weights + int8 activations (w4a8). Quantization accelerates the matvec
//! projections only; the SSM recurrence (softplus, exp, state update) stays in
//! floating point in every variant -- the honest dilution a fixed-point flight
//! implementation would see. Weights are deterministic pseudo-random: the cost
//! of a matvec does not depend on the values.
//!
//! Spec: docs/superpowers/specs/2026-07-10-quantization-study-appendix-design.md

use aerocapture::data::neural::NeuralNetModel;
use aerocapture::data::nn_state::NnState;
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

// Champion shapes: Dense(17 -> 16, swish) -> Mamba(16, d_state 12, dt_rank 1) -> Dense(16 -> 2, asinh)
const N_IN: usize = 17;
const H: usize = 16;
const D_STATE: usize = 12;
const DT_RANK: usize = 1;
const XPROJ_ROWS: usize = DT_RANK + 2 * D_STATE; // 25
const N_OUT: usize = 2;

fn pseudo(seed: u64, n: usize) -> Vec<f64> {
    // Deterministic hash-noise in [-0.5, 0.5]; no rand dependency.
    (0..n)
        .map(|k| {
            let x = ((seed + k as u64 + 1) as f64 * 12.9898).sin() * 43758.5453;
            x - x.floor() - 0.5
        })
        .collect()
}

fn mat_json(v: &[f64], rows: usize, cols: usize) -> serde_json::Value {
    serde_json::Value::Array(
        (0..rows)
            .map(|r| {
                serde_json::Value::Array(
                    (0..cols)
                        .map(|c| serde_json::json!(v[r * cols + c]))
                        .collect(),
                )
            })
            .collect(),
    )
}

struct Weights {
    d0_w: Vec<f64>, // (H, N_IN) row-major
    d0_b: Vec<f64>,
    x_proj: Vec<f64>, // (XPROJ_ROWS, H)
    dt_w: Vec<f64>,   // (H, DT_RANK)
    dt_b: Vec<f64>,
    a_log: Vec<f64>, // (H, D_STATE)
    d_skip: Vec<f64>,
    d2_w: Vec<f64>, // (N_OUT, H)
    d2_b: Vec<f64>,
}

fn make_weights() -> Weights {
    Weights {
        d0_w: pseudo(1, H * N_IN),
        d0_b: pseudo(2, H),
        x_proj: pseudo(3, XPROJ_ROWS * H),
        dt_w: pseudo(4, H * DT_RANK),
        dt_b: pseudo(5, H),
        a_log: pseudo(6, H * D_STATE).iter().map(|v| v + 1.0).collect(), // positive-ish, HiPPO-like
        d_skip: pseudo(7, H).iter().map(|v| v + 1.0).collect(),
        d2_w: pseudo(8, N_OUT * H),
        d2_b: pseudo(9, N_OUT),
    }
}

fn model_json(w: &Weights) -> String {
    serde_json::json!({
        "format_version": 2,
        "architecture": [
            {"type": "dense", "input_size": N_IN, "output_size": H, "activation": "swish"},
            {"type": "mamba", "input_size": H, "d_state": D_STATE, "dt_rank": DT_RANK},
            {"type": "dense", "input_size": H, "output_size": N_OUT, "activation": "asinh"},
        ],
        "weights": {
            "layer_0": {"w": mat_json(&w.d0_w, H, N_IN), "b": w.d0_b},
            "layer_1": {
                "x_proj_w": mat_json(&w.x_proj, XPROJ_ROWS, H),
                "dt_proj_w": mat_json(&w.dt_w, H, DT_RANK),
                "dt_proj_b": w.dt_b,
                "a_log": mat_json(&w.a_log, H, D_STATE),
                "d_skip": w.d_skip,
            },
            "layer_2": {"w": mat_json(&w.d2_w, N_OUT, H), "b": w.d2_b},
        },
    })
    .to_string()
}

fn swish(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

fn softplus(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

fn expm1_over_x(z: f64) -> f64 {
    if z.abs() < 1e-8 {
        1.0 + z / 2.0
    } else {
        z.exp_m1() / z
    }
}

/// Hand-rolled f64 full-tick forward -- validated against `NeuralNetModel::forward`.
struct HandF64 {
    w: Weights,
    h: Vec<f64>, // (H, D_STATE) row-major state
}

impl HandF64 {
    fn forward(&mut self, input: &[f64]) -> [f64; N_OUT] {
        let mut a1 = [0.0f64; H];
        for ((slot, row), &bias) in a1
            .iter_mut()
            .zip(self.w.d0_w.chunks(N_IN))
            .zip(self.w.d0_b.iter())
        {
            let acc: f64 = bias
                + row
                    .iter()
                    .zip(input.iter())
                    .map(|(w, x)| w * x)
                    .sum::<f64>();
            *slot = swish(acc);
        }
        let y1 = mamba_step_f64(&self.w, &a1, &mut self.h);
        let mut out = [0.0f64; N_OUT];
        for ((slot, row), &bias) in out
            .iter_mut()
            .zip(self.w.d2_w.chunks(H))
            .zip(self.w.d2_b.iter())
        {
            let acc: f64 = bias + row.iter().zip(y1.iter()).map(|(w, x)| w * x).sum::<f64>();
            *slot = acc.asinh();
        }
        out
    }
}

/// f64 x_proj matvec + the shared fp SSM core (`finish_mamba`), which the
/// quantized variants reuse verbatim -- the fp-dilution the appendix reports.
fn mamba_step_f64(w: &Weights, x: &[f64; H], h: &mut [f64]) -> [f64; H] {
    let mut proj = [0.0f64; XPROJ_ROWS];
    for (slot, row) in proj.iter_mut().zip(w.x_proj.chunks(H)) {
        *slot = row.iter().zip(x.iter()).map(|(wv, xv)| wv * xv).sum();
    }
    finish_mamba(w, x, h, &proj)
}

/// Everything after the x_proj matvec: dt lift + softplus + ZOH recurrence + readout.
fn finish_mamba(w: &Weights, x: &[f64; H], h: &mut [f64], proj: &[f64; XPROJ_ROWS]) -> [f64; H] {
    let dt_pre = &proj[0..DT_RANK];
    let b_vec = &proj[DT_RANK..DT_RANK + D_STATE];
    let c_vec = &proj[DT_RANK + D_STATE..XPROJ_ROWS];
    let mut y = [0.0f64; H];
    for d in 0..H {
        let dt_row = &w.dt_w[d * DT_RANK..(d + 1) * DT_RANK];
        let lift = w.dt_b[d]
            + dt_row
                .iter()
                .zip(dt_pre.iter())
                .map(|(wv, dv)| wv * dv)
                .sum::<f64>();
        let delta = softplus(lift);
        let mut acc = 0.0;
        for n in 0..D_STATE {
            let a_dn = -w.a_log[d * D_STATE + n].exp();
            let za = delta * a_dn;
            let a_bar = za.exp();
            let b_bar = delta * b_vec[n] * expm1_over_x(za);
            let idx = d * D_STATE + n;
            h[idx] = a_bar * h[idx] + b_bar * x[d];
            acc += h[idx] * c_vec[n];
        }
        y[d] = acc + w.d_skip[d] * x[d];
    }
    y
}

/// f32 variant: same structure, all math in f32 (dtype-width reference row).
struct HandF32 {
    d0_w: Vec<f32>,
    d0_b: Vec<f32>,
    x_proj: Vec<f32>,
    dt_w: Vec<f32>,
    dt_b: Vec<f32>,
    a_log: Vec<f32>,
    d_skip: Vec<f32>,
    d2_w: Vec<f32>,
    d2_b: Vec<f32>,
    h: Vec<f32>,
}

impl HandF32 {
    fn new(w: &Weights) -> Self {
        let c = |v: &[f64]| v.iter().map(|x| *x as f32).collect::<Vec<f32>>();
        Self {
            d0_w: c(&w.d0_w),
            d0_b: c(&w.d0_b),
            x_proj: c(&w.x_proj),
            dt_w: c(&w.dt_w),
            dt_b: c(&w.dt_b),
            a_log: c(&w.a_log),
            d_skip: c(&w.d_skip),
            d2_w: c(&w.d2_w),
            d2_b: c(&w.d2_b),
            h: vec![0.0f32; H * D_STATE],
        }
    }

    fn forward(&mut self, input: &[f64]) -> [f64; N_OUT] {
        let xin: Vec<f32> = input.iter().map(|v| *v as f32).collect();
        let mut a1 = [0.0f32; H];
        for ((slot, row), &bias) in a1
            .iter_mut()
            .zip(self.d0_w.chunks(N_IN))
            .zip(self.d0_b.iter())
        {
            let acc: f32 = bias + row.iter().zip(xin.iter()).map(|(w, x)| w * x).sum::<f32>();
            *slot = acc / (1.0 + (-acc).exp());
        }
        let mut proj = [0.0f32; XPROJ_ROWS];
        for (slot, row) in proj.iter_mut().zip(self.x_proj.chunks(H)) {
            *slot = row.iter().zip(a1.iter()).map(|(w, x)| w * x).sum();
        }
        let dt_pre = &proj[0..DT_RANK];
        let b_vec = &proj[DT_RANK..DT_RANK + D_STATE];
        let c_vec = &proj[DT_RANK + D_STATE..XPROJ_ROWS];
        let mut y = [0.0f32; H];
        for d in 0..H {
            let dt_row = &self.dt_w[d * DT_RANK..(d + 1) * DT_RANK];
            let lift = self.dt_b[d]
                + dt_row
                    .iter()
                    .zip(dt_pre.iter())
                    .map(|(w, x)| w * x)
                    .sum::<f32>();
            let delta = lift.max(0.0) + (-lift.abs()).exp().ln_1p();
            let mut acc = 0.0f32;
            for n in 0..D_STATE {
                let za = delta * (-self.a_log[d * D_STATE + n].exp());
                let a_bar = za.exp();
                let bz = if za.abs() < 1e-4 {
                    1.0 + za / 2.0
                } else {
                    za.exp_m1() / za
                };
                let idx = d * D_STATE + n;
                self.h[idx] = a_bar * self.h[idx] + delta * b_vec[n] * bz * a1[d];
                acc += self.h[idx] * c_vec[n];
            }
            y[d] = acc + self.d_skip[d] * a1[d];
        }
        let mut out = [0.0f64; N_OUT];
        for ((slot, row), &bias) in out
            .iter_mut()
            .zip(self.d2_w.chunks(H))
            .zip(self.d2_b.iter())
        {
            let acc: f32 = bias + row.iter().zip(y.iter()).map(|(w, x)| w * x).sum::<f32>();
            *slot = (acc as f64).asinh();
        }
        out
    }
}

/// Per-row symmetric int8 quantization of a (rows, cols) row-major matrix.
fn quant_i8(w: &[f64], rows: usize, cols: usize) -> (Vec<i8>, Vec<f64>) {
    let mut q = vec![0i8; rows * cols];
    let mut scales = vec![1.0f64; rows];
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        let amax = row.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let s = if amax == 0.0 { 1.0 } else { amax / 127.0 };
        scales[r] = s;
        for c in 0..cols {
            q[r * cols + c] = (row[c] / s).round().clamp(-127.0, 127.0) as i8;
        }
    }
    (q, scales)
}

/// Per-row symmetric int4, packed two nibbles per byte (row-major, low nibble first).
fn quant_i4(w: &[f64], rows: usize, cols: usize) -> (Vec<u8>, Vec<f64>) {
    let packed_cols = cols.div_ceil(2);
    let mut q = vec![0u8; rows * packed_cols];
    let mut scales = vec![1.0f64; rows];
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        let amax = row.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let s = if amax == 0.0 { 1.0 } else { amax / 7.0 };
        scales[r] = s;
        for c in 0..cols {
            let v = ((row[c] / s).round().clamp(-7.0, 7.0) as i8) as u8 & 0x0F;
            let byte = &mut q[r * packed_cols + c / 2];
            if c % 2 == 0 {
                *byte |= v
            } else {
                *byte |= v << 4
            }
        }
    }
    (q, scales)
}

/// Dynamic per-vector int8 activation quantization.
fn quant_act(x: &[f64], out: &mut [i8]) -> f64 {
    let amax = x.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let s = if amax == 0.0 { 1.0 } else { amax / 127.0 };
    for (o, v) in out.iter_mut().zip(x.iter()) {
        *o = (v / s).round().clamp(-127.0, 127.0) as i8;
    }
    s
}

fn dot_i8(w: &[i8], x: &[i8]) -> i32 {
    let mut acc = 0i32;
    for (wi, xi) in w.iter().zip(x.iter()) {
        acc += (*wi as i32) * (*xi as i32);
    }
    acc
}

fn dot_i4(packed: &[u8], x: &[i8]) -> i32 {
    let mut acc = 0i32;
    for (k, xi) in x.iter().enumerate() {
        let byte = packed[k / 2];
        let nib = if k % 2 == 0 { byte & 0x0F } else { byte >> 4 };
        let w = ((nib << 4) as i8) >> 4; // sign-extend 4-bit
        acc += (w as i32) * (*xi as i32);
    }
    acc
}

/// w8a8 / w4a8: int projections (dense0, x_proj, dt lift folded into fp, dense2),
/// f64 SSM recurrence via `finish_mamba` -- the same fp core as the references.
struct QuantNet {
    w: Weights,
    d0_q: Vec<i8>,
    d0_s: Vec<f64>,
    xp_q: Vec<i8>,
    xp_s: Vec<f64>,
    d2_q: Vec<i8>,
    d2_s: Vec<f64>,
    d0_q4: Vec<u8>,
    d0_s4: Vec<f64>,
    xp_q4: Vec<u8>,
    xp_s4: Vec<f64>,
    d2_q4: Vec<u8>,
    d2_s4: Vec<f64>,
    h8: Vec<f64>,
    h4: Vec<f64>,
}

impl QuantNet {
    fn new(w: Weights) -> Self {
        let (d0_q, d0_s) = quant_i8(&w.d0_w, H, N_IN);
        let (xp_q, xp_s) = quant_i8(&w.x_proj, XPROJ_ROWS, H);
        let (d2_q, d2_s) = quant_i8(&w.d2_w, N_OUT, H);
        let (d0_q4, d0_s4) = quant_i4(&w.d0_w, H, N_IN);
        let (xp_q4, xp_s4) = quant_i4(&w.x_proj, XPROJ_ROWS, H);
        let (d2_q4, d2_s4) = quant_i4(&w.d2_w, N_OUT, H);
        Self {
            w,
            d0_q,
            d0_s,
            xp_q,
            xp_s,
            d2_q,
            d2_s,
            d0_q4,
            d0_s4,
            xp_q4,
            xp_s4,
            d2_q4,
            d2_s4,
            h8: vec![0.0; H * D_STATE],
            h4: vec![0.0; H * D_STATE],
        }
    }

    fn forward_w8a8(&mut self, input: &[f64]) -> [f64; N_OUT] {
        let mut xq = [0i8; N_IN];
        let sx = quant_act(input, &mut xq);
        let mut a1 = [0.0f64; H];
        for (((slot, row), &scale), &bias) in a1
            .iter_mut()
            .zip(self.d0_q.chunks(N_IN))
            .zip(self.d0_s.iter())
            .zip(self.w.d0_b.iter())
        {
            let acc = dot_i8(row, &xq) as f64;
            *slot = swish(acc * scale * sx + bias);
        }
        let mut a1q = [0i8; H];
        let s1 = quant_act(&a1, &mut a1q);
        let mut proj = [0.0f64; XPROJ_ROWS];
        for ((slot, row), &scale) in proj
            .iter_mut()
            .zip(self.xp_q.chunks(H))
            .zip(self.xp_s.iter())
        {
            *slot = dot_i8(row, &a1q) as f64 * scale * s1;
        }
        let y1 = finish_mamba(&self.w, &a1, &mut self.h8, &proj);
        let mut y1q = [0i8; H];
        let s2 = quant_act(&y1, &mut y1q);
        let mut out = [0.0f64; N_OUT];
        for (((slot, row), &scale), &bias) in out
            .iter_mut()
            .zip(self.d2_q.chunks(H))
            .zip(self.d2_s.iter())
            .zip(self.w.d2_b.iter())
        {
            let acc = dot_i8(row, &y1q) as f64;
            *slot = (acc * scale * s2 + bias).asinh();
        }
        out
    }

    fn forward_w4a8(&mut self, input: &[f64]) -> [f64; N_OUT] {
        let pc_in = N_IN.div_ceil(2);
        let pc_h = H.div_ceil(2);
        let mut xq = [0i8; N_IN];
        let sx = quant_act(input, &mut xq);
        let mut a1 = [0.0f64; H];
        for (((slot, row), &scale), &bias) in a1
            .iter_mut()
            .zip(self.d0_q4.chunks(pc_in))
            .zip(self.d0_s4.iter())
            .zip(self.w.d0_b.iter())
        {
            let acc = dot_i4(row, &xq) as f64;
            *slot = swish(acc * scale * sx + bias);
        }
        let mut a1q = [0i8; H];
        let s1 = quant_act(&a1, &mut a1q);
        let mut proj = [0.0f64; XPROJ_ROWS];
        for ((slot, row), &scale) in proj
            .iter_mut()
            .zip(self.xp_q4.chunks(pc_h))
            .zip(self.xp_s4.iter())
        {
            *slot = dot_i4(row, &a1q) as f64 * scale * s1;
        }
        let y1 = finish_mamba(&self.w, &a1, &mut self.h4, &proj);
        let mut y1q = [0i8; H];
        let s2 = quant_act(&y1, &mut y1q);
        let mut out = [0.0f64; N_OUT];
        for (((slot, row), &scale), &bias) in out
            .iter_mut()
            .zip(self.d2_q4.chunks(pc_h))
            .zip(self.d2_s4.iter())
            .zip(self.w.d2_b.iter())
        {
            let acc = dot_i4(row, &y1q) as f64;
            *slot = (acc * scale * s2 + bias).asinh();
        }
        out
    }
}

fn inputs(n: usize) -> Vec<Vec<f64>> {
    (0..n).map(|k| pseudo(100 + k as u64, N_IN)).collect()
}

fn self_check(model: &NeuralNetModel) {
    // (a) hand-rolled f64 must match the deployed forward to fp-noise level.
    let w = make_weights();
    let mut hand = HandF64 {
        w,
        h: vec![0.0; H * D_STATE],
    };
    let mut state = NnState::for_model(model);
    for x in inputs(200) {
        let a = model.forward(&mut state, &x);
        let b = hand.forward(&x);
        for o in 0..N_OUT {
            assert!(
                (a[o] - b[o]).abs() < 1e-9,
                "handrolled f64 diverges: {} vs {}",
                a[o],
                b[o]
            );
        }
    }
    // (b) quantized kernels: finite and loosely sane vs f64 (activation quant adds real error).
    let mut hand2 = HandF64 {
        w: make_weights(),
        h: vec![0.0; H * D_STATE],
    };
    let mut qn = QuantNet::new(make_weights());
    for x in inputs(200) {
        let r = hand2.forward(&x);
        let q8 = qn.forward_w8a8(&x);
        let q4 = qn.forward_w4a8(&x);
        for o in 0..N_OUT {
            assert!(q8[o].is_finite() && q4[o].is_finite());
            assert!(
                (q8[o] - r[o]).abs() < 0.5,
                "w8a8 wildly off: {} vs {}",
                q8[o],
                r[o]
            );
        }
    }
}

fn bench_forward(c: &mut Criterion) {
    let w = make_weights();
    let json = model_json(&w);
    let model = NeuralNetModel::from_json_str(&json, "bench_model.json").expect("model parse");
    self_check(&model);

    let xs = inputs(64);
    let mut g = c.benchmark_group("forward");

    let mut state = NnState::for_model(&model);
    let mut k = 0usize;
    g.bench_function("f64_model", |b| {
        b.iter(|| {
            k = (k + 1) % xs.len();
            black_box(model.forward(&mut state, black_box(&xs[k])))
        })
    });

    let mut hand = HandF64 {
        w: make_weights(),
        h: vec![0.0; H * D_STATE],
    };
    g.bench_function("f64_handrolled", |b| {
        b.iter(|| {
            k = (k + 1) % xs.len();
            black_box(hand.forward(black_box(&xs[k])))
        })
    });

    let mut hand32 = HandF32::new(&make_weights());
    g.bench_function("f32_handrolled", |b| {
        b.iter(|| {
            k = (k + 1) % xs.len();
            black_box(hand32.forward(black_box(&xs[k])))
        })
    });

    let mut qn = QuantNet::new(make_weights());
    g.bench_function("w8a8", |b| {
        b.iter(|| {
            k = (k + 1) % xs.len();
            black_box(qn.forward_w8a8(black_box(&xs[k])))
        })
    });
    g.bench_function("w4a8", |b| {
        b.iter(|| {
            k = (k + 1) % xs.len();
            black_box(qn.forward_w4a8(black_box(&xs[k])))
        })
    });
    g.finish();
}

criterion_group!(benches, bench_forward);
criterion_main!(benches);
