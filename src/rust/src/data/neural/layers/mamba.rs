//! Selective SSM core (Mamba S6) -- Phase 4a PSO-only MVP.

use super::helpers::{expm1_over_x, softplus};
use super::tensor::tensor_table;

/// Selective SSM core (Mamba S6) -- Phase 4a PSO-only MVP.
///
/// Per-tick forward computes input-dependent Δ, B, C from x via a fused `x_proj`
/// linear projection, discretizes A via ZOH (`A = -exp(a_log)`, diagonal),
/// updates per-channel state `h: (input_size, d_state)`, and emits
/// `y = h @ C + D * x` (skip residual per channel).
///
/// No conv1d, no SiLU gating -- those are the full Mamba block, deferred to
/// Phase 4c. No in/out expansion linears -- user stacks Dense before/after.
#[derive(Debug, Clone)]
pub struct MambaLayer {
    /// d_inner in the paper. Layer fan-in = fan-out = input_size.
    pub input_size: usize,
    /// N in the paper. SSM state dim per channel.
    pub d_state: usize,
    /// Bottleneck rank for the Δ projection (paper default: max(1, input_size / 16)).
    pub dt_rank: usize,

    /// Fused (Δ_pre, B, C) projection. Shape: (dt_rank + 2*d_state, input_size).
    pub x_proj_w: nalgebra::DMatrix<f64>,
    /// Δ lift projection. Shape: (input_size, dt_rank).
    pub dt_proj_w: nalgebra::DMatrix<f64>,
    /// Δ bias (critical init: inv_softplus(uniform(dt_min, dt_max)) per channel).
    pub dt_proj_b: nalgebra::DVector<f64>,
    /// HiPPO log-space reparameterization of A. Physical A = -exp(a_log).
    /// Shape: (input_size, d_state). Strictly negative A ensures stable contraction.
    pub a_log: nalgebra::DMatrix<f64>,
    /// Per-channel skip-residual scalar. Paper default init: 1.0.
    pub d_skip: nalgebra::DVector<f64>,
}

// Flat order: x_proj_w, dt_proj_w (row-major), dt_proj_b, a_log (row-major), d_skip.
tensor_table!(MambaLayer {
    x_proj_w,
    dt_proj_w,
    dt_proj_b,
    a_log,
    d_skip
});

impl MambaLayer {
    pub fn zeros(input_size: usize, d_state: usize, dt_rank: usize) -> Self {
        Self {
            input_size,
            d_state,
            dt_rank,
            x_proj_w: nalgebra::DMatrix::zeros(dt_rank + 2 * d_state, input_size),
            dt_proj_w: nalgebra::DMatrix::zeros(input_size, dt_rank),
            dt_proj_b: nalgebra::DVector::zeros(input_size),
            a_log: nalgebra::DMatrix::zeros(input_size, d_state),
            d_skip: nalgebra::DVector::zeros(input_size),
        }
    }

    /// Single-tick forward. Mutates `h` in place (state update), returns `y`.
    ///
    /// Shapes: `x: [f64; input_size]`, `h: DMatrix<f64> (input_size, d_state)`,
    /// returns `Vec<f64>` length `input_size`.
    ///
    /// Numerical contract: Python mirror (`rl/layers/mamba.py`) agrees to machine
    /// epsilon. The helpers match torch's reduction order, so the residual is general
    /// f64 non-associativity in the softplus/scan path, not an addmm ordering choice.
    /// Uses `softplus` and `expm1_over_x` helpers (free functions in this module).
    pub fn forward(&self, x: &[f64], h: &mut nalgebra::DMatrix<f64>) -> Vec<f64> {
        debug_assert_eq!(x.len(), self.input_size);
        debug_assert_eq!(h.nrows(), self.input_size);
        debug_assert_eq!(h.ncols(), self.d_state);

        let x_vec = nalgebra::DVector::from_row_slice(x);

        // 1. Fused x_proj: produces (dt_rank + 2*d_state,)
        let proj = &self.x_proj_w * &x_vec;
        let dt_pre: Vec<f64> = (0..self.dt_rank).map(|i| proj[i]).collect();
        let b_vec: Vec<f64> = (0..self.d_state).map(|i| proj[self.dt_rank + i]).collect();
        let c_vec: Vec<f64> = (0..self.d_state)
            .map(|i| proj[self.dt_rank + self.d_state + i])
            .collect();

        // 2. dt_proj + softplus -> per-channel positive Δ
        let dt_pre_v = nalgebra::DVector::from_row_slice(&dt_pre);
        let dt_lifted = &self.dt_proj_w * &dt_pre_v + &self.dt_proj_b;
        let delta: Vec<f64> = (0..self.input_size)
            .map(|i| softplus(dt_lifted[i]))
            .collect();

        // 3. ZOH discretization + state update, per (d, n).
        //    A = -exp(a_log), Ā = exp(Δ·A), B̄ = Δ·B · expm1_over_x(Δ·A)
        //    h_new = Ā*h + B̄*x[d]
        //    y[d]   = Σ_n h_new[d, n] * C[n] + D[d] * x[d]
        let mut y = vec![0.0_f64; self.input_size];
        for d in 0..self.input_size {
            let delta_d = delta[d];
            let x_d = x[d];
            let mut acc = 0.0;
            for n in 0..self.d_state {
                let a_dn = -self.a_log[(d, n)].exp();
                let za = delta_d * a_dn;
                let a_bar = za.exp();
                let b_bar = delta_d * b_vec[n] * expm1_over_x(za);
                h[(d, n)] = a_bar * h[(d, n)] + b_bar * x_d;
                acc += h[(d, n)] * c_vec[n];
            }
            y[d] = acc + self.d_skip[d] * x_d;
        }
        y
    }
}
