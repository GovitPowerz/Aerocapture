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
    /// Private: the only write path is a load, which rebuilds `a`.
    a_log: nalgebra::DMatrix<f64>,
    /// Per-channel skip-residual scalar. Paper default init: 1.0.
    pub d_skip: nalgebra::DVector<f64>,

    /// Derived `A = -exp(a_log)`, rebuilt by `rebuild_a` on every load; not in
    /// the flat chromosome or the JSON.
    a: nalgebra::DMatrix<f64>,
}

// Flat order: x_proj_w, dt_proj_w (row-major), dt_proj_b, a_log (row-major), d_skip.
tensor_table!(
    MambaLayer {
        x_proj_w,
        dt_proj_w,
        dt_proj_b,
        a_log,
        d_skip
    },
    post_load = rebuild_a
);

fn neg_exp(l: f64) -> f64 {
    -l.exp()
}

/// `A = -exp(a_log)`, evaluated at load instead of per tick.
pub(super) fn a_from_log(a_log: &nalgebra::DMatrix<f64>) -> nalgebra::DMatrix<f64> {
    a_log.map(neg_exp)
}

/// True when `a` is bit for bit the `A` derived from `a_log`.
pub(super) fn a_is_current(a: &nalgebra::DMatrix<f64>, a_log: &nalgebra::DMatrix<f64>) -> bool {
    a.shape() == a_log.shape()
        && a.iter()
            .zip(a_log.iter())
            .all(|(a, l)| a.to_bits() == neg_exp(*l).to_bits())
}

/// Per-sim forward scratch shared by `MambaLayer` and `Mamba3Layer`, kept in the
/// layer's `LayerState` so a tick allocates nothing. Every slot is overwritten
/// before it is read, so a state reset leaves it alone.
#[derive(Debug, Clone)]
pub struct MambaScratch {
    /// `x_proj_w @ x`: `[Δ_pre (dt_rank), B (d_state), C (d_state)]`.
    proj: nalgebra::DVector<f64>,
    /// Per-channel positive Δ.
    delta: nalgebra::DVector<f64>,
    y: Vec<f64>,
}

impl MambaScratch {
    pub fn new(input_size: usize, d_state: usize, dt_rank: usize) -> Self {
        Self {
            proj: nalgebra::DVector::zeros(dt_rank + 2 * d_state),
            delta: nalgebra::DVector::zeros(input_size),
            y: vec![0.0; input_size],
        }
    }

    /// `proj = x_proj_w @ x`, `delta = softplus(dt_proj_w @ proj[..dt_rank] + dt_proj_b)`.
    ///
    /// `gemv` into a held destination is the kernel `&a * &x` runs (`beta = 0`
    /// never reads the destination), so the sums keep their order.
    pub(super) fn project(
        &mut self,
        x: &[f64],
        x_proj_w: &nalgebra::DMatrix<f64>,
        dt_proj_w: &nalgebra::DMatrix<f64>,
        dt_proj_b: &nalgebra::DVector<f64>,
    ) {
        let x_vec = nalgebra::DVectorView::from_slice(x, x.len());
        self.proj.gemv(1.0, x_proj_w, &x_vec, 0.0);
        let dt_pre = self.proj.rows(0, dt_proj_w.ncols());
        self.delta.gemv(1.0, dt_proj_w, &dt_pre, 0.0);
        self.delta += dt_proj_b;
        self.delta.apply(|v| *v = softplus(*v));
    }

    /// `(delta, B, C, y)` after `project`.
    pub(super) fn split(&mut self, d_state: usize) -> (&[f64], &[f64], &[f64], &mut [f64]) {
        let bc = &self.proj.as_slice()[self.proj.len() - 2 * d_state..];
        let (b, c) = bc.split_at(d_state);
        (self.delta.as_slice(), b, c, &mut self.y)
    }
}

impl MambaLayer {
    pub fn zeros(input_size: usize, d_state: usize, dt_rank: usize) -> Self {
        let a_log = nalgebra::DMatrix::zeros(input_size, d_state);
        Self {
            input_size,
            d_state,
            dt_rank,
            x_proj_w: nalgebra::DMatrix::zeros(dt_rank + 2 * d_state, input_size),
            dt_proj_w: nalgebra::DMatrix::zeros(input_size, dt_rank),
            dt_proj_b: nalgebra::DVector::zeros(input_size),
            a: a_from_log(&a_log),
            a_log,
            d_skip: nalgebra::DVector::zeros(input_size),
        }
    }

    pub fn a_log(&self) -> &nalgebra::DMatrix<f64> {
        &self.a_log
    }

    /// Recompute the derived `A`. Both load paths (JSON and PSO chromosome)
    /// reach it through the table's `post_load` hook.
    fn rebuild_a(&mut self) {
        self.a = a_from_log(&self.a_log);
    }

    /// Single-tick forward. Mutates `h` in place (state update) and returns `y`
    /// (length `input_size`), borrowed from `scratch`. Allocates nothing.
    ///
    /// Shapes: `x: [f64; input_size]`, `h: DMatrix<f64> (input_size, d_state)`,
    /// `scratch: MambaScratch::new(input_size, d_state, dt_rank)`.
    ///
    /// Numerical contract: Python mirror (`torch_mirror/layers/mamba.py`) agrees to machine
    /// epsilon. The helpers match torch's reduction order, so the residual is general
    /// f64 non-associativity in the softplus/scan path, not an addmm ordering choice.
    /// Uses `softplus` and `expm1_over_x` helpers (free functions in this module).
    pub fn forward<'s>(
        &self,
        x: &[f64],
        h: &mut nalgebra::DMatrix<f64>,
        scratch: &'s mut MambaScratch,
    ) -> &'s [f64] {
        debug_assert_eq!(x.len(), self.input_size);
        debug_assert_eq!(h.nrows(), self.input_size);
        debug_assert_eq!(h.ncols(), self.d_state);
        debug_assert!(
            a_is_current(&self.a, &self.a_log),
            "MambaLayer: derived A is stale (a_log written without a rebuild)"
        );

        // 1. Fused x_proj -> (Δ_pre, B, C); 2. dt_proj + softplus -> per-channel positive Δ.
        scratch.project(x, &self.x_proj_w, &self.dt_proj_w, &self.dt_proj_b);
        let (delta, b_vec, c_vec, y) = scratch.split(self.d_state);

        // 3. ZOH discretization + state update, per (d, n).
        //    A = -exp(a_log), Ā = exp(Δ·A), B̄ = Δ·B · expm1_over_x(Δ·A)
        //    h_new = Ā*h + B̄*x[d]
        //    y[d]   = Σ_n h_new[d, n] * C[n] + D[d] * x[d]
        for d in 0..self.input_size {
            let delta_d = delta[d];
            let x_d = x[d];
            let mut acc = 0.0;
            for n in 0..self.d_state {
                let za = delta_d * self.a[(d, n)];
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::neural::LayerWeights;

    fn loaded(a_log: f64) -> MambaLayer {
        let mut m = MambaLayer::zeros(3, 2, 1);
        let mut flat: Vec<f64> = (0..m.n_params())
            .map(|i| 0.07 * (i % 11) as f64 - 0.3)
            .collect();
        // a_log sits after x_proj_w (5x3), dt_proj_w (3x1), dt_proj_b (3).
        flat[21..27].fill(a_log);
        m.from_flat(&flat);
        m
    }

    fn fly(m: &MambaLayer) -> Vec<f64> {
        let mut h = nalgebra::DMatrix::zeros(3, 2);
        let mut scratch = MambaScratch::new(3, 2, 1);
        let mut y = Vec::new();
        for t in 0..5 {
            let x = [0.4, -0.2 * t as f64, 0.9];
            y = m.forward(&x, &mut h, &mut scratch).to_vec();
        }
        y
    }

    #[test]
    fn reload_rebuilds_a() {
        let fresh = loaded(0.8);
        assert_eq!(fresh.a, a_from_log(&fresh.a_log));

        // Reload over a layer that already flew a different a_log.
        let mut reloaded = loaded(-0.5);
        assert_ne!(fly(&reloaded), fly(&fresh));
        reloaded.from_flat(&fresh.to_flat());
        assert_eq!(fly(&reloaded), fly(&fresh));
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "derived A is stale")]
    fn forward_refuses_a_stale_a() {
        let mut m = loaded(0.8);
        m.a_log[(1, 1)] = -0.5; // in-module write that skips the rebuild
        fly(&m);
    }
}
