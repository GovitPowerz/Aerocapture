//! Shared numerical helpers for the layer forward passes.
//!
//! All reductions are sequential FIFO so the Rust output is bit-identical to
//! the PyTorch mirror in `src/python/aerocapture/training/rl/layers/`.

#[inline]
pub(crate) fn gelu_exact(z: f64) -> f64 {
    // Exact GELU: 0.5 * z * (1 + erf(z / sqrt(2)))
    // Uses libm::erf for IEEE-754 correct rounding; matches torch.special.erf.
    const INV_SQRT2: f64 = 0.7071067811865475_f64;
    0.5 * z * (1.0 + libm::erf(z * INV_SQRT2))
}

pub(crate) fn layer_norm_biased(x: &[f64], gamma: &[f64], beta: &[f64], eps: f64) -> Vec<f64> {
    debug_assert_eq!(x.len(), gamma.len());
    debug_assert_eq!(x.len(), beta.len());
    let n = x.len() as f64;
    // Sequential reduction for cross-language bit-identity.
    let mut mean = 0.0;
    for v in x {
        mean += *v;
    }
    mean /= n;
    let mut var = 0.0;
    for v in x {
        let d = *v - mean;
        var += d * d;
    }
    var /= n; // biased: 1/N, NOT Bessel 1/(N-1); matches torch nn.LayerNorm default.
    let inv_std = 1.0 / (var + eps).sqrt();
    x.iter()
        .zip(gamma)
        .zip(beta)
        .map(|((xi, g), b)| ((*xi - mean) * inv_std) * g + b)
        .collect()
}

pub(crate) fn build_pe_table(n_seq: usize, d_model: usize) -> Vec<Vec<f64>> {
    // Standard Vaswani et al. 2017 sinusoidal positional encoding.
    // PE[pos, 2k]   = sin(pos / 10000^(2k / d_model))
    // PE[pos, 2k+1] = cos(pos / 10000^(2k / d_model))
    // Iteration order: pos outer, i inner. Matches Python mirror for bit-identity.
    (0..n_seq)
        .map(|pos| {
            (0..d_model)
                .map(|i| {
                    let k = i / 2;
                    let div = 10000.0_f64.powf((2.0 * k as f64) / d_model as f64);
                    let angle = pos as f64 / div;
                    if i % 2 == 0 { angle.sin() } else { angle.cos() }
                })
                .collect()
        })
        .collect()
}

/// Numerically stable softplus: `log(1 + exp(x))`.
///
/// Uses `max(x, 0) + log1p(exp(-|x|))` to avoid overflow for large positive x
/// and underflow for large negative x. The Python mirror in `rl/layers/mamba.py`
/// uses the identical manual form (NOT `torch.nn.functional.softplus`, which has a
/// `threshold=20` linear-branch fallback we do not want for bit-equivalence).
pub(crate) fn softplus(x: f64) -> f64 {
    let a = x.abs();
    x.max(0.0) + (-a).exp().ln_1p()
}

/// Stable `(exp(z) - 1) / z` with Taylor fallback for |z| < 1e-8.
///
/// For |z| < 1e-8 the exact form suffers from catastrophic cancellation and we
/// use `1 + z/2 + z^2/6` (Taylor expansion, error ~ z^3/24 which is machine
/// epsilon at |z| < 1e-5). The Python mirror uses `torch.where` to switch
/// between the same two branches.
pub(crate) fn expm1_over_x(z: f64) -> f64 {
    if z.abs() < 1e-8 {
        1.0 + z * 0.5 + z * z / 6.0
    } else {
        z.exp_m1() / z
    }
}

/// Dot product `row . vec + bias`. Helper for per-gate pre-activation sums.
#[inline]
pub(crate) fn dot_plus_bias(row: &[f64], vec: &[f64], bias: f64) -> f64 {
    bias + row.iter().zip(vec).map(|(w, v)| w * v).sum::<f64>()
}

/// Sequential matrix-vector product: m is [rows][cols] (row-major), v is [cols].
/// Deterministic FIFO reduction for cross-language bit-identity.
pub(crate) fn matvec(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    m.iter()
        .map(|row| {
            debug_assert_eq!(row.len(), v.len());
            let mut acc = 0.0_f64;
            for (a, b) in row.iter().zip(v) {
                acc += a * b;
            }
            acc
        })
        .collect()
}

/// LeCun-scaled tanh used by the CfC backbone: 1.7159 * tanh(2z/3).
/// Constant order matches the Python mirror (`1.7159 * torch.tanh(2.0 * z / 3.0)`).
pub(crate) fn lecun_tanh(z: f64) -> f64 {
    1.7159 * (2.0 * z / 3.0).tanh()
}

/// xLSTM stabilized exponential gating (Beck et al. 2024, eq. 15-17).
/// m_new = max(f_pre + m_prev, i_pre); both exp arguments are <= 0 by
/// construction, so the returned gates are finite for arbitrarily large
/// preactivations. Returns (i_gate, f_gate, m_new).
#[allow(dead_code)]
pub(crate) fn stabilized_exp_gates(i_pre: f64, f_pre: f64, m_prev: f64) -> (f64, f64, f64) {
    let m_new = (f_pre + m_prev).max(i_pre);
    ((i_pre - m_new).exp(), (f_pre + m_prev - m_new).exp(), m_new)
}

/// Copy a row-major matrix slab out of `flat`, advancing the cursor.
pub(crate) fn copy_mat_from_flat(mat: &mut [Vec<f64>], flat: &[f64], idx: &mut usize) {
    for row in mat.iter_mut() {
        let n = row.len();
        row.copy_from_slice(&flat[*idx..*idx + n]);
        *idx += n;
    }
}

/// Copy a vector slab out of `flat`, advancing the cursor.
pub(crate) fn copy_vec_from_flat(v: &mut [f64], flat: &[f64], idx: &mut usize) {
    let n = v.len();
    v.copy_from_slice(&flat[*idx..*idx + n]);
    *idx += n;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lecun_tanh_matches_definition() {
        assert!((lecun_tanh(0.0)).abs() < 1e-15);
        let z: f64 = 0.7;
        let expected = 1.7159 * (2.0_f64 * z / 3.0_f64).tanh();
        assert_eq!(lecun_tanh(z), expected);
        assert_eq!(lecun_tanh(-z), -expected);
    }

    #[test]
    fn stabilized_exp_gates_both_args_nonpositive() {
        // Stabilizer guarantees exp arguments <= 0, so gates are in (0, 1].
        for (i_pre, f_pre, m_prev) in [(0.0, 0.0, 0.0), (50.0, -50.0, 10.0), (-300.0, 300.0, -5.0)]
        {
            let (ig, fg, m_new) = stabilized_exp_gates(i_pre, f_pre, m_prev);
            assert!(ig.is_finite() && fg.is_finite() && m_new.is_finite());
            assert!(ig > 0.0 && ig <= 1.0, "ig={ig}");
            assert!(fg > 0.0 && fg <= 1.0, "fg={fg}");
            assert_eq!(m_new, (f_pre + m_prev).max(i_pre));
            // One of the two gates is exactly exp(0) = 1 (the max branch).
            assert!(ig == 1.0 || fg == 1.0);
        }
    }

    #[test]
    fn copy_helpers_advance_cursor() {
        let flat = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut mat = vec![vec![0.0; 2]; 2];
        let mut v = vec![0.0; 2];
        let mut idx = 0;
        copy_mat_from_flat(&mut mat, &flat, &mut idx);
        copy_vec_from_flat(&mut v, &flat, &mut idx);
        assert_eq!(idx, 6);
        assert_eq!(mat, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert_eq!(v, vec![5.0, 6.0]);
    }
}
