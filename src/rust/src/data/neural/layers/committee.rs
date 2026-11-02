//! Committee layer (#232): N member layer stacks flown in parallel on the same
//! input, each with its own recurrent state, their 2-output `(sin, cos)` heads
//! combined into one head for the atan2 bank decoder. Deploy-only: no torch
//! mirror, no chromosome layout; the Python builder assembles one from deployed
//! `best_model.json` files.

use super::tensor::Tensor;
use crate::data::neural::{Layer, LayerWeights, forward_layer};
use crate::data::nn_state::LayerState;

/// How the member heads are combined before the one atan2 decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitteeRule {
    /// Sum of the unit-normalised heads: one vote per member (a circular mean
    /// once decoded). A zero head casts no vote.
    CircularMean,
    /// Sum of the raw heads: a vote weighted by head magnitude.
    RawSum,
    /// The raw head of the member whose angle is closest to the circular mean
    /// (first on ties); bit-identical to that member alone on that tick.
    Medoid,
    /// Post-hoc rule (#232 pilot): the members agree on the bank magnitude and
    /// not on its sign (independent lateral strategies), so the magnitude is the
    /// circular mean of the unit heads reflected into the right half-plane
    /// (|sin|, cos) and the sign is member 0's (the deployed seed leads).
    SignLeader,
}

impl CommitteeRule {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "circular_mean" => Ok(Self::CircularMean),
            "raw_sum" => Ok(Self::RawSum),
            "medoid" => Ok(Self::Medoid),
            "sign_leader" => Ok(Self::SignLeader),
            other => Err(format!(
                "rule must be circular_mean|raw_sum|medoid|sign_leader, got {other:?}"
            )),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CommitteeLayer {
    pub rule: CommitteeRule,
    /// Shared fan-in of every member (= the committee's input size).
    pub input_size: usize,
    /// Member stacks, each ending in a 2-output head.
    pub members: Vec<Vec<Layer>>,
}

impl CommitteeLayer {
    /// One tick: every member advances its own state on `x`; the heads combine.
    pub fn forward(&self, x: &[f64], states: &mut [Vec<LayerState>]) -> Vec<f64> {
        debug_assert_eq!(states.len(), self.members.len());
        let mut heads = Vec::with_capacity(self.members.len());
        for (layers, member_states) in self.members.iter().zip(states.iter_mut()) {
            let mut current = x.to_vec();
            for (layer, state) in layers.iter().zip(member_states.iter_mut()) {
                current = forward_layer(layer, state, current);
            }
            heads.push(current);
        }
        combine(self.rule, &heads)
    }
}

/// Combine 2-vector heads under `rule`. Sequential FIFO sums (cross-language
/// bit-identity convention).
pub fn combine(rule: CommitteeRule, heads: &[Vec<f64>]) -> Vec<f64> {
    match rule {
        CommitteeRule::RawSum => {
            let mut s = 0.0;
            let mut c = 0.0;
            for h in heads {
                s += h[0];
                c += h[1];
            }
            vec![s, c]
        }
        CommitteeRule::CircularMean => unit_sum(heads),
        CommitteeRule::Medoid => {
            let mean = unit_sum(heads);
            let theta = mean[0].atan2(mean[1]);
            let mut best = 0;
            let mut best_d = f64::INFINITY;
            for (k, h) in heads.iter().enumerate() {
                let d = angular_distance(h[0].atan2(h[1]), theta);
                if d < best_d {
                    best_d = d;
                    best = k;
                }
            }
            heads[best].clone()
        }
        CommitteeRule::SignLeader => {
            let mut s = 0.0;
            let mut c = 0.0;
            for h in heads {
                let n = h[0].hypot(h[1]);
                if n > 0.0 {
                    s += (h[0] / n).abs();
                    c += h[1] / n;
                }
            }
            let sign = if heads[0][0] < 0.0 { -1.0 } else { 1.0 };
            vec![sign * s, c]
        }
    }
}

fn unit_sum(heads: &[Vec<f64>]) -> Vec<f64> {
    let mut s = 0.0;
    let mut c = 0.0;
    for h in heads {
        let n = h[0].hypot(h[1]);
        if n > 0.0 {
            s += h[0] / n;
            c += h[1] / n;
        }
    }
    vec![s, c]
}

fn angular_distance(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(std::f64::consts::TAU);
    d.min(std::f64::consts::TAU - d)
}

impl LayerWeights for CommitteeLayer {
    fn tensors(&self) -> Vec<(&'static str, &dyn Tensor)> {
        let mut out = Vec::new();
        for member in &self.members {
            for layer in member {
                out.extend(layer.tensors());
            }
        }
        out
    }

    fn tensors_mut(&mut self) -> Vec<(&'static str, &mut dyn Tensor)> {
        let mut out = Vec::new();
        for member in &mut self.members {
            for layer in member {
                out.extend(layer.tensors_mut());
            }
        }
        out
    }

    // Delegated so every member layer's `post_load` hook runs (the default
    // walks the flattened table and would skip them).
    fn from_flat(&mut self, flat: &[f64]) -> usize {
        let mut idx = 0;
        for member in &mut self.members {
            for layer in member {
                idx += layer.from_flat(&flat[idx..]);
            }
        }
        idx
    }
}
