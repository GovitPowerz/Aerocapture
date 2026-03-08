//! PredGuid — Apollo/Shuttle-heritage drag tracking guidance.
//!
//! Tracks a reference drag acceleration profile as a function of velocity.
//! The reference drag profile is derived from the reference trajectory
//! (tables_energie_gains), which stores dynamic pressure vs energy. Since
//! drag = q * S * Cx / m, and energy is monotonically related to velocity
//! during atmospheric flight, we can map between them.
//!
//! The control law modulates bank angle to match the reference drag level:
//!
//!   D_cmd = D_ref + K * (D - D_ref)   [drag error feedback]
//!   cos(bank) = D_cmd / L              [bank angle from drag command]
//!
//! In practice, we use dynamic pressure as a proxy for drag (proportional
//! for constant ballistic coefficient) and compute:
//!
//!   cos(bank) = cos_ref + K_d * (D - D_ref) / (L/m)
//!
//! This is conceptually similar to FTC but uses a simpler feedback structure
//! without the altitude-rate damping term.

use crate::config::Planet;
use crate::data::SimData;
use crate::gnc::navigation::coordinates::total_energy;
use crate::gnc::navigation::estimator::NavigationOutput;

/// PredGuid persistent state (runtime-only, no tunable params).
#[derive(Debug, Clone)]
pub struct PredGuidState {
    _placeholder: (),
}

impl PredGuidState {
    pub fn new() -> Self {
        Self { _placeholder: () }
    }
}

/// Compute PredGuid bank angle command.
///
/// Tracks reference drag acceleration profile using bank angle modulation.
/// Returns bank angle magnitude in radians.
pub fn predguid_bank(
    nav: &NavigationOutput,
    _state: &PredGuidState,
    data: &SimData,
    planet: &Planet,
) -> f64 {
    let ref_traj = &data.guidance.ref_trajectory;
    if ref_traj.n_points == 0 {
        return 60.0_f64.to_radians();
    }

    // Current energy for reference lookup
    let energy = total_energy(
        nav.positn[0],
        nav.positn[1],
        nav.positn[2],
        nav.vitesn[0],
        nav.vitesn[1],
        nav.vitesn[2],
        planet,
    );

    // Reference values
    let cos_bank_ref = ref_traj.interpolate(energy, &ref_traj.cos_bank);
    let pdyn_ref = ref_traj.interpolate(energy, &ref_traj.pressure);

    // Current drag and lift accelerations from navigation
    let drag_accel = nav.acceln[0]; // D/m
    let lift_accel = nav.acceln[1]; // L/m

    // Current dynamic pressure
    let v = nav.vitesn[0];
    let pdyn = 0.5 * nav.roguid * v * v;

    // Reference drag acceleration: D_ref/m = pdyn_ref * S * Cx / m
    let cx = nav.coefan[0];
    let sref = data.capsule.reference_area;
    let mass = data.capsule.mass;
    let drag_ref = pdyn_ref * sref * cx / mass;

    // Drag error
    let drag_err = drag_accel - drag_ref;

    // Lift acceleration (absolute value for bank angle computation)
    let lift_abs = lift_accel.abs().max(1e-10);

    // PredGuid control law:
    // The reference cos(bank) already accounts for the nominal drag.
    // We add a correction proportional to the drag error, normalized by lift.
    //
    // If drag > ref: we're too deep → increase bank (more lift-down) → increase cos_bank
    // If drag < ref: we're too high → decrease bank (more lift-up) → decrease cos_bank
    //
    // The sign convention: cos(bank) = 1 means full lift-up (min drag exposure),
    // cos(bank) = -1 means full lift-down (max drag exposure).
    // So drag error should DECREASE cos_bank (bank toward lift-up to reduce drag).
    let params = &data.guidance.pred_guid;
    let k_drag = if pdyn > params.pdyn_threshold {
        params.k_drag_high
    } else {
        params.k_drag_low
    };
    let cos_bank = cos_bank_ref - k_drag * drag_err / lift_abs;

    let cos_bank = cos_bank.clamp(-1.0, 1.0);
    cos_bank.acos()
}
