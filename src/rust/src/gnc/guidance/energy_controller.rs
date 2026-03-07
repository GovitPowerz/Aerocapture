//! Energy Controller guidance.
//!
//! Tracks a reference energy dissipation profile using bank angle feedback.
//! The idea is simple: the reference trajectory defines what the orbital
//! energy should be at each point in time. If the vehicle has too much
//! energy, increase drag (bank toward lift-down); if too little, decrease
//! drag (bank toward lift-up).
//!
//! The bank angle command is:
//!   cos(bank) = cos(bank_ref) + K_e * (E - E_ref) / q_dyn
//!
//! where E_ref is interpolated from the reference trajectory at the current
//! energy level, and K_e is a tunable gain.

use crate::config::Planet;
use crate::data::SimData;
use crate::gnc::navigation::coordinates::total_energy;
use crate::gnc::navigation::estimator::NavigationOutput;

/// Energy controller persistent state.
#[derive(Debug, Clone)]
pub struct EnergyControllerState {
    /// Gain on energy error (1/Pa). Typical: 1e-7 to 1e-5.
    pub gain: f64,
}

impl EnergyControllerState {
    pub fn new() -> Self {
        Self { gain: 5e-7 }
    }
}

/// Compute energy-tracking bank angle.
///
/// Uses the same reference trajectory tables as FTC (tables_energie_gains).
/// Returns the bank angle magnitude in radians.
pub fn energy_controller_bank(
    nav: &NavigationOutput,
    _state: &EnergyControllerState,
    data: &SimData,
    planet: &Planet,
) -> f64 {
    let ref_traj = &data.guidance.ref_trajectory;
    if ref_traj.n_points == 0 {
        // No reference trajectory loaded — fall back to 60° bank
        return 60.0_f64.to_radians();
    }

    // Current energy
    let energy = total_energy(
        nav.positn[0],
        nav.positn[1],
        nav.positn[2],
        nav.vitesn[0],
        nav.vitesn[1],
        nav.vitesn[2],
        planet,
    );

    // Reference values at current energy
    let cos_bank_ref = ref_traj.interpolate(energy, &ref_traj.cos_bank);
    let pdyn_ref = ref_traj.interpolate(energy, &ref_traj.pressure);
    let hdot_ref = ref_traj.interpolate(energy, &ref_traj.radial_vel);

    // Current dynamic pressure and radial velocity
    let v = nav.vitesn[0];
    let pdyn = 0.5 * nav.roguid * v * v;
    let hdot = v * nav.vitesn[1].sin();

    // Energy-based correction: if we have too much pressure (too deep),
    // reduce bank to increase lift-up. If too little (too high), increase bank.
    // Also correct for radial velocity error (helps dampen oscillations).
    let pdyn_safe = pdyn.max(1e-3);

    // Gains: pressure error gain and radial velocity damping gain
    let kp = 1.0; // pressure proportional gain (dimensionless)
    let kd = 0.5; // radial velocity damping gain (dimensionless, acts like derivative)

    let cos_bank = cos_bank_ref
        + kp * (pdyn - pdyn_ref) / pdyn_safe
        + kd * (hdot - hdot_ref) / pdyn_safe;

    let cos_bank = cos_bank.clamp(-1.0, 1.0);
    cos_bank.acos()
}
