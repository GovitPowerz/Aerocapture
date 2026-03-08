//! Neural network guidance.
//!
//! Feedforward network computing bank angle from navigation state.
//! Supports arbitrary layer architectures via NeuralNetModel.
//! Default: 6 inputs → 12 hidden (tanh) → 2 outputs (asinh) → atan2 bank angle.

use crate::config::Planet;
use crate::data::neural::NeuralNetModel;
use crate::gnc::navigation::estimator::NavigationOutput;
use crate::orbit::elements;

/// Compute NN-guided longitudinal bank angle.
///
/// Input normalization matches Fortran guidnn.f:
/// - Computes orbital elements from navigation state
/// - Normalizes 6 inputs from orbital/aerodynamic quantities
/// - Forward pass through the network
/// - Bank angle = atan2(out[0], out[1])
///
/// Returns the bank angle magnitude (gitlon) in radians.
pub fn nn_bank_angle(
    nav: &NavigationOutput,
    nn: &NeuralNetModel,
    planet: &Planet,
    target_inclination: f64, // radians (xincli from /orbvis/)
) -> f64 {
    let mu = planet.mu();
    let degrad = std::f64::consts::PI / 180.0;

    // Radial velocity: V * sin(gamma)
    let vitrad = nav.vitesn[0] * nav.vitesn[1].sin();

    // Orbital elements
    let orbit = elements::from_spherical(
        nav.positn[0],
        nav.positn[1],
        nav.positn[2],
        nav.vitesn[0],
        nav.vitesn[1],
        nav.vitesn[2],
        planet,
    );

    // Acceleration magnitude: sqrt(drag^2 + lift^2)
    let accel_mag = (nav.acceln[0] * nav.acceln[0] + nav.acceln[1] * nav.acceln[1]).sqrt();

    // 6 normalized inputs (matching guidnn.f)
    let input = [
        orbit.eccentricity - 1.0,
        (orbit.inclination - target_inclination) / degrad * 3.0 / 5.0,
        2.0 * (vitrad / 1e3 + 1.2) / 1.5 - 1.0,
        -mu / (2.0 * orbit.semi_major_axis) / 6e6,
        (nav.vitesn[0] / 3e3 - 1.5) * 2.0,
        accel_mag / 20.0 - 1.0,
    ];

    let output = nn.forward(&input);

    // Bank angle from atan2
    output[0].atan2(output[1])
}
