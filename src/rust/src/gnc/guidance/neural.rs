//! Neural network guidance.
//!
//! Matches Fortran guidnn.f — 2-layer feedforward network computing bank angle.
//! Architecture: 6 inputs → 12 hidden (tanh) → 2 outputs (asinh) → atan2 bank angle.

use crate::config::Planet;
use crate::data::neural::NeuralNetParams;
use crate::gnc::navigation::estimator::NavigationOutput;
use crate::orbit::elements;

/// Compute NN-guided longitudinal bank angle.
///
/// Matches Fortran guidnn.f exactly:
/// - Computes orbital elements from navigation state
/// - Normalizes 6 inputs from orbital/aerodynamic quantities
/// - Forward pass: hidden layer (tanh), output layer (asinh)
/// - Bank angle = atan2(out[0], out[1])
///
/// Returns the bank angle magnitude (gitlon) in radians.
pub fn nn_bank_angle(
    nav: &NavigationOutput,
    nn: &NeuralNetParams,
    planet: &Planet,
    target_inclination: f64, // radians (xincli from /orbvis/)
) -> f64 {
    let mu = planet.mu();
    let degrad = std::f64::consts::PI / 180.0;

    // Radial velocity: V * sin(gamma)
    let vitrad = nav.vitesn[0] * nav.vitesn[1].sin();

    // Orbital elements (matches Fortran: call orbito(positn, vitesn, xorbit))
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

    // 6 normalized inputs (matching guidnn.f lines 86-91)
    let input = [
        orbit.eccentricity - 1.0, // xorbit(2) - 1
        (orbit.inclination - target_inclination) / degrad * 3.0 / 5.0, // (incl - target) scaled
        2.0 * (vitrad / 1e3 + 1.2) / 1.5 - 1.0, // radial vel scaled
        -mu / (2.0 * orbit.semi_major_axis) / 6e6, // orbital energy scaled
        (nav.vitesn[0] / 3e3 - 1.5) * 2.0, // velocity scaled
        accel_mag / 20.0 - 1.0,   // acceleration scaled
    ];

    // Hidden layer: tanh activation
    let mut hidden = [0.0_f64; crate::data::neural::N_HIDDEN];
    for (j, h) in hidden.iter_mut().enumerate() {
        let mut sum = 0.0;
        for (i, inp) in input.iter().enumerate() {
            sum += nn.lw1[j][i] * inp;
        }
        *h = (sum + nn.bias1[j]).tanh();
    }

    // Output layer: asinh activation
    let mut output = [0.0_f64; crate::data::neural::N_OUTPUT];
    for (j, out) in output.iter_mut().enumerate() {
        let mut sum = 0.0;
        for (i, h) in hidden.iter().enumerate() {
            sum += nn.lw4[j][i] * h;
        }
        *out = (sum + nn.bias4[j]).asinh();
    }

    // Bank angle from atan2 (matches guidnn.f line 113)
    output[0].atan2(output[1])
}
