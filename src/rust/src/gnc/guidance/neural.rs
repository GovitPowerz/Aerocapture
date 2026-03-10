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
/// - Computes orbital elements from navigation state
/// - Normalizes 6 inputs from orbital/aerodynamic quantities
/// - Forward pass through the network
/// - Bank angle = atan2(out[0], out[1])
///
/// Returns the bank angle magnitude in radians.
pub fn nn_bank_angle(
    nav: &NavigationOutput,
    nn: &NeuralNetModel,
    planet: &Planet,
    target_inclination: f64, // radians
) -> f64 {
    let mu = planet.mu();

    // Radial velocity: V * sin(gamma)
    let velocity_radial = nav.velocity_estimated[0] * nav.velocity_estimated[1].sin();

    // Orbital elements
    let orbit = elements::from_spherical(
        nav.position_estimated[0],
        nav.position_estimated[1],
        nav.position_estimated[2],
        nav.velocity_estimated[0],
        nav.velocity_estimated[1],
        nav.velocity_estimated[2],
        planet,
    );

    // Acceleration magnitude: sqrt(drag^2 + lift^2)
    let accel_mag = (nav.acceleration_estimated[0] * nav.acceleration_estimated[0]
        + nav.acceleration_estimated[1] * nav.acceleration_estimated[1])
        .sqrt();

    // 6 normalized inputs
    let input = [
        orbit.eccentricity - 1.0,
        (orbit.inclination - target_inclination).to_degrees() * 3.0 / 5.0,
        2.0 * (velocity_radial / 1e3 + 1.2) / 1.5 - 1.0,
        -mu / (2.0 * orbit.semi_major_axis) / 6e6,
        (nav.velocity_estimated[0] / 3e3 - 1.5) * 2.0,
        accel_mag / 20.0 - 1.0,
    ];

    let output = nn.forward(&input);

    // Bank angle from atan2
    output[0].atan2(output[1])
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    use crate::data::neural::{Activation, Layer, NeuralNetModel};
    use crate::gnc::navigation::estimator::NavigationOutput;

    fn test_nav() -> NavigationOutput {
        let r = 3_396_200.0 + 50_000.0; // Mars radius + 50 km
        let velocity = 5000.0;
        NavigationOutput {
            position_estimated: [r, 0.1, 0.05],
            velocity_estimated: [velocity, -0.10, 0.5],
            acceleration_estimated: [80.0, -12.0],
            aero_coefficients: [1.269, -0.205],
            density_guidance: 0.001,
            density_exit: 1e-6,
            dynamic_pressure_estimated: 0.5 * 0.001 * velocity * velocity,
            energy_estimated: -1e6,
            ..Default::default()
        }
    }

    /// Build a minimal 6→2 network (one layer, all-zero weights).
    ///
    /// Forward pass: output[j] = activation(sum(0 * input) + bias[j]) = activation(bias[j])
    /// With Linear activation: output = bias directly.
    /// Bank angle = atan2(b[0], b[1])
    fn zero_weight_nn(bias0: f64, bias1: f64) -> NeuralNetModel {
        NeuralNetModel {
            layer_sizes: vec![6, 2],
            layers: vec![Layer {
                w: vec![vec![0.0; 6], vec![0.0; 6]],
                b: vec![bias0, bias1],
                activation: Activation::Linear,
            }],
            output_interpretation: "atan2".to_string(),
        }
    }

    #[test]
    fn zero_weights_known_output() {
        let bias0 = 1.0_f64;
        let bias1 = 1.0_f64;
        let nn = zero_weight_nn(bias0, bias1);
        let nav = test_nav();
        let planet = Planet::Mars;

        // With zero weights + linear activation: output = [bias0, bias1]
        // Bank angle = atan2(1.0, 1.0) = PI/4
        let bank = nn_bank_angle(&nav, &nn, &planet, 50.0_f64.to_radians());
        let expected = bias0.atan2(bias1); // PI/4
        assert_relative_eq!(bank, expected, epsilon = 1e-12);
    }

    #[test]
    fn zero_weights_negative_bias() {
        // Verify atan2 sign handling: atan2(-1, 1) = -PI/4
        let nn = zero_weight_nn(-1.0, 1.0);
        let nav = test_nav();
        let planet = Planet::Mars;

        let bank = nn_bank_angle(&nav, &nn, &planet, 50.0_f64.to_radians());
        assert_relative_eq!(bank, (-1.0_f64).atan2(1.0), epsilon = 1e-12);
    }

    #[test]
    fn output_in_valid_range() {
        // Small 6→3→2 network with tanh hidden layer and asinh output
        let layer0 = Layer {
            w: vec![
                vec![0.1, -0.2, 0.3, -0.1, 0.2, -0.3],
                vec![-0.2, 0.1, -0.1, 0.3, -0.2, 0.1],
                vec![0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
            ],
            b: vec![0.1, -0.1, 0.0],
            activation: Activation::Tanh,
        };
        let layer1 = Layer {
            w: vec![vec![0.5, -0.5, 0.2], vec![-0.3, 0.3, -0.1]],
            b: vec![0.0, 0.0],
            activation: Activation::Asinh,
        };
        let nn = NeuralNetModel {
            layer_sizes: vec![6, 3, 2],
            layers: vec![layer0, layer1],
            output_interpretation: "atan2".to_string(),
        };

        let nav = test_nav();
        let planet = Planet::Mars;
        let bank = nn_bank_angle(&nav, &nn, &planet, 50.0_f64.to_radians());

        assert!(bank.is_finite(), "bank angle must be finite, got: {}", bank);
        // atan2 always produces values in (-PI, PI]
        assert!(
            bank > -std::f64::consts::PI - 1e-10 && bank <= std::f64::consts::PI + 1e-10,
            "bank angle out of atan2 range: {}",
            bank
        );
    }

    mod prop {
        use super::*;
        use proptest::prelude::*;

        fn fixed_small_nn() -> NeuralNetModel {
            NeuralNetModel {
                layer_sizes: vec![6, 2],
                layers: vec![Layer {
                    w: vec![
                        vec![0.1, -0.1, 0.2, -0.2, 0.05, -0.05],
                        vec![-0.1, 0.1, -0.05, 0.05, 0.15, -0.15],
                    ],
                    b: vec![0.3, -0.2],
                    activation: Activation::Tanh,
                }],
                output_interpretation: "atan2".to_string(),
            }
        }

        proptest! {
            #[test]
            fn output_always_finite(
                alt in 10_000.0..130_000.0_f64,
                vel in 2000.0..7000.0_f64,
                fpa in -0.3..0.05_f64,
                az  in -1.0..1.0_f64,
            ) {
                let r = Planet::Mars.equatorial_radius() + alt;
                let nav = NavigationOutput {
                    position_estimated: [r, 0.1, 0.05],
                    velocity_estimated: [vel, fpa, az],
                    acceleration_estimated: [50.0, -8.0],
                    aero_coefficients: [1.269, -0.205],
                    density_guidance: 0.001,
                    dynamic_pressure_estimated: 0.5 * 0.001 * vel * vel,
                    energy_estimated: -1e6,
                    ..Default::default()
                };

                let nn = fixed_small_nn();
                let planet = Planet::Mars;
                let bank = nn_bank_angle(&nav, &nn, &planet, 50.0_f64.to_radians());

                prop_assert!(bank.is_finite(), "bank not finite: {}", bank);
            }
        }
    }
}
