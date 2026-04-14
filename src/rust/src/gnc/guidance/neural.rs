//! Neural network guidance.
//!
//! Feedforward network computing bank angle from navigation state.
//! Supports arbitrary layer architectures via NeuralNetModel.
//!
//! 23 candidate inputs (selected by configurable `input_mask`):
//!   0  eccentricity_excess    8  altitude              16 cos_bank_nominal
//!   1  inclination_error      9  fpa                   17 pdyn_nominal
//!   2  radial_velocity       10  latitude               18 hdot_nominal
//!   3  orbital_energy        11  drag_accel             19 pdyn_error
//!   4  velocity              12  lift_accel             20 exit_bank_angle
//!   5  accel_magnitude       13  sma_error              21 density_exit
//!   6  heat_flux_fraction    14  apoapsis_alt           22 ref_velocity_latched
//!   7  heat_load_fraction    15  bounce_flag
//!
//! Output modes: "atan2" (2 outputs, signed bank via atan2) or "direct" (1 output).

use crate::config::PlanetConfig;
use crate::data::SimData;
use crate::data::neural::{NN_FULL_INPUT_SIZE, NeuralNetModel};
use crate::gnc::guidance::exit;
use crate::gnc::navigation::coordinates::total_energy;
use crate::gnc::navigation::estimator::NavigationOutput;
use crate::orbit::elements;

/// Compute NN-guided longitudinal bank angle.
///
/// Builds a 23-element candidate input vector, applies the model's input_mask
/// (or legacy [0..16] default), runs a forward pass, and interprets the output
/// as either atan2(out[0], out[1]) or direct out[0] depending on the model config.
///
/// Returns the **signed** bank angle in radians.
/// Lateral guidance is bypassed for this scheme -- the NN controls roll direction directly.
pub fn nn_bank_angle(
    nav: &NavigationOutput,
    nn: &NeuralNetModel,
    data: &SimData,
    planet: &PlanetConfig,
    target_inclination: f64, // radians
    ref_velocity_latched: f64,
) -> f64 {
    let mu = planet.mu;

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

    // Altitude in km
    let altitude_km = (nav.position_estimated[0] - planet.equatorial_radius) / 1e3;

    // Build full 23-element input vector
    let mut full_input = [0.0_f64; NN_FULL_INPUT_SIZE];

    // -- 16 existing inputs (indices 0-15) --
    full_input[0] = orbit.eccentricity - 1.0; // eccentricity excess
    full_input[1] = (orbit.inclination - target_inclination).to_degrees() * 3.0 / 5.0; // inclination error
    full_input[2] = 2.0 * (velocity_radial / 1e3 + 1.2) / 1.5 - 1.0; // radial velocity
    full_input[3] = -mu / (2.0 * orbit.semi_major_axis) / 6e6; // orbital energy
    full_input[4] = (nav.velocity_estimated[0] / 3e3 - 1.5) * 2.0; // velocity
    full_input[5] = accel_mag / 20.0 - 1.0; // accel magnitude
    full_input[6] = nav.heat_flux_fraction * 2.0 - 1.0; // heat flux fraction
    full_input[7] = nav.heat_load_fraction * 2.0 - 1.0; // heat load fraction
    full_input[8] = (altitude_km - 65.0) / 65.0; // altitude
    full_input[9] = nav.velocity_estimated[1] / 0.3; // flight path angle
    full_input[10] = nav.position_estimated[2] / std::f64::consts::FRAC_PI_2; // latitude
    full_input[11] = nav.acceleration_estimated[0] / 50.0 - 1.0; // drag acceleration
    full_input[12] = nav.acceleration_estimated[1] / 10.0; // lift acceleration
    full_input[13] = nav.orbital_errors[0] / 5e5; // SMA error
    full_input[14] = orbit.apoapsis_alt.clamp(-10e6, 10e6) / 1e6 - 1.0; // apoapsis altitude
    full_input[15] = nav.bounce_flag as f64 * 2.0 - 1.0; // bounce flag

    // -- 4 reference trajectory inputs (indices 16-19) --
    let energy = total_energy(
        nav.position_estimated[0],
        nav.position_estimated[1],
        nav.position_estimated[2],
        nav.velocity_estimated[0],
        nav.velocity_estimated[1],
        nav.velocity_estimated[2],
        planet,
    );
    let ref_traj = &data.guidance.ref_trajectory;
    let cos_bank_nominal = ref_traj.interpolate(energy, &ref_traj.cos_bank);
    let pdyn_nominal = ref_traj.interpolate(energy, &ref_traj.pressure);
    let hdot_nominal = ref_traj.interpolate(energy, &ref_traj.radial_vel);
    let pdyn_current =
        0.5 * nav.density_guidance * nav.velocity_estimated[0] * nav.velocity_estimated[0];
    let pdyn_error = pdyn_current - pdyn_nominal;

    full_input[16] = cos_bank_nominal; // ref cos(bank)
    full_input[17] = pdyn_nominal / 2e3 - 1.0; // ref dynamic pressure
    full_input[18] = hdot_nominal / 500.0; // ref radial velocity
    full_input[19] = pdyn_error / 2e3; // dynamic pressure error

    // -- 3 exit-related inputs (indices 20-22), gated by bounce flag --
    let bf = nav.bounce_flag as f64;
    let exit_bank = exit::exit_guidance(nav, data, planet, ref_velocity_latched);

    full_input[20] = (exit_bank / std::f64::consts::PI * 2.0 - 1.0) * bf; // exit bank angle
    full_input[21] = ((nav.density_exit.max(1e-12).log10() + 7.0) / 5.0) * bf; // exit density
    full_input[22] = (ref_velocity_latched / 500.0) * bf; // ref velocity

    // Apply ablation: zero out a single input for sensitivity analysis
    if let Some(idx) = nn.ablated_input {
        full_input[idx] = 0.0;
    }

    // Apply input mask: select subset of inputs, or default to first 16 for backward compat
    let masked: Vec<f64> = match &nn.input_mask {
        Some(mask) => mask.iter().map(|&i| full_input[i]).collect(),
        None => full_input[..16].to_vec(),
    };

    let output = nn.forward(&masked);

    // Interpret output based on model configuration
    match nn.output_interpretation.as_str() {
        "direct" => output[0],
        _ => output[0].atan2(output[1]), // "atan2" (legacy default)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    use crate::data::aerodynamics::AeroTables;
    use crate::data::atmosphere::{AtmosphereModel, DensityProfile};
    use crate::data::capsule::Capsule;
    use crate::data::guidance_params::{GuidanceParams, ReferenceTrajectory};
    use crate::data::incidence::IncidenceProfile;
    use crate::data::neural::{Activation, Layer, NN_FULL_INPUT_SIZE, NeuralNetModel};
    use crate::data::pilot::{PilotModel, PilotType};
    use crate::data::{
        Constraints, EntryConditions, FinalConditions, OrbitalTarget, ParkingOrbit, SimData,
        SphericalState, SuccessCriteria, TimePeriods,
    };
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
            orbital_errors: [1000.0, 0.01, 0.001, 0.002],
            ..Default::default()
        }
    }

    fn test_sim_data() -> SimData {
        SimData {
            capsule: Capsule {
                mass: 1089.0,
                reference_area: 14.7,
                cq: 0.00008242,
                max_bank_rate: 15.0_f64.to_radians(),
                periods: TimePeriods::default(),
            },
            aero: AeroTables {
                n_points: 2,
                incidence: vec![-0.5, 0.0],
                cx: vec![1.269, 1.269],
                cz: vec![-0.205, -0.205],
                equilibrium_aoa: -0.48,
                ..Default::default()
            },
            atmosphere: AtmosphereModel {
                n_points: 3,
                altitudes: vec![0.0, 50_000.0, 130_000.0],
                densities: vec![0.02, 0.001, 1e-8],
                ref_density: 1e-8,
                scale_factor: 1e-4,
                ref_altitude: 130_000.0,
                gas_constant: 1.3,
                density_profile: DensityProfile::default(),
            },
            atmosphere_onboard: crate::data::atmosphere::OnboardAtmosphereModel::Identical,
            entry: EntryConditions {
                state: SphericalState {
                    altitude: 130_000.0,
                    velocity: 5687.0,
                    flight_path: -10.8_f64.to_radians(),
                    ..Default::default()
                },
                initial_bank: 64.77_f64.to_radians(),
                initial_aoa: -27.5_f64.to_radians(),
                initial_date: 0.0,
            },
            guidance: GuidanceParams {
                density_filter_gain: 0.8,
                exit_velocity_threshold: 4400.0,
                exit_pdyn_margin: 1.75,
                exit_altitude_threshold: 60_000.0,
                exit_radial_vel_gain: 10.0,
                exit_apoapsis_threshold: 100.0,
                ..Default::default()
            },
            incidence: IncidenceProfile {
                n_points: 2,
                altitudes: vec![-10_000.0, 150_000.0],
                incidences: vec![-0.48, -0.48],
            },
            periods: TimePeriods::default(),
            pilot: PilotModel {
                pilot_type: PilotType::Perfect,
                time_constant: 0.0,
                damping: 0.0,
                frequency: 0.0,
            },
            target_orbit: OrbitalTarget {
                semi_major_axis: 3_649_622.0,
                eccentricity: 0.067,
                inclination: 50.0_f64.to_radians(),
                raan: -7.612_f64.to_radians(),
                apoapsis: 500_130.0,
                periapsis: 11_233.0,
            },
            final_conditions: FinalConditions::default(),
            parking_orbit: ParkingOrbit::default(),
            constraints: Constraints::default(),
            success: SuccessCriteria::default(),
            wind_enabled: false,
            wind_table: None,
            neural_net: None,
            dispersion_config: None,
            nav_mode: crate::data::NavMode::Bias,
            nav_config: None,
            integration_mode: crate::config::IntegrationMode::FixedGill,
            sim_phase: crate::config::SimPhase::Full,
            density_perturbation: None,
        }
    }

    fn test_sim_data_with_ref_traj() -> SimData {
        let mut data = test_sim_data();
        data.guidance.ref_trajectory = ReferenceTrajectory {
            n_points: 3,
            energy: vec![-8.0e6, -5.0e6, -2.0e6],
            pressure: vec![500.0, 2000.0, 500.0],
            radial_vel: vec![-200.0, 0.0, 100.0],
            altitude_rate: vec![-200.0, 0.0, 100.0],
            inclination: vec![0.87, 0.87, 0.87],
            time: vec![0.0, 300.0, 600.0],
            cos_bank: vec![0.4, 0.6, 0.8],
        };
        data
    }

    /// Build a minimal 16->2 network (one layer, all-zero weights).
    ///
    /// Forward pass: output[j] = activation(sum(0 * input) + bias[j]) = activation(bias[j])
    /// With Linear activation: output = bias directly.
    /// Bank angle = atan2(b[0], b[1])
    fn zero_weight_nn(bias0: f64, bias1: f64) -> NeuralNetModel {
        NeuralNetModel {
            layer_sizes: vec![16, 2],
            layers: vec![Layer {
                w: vec![vec![0.0; 16], vec![0.0; 16]],
                b: vec![bias0, bias1],
                activation: Activation::Linear,
            }],
            output_interpretation: "atan2".to_string(),
            input_mask: None,
            ablated_input: None,
        }
    }

    #[test]
    fn zero_weights_known_output() {
        let bias0 = 1.0_f64;
        let bias1 = 1.0_f64;
        let nn = zero_weight_nn(bias0, bias1);
        let nav = test_nav();
        let data = test_sim_data();
        let planet = PlanetConfig::mars();

        // With zero weights + linear activation: output = [bias0, bias1]
        // Bank angle = atan2(1.0, 1.0) = PI/4
        let bank = nn_bank_angle(&nav, &nn, &data, &planet, 50.0_f64.to_radians(), 0.0);
        let expected = bias0.atan2(bias1); // PI/4
        assert_relative_eq!(bank, expected, epsilon = 1e-12);
    }

    #[test]
    fn zero_weights_negative_bias() {
        // Verify atan2 sign handling: atan2(-1, 1) = -PI/4
        let nn = zero_weight_nn(-1.0, 1.0);
        let nav = test_nav();
        let data = test_sim_data();
        let planet = PlanetConfig::mars();

        let bank = nn_bank_angle(&nav, &nn, &data, &planet, 50.0_f64.to_radians(), 0.0);
        assert_relative_eq!(bank, (-1.0_f64).atan2(1.0), epsilon = 1e-12);
    }

    #[test]
    fn direct_output_returns_raw_value() {
        // Single-output network with "direct" interpretation: output = bias directly
        let nn = NeuralNetModel {
            layer_sizes: vec![16, 1],
            layers: vec![Layer {
                w: vec![vec![0.0; 16]],
                b: vec![1.5],
                activation: Activation::Linear,
            }],
            output_interpretation: "direct".to_string(),
            input_mask: None,
            ablated_input: None,
        };
        let nav = test_nav();
        let data = test_sim_data();
        let planet = PlanetConfig::mars();

        let bank = nn_bank_angle(&nav, &nn, &data, &planet, 50.0_f64.to_radians(), 0.0);
        assert_relative_eq!(bank, 1.5, epsilon = 1e-12);
    }

    #[test]
    fn direct_output_unbounded() {
        // Verify direct output can exceed [-PI, PI]
        let nn = NeuralNetModel {
            layer_sizes: vec![16, 1],
            layers: vec![Layer {
                w: vec![vec![0.0; 16]],
                b: vec![10.0], // > PI
                activation: Activation::Linear,
            }],
            output_interpretation: "direct".to_string(),
            input_mask: None,
            ablated_input: None,
        };
        let nav = test_nav();
        let data = test_sim_data();
        let planet = PlanetConfig::mars();

        let bank = nn_bank_angle(&nav, &nn, &data, &planet, 50.0_f64.to_radians(), 0.0);
        assert_relative_eq!(bank, 10.0, epsilon = 1e-12);
    }

    #[test]
    fn output_in_valid_range() {
        // Small 16->3->2 network with tanh hidden layer and asinh output
        let layer0 = Layer {
            w: vec![
                vec![
                    0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.05, -0.05, 0.02, -0.03, 0.04, -0.01, 0.03,
                    -0.02, 0.01, -0.04,
                ],
                vec![
                    -0.2, 0.1, -0.1, 0.3, -0.2, 0.1, 0.05, -0.05, -0.01, 0.02, -0.03, 0.04, -0.02,
                    0.01, -0.04, 0.03,
                ],
                vec![
                    0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02,
                    0.02, 0.02, 0.02,
                ],
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
            layer_sizes: vec![16, 3, 2],
            layers: vec![layer0, layer1],
            output_interpretation: "atan2".to_string(),
            input_mask: None,
            ablated_input: None,
        };

        let nav = test_nav();
        let data = test_sim_data();
        let planet = PlanetConfig::mars();
        let bank = nn_bank_angle(&nav, &nn, &data, &planet, 50.0_f64.to_radians(), 0.0);

        assert!(bank.is_finite(), "bank angle must be finite, got: {}", bank);
        // atan2 always produces values in (-PI, PI]
        assert!(
            bank > -std::f64::consts::PI - 1e-10 && bank <= std::f64::consts::PI + 1e-10,
            "bank angle out of atan2 range: {}",
            bank
        );
    }

    #[test]
    fn sixteen_input_network_produces_valid_output() {
        // 16->24->2 network with tanh hidden, asinh output
        let mut layer0_weights = Vec::new();
        for row in 0..24 {
            let w: Vec<f64> = (0..16)
                .map(|col| {
                    let sign = if (row + col) % 2 == 0 { 1.0 } else { -1.0 };
                    sign * 0.05 * ((row * 16 + col) as f64 % 7.0 + 1.0) / 7.0
                })
                .collect();
            layer0_weights.push(w);
        }
        let layer0 = Layer {
            w: layer0_weights,
            b: vec![0.0; 24],
            activation: Activation::Tanh,
        };
        let layer1 = Layer {
            w: vec![
                (0..24)
                    .map(|i| 0.1 * if i % 2 == 0 { 1.0 } else { -1.0 })
                    .collect(),
                (0..24)
                    .map(|i| 0.1 * if i % 3 == 0 { 1.0 } else { -1.0 })
                    .collect(),
            ],
            b: vec![0.0, 0.0],
            activation: Activation::Asinh,
        };
        let nn = NeuralNetModel {
            layer_sizes: vec![16, 24, 2],
            layers: vec![layer0, layer1],
            output_interpretation: "atan2".to_string(),
            input_mask: None,
            ablated_input: None,
        };

        let nav = test_nav();
        let data = test_sim_data();
        let planet = PlanetConfig::mars();
        let bank = nn_bank_angle(&nav, &nn, &data, &planet, 50.0_f64.to_radians(), 0.0);

        assert!(bank.is_finite(), "bank angle must be finite, got: {bank}");
        assert!(
            bank > -std::f64::consts::PI - 1e-10 && bank <= std::f64::consts::PI + 1e-10,
            "bank angle out of atan2 range: {bank}",
        );
    }

    // ── New tests for 23-input expansion, mask, and ablation ──

    #[test]
    fn full_23_input_vector_is_finite() {
        // 23->2 network with explicit full mask
        let nn = NeuralNetModel {
            layer_sizes: vec![NN_FULL_INPUT_SIZE, 2],
            layers: vec![Layer {
                w: vec![
                    vec![0.01; NN_FULL_INPUT_SIZE],
                    vec![0.01; NN_FULL_INPUT_SIZE],
                ],
                b: vec![0.1, 0.2],
                activation: Activation::Linear,
            }],
            output_interpretation: "atan2".to_string(),
            input_mask: Some((0..NN_FULL_INPUT_SIZE).collect()),
            ablated_input: None,
        };

        let nav = test_nav();
        let data = test_sim_data_with_ref_traj();
        let planet = PlanetConfig::mars();
        let bank = nn_bank_angle(&nav, &nn, &data, &planet, 50.0_f64.to_radians(), -50.0);

        assert!(
            bank.is_finite(),
            "bank angle must be finite with 23 inputs, got: {bank}"
        );
    }

    #[test]
    fn mask_selects_correct_inputs() {
        // 3->2 network with mask [0, 8, 15]
        let nn = NeuralNetModel {
            layer_sizes: vec![3, 2],
            layers: vec![Layer {
                w: vec![vec![0.1; 3], vec![0.1; 3]],
                b: vec![0.0, 1.0],
                activation: Activation::Linear,
            }],
            output_interpretation: "atan2".to_string(),
            input_mask: Some(vec![0, 8, 15]),
            ablated_input: None,
        };

        let nav = test_nav();
        let data = test_sim_data();
        let planet = PlanetConfig::mars();
        let bank = nn_bank_angle(&nav, &nn, &data, &planet, 50.0_f64.to_radians(), 0.0);

        assert!(
            bank.is_finite(),
            "bank angle must be finite with mask, got: {bank}"
        );
    }

    #[test]
    fn ablation_zeros_target_input() {
        // Two networks: one with ablated_input=Some(0), one without.
        // Weights only on input 0 so ablation changes the output.
        let mut w_row = vec![0.0; 16];
        w_row[0] = 5.0; // large weight on input 0

        let nn_normal = NeuralNetModel {
            layer_sizes: vec![16, 2],
            layers: vec![Layer {
                w: vec![w_row.clone(), vec![0.0; 16]],
                b: vec![0.0, 1.0],
                activation: Activation::Linear,
            }],
            output_interpretation: "atan2".to_string(),
            input_mask: None,
            ablated_input: None,
        };

        let nn_ablated = NeuralNetModel {
            layer_sizes: vec![16, 2],
            layers: vec![Layer {
                w: vec![w_row, vec![0.0; 16]],
                b: vec![0.0, 1.0],
                activation: Activation::Linear,
            }],
            output_interpretation: "atan2".to_string(),
            input_mask: None,
            ablated_input: Some(0),
        };

        let nav = test_nav();
        let data = test_sim_data();
        let planet = PlanetConfig::mars();
        let target_inc = 50.0_f64.to_radians();

        let bank_normal = nn_bank_angle(&nav, &nn_normal, &data, &planet, target_inc, 0.0);
        let bank_ablated = nn_bank_angle(&nav, &nn_ablated, &data, &planet, target_inc, 0.0);

        // Ablated version zeros input 0, so output[0] = 0.0, bank = atan2(0, 1) = 0
        assert_relative_eq!(bank_ablated, 0.0, epsilon = 1e-12);
        // Normal version has non-zero input 0 (eccentricity excess), so bank differs
        assert_ne!(bank_normal, bank_ablated, "ablation should change output");
    }

    #[test]
    fn backward_compat_16_input_mask() {
        // 16->2 network with explicit mask [0..16] should behave same as None
        let nn_explicit = NeuralNetModel {
            layer_sizes: vec![16, 2],
            layers: vec![Layer {
                w: vec![vec![0.01; 16], vec![-0.01; 16]],
                b: vec![0.1, 0.2],
                activation: Activation::Linear,
            }],
            output_interpretation: "atan2".to_string(),
            input_mask: Some((0..16).collect()),
            ablated_input: None,
        };

        let nav = test_nav();
        let data = test_sim_data();
        let planet = PlanetConfig::mars();
        let bank = nn_bank_angle(
            &nav,
            &nn_explicit,
            &data,
            &planet,
            50.0_f64.to_radians(),
            0.0,
        );

        assert!(bank.is_finite(), "bank must be finite, got: {bank}");
        assert!(
            bank > -std::f64::consts::PI - 1e-10 && bank <= std::f64::consts::PI + 1e-10,
            "bank out of atan2 range: {bank}",
        );
    }

    #[test]
    fn bounce_gated_inputs_zero_pre_bounce() {
        // 23->2 network with weights only on indices 20,21,22 and bias [0, 1].
        // bounce_flag=0 should gate exit inputs to zero, so output = [0, 1], bank = 0.
        let mut w0 = vec![0.0; NN_FULL_INPUT_SIZE];
        w0[20] = 10.0;
        w0[21] = 10.0;
        w0[22] = 10.0;
        let nn = NeuralNetModel {
            layer_sizes: vec![NN_FULL_INPUT_SIZE, 2],
            layers: vec![Layer {
                w: vec![w0, vec![0.0; NN_FULL_INPUT_SIZE]],
                b: vec![0.0, 1.0],
                activation: Activation::Linear,
            }],
            output_interpretation: "atan2".to_string(),
            input_mask: Some((0..NN_FULL_INPUT_SIZE).collect()),
            ablated_input: None,
        };

        let mut nav = test_nav();
        nav.bounce_flag = 0; // pre-bounce
        let data = test_sim_data_with_ref_traj();
        let planet = PlanetConfig::mars();
        let bank = nn_bank_angle(&nav, &nn, &data, &planet, 50.0_f64.to_radians(), -50.0);

        // With bounce_flag=0, inputs 20-22 are zero, so output[0] = 0.0, bank = atan2(0, 1) = 0
        assert_relative_eq!(bank, 0.0, epsilon = 1e-12,);
    }

    mod prop {
        use super::*;
        use proptest::prelude::*;

        fn fixed_small_nn() -> NeuralNetModel {
            NeuralNetModel {
                layer_sizes: vec![16, 2],
                layers: vec![Layer {
                    w: vec![
                        vec![
                            0.1, -0.1, 0.2, -0.2, 0.05, -0.05, 0.1, -0.1, 0.02, -0.03, 0.04, -0.01,
                            0.03, -0.02, 0.01, -0.04,
                        ],
                        vec![
                            -0.1, 0.1, -0.05, 0.05, 0.15, -0.15, 0.05, -0.05, -0.02, 0.03, -0.01,
                            0.04, -0.03, 0.02, -0.04, 0.01,
                        ],
                    ],
                    b: vec![0.3, -0.2],
                    activation: Activation::Tanh,
                }],
                output_interpretation: "atan2".to_string(),
                input_mask: None,
                ablated_input: None,
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
                let r = PlanetConfig::mars().equatorial_radius + alt;
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
                let data = test_sim_data();
                let planet = PlanetConfig::mars();
                let bank = nn_bank_angle(&nav, &nn, &data, &planet, 50.0_f64.to_radians(), 0.0);

                prop_assert!(bank.is_finite(), "bank not finite: {}", bank);
            }
        }
    }
}
