//! Neural network guidance.
//!
//! Feedforward network computing bank angle from navigation state.
//! Supports arbitrary layer architectures via NeuralNetModel.
//!
//! 32 candidate inputs (selected by configurable `input_mask`):
//!   0  eccentricity_excess    8  altitude              16 cos_bank_nominal       21 inclination_err_rate
//!   1  inclination_error      9  fpa                   17 pdyn_nominal           22 prev_bank_signed
//!   2  radial_velocity       10  latitude              18 hdot_nominal           23 time_since_sign_flip
//!   3  orbital_energy        11  drag_accel            19 pdyn_error             24 inclination_err_integral
//!   4  velocity              12  lift_accel            20 exit_bank_teacher      25 exit_bank_sin
//!   5  accel_magnitude       13  sma_error             26 exit_bank_cos          27 prev_bank_signed_sin
//!   6  heat_flux_fraction    14  apoapsis_alt          28 prev_bank_signed_cos   29 prev_realized_sin
//!   7  heat_load_fraction    15  bounce_flag           30 prev_realized_cos      31 periapsis_alt
//!
//! Indices 2, 3, 13, 14, 18 use a signed-log `asinh(raw / s)` transform (scales `S_*`
//! measured from an MC ensemble) instead of the old clamps / linear scalings, so the
//! operating range sits in ~[-1, 1] with compressed tails rather than saturating.
//! Index 31 (`periapsis_alt`) is also `asinh`-scaled.
//!
//! Index 20 is the closed-loop FTC exit-phase pdyn-feedback law, fed every step
//! as a teacher signal (always live, not bounce-gated). Pre-bounce, with
//! `ref_velocity_latched = 0`, it degenerates to pure radial-velocity damping.
//!
//! Indices 21-24 expose the lateral-reversal state machine's Markovian state so
//! the NN can learn (and reproduce) signed bank decisions from inputs alone --
//! see GuidanceState::{prev_inclination_error_for_nn, prev_bank_for_nn,
//! last_sign_flip_time_for_nn, inclination_error_integral}. Without these, the
//! supervisor's signed target is bimodal for near-duplicate states (post-reversal
//! pairs collapse under MSE to ~0).
//!
//! Output mapping: dispatched on `nn.output_param` (`OutputParam`).
//! - `Atan2Signed` (default): network emits 2 outputs; signed bank is
//!   `atan2(out[0], out[1]) ∈ (-π, π]`.
//! - `AcosTanh`: network emits 1 output through a `tanh` last layer (so
//!   `out[0] ∈ [-1, 1]`); bank magnitude is `out[0].acos() ∈ [0, π]`.
//!   Only legal in `magnitude_only` mode.
//! - `ScaledPi`: 1 tanh output; `bank = wrap_to_pi(scaled_pi_n * π * out[0]) ∈ (-π, π]`.
//!   Only legal in `full_neural` mode.
//! - `Delta`: 1 tanh output; `bank = wrap_to_pi(prev_realized + delta_max * out[0]) ∈ (-π, π]`.
//!   Only legal in `full_neural` mode.

use crate::config::PlanetConfig;
use crate::data::SimData;
use crate::data::neural::{NN_FULL_INPUT_SIZE, NeuralNetModel};
use crate::data::nn_state::NnState;
use crate::gnc::guidance::exit;
use crate::gnc::navigation::coordinates::total_energy;
use crate::gnc::navigation::estimator::NavigationOutput;
use crate::orbit::elements;

// asinh signed-log scale factors: s = max(|p1|,|p99|)/sinh(1.0), measured from a
// representative MC ensemble (plan Task 1). asinh(raw/s) keeps the operating range
// within ~[-1,1] without clamping; tails compress.
const S_RADIAL_VELOCITY: f64 = 8.802043e+02;
const S_ORBITAL_ENERGY: f64 = 5.554906e+06;
const S_SMA_ERROR: f64 = 3.259362e+07;
const S_APOAPSIS_ALT: f64 = 6.626041e+07;
const S_HDOT_NOMINAL: f64 = 7.333648e+02;
const S_PERIAPSIS_ALT: f64 = 9.158960e+05;

/// Build the masked NN input vector from navigation state.
///
/// Constructs the full 32-element candidate input vector, applies ablation zeroing
/// (if configured), then applies the input_mask (or legacy [0..16] default).
/// Returns the masked `Vec<f64>` ready for `nn.forward()`.
///
/// `input_mask` and `ablated_input` are taken directly so this function can be
/// called without a `NeuralNetModel` (e.g. supervised-trace capture during FTC runs).
///
/// The last 4 parameters (`prev_inclination_error`, `prev_bank_signed`,
/// `time_since_last_sign_flip`, `inclination_error_integral`) populate indices
/// 21-24. They MUST be sourced from GuidanceState (telemetry updated each tick
/// by `tick.rs`) so they reflect the previous-tick state at NN-evaluation time.
///
/// `prev_realized_bank` (radians) is the pilot-realized bank angle from the
/// PREVIOUS tick. It populates indices 29-30 as a (sin, cos) pair and also
/// serves as the base for the `Delta` output decoder in `nn_bank_angle`.
#[allow(clippy::too_many_arguments)]
pub fn build_nn_input(
    nav: &NavigationOutput,
    input_mask: Option<&[usize]>,
    ablated_input: Option<usize>,
    data: &SimData,
    planet: &PlanetConfig,
    target_inclination: f64,
    ref_velocity_latched: f64,
    prev_inclination_error: Option<f64>,
    prev_bank_signed: f64,
    time_since_last_sign_flip: f64,
    inclination_error_integral: f64,
    prev_realized_bank: f64,
) -> Vec<f64> {
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

    // Build full 32-element input vector
    let mut full_input = [0.0_f64; NN_FULL_INPUT_SIZE];

    // -- 16 existing inputs (indices 0-15) --
    full_input[0] = orbit.eccentricity - 1.0; // eccentricity excess
    full_input[1] = (orbit.inclination - target_inclination).to_degrees() * 3.0 / 5.0; // inclination error
    full_input[2] = (velocity_radial / S_RADIAL_VELOCITY).asinh(); // radial velocity
    full_input[3] = (-mu / (2.0 * orbit.semi_major_axis) / S_ORBITAL_ENERGY).asinh(); // orbital energy
    full_input[4] = (nav.velocity_estimated[0] / 3e3 - 1.5) * 2.0; // velocity
    full_input[5] = accel_mag / 20.0 - 1.0; // accel magnitude
    full_input[6] = nav.heat_flux_fraction * 2.0 - 1.0; // heat flux fraction
    full_input[7] = nav.heat_load_fraction * 2.0 - 1.0; // heat load fraction
    full_input[8] = (altitude_km - 65.0) / 65.0; // altitude
    full_input[9] = nav.velocity_estimated[1] / 0.3; // flight path angle
    full_input[10] = nav.position_estimated[2] / std::f64::consts::FRAC_PI_2; // latitude
    full_input[11] = nav.acceleration_estimated[0] / 50.0 - 1.0; // drag acceleration
    full_input[12] = nav.acceleration_estimated[1] / 10.0; // lift acceleration
    full_input[13] = (nav.orbital_errors[0] / S_SMA_ERROR).asinh(); // SMA error
    full_input[14] = (orbit.apoapsis_alt / S_APOAPSIS_ALT).asinh(); // apoapsis altitude
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
    full_input[18] = (hdot_nominal / S_HDOT_NOMINAL).asinh(); // ref radial velocity
    full_input[19] = pdyn_error / 2e3; // dynamic pressure error

    // -- Exit-bank teacher signal (index 20), always live --
    // Closed-loop FTC exit-phase pdyn-feedback law, fed every step as a
    // teacher signal. Pre-bounce, ref_velocity_latched = 0 so this degenerates
    // to pure radial-velocity damping.
    let exit_bank = exit::exit_guidance(nav, data, planet, ref_velocity_latched);
    full_input[20] = exit_bank / std::f64::consts::PI * 2.0 - 1.0;

    // -- Lateral-state telemetry (indices 21-24) --
    // These make the supervisor's signed-bank decision a Markovian function of
    // inputs. Without them, post-reversal near-duplicate states collapse under
    // MSE -> NN can't learn signed targets. See module doc-comment.
    let current_inclination_error = orbit.inclination - target_inclination;
    let di_err_dt = match prev_inclination_error {
        Some(prev) => (current_inclination_error - prev) / data.periods.guidance,
        None => 0.0,
    };
    // Index 21: inclination error rate (deg/s), scaled so typical ~ [-1, 1].
    full_input[21] = di_err_dt.to_degrees() * 10.0;
    // Index 22: previous bank command, normalized to [-1, 1].
    full_input[22] = prev_bank_signed / std::f64::consts::PI;
    // Index 23: time since last bank-command sign flip, tanh-bounded with 30s scale.
    full_input[23] = (time_since_last_sign_flip / 30.0).tanh();
    // Index 24: integrated inclination error (deg·s), tanh-bounded with 100 deg·s scale.
    full_input[24] = (inclination_error_integral.to_degrees() / 100.0).tanh();

    // -- (sin, cos) bank-history pairs (indices 25-30), seam-free cyclic encoding --
    // Bank angles fed as (sin, cos) pairs avoid the +π/-π representation seam.
    // `exit_bank` is the raw radian value already computed above (before its normalization at index 20).
    full_input[25] = exit_bank.sin();
    full_input[26] = exit_bank.cos();
    full_input[27] = prev_bank_signed.sin();
    full_input[28] = prev_bank_signed.cos();
    full_input[29] = prev_realized_bank.sin();
    full_input[30] = prev_realized_bank.cos();

    // -- Periapsis altitude (index 31), asinh signed-log scaled --
    full_input[31] = (orbit.periapsis_alt / S_PERIAPSIS_ALT).asinh(); // periapsis altitude

    // Apply ablation: zero out a single input for sensitivity analysis
    if let Some(idx) = ablated_input {
        full_input[idx] = 0.0;
    }

    // Apply input mask: select subset of inputs, or default to first 16 for backward compat
    match input_mask {
        Some(mask) => mask.iter().map(|&i| full_input[i]).collect(),
        None => full_input[..16].to_vec(),
    }
}

/// Compute NN-guided longitudinal bank angle.
///
/// Builds the masked input vector via `build_nn_input`, runs a forward pass,
/// and returns the decoded bank angle per `nn.output_param`.
///
/// Returns the **signed** bank angle in radians.
/// Lateral guidance is bypassed for this scheme -- the NN controls roll direction directly.
///
/// `prev_inclination_error`, `prev_bank_signed`, `time_since_last_sign_flip`, and
/// `inclination_error_integral` are pulled from GuidanceState by the caller
/// (dispatch.rs) and populate input indices 21-24 -- see build_nn_input.
///
/// `prev_realized_bank` (radians) is the pilot-realized bank angle from the
/// PREVIOUS tick. Populates input indices 29-30 and serves as the base for
/// the `Delta` output decoder.
#[allow(clippy::too_many_arguments)]
pub fn nn_bank_angle(
    nav: &NavigationOutput,
    nn: &NeuralNetModel,
    nn_state: &mut NnState,
    data: &SimData,
    planet: &PlanetConfig,
    target_inclination: f64, // radians
    ref_velocity_latched: f64,
    prev_inclination_error: Option<f64>,
    prev_bank_signed: f64,
    time_since_last_sign_flip: f64,
    inclination_error_integral: f64,
    prev_realized_bank: f64,
) -> f64 {
    let masked = build_nn_input(
        nav,
        nn.input_mask.as_deref(),
        nn.ablated_input,
        data,
        planet,
        target_inclination,
        ref_velocity_latched,
        prev_inclination_error,
        prev_bank_signed,
        time_since_last_sign_flip,
        inclination_error_integral,
        prev_realized_bank,
    );
    use crate::data::neural::OutputParam;
    use crate::gnc::control::angle_utils::wrap_to_pi;
    use std::f64::consts::PI;
    let output = nn.forward(nn_state, &masked);
    match nn.output_param {
        OutputParam::Atan2Signed => output[0].atan2(output[1]),
        OutputParam::AcosTanh => output[0].acos(),
        OutputParam::ScaledPi => wrap_to_pi(nn.scaled_pi_n * PI * output[0]),
        OutputParam::Delta => wrap_to_pi(prev_realized_bank + nn.delta_max * output[0]),
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
    use crate::data::neural::{
        Activation, DenseLayer, Layer, LayerSpec, NN_FULL_INPUT_SIZE, NeuralNetModel, OutputParam,
    };
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
            architecture: vec![LayerSpec::Dense {
                input_size: 16,
                output_size: 2,
                activation: Activation::Linear,
            }],
            layer_sizes: vec![16, 2],
            layers: vec![Layer::Dense(DenseLayer {
                w: vec![vec![0.0; 16], vec![0.0; 16]],
                b: vec![bias0, bias1],
                activation: Activation::Linear,
            })],
            input_mask: None,
            ablated_input: None,
            output_param: OutputParam::default(),
            scaled_pi_n: 1.0,
            delta_max: 0.35,
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
        let mut state = NnState::for_model(&nn);
        let bank = nn_bank_angle(
            &nav,
            &nn,
            &mut state,
            &data,
            &planet,
            50.0_f64.to_radians(),
            0.0,
            None,
            0.0,
            0.0,
            0.0,
            0.0,
        );
        let expected = bias0.atan2(bias1); // PI/4
        assert_relative_eq!(bank, expected, epsilon = 1e-12);
    }

    #[test]
    fn stateful_forward_with_empty_state_matches_stateless() {
        use crate::data::nn_state::NnState;

        let nn = zero_weight_nn(0.5, 0.5);
        let nav = test_nav();
        let data = test_sim_data();
        let planet = PlanetConfig::mars();
        let mut state = NnState::for_model(&nn);

        let bank = nn_bank_angle(
            &nav,
            &nn,
            &mut state,
            &data,
            &planet,
            50.0_f64.to_radians(),
            0.0,
            None,
            0.0,
            0.0,
            0.0,
            0.0,
        );
        assert_relative_eq!(bank, 0.5_f64.atan2(0.5), epsilon = 1e-12);
    }

    #[test]
    fn zero_weights_negative_bias() {
        // Verify atan2 sign handling: atan2(-1, 1) = -PI/4
        let nn = zero_weight_nn(-1.0, 1.0);
        let nav = test_nav();
        let data = test_sim_data();
        let planet = PlanetConfig::mars();

        let mut state = NnState::for_model(&nn);
        let bank = nn_bank_angle(
            &nav,
            &nn,
            &mut state,
            &data,
            &planet,
            50.0_f64.to_radians(),
            0.0,
            None,
            0.0,
            0.0,
            0.0,
            0.0,
        );
        assert_relative_eq!(bank, (-1.0_f64).atan2(1.0), epsilon = 1e-12);
    }

    #[test]
    fn output_in_valid_range() {
        // Small 16->3->2 network with tanh hidden layer and asinh output
        let layer0 = Layer::Dense(DenseLayer {
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
        });
        let layer1 = Layer::Dense(DenseLayer {
            w: vec![vec![0.5, -0.5, 0.2], vec![-0.3, 0.3, -0.1]],
            b: vec![0.0, 0.0],
            activation: Activation::Asinh,
        });
        let nn = NeuralNetModel {
            architecture: vec![
                LayerSpec::Dense {
                    input_size: 16,
                    output_size: 3,
                    activation: Activation::Tanh,
                },
                LayerSpec::Dense {
                    input_size: 3,
                    output_size: 2,
                    activation: Activation::Asinh,
                },
            ],
            layer_sizes: vec![16, 3, 2],
            layers: vec![layer0, layer1],
            input_mask: None,
            ablated_input: None,
            output_param: OutputParam::default(),
            scaled_pi_n: 1.0,
            delta_max: 0.35,
        };

        let nav = test_nav();
        let data = test_sim_data();
        let planet = PlanetConfig::mars();
        let mut state = NnState::for_model(&nn);
        let bank = nn_bank_angle(
            &nav,
            &nn,
            &mut state,
            &data,
            &planet,
            50.0_f64.to_radians(),
            0.0,
            None,
            0.0,
            0.0,
            0.0,
            0.0,
        );

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
        let layer0 = Layer::Dense(DenseLayer {
            w: layer0_weights,
            b: vec![0.0; 24],
            activation: Activation::Tanh,
        });
        let layer1 = Layer::Dense(DenseLayer {
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
        });
        let nn = NeuralNetModel {
            architecture: vec![
                LayerSpec::Dense {
                    input_size: 16,
                    output_size: 24,
                    activation: Activation::Tanh,
                },
                LayerSpec::Dense {
                    input_size: 24,
                    output_size: 2,
                    activation: Activation::Asinh,
                },
            ],
            layer_sizes: vec![16, 24, 2],
            layers: vec![layer0, layer1],
            input_mask: None,
            ablated_input: None,
            output_param: OutputParam::default(),
            scaled_pi_n: 1.0,
            delta_max: 0.35,
        };

        let nav = test_nav();
        let data = test_sim_data();
        let planet = PlanetConfig::mars();
        let mut state = NnState::for_model(&nn);
        let bank = nn_bank_angle(
            &nav,
            &nn,
            &mut state,
            &data,
            &planet,
            50.0_f64.to_radians(),
            0.0,
            None,
            0.0,
            0.0,
            0.0,
            0.0,
        );

        assert!(bank.is_finite(), "bank angle must be finite, got: {bank}");
        assert!(
            bank > -std::f64::consts::PI - 1e-10 && bank <= std::f64::consts::PI + 1e-10,
            "bank angle out of atan2 range: {bank}",
        );
    }

    // ── Tests for full-input expansion, mask, and ablation ──

    #[test]
    fn full_input_vector_is_finite() {
        // 23->2 network with explicit full mask
        let nn = NeuralNetModel {
            architecture: vec![LayerSpec::Dense {
                input_size: NN_FULL_INPUT_SIZE,
                output_size: 2,
                activation: Activation::Linear,
            }],
            layer_sizes: vec![NN_FULL_INPUT_SIZE, 2],
            layers: vec![Layer::Dense(DenseLayer {
                w: vec![
                    vec![0.01; NN_FULL_INPUT_SIZE],
                    vec![0.01; NN_FULL_INPUT_SIZE],
                ],
                b: vec![0.1, 0.2],
                activation: Activation::Linear,
            })],
            input_mask: Some((0..NN_FULL_INPUT_SIZE).collect()),
            ablated_input: None,
            output_param: OutputParam::default(),
            scaled_pi_n: 1.0,
            delta_max: 0.35,
        };

        let nav = test_nav();
        let data = test_sim_data_with_ref_traj();
        let planet = PlanetConfig::mars();
        let mut state = NnState::for_model(&nn);
        let bank = nn_bank_angle(
            &nav,
            &nn,
            &mut state,
            &data,
            &planet,
            50.0_f64.to_radians(),
            -50.0,
            None,
            0.0,
            0.0,
            0.0,
            0.0,
        );

        assert!(
            bank.is_finite(),
            "bank angle must be finite with full input, got: {bank}"
        );
    }

    #[test]
    fn mask_selects_correct_inputs() {
        // 3->2 network with mask [0, 8, 15] selecting inputs at those indices
        // from the full 21-element candidate vector. Each output neuron has
        // a UNIQUE weight per masked column so a misordered or wrong-index
        // mask would produce a different result.
        //
        // We verify this by computing the expected output in closed form
        // from the inputs we know `build_nn_input` produces for `test_nav()`.
        let nn = NeuralNetModel {
            architecture: vec![LayerSpec::Dense {
                input_size: 3,
                output_size: 2,
                activation: Activation::Linear,
            }],
            layer_sizes: vec![3, 2],
            layers: vec![Layer::Dense(DenseLayer {
                // Distinct weights per column so any reordering / misindexing changes the output:
                //   y[0] = 1.0 * x[0] + 2.0 * x[1] + 3.0 * x[2] + 0.0
                //   y[1] = 4.0 * x[0] + 5.0 * x[1] + 6.0 * x[2] + 1.0
                w: vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
                b: vec![0.0, 1.0],
                activation: Activation::Linear,
            })],
            input_mask: Some(vec![0, 8, 15]),
            ablated_input: None,
            output_param: OutputParam::default(),
            scaled_pi_n: 1.0,
            delta_max: 0.35,
        };

        let nav = test_nav();
        let data = test_sim_data();
        let planet = PlanetConfig::mars();
        let target_inc = 50.0_f64.to_radians();
        let ref_velocity = 0.0;

        // Compute the expected masked input vector by calling build_nn_input
        // with the full mask first, then picking indices 0, 8, 15 manually.
        let full_mask: Vec<usize> = (0..21).collect();
        let full_input = build_nn_input(
            &nav,
            Some(&full_mask),
            None,
            &data,
            &planet,
            target_inc,
            ref_velocity,
            None,
            0.0,
            0.0,
            0.0,
            0.0,
        );
        let x0 = full_input[0];
        let x8 = full_input[8];
        let x15 = full_input[15];
        let expected_y0 = 1.0 * x0 + 2.0 * x8 + 3.0 * x15;
        let expected_y1 = 4.0 * x0 + 5.0 * x8 + 6.0 * x15 + 1.0;
        let expected_bank = expected_y0.atan2(expected_y1);

        // Run the actual code path with the same nav state.
        let mut state = NnState::for_model(&nn);
        let bank = nn_bank_angle(
            &nav,
            &nn,
            &mut state,
            &data,
            &planet,
            target_inc,
            ref_velocity,
            None,
            0.0,
            0.0,
            0.0,
            0.0,
        );

        assert!(
            (bank - expected_bank).abs() < 1e-12,
            "mask must select inputs at [0, 8, 15] in order: expected atan2({expected_y0}, {expected_y1})={expected_bank}, got {bank}"
        );
    }

    #[test]
    fn ablation_zeros_target_input() {
        // Two networks: one with ablated_input=Some(0), one without.
        // Weights only on input 0 so ablation changes the output.
        let mut w_row = vec![0.0; 16];
        w_row[0] = 5.0; // large weight on input 0

        let nn_normal = NeuralNetModel {
            architecture: vec![LayerSpec::Dense {
                input_size: 16,
                output_size: 2,
                activation: Activation::Linear,
            }],
            layer_sizes: vec![16, 2],
            layers: vec![Layer::Dense(DenseLayer {
                w: vec![w_row.clone(), vec![0.0; 16]],
                b: vec![0.0, 1.0],
                activation: Activation::Linear,
            })],
            input_mask: None,
            ablated_input: None,
            output_param: OutputParam::default(),
            scaled_pi_n: 1.0,
            delta_max: 0.35,
        };

        let nn_ablated = NeuralNetModel {
            architecture: vec![LayerSpec::Dense {
                input_size: 16,
                output_size: 2,
                activation: Activation::Linear,
            }],
            layer_sizes: vec![16, 2],
            layers: vec![Layer::Dense(DenseLayer {
                w: vec![w_row, vec![0.0; 16]],
                b: vec![0.0, 1.0],
                activation: Activation::Linear,
            })],
            input_mask: None,
            ablated_input: Some(0),
            output_param: OutputParam::default(),
            scaled_pi_n: 1.0,
            delta_max: 0.35,
        };

        let nav = test_nav();
        let data = test_sim_data();
        let planet = PlanetConfig::mars();
        let target_inc = 50.0_f64.to_radians();

        let mut state_normal = NnState::for_model(&nn_normal);
        let mut state_ablated = NnState::for_model(&nn_ablated);
        let bank_normal = nn_bank_angle(
            &nav,
            &nn_normal,
            &mut state_normal,
            &data,
            &planet,
            target_inc,
            0.0,
            None,
            0.0,
            0.0,
            0.0,
            0.0,
        );
        let bank_ablated = nn_bank_angle(
            &nav,
            &nn_ablated,
            &mut state_ablated,
            &data,
            &planet,
            target_inc,
            0.0,
            None,
            0.0,
            0.0,
            0.0,
            0.0,
        );

        // Ablated version zeros input 0, so output[0] = 0.0, bank = atan2(0, 1) = 0
        assert_relative_eq!(bank_ablated, 0.0, epsilon = 1e-12);
        // Normal version has non-zero input 0 (eccentricity excess), so bank differs
        assert_ne!(bank_normal, bank_ablated, "ablation should change output");
    }

    #[test]
    fn backward_compat_16_input_mask() {
        // 16->2 network with explicit mask [0..16] should behave same as None
        let nn_explicit = NeuralNetModel {
            architecture: vec![LayerSpec::Dense {
                input_size: 16,
                output_size: 2,
                activation: Activation::Linear,
            }],
            layer_sizes: vec![16, 2],
            layers: vec![Layer::Dense(DenseLayer {
                w: vec![vec![0.01; 16], vec![-0.01; 16]],
                b: vec![0.1, 0.2],
                activation: Activation::Linear,
            })],
            input_mask: Some((0..16).collect()),
            ablated_input: None,
            output_param: OutputParam::default(),
            scaled_pi_n: 1.0,
            delta_max: 0.35,
        };

        let nav = test_nav();
        let data = test_sim_data();
        let planet = PlanetConfig::mars();
        let mut state = NnState::for_model(&nn_explicit);
        let bank = nn_bank_angle(
            &nav,
            &nn_explicit,
            &mut state,
            &data,
            &planet,
            50.0_f64.to_radians(),
            0.0,
            None,
            0.0,
            0.0,
            0.0,
            0.0,
        );

        assert!(bank.is_finite(), "bank must be finite, got: {bank}");
        assert!(
            bank > -std::f64::consts::PI - 1e-10 && bank <= std::f64::consts::PI + 1e-10,
            "bank out of atan2 range: {bank}",
        );
    }

    #[test]
    fn exit_bank_teacher_signal_is_finite_pre_bounce() {
        // Exit-bank teacher (index 20) is always live, including pre-bounce.
        // With ref_velocity_latched = 0, exit_guidance degenerates to pure
        // radial-velocity damping; verify the resulting input still produces a
        // finite bank command.
        let mut w0 = vec![0.0; NN_FULL_INPUT_SIZE];
        w0[20] = 10.0;
        let nn = NeuralNetModel {
            architecture: vec![LayerSpec::Dense {
                input_size: NN_FULL_INPUT_SIZE,
                output_size: 2,
                activation: Activation::Linear,
            }],
            layer_sizes: vec![NN_FULL_INPUT_SIZE, 2],
            layers: vec![Layer::Dense(DenseLayer {
                w: vec![w0, vec![0.0; NN_FULL_INPUT_SIZE]],
                b: vec![0.0, 1.0],
                activation: Activation::Linear,
            })],
            input_mask: Some((0..NN_FULL_INPUT_SIZE).collect()),
            ablated_input: None,
            output_param: OutputParam::default(),
            scaled_pi_n: 1.0,
            delta_max: 0.35,
        };

        let mut nav = test_nav();
        nav.bounce_flag = 0; // pre-bounce
        let data = test_sim_data_with_ref_traj();
        let planet = PlanetConfig::mars();
        let mut state = NnState::for_model(&nn);
        let bank = nn_bank_angle(
            &nav,
            &nn,
            &mut state,
            &data,
            &planet,
            50.0_f64.to_radians(),
            0.0, // ref_velocity_latched = 0 pre-bounce,
            None,
            0.0,
            0.0,
            0.0,
            0.0,
        );

        assert!(
            bank.is_finite(),
            "exit-bank teacher signal must produce finite bank pre-bounce, got: {bank}"
        );
    }

    // ── Tests for indices 21-24 (Markovian lateral-state telemetry) ──

    #[test]
    fn input_21_rate_is_zero_when_prev_incl_err_is_none() {
        // First tick of a trajectory: prev_inclination_error is None, so the
        // finite-diff rate input MUST be exactly 0.0. Otherwise NaN from
        // (current - None) would propagate.
        let nav = test_nav();
        let data = test_sim_data_with_ref_traj();
        let planet = PlanetConfig::mars();
        let full_mask: Vec<usize> = (0..NN_FULL_INPUT_SIZE).collect();
        let inp = build_nn_input(
            &nav,
            Some(&full_mask),
            None,
            &data,
            &planet,
            50.0_f64.to_radians(),
            0.0,
            None, // first-tick: no prev incl err
            0.0,
            0.0,
            0.0,
            0.0,
        );
        assert_eq!(inp[21], 0.0, "first-tick rate must be 0.0, got {}", inp[21]);
    }

    #[test]
    fn input_21_rate_finite_diff_matches_formula() {
        // Pass an explicit prev_inclination_error; verify input[21] equals
        // ((current - prev) / guidance_period).to_degrees() * 10.0.
        let nav = test_nav();
        let data = test_sim_data_with_ref_traj();
        let planet = PlanetConfig::mars();
        let target = 50.0_f64.to_radians();
        // Compute current inclination error to set up prev that produces a known rate
        let orbit = elements::from_spherical(
            nav.position_estimated[0],
            nav.position_estimated[1],
            nav.position_estimated[2],
            nav.velocity_estimated[0],
            nav.velocity_estimated[1],
            nav.velocity_estimated[2],
            &planet,
        );
        let current_err = orbit.inclination - target;
        // Choose prev so (current - prev) / guidance_period = 0.001 rad/s
        let dt = data.periods.guidance;
        let target_rate = 0.001_f64;
        let prev_err = current_err - target_rate * dt;

        let full_mask: Vec<usize> = (0..NN_FULL_INPUT_SIZE).collect();
        let inp = build_nn_input(
            &nav,
            Some(&full_mask),
            None,
            &data,
            &planet,
            target,
            0.0,
            Some(prev_err),
            0.0,
            0.0,
            0.0,
            0.0,
        );
        let expected = target_rate.to_degrees() * 10.0;
        assert_relative_eq!(inp[21], expected, epsilon = 1e-12);
    }

    #[test]
    fn input_22_prev_bank_normalized_to_pm_one() {
        // prev_bank_signed of PI must produce input[22] = 1.0;
        // -PI/2 must produce -0.5; 0.0 must produce 0.0.
        let nav = test_nav();
        let data = test_sim_data_with_ref_traj();
        let planet = PlanetConfig::mars();
        let target = 50.0_f64.to_radians();
        let full_mask: Vec<usize> = (0..NN_FULL_INPUT_SIZE).collect();
        let cases = [
            (std::f64::consts::PI, 1.0),
            (-std::f64::consts::FRAC_PI_2, -0.5),
            (0.0, 0.0),
        ];
        for (prev_bank, expected) in cases {
            let inp = build_nn_input(
                &nav,
                Some(&full_mask),
                None,
                &data,
                &planet,
                target,
                0.0,
                None,
                prev_bank,
                0.0,
                0.0,
                0.0,
            );
            assert_relative_eq!(inp[22], expected, epsilon = 1e-15);
        }
    }

    #[test]
    fn input_23_time_since_flip_tanh_bounded() {
        // tanh(0/30) = 0; tanh(30/30) = tanh(1) ≈ 0.7616; tanh(very large) → 1.0
        let nav = test_nav();
        let data = test_sim_data_with_ref_traj();
        let planet = PlanetConfig::mars();
        let target = 50.0_f64.to_radians();
        let full_mask: Vec<usize> = (0..NN_FULL_INPUT_SIZE).collect();
        let cases = [(0.0, 0.0), (30.0, 1.0_f64.tanh()), (1e6, 1.0)];
        for (t_since, expected) in cases {
            let inp = build_nn_input(
                &nav,
                Some(&full_mask),
                None,
                &data,
                &planet,
                target,
                0.0,
                None,
                0.0,
                t_since,
                0.0,
                0.0,
            );
            assert_relative_eq!(inp[23], expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn input_24_integral_tanh_bounded() {
        // tanh(integral_deg_s / 100); integral in radians, converted to deg internally.
        // integral = 100 * pi/180 rad·s → 100 deg·s → tanh(1) ≈ 0.7616
        let nav = test_nav();
        let data = test_sim_data_with_ref_traj();
        let planet = PlanetConfig::mars();
        let target = 50.0_f64.to_radians();
        let full_mask: Vec<usize> = (0..NN_FULL_INPUT_SIZE).collect();
        let integral_rad_s = 100.0_f64.to_radians(); // 100 deg·s in rad·s
        let inp = build_nn_input(
            &nav,
            Some(&full_mask),
            None,
            &data,
            &planet,
            target,
            0.0,
            None,
            0.0,
            0.0,
            integral_rad_s,
            0.0,
        );
        assert_relative_eq!(inp[24], 1.0_f64.tanh(), epsilon = 1e-12);

        // Negative integral → negative output (tanh is antisymmetric).
        let inp_neg = build_nn_input(
            &nav,
            Some(&full_mask),
            None,
            &data,
            &planet,
            target,
            0.0,
            None,
            0.0,
            0.0,
            -integral_rad_s,
            0.0,
        );
        assert_relative_eq!(inp_neg[24], -1.0_f64.tanh(), epsilon = 1e-12);
    }

    #[test]
    fn legacy_mask_0_to_20_unchanged_by_new_inputs() {
        // Backward compat: input_mask = [0..21] must yield identical values for
        // indices 0-20 regardless of the new telemetry params (which only affect
        // indices 21-24). This is the load-bearing guarantee for existing trained
        // models with input_size=21.
        let nav = test_nav();
        let data = test_sim_data_with_ref_traj();
        let planet = PlanetConfig::mars();
        let target = 50.0_f64.to_radians();
        let mask_21: Vec<usize> = (0..21).collect();

        let inp_a = build_nn_input(
            &nav,
            Some(&mask_21),
            None,
            &data,
            &planet,
            target,
            0.0,
            None,
            0.0,
            0.0,
            0.0,
            0.0,
        );
        let inp_b = build_nn_input(
            &nav,
            Some(&mask_21),
            None,
            &data,
            &planet,
            target,
            0.0,
            Some(0.1234), // arbitrary non-zero
            std::f64::consts::PI,
            42.5,
            std::f64::consts::E,
            0.0,
        );
        assert_eq!(inp_a.len(), 21);
        assert_eq!(inp_b.len(), 21);
        for i in 0..21 {
            assert_relative_eq!(inp_a[i], inp_b[i], epsilon = 1e-15);
        }
    }

    // ── Tests for Tasks 3 & 4: prev_realized_bank param + (sin,cos) pairs + ScaledPi/Delta decoders ──

    #[test]
    fn full_input_default_mask_is_16_and_ignores_prev_realized() {
        let nav = test_nav();
        let data = test_sim_data();
        let planet = PlanetConfig::mars();
        // A wild prev_realized (9.9 rad) must never leak into the default [..16] slice.
        let v = build_nn_input(
            &nav,
            None,
            None,
            &data,
            &planet,
            0.0,
            0.0,
            Some(0.0),
            0.3,
            0.0,
            0.0,
            /*prev_realized*/ 9.9,
        );
        assert_eq!(v.len(), 16, "default mask must return exactly 16 inputs");
        assert!(
            v.iter().all(|x| x.is_finite()),
            "wild prev_realized must never leak in"
        );
    }

    #[test]
    fn prev_realized_sincos_roundtrips() {
        let nav = test_nav();
        let data = test_sim_data_with_ref_traj();
        let planet = PlanetConfig::mars();
        let prev_realized = 2.5_f64;
        let mask: Vec<usize> = (0..NN_FULL_INPUT_SIZE).collect();
        let v = build_nn_input(
            &nav,
            Some(&mask),
            None,
            &data,
            &planet,
            0.0,
            0.0,
            Some(0.0),
            0.3,
            0.0,
            0.0,
            prev_realized,
        );
        assert_eq!(v.len(), NN_FULL_INPUT_SIZE);
        // sin at index 29, cos at index 30 — atan2(sin, cos) must recover the angle.
        let recovered = v[29].atan2(v[30]);
        assert_relative_eq!(recovered, prev_realized, epsilon = 1e-12);
    }

    #[test]
    fn scaled_pi_scales_by_n_pi_and_wraps() {
        // n=2, single-output tanh net with zero weights, bias=0 → tanh(0)=0 → bank = 2π*0 = 0
        let nn = NeuralNetModel {
            architecture: vec![LayerSpec::Dense {
                input_size: NN_FULL_INPUT_SIZE,
                output_size: 1,
                activation: Activation::Tanh,
            }],
            layer_sizes: vec![NN_FULL_INPUT_SIZE, 1],
            layers: vec![Layer::Dense(DenseLayer {
                w: vec![vec![0.0; NN_FULL_INPUT_SIZE]],
                b: vec![0.0],
                activation: Activation::Tanh,
            })],
            input_mask: Some((0..NN_FULL_INPUT_SIZE).collect()),
            ablated_input: None,
            output_param: OutputParam::ScaledPi,
            scaled_pi_n: 2.0,
            delta_max: 0.0,
        };
        let mut st = NnState::for_model(&nn);
        let nav = test_nav();
        let data = test_sim_data_with_ref_traj();
        let planet = PlanetConfig::mars();
        let b = nn_bank_angle(
            &nav,
            &nn,
            &mut st,
            &data,
            &planet,
            0.0,
            0.0,
            Some(0.0),
            0.0,
            0.0,
            0.0,
            0.0,
        );
        assert_relative_eq!(b, 0.0, epsilon = 1e-12);
        assert!(
            b > -std::f64::consts::PI && b <= std::f64::consts::PI,
            "ScaledPi output must be in (-π, π]: {}",
            b
        );
    }

    #[test]
    fn scaled_pi_wraps_when_product_exceeds_pi() {
        use crate::gnc::control::angle_utils::wrap_to_pi;
        use std::f64::consts::PI;
        // n=1.5, large positive bias → tanh(bias) ≈ 1.0 → raw = 1.5*π ≈ 4.71 > π → must wrap.
        let bias = 20.0_f64;
        let nn = NeuralNetModel {
            architecture: vec![LayerSpec::Dense {
                input_size: NN_FULL_INPUT_SIZE,
                output_size: 1,
                activation: Activation::Tanh,
            }],
            layer_sizes: vec![NN_FULL_INPUT_SIZE, 1],
            layers: vec![Layer::Dense(DenseLayer {
                w: vec![vec![0.0; NN_FULL_INPUT_SIZE]],
                b: vec![bias],
                activation: Activation::Tanh,
            })],
            input_mask: Some((0..NN_FULL_INPUT_SIZE).collect()),
            ablated_input: None,
            output_param: OutputParam::ScaledPi,
            scaled_pi_n: 1.5,
            delta_max: 0.0,
        };
        let mut st = NnState::for_model(&nn);
        let nav = test_nav();
        let data = test_sim_data_with_ref_traj();
        let planet = PlanetConfig::mars();
        let b = nn_bank_angle(
            &nav,
            &nn,
            &mut st,
            &data,
            &planet,
            0.0,
            0.0,
            Some(0.0),
            0.0,
            0.0,
            0.0,
            0.0,
        );
        let unwrapped = 1.5 * PI * bias.tanh();
        let expected = wrap_to_pi(unwrapped);
        assert!(
            (b - expected).abs() < 1e-9,
            "ScaledPi wrap: expected {expected}, got {b}"
        );
        assert!(
            b > -PI && b <= PI,
            "ScaledPi output must be in (-π, π]: {}",
            b
        );
        // Confirm wrapping actually changed the value (unwrapped > π, wrapped < 0).
        assert!(
            (b - unwrapped).abs() > 1e-9,
            "wrap_to_pi did not change the value: unwrapped={unwrapped}, wrapped={b}"
        );
    }

    #[test]
    fn delta_integrates_on_prev_realized_bounded_and_wrapped() {
        // delta_max=0.2, bias=5 → tanh(5)≈1 → step ≈ +0.2 added to prev_realized=1.0
        let nn = NeuralNetModel {
            architecture: vec![LayerSpec::Dense {
                input_size: NN_FULL_INPUT_SIZE,
                output_size: 1,
                activation: Activation::Tanh,
            }],
            layer_sizes: vec![NN_FULL_INPUT_SIZE, 1],
            layers: vec![Layer::Dense(DenseLayer {
                w: vec![vec![0.0; NN_FULL_INPUT_SIZE]],
                b: vec![5.0],
                activation: Activation::Tanh,
            })],
            input_mask: Some((0..NN_FULL_INPUT_SIZE).collect()),
            ablated_input: None,
            output_param: OutputParam::Delta,
            scaled_pi_n: 1.0,
            delta_max: 0.2,
        };
        let mut st = NnState::for_model(&nn);
        let nav = test_nav();
        let data = test_sim_data_with_ref_traj();
        let planet = PlanetConfig::mars();
        let prev_realized = 1.0_f64;
        let b = nn_bank_angle(
            &nav,
            &nn,
            &mut st,
            &data,
            &planet,
            0.0,
            0.0,
            Some(0.0),
            0.0,
            0.0,
            0.0,
            prev_realized,
        );
        let expected_raw = prev_realized + 0.2 * 5.0_f64.tanh();
        assert!(
            (b - expected_raw).abs() < 1e-9,
            "Delta decoder: expected {expected_raw}, got {b}"
        );
        assert!(
            b > -std::f64::consts::PI && b <= std::f64::consts::PI,
            "Delta output must be wrapped to (-π, π]: {}",
            b
        );
    }

    #[test]
    fn delta_wraps_when_sum_exceeds_pi() {
        use crate::gnc::control::angle_utils::wrap_to_pi;
        use std::f64::consts::PI;
        // prev_realized = π - 0.1, delta_max = 0.5, large bias → tanh ≈ 1.0
        // raw sum ≈ π - 0.1 + 0.5 = π + 0.4 > π → must wrap to negative.
        let bias = 20.0_f64;
        let prev_realized = PI - 0.1;
        let nn = NeuralNetModel {
            architecture: vec![LayerSpec::Dense {
                input_size: NN_FULL_INPUT_SIZE,
                output_size: 1,
                activation: Activation::Tanh,
            }],
            layer_sizes: vec![NN_FULL_INPUT_SIZE, 1],
            layers: vec![Layer::Dense(DenseLayer {
                w: vec![vec![0.0; NN_FULL_INPUT_SIZE]],
                b: vec![bias],
                activation: Activation::Tanh,
            })],
            input_mask: Some((0..NN_FULL_INPUT_SIZE).collect()),
            ablated_input: None,
            output_param: OutputParam::Delta,
            scaled_pi_n: 1.0,
            delta_max: 0.5,
        };
        let mut st = NnState::for_model(&nn);
        let nav = test_nav();
        let data = test_sim_data_with_ref_traj();
        let planet = PlanetConfig::mars();
        let b = nn_bank_angle(
            &nav,
            &nn,
            &mut st,
            &data,
            &planet,
            0.0,
            0.0,
            Some(0.0),
            0.0,
            0.0,
            0.0,
            prev_realized,
        );
        let expected = wrap_to_pi(prev_realized + 0.5 * bias.tanh());
        assert!(
            (b - expected).abs() < 1e-9,
            "Delta wrap: expected {expected}, got {b}"
        );
        // The wrapped result must be negative (folded below 0).
        assert!(
            b < 0.0,
            "Delta wrap: result should be negative after wrap, got {b}"
        );
        assert!(b > -PI && b <= PI, "Delta output must be in (-π, π]: {}", b);
    }

    mod prop {
        use super::*;
        use proptest::prelude::*;

        fn fixed_small_nn() -> NeuralNetModel {
            NeuralNetModel {
                architecture: vec![LayerSpec::Dense {
                    input_size: 16,
                    output_size: 2,
                    activation: Activation::Tanh,
                }],
                layer_sizes: vec![16, 2],
                layers: vec![Layer::Dense(DenseLayer {
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
                })],
                input_mask: None,
                ablated_input: None,
                output_param: OutputParam::default(),
                scaled_pi_n: 1.0,
                delta_max: 0.35,
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
                        let mut state = NnState::for_model(&nn);
                        let bank = nn_bank_angle(&nav, &nn, &mut state, &data, &planet, 50.0_f64.to_radians(), 0.0,
            None,
            0.0,
            0.0,
            0.0,
            0.0,
        );

                        prop_assert!(bank.is_finite(), "bank not finite: {}", bank);
                    }
                }

        #[test]
        fn acos_tanh_parameterization_emits_acos_of_output() {
            use crate::data::neural::{
                Activation, DenseLayer, Layer, LayerSpec, NeuralNetModel, OutputParam,
            };

            let nn = NeuralNetModel {
                architecture: vec![LayerSpec::Dense {
                    input_size: 16,
                    output_size: 1,
                    activation: Activation::Tanh,
                }],
                layer_sizes: vec![16, 1],
                layers: vec![Layer::Dense(DenseLayer {
                    w: vec![vec![0.0; 16]],
                    b: vec![0.5],
                    activation: Activation::Tanh,
                })],
                input_mask: None,
                ablated_input: None,
                output_param: OutputParam::AcosTanh,
                scaled_pi_n: 1.0,
                delta_max: 0.35,
            };

            let nav = test_nav();
            let data = test_sim_data();
            let planet = PlanetConfig::mars();
            let mut state = NnState::for_model(&nn);
            let bank = nn_bank_angle(
                &nav,
                &nn,
                &mut state,
                &data,
                &planet,
                50.0_f64.to_radians(),
                0.0,
                None,
                0.0,
                0.0,
                0.0,
                0.0,
            );

            let expected = (0.5_f64).tanh().acos();
            assert!(
                (bank - expected).abs() < 1e-12,
                "bank={bank} expected={expected}"
            );
        }
    }

    // ── Task 1: asinh-rescaled inputs + periapsis_alt at index 31 ──

    #[test]
    fn nn_full_input_size_is_32() {
        assert_eq!(NN_FULL_INPUT_SIZE, 32);
    }

    #[test]
    fn rescaled_inputs_use_asinh_and_periapsis_present() {
        let nav = test_nav();
        let data = test_sim_data_with_ref_traj();
        let planet = PlanetConfig::mars();
        let mask: Vec<usize> = (0..32).collect();
        let v = build_nn_input(
            &nav,
            Some(&mask),
            None,
            &data,
            &planet,
            0.0,
            0.0,
            Some(0.0),
            0.3,
            0.0,
            0.0,
            0.0,
        );
        assert_eq!(v.len(), 32);
        assert!(v.iter().all(|x| x.is_finite()));
        assert!(v[14].abs() < 5.0); // apoapsis now asinh-scaled, not pinned at the old ±9 clamp
        assert!(v[31].is_finite()); // periapsis present
    }

    #[test]
    fn asinh_rescale_bounds_huge_values() {
        assert!((100.0_f64).asinh() < 6.0);
    }

    #[test]
    fn default_mask_path_still_16() {
        let nav = test_nav();
        let data = test_sim_data_with_ref_traj();
        let planet = PlanetConfig::mars();
        let v = build_nn_input(
            &nav,
            None,
            None,
            &data,
            &planet,
            0.0,
            0.0,
            Some(0.0),
            0.3,
            0.0,
            0.0,
            9.9,
        );
        assert_eq!(v.len(), 16);
    }

    #[test]
    fn radial_velocity_input_is_asinh_of_raw() {
        let nav = test_nav();
        let data = test_sim_data_with_ref_traj();
        let planet = PlanetConfig::mars();
        let mask: Vec<usize> = (0..NN_FULL_INPUT_SIZE).collect();
        let v = build_nn_input(
            &nav,
            Some(&mask),
            None,
            &data,
            &planet,
            0.0,
            0.0,
            Some(0.0),
            0.3,
            0.0,
            0.0,
            0.0,
        );
        let velocity_radial = nav.velocity_estimated[0] * nav.velocity_estimated[1].sin();
        let expected = (velocity_radial / S_RADIAL_VELOCITY).asinh();
        assert!(
            (v[2] - expected).abs() < 1e-12,
            "index 2 must be asinh(velocity_radial/S_RADIAL_VELOCITY): expected {expected}, got {}",
            v[2]
        );
    }
}
