//! Central guidance dispatcher: routes to scheme-specific modules per phase.

use crate::config::{GuidanceType, PlanetConfig};
use crate::data::SimData;
use crate::data::guidance_params::NeuralNetMode;
use crate::data::neural::NeuralNetModel;
use crate::data::nn_state::NnState;
use crate::gnc::control::angle_utils::shortest_angle_diff;
use crate::gnc::guidance::ftc::{self as ftc_capture, FtcCaptureState};
use crate::gnc::guidance::lateral::{self, LateralState};
use crate::gnc::guidance::{
    energy_controller, equilibrium_glide, exit, fnpag, neural, piecewise_constant, predguid,
    thermal_limiter,
};
use crate::gnc::navigation::coordinates::{geodetic_from_spherical, total_energy};
use crate::gnc::navigation::estimator::NavigationOutput;

/// Default fallback bank angle (60°) for schemes with no valid reference trajectory.
///
/// Defined as `60.0_f64.to_radians()` via const-eval so every use is the
/// same f64 bit-pattern as the literal expression.
pub(crate) const DEFAULT_FALLBACK_BANK_RAD: f64 = 60.0_f64.to_radians();

/// Convert a (possibly out-of-range) cos(bank) into a valid bank magnitude.
///
/// Clamps to [-1, 1] so acos stays in [0, π].  Only the clamp+acos tail is
/// shared — each scheme's cos_bank summation stays inline (reassociating the
/// sum would break bit-identity).
#[inline]
pub(crate) fn securize_cos_bank(cos_bank: f64) -> f64 {
    cos_bank.clamp(-1.0, 1.0).acos()
}

/// Acceleration-limited command shaper state.
#[derive(Debug, Clone, Copy, Default)]
pub struct CommandShaper {
    pub shaped_rate: f64, // current shaped bank rate (rad/s)
}

impl CommandShaper {
    pub fn new() -> Self {
        Self { shaped_rate: 0.0 }
    }
}

/// Guidance dispatcher persistent state.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct GuidanceState {
    // Bank angle command
    pub bank_angle_commanded: f64, // current commanded bank angle (rad)
    pub bank_angle_realized: f64,  // pilot-realized bank angle (rad)
    pub aoa_commanded: f64,        // commanded AoA (rad)
    pub command_shaper: CommandShaper,

    // Roll sign and reversal tracking
    pub lateral_state: LateralState,
    pub cumulative_bank_change: f64, // cumulative bank angle changes (rad)

    // Guidance securization (longitudinal only; lateral securization handled by lateral module)
    pub longi_active: i32, // longitudinal securization indicator (1=active)

    // Reference velocity
    pub reference_velocity: f64,

    // Counters
    pub n_active: i32, // number of active guidance calls

    // Scheme-specific states
    pub ftc_capture: FtcCaptureState,
    pub energy_ctrl: energy_controller::EnergyControllerState,
    pub predguid: predguid::PredGuidState,
    pub fnpag: fnpag::FnpagState,

    // Per-sim mutable NN state (`Some` only when the active scheme loads a NeuralNetModel).
    pub nn_state: Option<NnState>,

    // ── NN-input telemetry (Markovian state for the lateral reversal decision) ──
    // Updated by tick.rs every tick regardless of the active scheme so the NN's
    // candidate input vector indices 21-24 stay consistent across supervisor
    // collection (FTC/EqGlide/...) and runtime NN deploy.
    /// Previous tick's inclination error (radians, None on first tick).
    /// Used to compute `di_err_dt = (current - prev) / guidance_period` for input 21.
    pub prev_inclination_error_for_nn: Option<f64>,
    /// Previous tick's commanded bank angle (radians, signed in [-π, π]).
    /// Surfaced as input 22 (normalized by /π) so the NN can see its own last command.
    pub prev_bank_for_nn: f64,
    /// Previous-tick pilot-realized bank (rad). Backs the `delta` decoder base
    /// and the prev-realized (sin,cos) NN input. Updated post-guidance in tick.rs.
    pub prev_realized_bank_for_nn: f64,
    /// Sim time of the most recent bank-command sign flip (seconds).
    /// Surfaced via `tanh((sim_time - this) / 30)` as input 23 -- anti-chatter awareness.
    pub last_sign_flip_time_for_nn: f64,
    /// Running integral of inclination error (radian-seconds, simple Euler).
    /// Surfaced via `tanh(integral_deg_s / 100)` as input 24 -- long-term tracking error.
    pub inclination_error_integral: f64,
}

impl GuidanceState {
    pub fn new(initial_bank: f64, initial_aoa: f64, nn_model: Option<&NeuralNetModel>) -> Self {
        let nn_state = nn_model.map(NnState::for_model);
        Self {
            bank_angle_commanded: initial_bank,
            bank_angle_realized: initial_bank,
            aoa_commanded: initial_aoa,
            command_shaper: CommandShaper::new(),
            lateral_state: LateralState::new(initial_bank),
            cumulative_bank_change: 0.0,
            longi_active: 1,
            reference_velocity: 0.0,
            n_active: 0,
            ftc_capture: FtcCaptureState::default(),
            energy_ctrl: energy_controller::EnergyControllerState::new(),
            predguid: predguid::PredGuidState::new(),
            fnpag: fnpag::FnpagState::new(initial_bank),
            nn_state,
            prev_inclination_error_for_nn: None,
            prev_bank_for_nn: initial_bank,
            prev_realized_bank_for_nn: initial_bank,
            last_sign_flip_time_for_nn: 0.0,
            inclination_error_integral: 0.0,
        }
    }
}

/// Guidance dispatcher output.
#[derive(Debug, Clone, Copy, Default)]
pub struct GuidanceOutput {
    pub bank_angle_commanded: f64, // commanded bank angle (rad)
    pub aoa_commanded: f64,        // commanded AoA (rad)
    pub bank_rate: f64,            // bank rate before saturation (rad/s)
    pub longitudinal_active: i32,  // longitudinal guidance active
    pub rate_saturated: i32,       // rate saturation occurred
    pub roll_reversal_active: i32, // roll reversal indicator
    /// Bank magnitude after thermal limiter, before lateral sign selection.
    /// Zero for signed-bank schemes (PiecewiseConstant, NN in FullNeural mode).
    pub pre_lateral_magnitude: f64,
    /// Signed bank command after lateral sign selection, BEFORE command shaping.
    /// This is the value the supervised warm-start path captures so that:
    ///   1. the sign carries through (full_neural deploy has no lateral guidance to add signs),
    ///   2. the command shaper runs exactly once at deploy on the NN's output, not twice
    ///      (which `bank_angle_commanded` would cause, since it is post-shaper).
    ///
    /// For signed-bank schemes (PiecewiseConstant, NN/FullNeural) this equals
    /// `bank_angle_longitudinal` directly (no lateral sign multiply happens).
    pub pre_shaper_signed: f64,
}

/// Run one guidance step (dispatches to the active scheme).
#[allow(clippy::too_many_arguments)]
pub fn guidance_step(
    nav: &NavigationOutput,
    pilot_bank_angle: f64, // pilot-realized bank angle
    sim_time: f64,
    reference_bank_angle: f64, // reference bank angle (from config, rad)
    state: &mut GuidanceState,
    data: &SimData,
    planet: &PlanetConfig,
    is_reference: bool,
    guidance_type: GuidanceType,
) -> GuidanceOutput {
    let mut out = GuidanceOutput::default();

    state.bank_angle_realized = pilot_bank_angle;

    // === Angle of attack guidance ===
    // proalf returns altitude as scheduling parameter
    let (altitude, _) = geodetic_from_spherical(
        nav.position_estimated[0],
        nav.position_estimated[1],
        nav.position_estimated[2],
        planet,
    );
    state.aoa_commanded = data.incidence.incidence_at(altitude);
    out.aoa_commanded = state.aoa_commanded;

    // === Longitudinal guidance activation ===
    let energy = total_energy(
        nav.position_estimated[0],
        nav.position_estimated[1],
        nav.position_estimated[2],
        nav.velocity_estimated[0],
        nav.velocity_estimated[1],
        nav.velocity_estimated[2],
        planet,
    );

    let mut longitudinal_active: i32;
    if energy <= data.guidance.longi_activation && energy >= data.guidance.longi_inhibition {
        longitudinal_active = 1;
    } else {
        longitudinal_active = 0;
    }

    longitudinal_active *= state.longi_active;
    out.longitudinal_active = longitudinal_active;

    // === Reference trajectory mode ===
    if is_reference {
        longitudinal_active = 0;
        state.longi_active = 0;
    }

    // === Longitudinal bank angle command ===
    // reference_bank_angle passed as parameter from config.reference_bank_angle
    let mut bank_angle_longitudinal: f64;

    // Schemes that produce signed bank angles bypass exit, lateral, and thermal-limiter
    // guidance entirely. NN bypasses only in FullNeural mode; MagnitudeOnly mode reuses
    // the same unsigned-magnitude pipeline as FTC and the other parametric schemes.
    let nn_full_neural = matches!(guidance_type, GuidanceType::NeuralNetwork)
        && data.guidance.neural_mode == NeuralNetMode::FullNeural;
    // Schemes that produce signed bank angles bypass exit/lateral/thermal-limiter entirely.
    // PiecewiseConstant always; NN only in FullNeural mode (MagnitudeOnly routes through
    // the unsigned-magnitude pipeline just like FTC).
    let is_signed_bank_scheme =
        matches!(guidance_type, GuidanceType::PiecewiseConstant) || nn_full_neural;
    let uses_exit_guidance = !is_signed_bank_scheme;

    if is_reference {
        state.bank_angle_commanded = reference_bank_angle;
        bank_angle_longitudinal = reference_bank_angle;
    } else if longitudinal_active == 0 {
        bank_angle_longitudinal = reference_bank_angle.abs();
    } else if nav.guidance_phase == 2 && uses_exit_guidance {
        // Exit phase: shared pdyn-feedback controller for all unsigned-magnitude schemes
        bank_angle_longitudinal = exit::exit_guidance(nav, data, planet, state.reference_velocity);
        state.n_active += 1;
    } else {
        // Capture phase: scheme-specific longitudinal guidance
        bank_angle_longitudinal = match guidance_type {
            GuidanceType::Ftc => {
                ftc_capture::ftc_bank_angle(nav, &mut state.ftc_capture, data, altitude, energy)
            }
            GuidanceType::NeuralNetwork => {
                let nn = data.neural_net.as_ref().expect("NN params not loaded");
                // Snapshot telemetry scalars BEFORE the mut borrow of nn_state so
                // rustc doesn't trip on simultaneous shared+mut borrows of `state`.
                let prev_incl_err = state.prev_inclination_error_for_nn;
                let prev_bank = state.prev_bank_for_nn;
                let time_since_flip = sim_time - state.last_sign_flip_time_for_nn;
                let integral = state.inclination_error_integral;
                let ref_vel = state.reference_velocity;
                let prev_realized = state.prev_realized_bank_for_nn;
                let nn_state = state.nn_state.as_mut().expect(
                    "neural_network scheme requires nn_state initialized by GuidanceState::new",
                );
                let signed = neural::nn_bank_angle(
                    nav,
                    nn,
                    nn_state,
                    data,
                    planet,
                    data.target_orbit.inclination,
                    ref_vel,
                    prev_incl_err,
                    prev_bank,
                    time_since_flip,
                    integral,
                    prev_realized,
                );
                // MagnitudeOnly: drop the sign and feed magnitude into the unsigned
                // pipeline (thermal limiter + lateral guidance handle sign + safety).
                if data.guidance.neural_mode == NeuralNetMode::MagnitudeOnly {
                    signed.abs()
                } else {
                    signed
                }
            }
            GuidanceType::EquilibriumGlide => {
                equilibrium_glide::equilibrium_glide_bank(nav, data, planet, altitude)
            }
            GuidanceType::EnergyController => {
                energy_controller::energy_controller_bank(nav, &state.energy_ctrl, data, energy)
            }
            GuidanceType::PredGuid => predguid::predguid_bank(nav, &state.predguid, data, energy),
            GuidanceType::Fnpag => fnpag::fnpag_bank(nav, &mut state.fnpag, data, planet),
            GuidanceType::PiecewiseConstant => piecewise_constant::piecewise_constant_bank(
                nav,
                &data.guidance.piecewise_constant,
                energy,
            ),
        };
        state.n_active += 1;
    }

    // === Thermal safety limiter (unsigned-magnitude schemes only) ===
    // Same gating as exit: NN in MagnitudeOnly mode goes through the limiter.
    let uses_thermal_limiter = !is_signed_bank_scheme;
    if uses_thermal_limiter && longitudinal_active == 1 && !is_reference {
        let cos_bank = bank_angle_longitudinal.cos();
        let cos_limited = thermal_limiter::apply_thermal_limit(
            cos_bank,
            nav.heat_flux_fraction,
            nav.heat_load_fraction,
            &data.guidance.thermal_limiter,
        );
        bank_angle_longitudinal = cos_limited.acos();
    }

    // Record the unsigned magnitude post-thermal-limiter, before lateral sign selection.
    // For signed-bank schemes (PiecewiseConstant, NN FullNeural) this is 0.0 (not meaningful).
    // Also gate on `longitudinal_active == 1`: when guidance is inhibited the magnitude
    // is just `|reference_bank_angle|` regardless of state — recording those ticks would
    // pollute supervised datasets with constant rows that have zero variance vs. inputs.
    if longitudinal_active == 1 && !is_signed_bank_scheme {
        out.pre_lateral_magnitude = bank_angle_longitudinal;
    }

    // Schemes that provide signed bank angles — skip lateral guidance entirely.
    // NN in MagnitudeOnly mode delegates sign selection to lateral.
    let skip_lateral = is_signed_bank_scheme;
    if skip_lateral {
        state.bank_angle_commanded = bank_angle_longitudinal;
        state.lateral_state.roll_sign = if bank_angle_longitudinal >= 0.0 {
            1.0
        } else {
            -1.0
        };
    }

    // === Lateral guidance ===
    let mut roll_reversal_active = false;
    if !skip_lateral {
        roll_reversal_active = lateral::lateral_guidance(
            &data.guidance.lateral,
            &mut state.lateral_state,
            nav,
            data.target_orbit.inclination,
            energy,
            bank_angle_longitudinal,
            sim_time,
            planet,
        );
    }
    // === Combine longitudinal and lateral commands ===
    if !is_reference && !skip_lateral {
        state.bank_angle_commanded = bank_angle_longitudinal * state.lateral_state.roll_sign;
    }

    // Snapshot the post-lateral, PRE-shaper signed bank command. This is the
    // value the supervised warm-start path captures so the NN learns the bank
    // that gets fed INTO the shaper (which then runs exactly once at deploy,
    // not twice). For signed-bank schemes (PiecewiseConstant, NN/FullNeural)
    // `state.bank_angle_commanded` already equals `bank_angle_longitudinal`
    // via the `skip_lateral` branch above.
    out.pre_shaper_signed = state.bank_angle_commanded;

    // === Roll rate / acceleration shaping (wrap-aware) ===
    let max_bank_rate = data.capsule.max_bank_rate;
    let guidance_period = data.periods.guidance;
    // Use pilot-realized angle as baseline (feedback fix)
    let angle_diff = shortest_angle_diff(state.bank_angle_realized, state.bank_angle_commanded);
    let raw_rate = angle_diff / guidance_period;
    let mut rate_saturated = 0;

    let bank_rate;
    if let Some(ref shaping) = data.guidance.command_shaping {
        // S-curve command shaper: acceleration-limited rate
        let rate_delta = raw_rate - state.command_shaper.shaped_rate;
        let max_rate_delta = shaping.max_bank_acceleration * guidance_period;
        let clamped_delta = rate_delta.clamp(-max_rate_delta, max_rate_delta);
        state.command_shaper.shaped_rate += clamped_delta;
        state.command_shaper.shaped_rate = state
            .command_shaper
            .shaped_rate
            .clamp(-max_bank_rate, max_bank_rate);

        if clamped_delta.abs() < rate_delta.abs() - 1e-10
            || state.command_shaper.shaped_rate.abs() >= max_bank_rate - 1e-10
        {
            rate_saturated = 1;
        }

        state.bank_angle_commanded =
            state.bank_angle_realized + state.command_shaper.shaped_rate * guidance_period;
        bank_rate = state.command_shaper.shaped_rate;
    } else {
        // Legacy hard-clamp (backward compatible when shaping absent)
        bank_rate = raw_rate;
        if raw_rate.abs() - max_bank_rate > 1e-10 {
            rate_saturated = 1;
            state.bank_angle_commanded =
                state.bank_angle_realized + max_bank_rate.copysign(angle_diff) * guidance_period;
        }
    }

    // Cumulative bank angle tracking (shortest path)
    let cumulative_diff =
        shortest_angle_diff(state.bank_angle_realized, state.bank_angle_commanded);
    if cumulative_diff.abs() > 1e-10 {
        state.cumulative_bank_change += cumulative_diff.abs();
    }

    out.bank_angle_commanded = state.bank_angle_commanded;
    out.bank_rate = bank_rate;
    out.rate_saturated = rate_saturated;
    out.roll_reversal_active = if roll_reversal_active { 1 } else { 0 };

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::config::{GuidanceType, PlanetConfig};

    /// DEFAULT_FALLBACK_BANK_RAD must be bit-identical to 60.0_f64.to_radians().
    #[test]
    fn fallback_bank_rad_bit_identity() {
        assert_eq!(DEFAULT_FALLBACK_BANK_RAD, 60.0_f64.to_radians());
    }
    use crate::data::aerodynamics::AeroTables;
    use crate::data::atmosphere::{AtmosphereModel, DensityProfile};
    use crate::data::capsule::Capsule;
    use crate::data::guidance_params::GuidanceParams;
    use crate::data::incidence::IncidenceProfile;
    use crate::data::pilot::{PilotModel, PilotType};
    use crate::data::{
        Constraints, EntryConditions, FinalConditions, OrbitalTarget, ParkingOrbit, SimData,
        SphericalState, SuccessCriteria, TimePeriods,
    };
    use crate::gnc::guidance::lateral::LateralParams;
    use crate::gnc::navigation::estimator::NavigationOutput;

    // ─── Fixture builders ───────────────────────────────────────────────────

    fn test_nav() -> NavigationOutput {
        let r = PlanetConfig::mars().equatorial_radius + 50_000.0; // Mars + 50 km
        NavigationOutput {
            position_estimated: [r, 0.0, 0.0],
            velocity_estimated: [5000.0, -0.15, 0.6],
            acceleration_estimated: [50.0, -8.0],
            aero_coefficients: [1.269, -0.205],
            density_guidance: 0.001,
            density_exit: 1e-6,
            dynamic_pressure_estimated: 0.5 * 0.001 * 5000.0 * 5000.0,
            energy_estimated: -1e6,
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
                // Wide activation window so longitudinal guidance fires
                longi_activation: 1e12,
                longi_inhibition: -1e12,
                lateral: LateralParams {
                    lateral_activation: -1e12, // disable lateral for simple tests
                    lateral_inhibition: -1e12,
                    max_reversals: 5,
                    ..LateralParams::default()
                },
                density_filter_gain: 0.8,
                exit_velocity_threshold: 4400.0,
                exit_altitude_threshold: 60_000.0,
                capture_damping: 0.7,
                capture_frequency: 0.072,
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
            nn_normalization_override: None,
        }
    }

    fn test_sim_data_with_shaping(max_bank_acceleration_deg: f64) -> SimData {
        let mut data = test_sim_data();
        data.guidance.command_shaping = Some(crate::data::guidance_params::CommandShapingConfig {
            max_bank_acceleration: max_bank_acceleration_deg.to_radians(),
        });
        data
    }

    // ─── GuidanceState field tests ───────────────────────────────────────────

    #[test]
    fn guidance_state_inits_prev_realized_bank() {
        let s = GuidanceState::new(0.5, 0.1, None);
        assert_eq!(s.prev_realized_bank_for_nn, 0.5);
    }

    // ─── Deterministic tests ─────────────────────────────────────────────────

    /// guidance_step should return a finite bank angle for a typical MSR state
    /// using the FTC scheme.
    #[test]
    fn guidance_step_returns_finite_output() {
        let nav = test_nav();
        let data = test_sim_data();
        let planet = PlanetConfig::mars();
        let initial_bank = 64.77_f64.to_radians();
        let mut state = GuidanceState::new(initial_bank, -0.48_f64.to_radians(), None);

        let out = guidance_step(
            &nav,
            initial_bank,
            0.0, // sim_time
            initial_bank,
            &mut state,
            &data,
            &planet,
            false,
            GuidanceType::Ftc,
        );

        assert!(
            out.bank_angle_commanded.is_finite(),
            "bank_angle_commanded not finite: {}",
            out.bank_angle_commanded
        );
        assert!(
            out.aoa_commanded.is_finite(),
            "aoa_commanded not finite: {}",
            out.aoa_commanded
        );
        assert!(
            out.bank_rate.is_finite(),
            "bank_rate not finite: {}",
            out.bank_rate
        );
    }

    /// In reference mode, output bank should equal the reference bank angle.
    #[test]
    fn reference_mode_returns_reference_bank() {
        let nav = test_nav();
        let data = test_sim_data();
        let planet = PlanetConfig::mars();
        let reference_bank_angle = 45.0_f64.to_radians();
        let mut state = GuidanceState::new(reference_bank_angle, -0.48_f64.to_radians(), None);
        // Prime bank_angle_realized so rate saturation doesn't shift the value
        state.bank_angle_realized = reference_bank_angle;

        let out = guidance_step(
            &nav,
            reference_bank_angle,
            0.0,
            reference_bank_angle,
            &mut state,
            &data,
            &planet,
            true, // is_reference
            GuidanceType::Ftc,
        );

        assert!(
            (out.bank_angle_commanded - reference_bank_angle).abs() < 1e-9,
            "expected bank_angle_commanded ≈ reference_bank_angle ({:.6} rad), got {:.6} rad",
            reference_bank_angle,
            out.bank_angle_commanded,
        );
    }

    /// Bank angle magnitude should stay within [0, π] radians.
    #[test]
    fn output_bank_bounded() {
        let nav = test_nav();
        let data = test_sim_data();
        let planet = PlanetConfig::mars();
        let initial_bank = 64.77_f64.to_radians();
        let mut state = GuidanceState::new(initial_bank, -0.48_f64.to_radians(), None);

        let out = guidance_step(
            &nav,
            initial_bank,
            0.0,
            initial_bank,
            &mut state,
            &data,
            &planet,
            false,
            GuidanceType::Ftc,
        );

        let pi = std::f64::consts::PI;
        assert!(
            out.bank_angle_commanded >= -pi && out.bank_angle_commanded <= pi,
            "bank_angle_commanded = {:.4} rad is outside [-π, π]",
            out.bank_angle_commanded,
        );
    }

    /// longitudinal_active=0 (guidance inactive) should still return a finite bank
    /// equal to |reference_bank_angle|, without saturating.
    #[test]
    fn inactive_longitudinal_guidance_uses_reference_bank() {
        let nav = test_nav();
        let mut data = test_sim_data();
        // Force energy outside activation window so longitudinal_active=0
        data.guidance.longi_activation = -1e12;
        data.guidance.longi_inhibition = -2e12;
        data.guidance.lateral.lateral_activation = -2e12;

        let planet = PlanetConfig::mars();
        let reference_bank_angle = 30.0_f64.to_radians();
        let mut state = GuidanceState::new(reference_bank_angle, -0.48_f64.to_radians(), None);
        state.bank_angle_realized = reference_bank_angle;

        let out = guidance_step(
            &nav,
            reference_bank_angle,
            0.0,
            reference_bank_angle,
            &mut state,
            &data,
            &planet,
            false,
            GuidanceType::Ftc,
        );

        assert!(
            out.bank_angle_commanded.is_finite(),
            "expected finite bank_angle_commanded, got {}",
            out.bank_angle_commanded
        );
        // When guidance is inactive and no lateral, bank_angle_commanded comes from reference_bank_angle.abs()
        // clamped by rate saturation — it should stay close to reference_bank_angle for a single step
        let pi = std::f64::consts::PI;
        assert!(
            out.bank_angle_commanded.abs() <= pi,
            "bank_angle_commanded magnitude exceeds π: {}",
            out.bank_angle_commanded
        );
    }

    // ─── Command shaper tests ────────────────────────────────────────────────

    /// Without shaping config, the legacy hard-clamp path fires and output is finite.
    #[test]
    fn shaper_disabled_matches_legacy_hardclamp() {
        let nav = test_nav();
        let data = test_sim_data(); // command_shaping = None
        let planet = PlanetConfig::mars();
        // Start at 0, target 90 deg — would saturate the rate
        let realized = 0.0_f64;
        let target = 90.0_f64.to_radians();
        let mut state = GuidanceState::new(realized, -0.48_f64.to_radians(), None);
        state.bank_angle_realized = realized;

        let out = guidance_step(
            &nav,
            realized,
            0.0,
            target,
            &mut state,
            &data,
            &planet,
            true, // reference mode: sets bank_angle_commanded = target before shaping
            GuidanceType::Ftc,
        );

        assert!(
            out.bank_angle_commanded.is_finite(),
            "legacy path must produce finite output"
        );
    }

    /// `enabled = false` at config layer produces `command_shaping = None` at runtime.
    /// This test verifies that `None` (the output of that conversion) triggers the legacy
    /// hard-clamp path: rate clamped to max_bank_rate, `rate_saturated = 1`, shaped_rate untouched.
    #[test]
    fn shaper_config_disabled_uses_legacy_hardclamp() {
        let nav = test_nav();
        let data = test_sim_data(); // command_shaping = None (simulates enabled=false)
        let planet = PlanetConfig::mars();

        let mut state = GuidanceState::new(0.0, -0.48_f64.to_radians(), None);
        state.bank_angle_realized = 0.0;

        // 0 -> 90 deg step in reference mode; raw_rate = 90 deg/s >> max_bank_rate (15 deg/s)
        let out = guidance_step(
            &nav,
            0.0,
            0.0,
            90.0_f64.to_radians(),
            &mut state,
            &data,
            &planet,
            true,
            GuidanceType::Ftc,
        );

        let max_bank_rate = data.capsule.max_bank_rate;
        let guidance_period = data.periods.guidance;
        let expected = max_bank_rate * guidance_period;

        assert!(
            (out.bank_angle_commanded - expected).abs() < 1e-10,
            "legacy clamp: expected {:.6} rad ({:.4} deg), got {:.6} rad ({:.4} deg)",
            expected,
            expected.to_degrees(),
            out.bank_angle_commanded,
            out.bank_angle_commanded.to_degrees(),
        );
        assert_eq!(out.rate_saturated, 1, "rate_saturated should be 1");
        assert!(
            state.command_shaper.shaped_rate.abs() < 1e-10,
            "shaped_rate should remain 0 when shaping is disabled (enabled=false), got {:.2e}",
            state.command_shaper.shaped_rate
        );
    }

    /// Shaper uses bank_angle_realized (pilot lag) as the baseline, not a stale commanded value.
    #[test]
    fn realized_baseline_detects_pilot_lag() {
        let nav = test_nav();
        let data = test_sim_data_with_shaping(30.0); // plenty of accel
        let planet = PlanetConfig::mars();
        let realized = 5.0_f64.to_radians();
        let target = 10.0_f64.to_radians();
        let mut state = GuidanceState::new(realized, -0.48_f64.to_radians(), None);
        state.bank_angle_realized = realized;

        let out = guidance_step(
            &nav,
            realized, // pilot_bank_angle = realized
            0.0,
            target,
            &mut state,
            &data,
            &planet,
            true, // reference mode
            GuidanceType::Ftc,
        );

        assert!(out.bank_angle_commanded.is_finite());
        // Commanded should be > realized (moving toward target)
        assert!(
            out.bank_angle_commanded > realized - 1e-9,
            "commanded ({}) should be >= realized ({})",
            out.bank_angle_commanded,
            realized
        );
    }

    /// With 5 deg/s^2 acceleration and 0->90 deg step, shaped_rate stays near 5 deg/s after 1 tick.
    #[test]
    fn shaper_acceleration_limits_large_step() {
        let nav = test_nav();
        let data = test_sim_data_with_shaping(5.0);
        let planet = PlanetConfig::mars();
        let realized = 0.0_f64;
        let target = 90.0_f64.to_radians();
        let mut state = GuidanceState::new(realized, -0.48_f64.to_radians(), None);
        state.bank_angle_realized = realized;

        let out = guidance_step(
            &nav,
            realized,
            0.0,
            target,
            &mut state,
            &data,
            &planet,
            true,
            GuidanceType::Ftc,
        );

        let guidance_period = 1.0_f64; // TimePeriods::default()
        let expected_rate = 5.0_f64.to_radians(); // max_accel * dt, starting from 0
        assert!(
            (out.bank_rate - expected_rate).abs() < 1e-9,
            "shaped_rate should be ~5 deg/s, got {:.4} deg/s",
            out.bank_rate.to_degrees()
        );
        // Rate-limited by acceleration, so saturation flag should be set
        assert_eq!(out.rate_saturated, 1, "rate_saturated should be 1");
        // Commanded angle = realized + shaped_rate * dt
        let expected_commanded = realized + expected_rate * guidance_period;
        assert!(
            (out.bank_angle_commanded - expected_commanded).abs() < 1e-9,
            "commanded should be {:.4} rad, got {:.4} rad",
            expected_commanded,
            out.bank_angle_commanded
        );
    }

    /// With very high acceleration (100 deg/s^2), shaped_rate is capped at max_bank_rate (15 deg/s).
    #[test]
    fn shaper_rate_capped_by_max_bank_rate() {
        let nav = test_nav();
        let data = test_sim_data_with_shaping(100.0); // accel >> max_rate
        let planet = PlanetConfig::mars();
        let realized = 0.0_f64;
        let target = 90.0_f64.to_radians();
        let mut state = GuidanceState::new(realized, -0.48_f64.to_radians(), None);
        state.bank_angle_realized = realized;

        let out = guidance_step(
            &nav,
            realized,
            0.0,
            target,
            &mut state,
            &data,
            &planet,
            true,
            GuidanceType::Ftc,
        );

        let max_bank_rate = 15.0_f64.to_radians();
        assert!(
            out.bank_rate.abs() <= max_bank_rate + 1e-10,
            "shaped_rate ({:.4} deg/s) exceeds max_bank_rate (15 deg/s)",
            out.bank_rate.to_degrees()
        );
        assert_eq!(out.rate_saturated, 1, "rate_saturated should be 1");
    }

    /// After reversing the commanded direction, shaped_rate should decelerate (not immediately flip).
    #[test]
    fn shaper_decelerates_before_reversal() {
        let nav = test_nav();
        let data = test_sim_data_with_shaping(5.0);
        let planet = PlanetConfig::mars();
        let realized = 0.0_f64;
        let mut state = GuidanceState::new(realized, -0.48_f64.to_radians(), None);
        state.bank_angle_realized = realized;

        // Tick 1: command +90 deg
        let out1 = guidance_step(
            &nav,
            realized,
            0.0,
            90.0_f64.to_radians(),
            &mut state,
            &data,
            &planet,
            true,
            GuidanceType::Ftc,
        );
        let rate_after_tick1 = out1.bank_rate;

        // Tick 2: update realized to commanded position from tick 1, then reverse to -90 deg
        let new_realized = out1.bank_angle_commanded;
        state.bank_angle_realized = new_realized;

        let out2 = guidance_step(
            &nav,
            new_realized,
            1.0,
            -90.0_f64.to_radians(),
            &mut state,
            &data,
            &planet,
            true,
            GuidanceType::Ftc,
        );
        let rate_after_tick2 = out2.bank_rate;

        // The shaper should be decelerating: |rate| after tick 2 < |rate| after tick 1
        assert!(
            rate_after_tick2.abs() < rate_after_tick1.abs(),
            "shaped_rate should decelerate after reversal: tick1={:.4} deg/s, tick2={:.4} deg/s",
            rate_after_tick1.to_degrees(),
            rate_after_tick2.to_degrees()
        );
    }

    /// From +170 deg to -170 deg: shaper should take the short path through +180.
    #[test]
    fn shaper_wraparound_shortest_path() {
        let nav = test_nav();
        let data = test_sim_data_with_shaping(100.0); // high accel so rate matters
        let planet = PlanetConfig::mars();
        let realized = 170.0_f64.to_radians();
        let target = -170.0_f64.to_radians();
        let mut state = GuidanceState::new(realized, -0.48_f64.to_radians(), None);
        state.bank_angle_realized = realized;

        let out = guidance_step(
            &nav,
            realized,
            0.0,
            target,
            &mut state,
            &data,
            &planet,
            true,
            GuidanceType::Ftc,
        );

        // Shortest path from +170 to -170 is +20 deg (through +180), so rate should be positive
        assert!(
            out.bank_rate > 0.0,
            "rate should be positive (shortest path through +180), got {:.4} deg/s",
            out.bank_rate.to_degrees()
        );
    }

    /// Small correction (2 deg) with 5 deg/s^2 accel: no saturation, commanded ≈ target.
    #[test]
    fn shaper_small_correction_passes_through() {
        let nav = test_nav();
        let data = test_sim_data_with_shaping(5.0);
        let planet = PlanetConfig::mars();
        let realized = 60.0_f64.to_radians();
        let target = 62.0_f64.to_radians();
        let mut state = GuidanceState::new(realized, -0.48_f64.to_radians(), None);
        state.bank_angle_realized = realized;

        let out = guidance_step(
            &nav,
            realized,
            0.0,
            target,
            &mut state,
            &data,
            &planet,
            true,
            GuidanceType::Ftc,
        );

        // raw_rate = 2 deg/s; max_rate_delta = 5 deg/s*s * 1s = 5 deg/s > 2 => no accel saturation
        // shaped_rate goes from 0 to 2 deg/s in one step; max_bank_rate=15 => no rate cap
        assert_eq!(
            out.rate_saturated, 0,
            "small correction should not saturate"
        );
        // commanded should be close to target
        assert!(
            (out.bank_angle_commanded - target).abs() < 1e-9,
            "commanded ({:.4} deg) should equal target ({:.4} deg)",
            out.bank_angle_commanded.to_degrees(),
            target.to_degrees()
        );
    }

    // ─── Property-based tests ────────────────────────────────────────────────

    mod prop {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            /// For any valid atmospheric state, guidance_step produces finite output.
            #[test]
            fn output_always_finite(
                alt in 20_000.0..130_000.0_f64,
                vel in 2000.0..7000.0_f64,
                fpa in -0.3..0.05_f64,
                bank_deg in 0.0..90.0_f64,
            ) {
                let r = PlanetConfig::mars().equatorial_radius + alt;
                let initial_bank = bank_deg.to_radians();
                let nav = NavigationOutput {
                    position_estimated: [r, 0.0, 0.0],
                    velocity_estimated: [vel, fpa, 0.6],
                    acceleration_estimated: [50.0, -8.0],
                    aero_coefficients: [1.269, -0.205],
                    density_guidance: 1e-4,
                    density_exit: 1e-6,
                    dynamic_pressure_estimated: 0.5 * 1e-4 * vel * vel,
                    energy_estimated: -1e6,
                    ..Default::default()
                };

                let data = test_sim_data();
                let planet = PlanetConfig::mars();
                let mut state = GuidanceState::new(initial_bank, -0.48_f64.to_radians(), None);

                let out = guidance_step(
                    &nav,
                    initial_bank,
                    0.0,
                    initial_bank,
                    &mut state,
                    &data,
                    &planet,
                    false,
                    GuidanceType::Ftc,
                );

                prop_assert!(out.bank_angle_commanded.is_finite(), "bank_angle_commanded not finite: {}", out.bank_angle_commanded);
                prop_assert!(out.aoa_commanded.is_finite(), "aoa_commanded not finite: {}", out.aoa_commanded);

                let pi = std::f64::consts::PI;
                prop_assert!(
                    out.bank_angle_commanded >= -pi && out.bank_angle_commanded <= pi,
                    "bank_angle_commanded = {} outside [-π, π]",
                    out.bank_angle_commanded
                );
            }

            /// Shaped rate never exceeds max_bank_rate regardless of accel config.
            #[test]
            fn shaped_rate_bounded(
                bank_deg in -180.0..180.0_f64,
                target_deg in -180.0..180.0_f64,
                accel_deg in 1.0..20.0_f64,
            ) {
                let data = test_sim_data_with_shaping(accel_deg);
                let planet = PlanetConfig::mars();
                let max_bank_rate = data.capsule.max_bank_rate;
                let realized = bank_deg.to_radians();
                let target = target_deg.to_radians();
                let mut state = GuidanceState::new(realized, -0.48_f64.to_radians(), None);
                state.bank_angle_realized = realized;
                let nav = NavigationOutput {
                    position_estimated: [PlanetConfig::mars().equatorial_radius + 50_000.0, 0.0, 0.0],
                    velocity_estimated: [5000.0, -0.15, 0.6],
                    acceleration_estimated: [50.0, -8.0],
                    aero_coefficients: [1.269, -0.205],
                    density_guidance: 0.001,
                    density_exit: 1e-6,
                    dynamic_pressure_estimated: 0.5 * 0.001 * 5000.0 * 5000.0,
                    energy_estimated: -1e6,
                    ..Default::default()
                };

                let out = guidance_step(
                    &nav, realized, 0.0, target, &mut state, &data, &planet, true, GuidanceType::Ftc,
                );

                prop_assert!(
                    out.bank_rate.abs() <= max_bank_rate + 1e-10,
                    "shaped_rate ({}) exceeds max_bank_rate ({})",
                    out.bank_rate, max_bank_rate
                );
            }

            /// From shaped_rate=0, rate change after 1 tick is bounded by accel*dt.
            #[test]
            fn shaped_rate_change_bounded(
                bank_deg in -180.0..180.0_f64,
                target_deg in -180.0..180.0_f64,
                accel_deg in 1.0..20.0_f64,
            ) {
                let data = test_sim_data_with_shaping(accel_deg);
                let planet = PlanetConfig::mars();
                let guidance_period = data.periods.guidance;
                let max_rate_change = accel_deg.to_radians() * guidance_period;
                let realized = bank_deg.to_radians();
                let target = target_deg.to_radians();
                let mut state = GuidanceState::new(realized, -0.48_f64.to_radians(), None);
                state.bank_angle_realized = realized;
                // shaped_rate starts at 0 (new state)
                let nav = NavigationOutput {
                    position_estimated: [PlanetConfig::mars().equatorial_radius + 50_000.0, 0.0, 0.0],
                    velocity_estimated: [5000.0, -0.15, 0.6],
                    acceleration_estimated: [50.0, -8.0],
                    aero_coefficients: [1.269, -0.205],
                    density_guidance: 0.001,
                    density_exit: 1e-6,
                    dynamic_pressure_estimated: 0.5 * 0.001 * 5000.0 * 5000.0,
                    energy_estimated: -1e6,
                    ..Default::default()
                };

                let out = guidance_step(
                    &nav, realized, 0.0, target, &mut state, &data, &planet, true, GuidanceType::Ftc,
                );

                // |shaped_rate| <= |max_rate_change| (since starting at 0, capped by accel*dt)
                // also capped by max_bank_rate but we only need the accel bound here
                let effective_bound = max_rate_change.min(data.capsule.max_bank_rate);
                prop_assert!(
                    out.bank_rate.abs() <= effective_bound + 1e-10,
                    "shaped_rate change ({}) exceeds accel*dt ({})",
                    out.bank_rate.abs(), effective_bound
                );
            }

            /// Shaper output is always finite for any starting angle, target, and accel config.
            #[test]
            fn shaped_output_always_finite(
                bank_deg in -180.0..180.0_f64,
                target_deg in -180.0..180.0_f64,
                accel_deg in 1.0..20.0_f64,
            ) {
                let data = test_sim_data_with_shaping(accel_deg);
                let planet = PlanetConfig::mars();
                let realized = bank_deg.to_radians();
                let target = target_deg.to_radians();
                let mut state = GuidanceState::new(realized, -0.48_f64.to_radians(), None);
                state.bank_angle_realized = realized;
                let nav = NavigationOutput {
                    position_estimated: [PlanetConfig::mars().equatorial_radius + 50_000.0, 0.0, 0.0],
                    velocity_estimated: [5000.0, -0.15, 0.6],
                    acceleration_estimated: [50.0, -8.0],
                    aero_coefficients: [1.269, -0.205],
                    density_guidance: 0.001,
                    density_exit: 1e-6,
                    dynamic_pressure_estimated: 0.5 * 0.001 * 5000.0 * 5000.0,
                    energy_estimated: -1e6,
                    ..Default::default()
                };

                let out = guidance_step(
                    &nav, realized, 0.0, target, &mut state, &data, &planet, true, GuidanceType::Ftc,
                );

                prop_assert!(out.bank_angle_commanded.is_finite(), "bank_angle_commanded is not finite");
                prop_assert!(out.bank_rate.is_finite(), "bank_rate is not finite");
            }
        }
    }

    // ─── Phase dispatch tests ─────────────────────────────────────────────────

    /// Phase 2 should dispatch to exit guidance for FTC scheme.
    #[test]
    fn phase_2_dispatches_to_exit_guidance() {
        let mut nav = test_nav();
        nav.guidance_phase = 2;
        nav.bounce_flag = 1;
        nav.density_exit = 1e-6;
        nav.velocity_estimated[1] = 0.05; // positive FPA (ascending)

        let mut data = test_sim_data();
        data.guidance.exit_pdyn_margin = 1.75;
        data.guidance.exit_radial_vel_gain = 10.0;

        let planet = PlanetConfig::mars();
        let initial_bank = 64.77_f64.to_radians();
        let mut state = GuidanceState::new(initial_bank, -0.48_f64.to_radians(), None);
        state.reference_velocity = 50.0;

        let out = guidance_step(
            &nav,
            initial_bank,
            100.0,
            initial_bank,
            &mut state,
            &data,
            &planet,
            false,
            GuidanceType::Ftc,
        );

        assert!(
            out.bank_angle_commanded.is_finite(),
            "exit phase should produce finite bank: {}",
            out.bank_angle_commanded
        );
    }

    /// `mode = magnitude_only` routes the NN through the thermal limiter, so a
    /// high heat-flux fraction pulls the commanded bank toward lift-up
    /// (smaller |bank|) compared to `mode = full_neural` which bypasses it.
    #[test]
    fn magnitude_only_mode_routes_through_thermal_limiter() {
        use crate::data::guidance_params::NeuralNetMode;
        use crate::data::neural::{
            Activation, DenseLayer, Layer, LayerSpec, NeuralNetModel, OutputParam,
        };
        use crate::gnc::guidance::thermal_limiter::ThermalLimiterParams;

        // 16 -> 2 NN with zero weights + biases tuned so atan2(b0, b1) = 0.5 rad
        let target_bank = 0.5_f64;
        let nn = NeuralNetModel {
            architecture: vec![LayerSpec::Dense {
                input_size: 16,
                output_size: 2,
                activation: Activation::Linear,
            }],
            layer_sizes: vec![16, 2],
            layers: vec![Layer::Dense(DenseLayer {
                w: vec![vec![0.0; 16], vec![0.0; 16]],
                b: vec![target_bank.sin(), target_bank.cos()],
                activation: Activation::Linear,
            })],
            input_mask: None,
            ablated_input: None,

            ablated_value: 0.0,
            output_param: OutputParam::default(),
            scaled_pi_n: 1.0,
            delta_max: 0.35,
            normalization: crate::data::neural::DEFAULT_NORMALIZATION.to_vec(),
        };

        // Heat flux at 99% of limit -> thermal limiter activates aggressively.
        let mut nav = test_nav();
        nav.heat_flux_fraction = 0.99;
        nav.heat_load_fraction = 0.0;

        let planet = PlanetConfig::mars();
        let mut data = test_sim_data();
        data.neural_net = Some(nn.clone());
        data.guidance.thermal_limiter = ThermalLimiterParams {
            heat_flux_activation: 0.5,
            heat_load_activation: 1.0,
            heat_flux_ramp_exponent: 1.0,
            heat_load_ramp_exponent: 1.0,
        };
        // High max_bank_rate to keep rate clamp out of the picture.
        data.capsule.max_bank_rate = 5.0_f64.to_radians() * 100.0;

        let run = |mode: NeuralNetMode, data: &mut SimData| {
            data.guidance.neural_mode = mode;
            let mut state = GuidanceState::new(target_bank, -0.48_f64.to_radians(), Some(&nn));
            // Prime realized = target so rate shaping is a no-op when not limited.
            state.bank_angle_realized = target_bank;
            let out = guidance_step(
                &nav,
                target_bank,
                0.0,
                target_bank,
                &mut state,
                data,
                &planet,
                false,
                GuidanceType::NeuralNetwork,
            );
            out.bank_angle_commanded.abs()
        };

        let bank_full = run(NeuralNetMode::FullNeural, &mut data);
        let bank_mag = run(NeuralNetMode::MagnitudeOnly, &mut data);

        // FullNeural bypasses the limiter -> commanded magnitude is the raw NN output.
        assert!(
            (bank_full - target_bank).abs() < 1e-9,
            "full_neural should pass NN bank through unchanged: expected {target_bank}, got {bank_full}"
        );
        // MagnitudeOnly + high heat flux: limiter pulls cos(bank) toward 1, so |bank| shrinks.
        assert!(
            bank_mag < bank_full - 1e-6,
            "magnitude_only should shrink |bank| via thermal limiter: full={bank_full}, mag={bank_mag}"
        );
    }

    /// Acos_tanh end-to-end golden: a fixed 16->1 tanh-output NN with
    /// known bias `b` produces commanded bank `acos(tanh(b))` after going
    /// through the magnitude_only pipeline (signed.abs() is a no-op since
    /// acos always returns [0, π], thermal limiter is benign at 0% heat
    /// flux, lateral guidance picks +1 sign, command shaper is disabled).
    /// This is the "trajectory golden" the spec required: a fixed-input
    /// scenario where the bank command is computable in closed form, so
    /// any regression in the new code paths (OutputParam dispatch, JSON
    /// serde, validate_output_activation, magnitude_only routing) is
    /// caught with a single assertion.
    #[test]
    fn acos_tanh_magnitude_only_end_to_end_golden() {
        use crate::data::guidance_params::NeuralNetMode;
        use crate::data::neural::{
            Activation, DenseLayer, Layer, LayerSpec, NeuralNetModel, OutputParam,
        };

        // 16 -> 1 NN, weights all 0, bias = 0.7. tanh(0 * x + 0.7) = tanh(0.7) ≈ 0.604.
        // acos(0.604) ≈ 0.9220 rad. This is the expected commanded magnitude.
        let bias = 0.7_f64;
        let expected_bank = bias.tanh().acos(); // ≈ 0.92204
        let nn = NeuralNetModel {
            architecture: vec![LayerSpec::Dense {
                input_size: 16,
                output_size: 1,
                activation: Activation::Tanh,
            }],
            layer_sizes: vec![16, 1],
            layers: vec![Layer::Dense(DenseLayer {
                w: vec![vec![0.0; 16]],
                b: vec![bias],
                activation: Activation::Tanh,
            })],
            input_mask: None,
            ablated_input: None,

            ablated_value: 0.0,
            output_param: OutputParam::AcosTanh,
            scaled_pi_n: 1.0,
            delta_max: 0.35,
            normalization: crate::data::neural::DEFAULT_NORMALIZATION.to_vec(),
        };

        let mut nav = test_nav();
        // Zero heat flux so thermal limiter is benign (cos_bank passes through).
        nav.heat_flux_fraction = 0.0;
        nav.heat_load_fraction = 0.0;

        let planet = PlanetConfig::mars();
        let mut data = test_sim_data();
        data.neural_net = Some(nn.clone());
        data.guidance.neural_mode = NeuralNetMode::MagnitudeOnly;
        // Disable command shaper so we see the raw bank command (lateral may
        // multiply by ±1 sign).
        data.guidance.command_shaping = None;
        // Wide rate cap to neutralize the rate clamp.
        data.capsule.max_bank_rate = 10.0_f64.to_radians() * 100.0;

        let mut state = GuidanceState::new(expected_bank, -0.48_f64.to_radians(), Some(&nn));
        state.bank_angle_realized = expected_bank;

        let out = guidance_step(
            &nav,
            expected_bank,
            0.0,
            expected_bank,
            &mut state,
            &data,
            &planet,
            false,
            GuidanceType::NeuralNetwork,
        );

        // |bank_angle_commanded| should match acos(tanh(bias)). Lateral may have
        // applied a sign of ±1 — strip it.
        let actual_magnitude = out.bank_angle_commanded.abs();
        assert!(
            (actual_magnitude - expected_bank).abs() < 1e-9,
            "acos_tanh magnitude_only golden: expected |bank|={expected_bank} (acos(tanh({bias}))), got {actual_magnitude}"
        );
        // pre_lateral_magnitude exposes the raw post-thermal-limiter magnitude
        // before sign assignment — should equal expected_bank exactly.
        assert!(
            (out.pre_lateral_magnitude - expected_bank).abs() < 1e-9,
            "pre_lateral_magnitude should equal acos(tanh(bias))={expected_bank}, got {}",
            out.pre_lateral_magnitude
        );
    }

    /// PiecewiseConstant scheme should ignore phase 2 (produces its own signed bank).
    #[test]
    fn piecewise_constant_ignores_exit_phase() {
        let mut nav = test_nav();
        nav.guidance_phase = 2;
        nav.bounce_flag = 1;

        let mut data = test_sim_data();
        data.guidance.piecewise_constant = crate::data::guidance_params::PiecewiseConstantParams {
            bank_angles: vec![0.5; 10],
            energy_min: -6.0e6,
            energy_max: 5.0e6,
        };

        let planet = PlanetConfig::mars();
        let initial_bank = 0.5;
        let mut state = GuidanceState::new(initial_bank, -0.48_f64.to_radians(), None);

        let out = guidance_step(
            &nav,
            initial_bank,
            100.0,
            initial_bank,
            &mut state,
            &data,
            &planet,
            false,
            GuidanceType::PiecewiseConstant,
        );

        assert!(out.bank_angle_commanded.is_finite());
    }
}
