//! CPAG — Convex Predictor-corrector Aerocapture Guidance.
//!
//! Rataczak, McMahon & Boyd (JGCD 2025, doi:10.2514/1.G008685; full math in
//! Rataczak's CU Boulder thesis Ch. 6 + App. B). Rust port of the validated
//! Stage C0 Python prototype (`aerocapture/cpag/scp.py`; findings and the
//! solver spike in `docs/plans/2026-07-16-cpag-c0-findings.md`).
//!
//! Per replan (SCP, box-trust QP variant — the C0 pick):
//! 1. PREDICT: integrate the nonlinear 8-state model (the FNPAG EOM with
//!    bank-as-state, lateral lift, and a heat-load state) from the nav state
//!    under the stored bank-RATE profile (ZOH segments) to atmospheric exit,
//!    crash, or horizon; the reference is dynamically feasible, so the delta
//!    formulation needs no virtual control, and the onboard atmosphere is
//!    scaled by the nav density factor.
//! 2. LINEARIZE: central-difference Jacobians of each segment map.
//! 3. CORRECT: one Clarabel QP for the rate correction du — control-effort
//!    objective, exact L1 penalties on the terminal energy-based apoapsis
//!    surrogate eps (smooth through v_esc) and terminal cos-inclination, a
//!    deadbanded intermediate inclination corridor, L1-slacked path rows
//!    (heat flux / g-load; pdyn off by default — the mission nominal exceeds
//!    the config value), terminal heat load, hard |sigma| <= sigma_max node
//!    box (the optimizer otherwise winds the bank through merit-free full
//!    turns), bank-rate bounds, and a hard |du| trust box.
//! 4. Greedy merit accept/reject with trust shrink/grow; the merit is
//!    grid-size-invariant (terminal values + node means/peaks, never sums)
//!    with a crash-tier offset so any exit outranks any crash.
//!
//! The solve returns a SIGNED bank profile — no lateral logic anywhere;
//! reversals emerge from the inclination terms. Between replans the commanded
//! bank plays back the planned profile (FNPAG's hold-throttle pattern, with
//! the profile as the held object); cold starts seed from a constant-bank
//! grid (FNPAG's monotone apoapsis-vs-bank bracket as an initializer).

use clarabel::algebra::CscMatrix;
use clarabel::solver::{DefaultSettings, DefaultSolver, IPSolver, SolverStatus, SupportedConeT};

use crate::config::PlanetConfig;
use crate::data::SimData;
use crate::data::guidance_params::CpagParams;
use crate::gnc::navigation::coordinates::geodetic_from_spherical;
use crate::gnc::navigation::estimator::NavigationOutput;
use crate::physics::gravity;
use crate::simulation::sim_types::G0;

const HEAT_FLUX_VEL_EXPONENT: f64 = 3.05;

// State indices: [r, lon, lat, v, gamma, psi, sigma, Q(J/m^2)]
const IR: usize = 0;
const ILAT: usize = 2;
const IV: usize = 3;
const IGAMMA: usize = 4;
const IPSI: usize = 5;
const ISIGMA: usize = 6;
const IQ: usize = 7;
const NS: usize = 8;

/// Column scaling (thesis Table 6.1 adapted; C0 values, Q in J/m^2 here).
const STATE_SCALE: [f64; NS] = [1e3, 3.5e-3, 3.5e-3, 50.0, 3.5e-3, 3.5e-3, 0.26, 1e6];
const U_SCALE: f64 = 0.05; // rad/s
const EPS_SCALE: f64 = 1e6; // eps rows in MJ/kg

/// Central-difference perturbation floors per state + control.
const FD_EPS: [f64; NS] = [1.0, 1e-8, 1e-8, 1e-4, 1e-8, 1e-8, 1e-7, 1.0];
const FD_EPS_U: f64 = 1e-7;

/// Any exit outranks any crash; eps still ranks within a tier.
const CRASH_MERIT_OFFSET: f64 = 1e4;

/// Constant-bank grid seeding targets (deg), C0's cold-start initializer.
const SEED_BANKS_DEG: [f64; 6] = [0.0, 45.0, 75.0, 105.0, 135.0, 180.0];

type State8 = [f64; 8];

/// CPAG persistent state: the planned bank-rate profile and its anchor.
#[derive(Debug, Clone)]
pub struct CpagState {
    /// Planned bank-rate segments (rad/s, ZOH on the absolute grid from `anchor_time`)
    pub u_profile: Vec<f64>,
    /// Signed bank at `anchor_time` (rad)
    pub sigma_anchor: f64,
    /// Sim time the current profile is anchored at (s)
    pub anchor_time: f64,
    /// Sim time of the last replan (throttle bookkeeping)
    pub last_replan_time: f64,
    /// Whether a real (in-atmosphere) replan has run yet
    pub initialized: bool,
    /// Diagnostics: SCP iterations of the last replan
    pub last_iters: usize,
    /// Diagnostics: terminal eps residual of the last replan (MJ/kg)
    pub last_eps_mj: f64,
}

impl CpagState {
    pub fn new(initial_bank: f64) -> Self {
        Self {
            u_profile: Vec::new(),
            sigma_anchor: initial_bank,
            anchor_time: 0.0,
            last_replan_time: f64::NEG_INFINITY,
            initialized: false,
            last_iters: 0,
            last_eps_mj: f64::NAN,
        }
    }

    /// Commanded signed bank at time `t`: piecewise-linear playback of the plan.
    fn sigma_at(&self, t: f64, params: &CpagParams) -> f64 {
        let sigma_max = params.sigma_max_deg.to_radians();
        let mut sigma = self.sigma_anchor;
        let mut remaining = (t - self.anchor_time).max(0.0);
        for &u in &self.u_profile {
            if remaining <= 0.0 {
                break;
            }
            let dt = remaining.min(params.seg_dt);
            sigma += u * dt;
            remaining -= dt;
        }
        sigma.clamp(-sigma_max, sigma_max)
    }
}

/// Model bundle threaded through the predictor (mirrors C0's CpagModel).
struct Model<'a> {
    data: &'a SimData,
    planet: &'a PlanetConfig,
    cx: f64,
    cz: f64,
    density_factor: f64,
    target_apo_radius: f64,
    exit_alt: f64,
}

impl<'a> Model<'a> {
    fn new(data: &'a SimData, planet: &'a PlanetConfig, density_factor: f64) -> Self {
        Self {
            data,
            planet,
            // Stability-axis coefficients at the entry AoA, RAW sign (the plant's
            // convention in runner.rs; cz > 0 = lift up for MSR).
            cx: data.aero.interpolate_cx(data.entry.initial_aoa),
            cz: data.aero.interpolate_cz(data.entry.initial_aoa),
            density_factor,
            target_apo_radius: planet.equatorial_radius + data.target_orbit.apoapsis,
            exit_alt: data.final_conditions.altitude,
        }
    }

    /// Dispersion-scaled onboard density at the state's geodetic altitude.
    fn density(&self, x: &State8) -> f64 {
        let (alt, _) = geodetic_from_spherical(x[IR], x[1], x[ILAT], self.planet);
        self.density_factor
            * self
                .data
                .atmosphere_onboard
                .density_at(alt, &self.data.atmosphere)
    }
}

// ── Dynamics (runner.rs EOM + bank-as-state + heat-load state) ──────────────

fn eom(x: &State8, u: f64, m: &Model) -> State8 {
    let r = x[IR];
    let lat = x[ILAT];
    let v = x[IV];
    let gamma = x[IGAMMA];
    let psi = x[IPSI];
    let sigma = x[ISIGMA];

    let rho = m.density(x);
    let veh = &m.data.capsule;
    let aero_factor = rho * veh.reference_area / (2.0 * veh.mass);
    let drag = aero_factor * m.cx * v * v;
    let lift = aero_factor * m.cz * v * v;

    let (gravtl, gravtr) = gravity::gravity(r, lat, m.planet);

    let cos_gamma = gamma.cos();
    let sin_gamma = gamma.sin();
    let cos_psi = psi.cos();
    let sin_psi = psi.sin();
    let cos_lat = lat.cos();
    let sin_lat = lat.sin();
    let tan_gamma = sin_gamma / cos_gamma;
    let tan_lat = sin_lat / cos_lat;
    let omega = m.planet.omega;

    let dr = v * sin_gamma;
    let dlon = v * cos_gamma * sin_psi / (r * cos_lat);
    let dlat = v * cos_gamma * cos_psi / r;

    let dv = -drag - gravtr * sin_gamma - gravtl * cos_gamma * cos_psi
        + omega * omega * r * cos_lat * (cos_lat * sin_gamma - sin_lat * cos_gamma * cos_psi);

    let dgamma = if v.abs() > 1.0 {
        (lift * sigma.cos() / v) + (v * cos_gamma / r)
            - ((gravtr * cos_gamma - gravtl * sin_gamma * cos_psi) / v)
            + (2.0 * omega * sin_psi * cos_lat)
            + (omega * omega * r * cos_lat * (sin_lat * sin_gamma * cos_psi + cos_lat * cos_gamma)
                / v)
    } else {
        0.0
    };

    let dpsi = if v.abs() > 1.0 && cos_gamma.abs() > 1e-10 {
        (lift * sigma.sin() / (v * cos_gamma))
            + (v * cos_gamma * sin_psi * tan_lat / r)
            + (2.0 * omega * (sin_lat - cos_psi * cos_lat * tan_gamma))
            + (gravtl * sin_psi / (v * cos_gamma))
            + (omega * omega * r * cos_lat * sin_lat * sin_psi / (v * cos_gamma))
    } else {
        0.0
    };

    let dq = veh.cq * rho.sqrt() * v.powf(HEAT_FLUX_VEL_EXPONENT); // W/m^2

    [dr, dlon, dlat, dv, dgamma, dpsi, u, dq]
}

fn rk4_step(x: &State8, u: f64, dt: f64, m: &Model) -> State8 {
    let k1 = eom(x, u, m);
    let k2 = eom(&add_scaled(x, &k1, 0.5 * dt), u, m);
    let k3 = eom(&add_scaled(x, &k2, 0.5 * dt), u, m);
    let k4 = eom(&add_scaled(x, &k3, dt), u, m);
    let mut out = *x;
    for i in 0..NS {
        out[i] += dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
    out
}

fn add_scaled(x: &State8, k: &State8, h: f64) -> State8 {
    let mut out = *x;
    for i in 0..NS {
        out[i] += h * k[i];
    }
    out
}

// ── Orbital quantities (frame-free, inertial) ───────────────────────────────

/// (v_abs^2, v_radial, v_east_abs, v_north) from the relative spherical state.
fn inertial_pieces(x: &State8, planet: &PlanetConfig) -> (f64, f64, f64, f64) {
    let we = planet.omega * x[IR] * x[ILAT].cos();
    let v_east = x[IV] * x[IGAMMA].cos() * x[IPSI].sin() + we;
    let v_north = x[IV] * x[IGAMMA].cos() * x[IPSI].cos();
    let v_radial = x[IV] * x[IGAMMA].sin();
    (
        v_east * v_east + v_north * v_north + v_radial * v_radial,
        v_radial,
        v_east,
        v_north,
    )
}

fn inertial_energy(x: &State8, planet: &PlanetConfig) -> f64 {
    let (v_abs2, _, _, _) = inertial_pieces(x, planet);
    0.5 * v_abs2 - planet.mu / x[IR]
}

fn angular_momentum_sq(x: &State8, planet: &PlanetConfig) -> f64 {
    let (v_abs2, v_radial, _, _) = inertial_pieces(x, planet);
    x[IR] * x[IR] * (v_abs2 - v_radial * v_radial).max(0.0)
}

/// CPAG's energy-based apoapsis targeting residual (J/kg), Keplerian form:
/// zero exactly when the osculating apoapsis radius equals the target; smooth
/// and well-defined for ALL orbits including hyperbolic (the paper's fix for
/// the apoapsis-Jacobian singularity at escape velocity).
fn eps_apoapsis(x: &State8, m: &Model) -> f64 {
    let ra = m.target_apo_radius;
    inertial_energy(x, m.planet) + m.planet.mu / ra
        - angular_momentum_sq(x, m.planet) / (2.0 * ra * ra)
}

/// Osculating apoapsis radius (m); unbound orbits get the FNPAG sentinel.
/// Test oracle for the eps identity — the SCP terminal signal is `eps_apoapsis`.
#[cfg(test)]
fn apoapsis_radius(x: &State8, planet: &PlanetConfig) -> f64 {
    let energy = inertial_energy(x, planet);
    let sma = -planet.mu / (2.0 * energy);
    let ecc = (1.0 - angular_momentum_sq(x, planet) / (planet.mu * sma))
        .abs()
        .sqrt();
    let r_apo = sma * (1.0 + ecc);
    if energy >= 0.0 || ecc >= 1.0 || !r_apo.is_finite() {
        1e9
    } else {
        r_apo
    }
}

/// cos(i) of the osculating orbit: h_z / |h| (frame-free, inertial).
fn cos_inclination(x: &State8, planet: &PlanetConfig) -> f64 {
    let (_, _, v_east, v_north) = inertial_pieces(x, planet);
    let h_horiz = (v_east * v_east + v_north * v_north).max(1e-30).sqrt();
    x[ILAT].cos() * v_east / h_horiz
}

/// (heat_flux W/m^2, g_load g, pdyn Pa) at the state, sim conventions.
fn path_quantities(x: &State8, m: &Model) -> (f64, f64, f64) {
    let rho = m.density(x);
    let v = x[IV];
    let veh = &m.data.capsule;
    let heat_flux = veh.cq * rho.sqrt() * v.powf(HEAT_FLUX_VEL_EXPONENT);
    let aero_accel = rho * veh.reference_area * v * v / (2.0 * veh.mass);
    let g_load = aero_accel * (m.cx * m.cx + m.cz * m.cz).sqrt() / G0;
    (heat_flux, g_load, 0.5 * rho * v * v)
}

// ── Predictor: shoot the profile to exit / crash / horizon ─────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Event {
    Exit,
    Crash,
    Horizon,
}

struct Shoot {
    x_nodes: Vec<State8>,
    u_nodes: Vec<f64>,
    dts: Vec<f64>,
    event: Event,
}

/// Propagate segment-wise with substep exit/crash detection (spherical
/// altitude, the FNPAG predictor convention; event time by linear
/// interpolation between substeps, terminal state by a partial RK4 step).
fn shoot_profile(x0: &State8, u_profile: &[f64], m: &Model, p: &CpagParams) -> Shoot {
    let req = m.planet.equatorial_radius;
    let dt_sub = p.seg_dt / p.n_sub as f64;
    let mut x_nodes = vec![*x0];
    let mut u_nodes: Vec<f64> = Vec::new();
    let mut dts: Vec<f64> = Vec::new();
    let mut x = *x0;
    let mut t = 0.0_f64;
    for &u in u_profile {
        let node_start_t = t;
        for _ in 0..p.n_sub {
            let x_new = rk4_step(&x, u, dt_sub, m);
            let alt_prev = x[IR] - req;
            let alt_new = x_new[IR] - req;
            let crash = alt_new <= 0.0;
            let exiting = alt_new >= m.exit_alt && x_new[IGAMMA].sin() > 0.0;
            if crash || exiting {
                let target = if crash { 0.0 } else { m.exit_alt };
                let denom = alt_new - alt_prev;
                let frac = if denom.abs() < 1e-12 {
                    0.5
                } else {
                    ((target - alt_prev) / denom).clamp(0.0, 1.0)
                };
                let dt_part = frac * dt_sub;
                let x_event = if dt_part > 1e-9 {
                    rk4_step(&x, u, dt_part, m)
                } else {
                    x
                };
                let dt_last = t + dt_part - node_start_t;
                if dt_last > 1e-6 {
                    x_nodes.push(x_event);
                    u_nodes.push(u);
                    dts.push(dt_last);
                }
                return Shoot {
                    x_nodes,
                    u_nodes,
                    dts,
                    event: if crash { Event::Crash } else { Event::Exit },
                };
            }
            x = x_new;
            t += dt_sub;
        }
        x_nodes.push(x);
        u_nodes.push(u);
        dts.push(t - node_start_t);
    }
    Shoot {
        x_nodes,
        u_nodes,
        dts,
        event: Event::Horizon,
    }
}

/// Bank-rate profile rolling sigma to `target_sigma` at max rate, then holding.
fn roll_to_bank_profile(
    sigma0: f64,
    target_sigma: f64,
    n_seg: usize,
    p: &CpagParams,
    max_rate: f64,
) -> Vec<f64> {
    let mut u = vec![0.0; n_seg];
    let mut sig = sigma0;
    for slot in u.iter_mut() {
        let step = (target_sigma - sig).clamp(-max_rate * p.seg_dt, max_rate * p.seg_dt);
        *slot = step / p.seg_dt;
        sig += step;
    }
    u
}

// ── Corrector: linearize + canonical QP + Clarabel ──────────────────────────

/// Central-difference Jacobians (A_k, B_k) of each segment map.
fn linearize(shoot: &Shoot, m: &Model, p: &CpagParams) -> (Vec<[[f64; NS]; NS]>, Vec<[f64; NS]>) {
    let m_seg = shoot.dts.len();
    let mut a_mats = Vec::with_capacity(m_seg);
    let mut b_mats = Vec::with_capacity(m_seg);
    for k in 0..m_seg {
        let dt_sub = shoot.dts[k] / p.n_sub as f64;
        let propagate = |x0: &State8, u: f64| -> State8 {
            let mut x = *x0;
            for _ in 0..p.n_sub {
                x = rk4_step(&x, u, dt_sub, m);
            }
            x
        };
        let x0 = &shoot.x_nodes[k];
        let u0 = shoot.u_nodes[k];
        let mut a = [[0.0; NS]; NS];
        for j in 0..NS {
            let mut xp = *x0;
            let mut xm = *x0;
            xp[j] += FD_EPS[j];
            xm[j] -= FD_EPS[j];
            let fp = propagate(&xp, u0);
            let fm = propagate(&xm, u0);
            for i in 0..NS {
                a[i][j] = (fp[i] - fm[i]) / (2.0 * FD_EPS[j]);
            }
        }
        let fp = propagate(x0, u0 + FD_EPS_U);
        let fm = propagate(x0, u0 - FD_EPS_U);
        let mut b = [0.0; NS];
        for i in 0..NS {
            b[i] = (fp[i] - fm[i]) / (2.0 * FD_EPS_U);
        }
        a_mats.push(a);
        b_mats.push(b);
    }
    (a_mats, b_mats)
}

fn fd_state_gradient(f: impl Fn(&State8) -> f64, x: &State8) -> [f64; NS] {
    let mut grad = [0.0; NS];
    for j in 0..NS {
        let mut xp = *x;
        let mut xm = *x;
        xp[j] += FD_EPS[j];
        xm[j] -= FD_EPS[j];
        grad[j] = (f(&xp) - f(&xm)) / (2.0 * FD_EPS[j]);
    }
    grad
}

/// Sparse-triplet accumulator for one stacked constraint block.
struct Rows {
    i: Vec<usize>,
    j: Vec<usize>,
    v: Vec<f64>,
    b: Vec<f64>,
}

impl Rows {
    fn new() -> Self {
        Self {
            i: Vec::new(),
            j: Vec::new(),
            v: Vec::new(),
            b: Vec::new(),
        }
    }

    fn add(&mut self, entries: &[(usize, f64)], rhs: f64) {
        let row = self.b.len();
        for &(c, val) in entries {
            self.i.push(row);
            self.j.push(c);
            self.v.push(val);
        }
        self.b.push(rhs);
    }
}

struct ReplanResult {
    u_profile: Vec<f64>,
    iters: usize,
    eps_mj: f64,
}

struct Merit {
    j: f64,
    eps_mj: f64,
}

/// Grid-size-invariant merit (terminal values + node means/peaks, never node
/// sums) with a crash-tier offset; inclination gated off on crash references.
fn merit(shoot: &Shoot, m: &Model, p: &CpagParams) -> Merit {
    let x_n = shoot.x_nodes.last().unwrap();
    let eps_mj = eps_apoapsis(x_n, m) / EPS_SCALE;
    let mut j = p.alpha3 * eps_mj.abs();

    let n_nodes = shoot.x_nodes.len();
    let limits = &m.data.constraints;
    // A non-positive limit means "no limit configured" (tick.rs convention).
    let enforced = [
        p.enforce_heat_flux && limits.max_heat_flux > 0.0,
        p.enforce_g_load && limits.max_load_factor > 0.0,
        p.enforce_pdyn && limits.max_dynamic_pressure > 0.0,
    ];
    let mut peak = [0.0_f64; 3];
    let mut mean_viol = [0.0_f64; 3];
    for x in &shoot.x_nodes {
        let (hf, gl, pd) = path_quantities(x, m);
        let fr = [
            hf / limits.max_heat_flux,
            gl * G0 / limits.max_load_factor,
            pd / limits.max_dynamic_pressure,
        ];
        for t in 0..3 {
            if enforced[t] {
                peak[t] = peak[t].max(fr[t]);
                mean_viol[t] += (fr[t] - 1.0).max(0.0) / n_nodes as f64;
            }
        }
    }
    for t in 0..3 {
        if enforced[t] {
            j += p.alpha5 * ((peak[t] - 1.0).max(0.0) + mean_viol[t]);
        }
    }
    if limits.max_heat_load > 0.0 {
        j += p.alpha5 * (x_n[IQ] / limits.max_heat_load - 1.0).max(0.0);
    }

    if shoot.event == Event::Crash {
        // Survival tier: inclination is meaningless on a spiraling-down arc.
        j += CRASH_MERIT_OFFSET;
    } else if p.enforce_inclination {
        let ci_t = m.data.target_orbit.inclination.cos();
        j += p.alpha2 * (cos_inclination(x_n, m.planet) - ci_t).abs();
        let m_seg = shoot.dts.len();
        let di_start = (p.di_node_fraction * m_seg as f64).ceil() as usize;
        let deadband = deadband_cos(m, p);
        let mut acc = 0.0;
        let mut n_di = 0usize;
        for x in &shoot.x_nodes[di_start.min(m_seg)..m_seg] {
            acc += ((cos_inclination(x, m.planet) - ci_t).abs() - deadband).max(0.0);
            n_di += 1;
        }
        if n_di > 0 {
            j += p.lambda_di * (acc / n_di as f64) * 10.0;
        }
    }
    Merit { j, eps_mj }
}

/// Deadband on the intermediate inclination rows (cos-space): mid-arc error
/// below the mission tolerance is free.
fn deadband_cos(m: &Model, p: &CpagParams) -> f64 {
    m.data.target_orbit.inclination.sin().abs() * p.di_deadband_deg.to_radians()
}

/// Build + solve the box-trust QP; returns the correction du per segment.
/// Layout: [dx (8(M+1)), du (M), nu_path (n_types(M+1)), nu_hl, s_eps, s_inc, nu_di].
#[allow(clippy::too_many_arguments)]
fn solve_subproblem(
    shoot: &Shoot,
    a_mats: &[[[f64; NS]; NS]],
    b_mats: &[[f64; NS]],
    m: &Model,
    p: &CpagParams,
    trust: f64,
    max_rate: f64,
) -> Option<(Vec<f64>, usize)> {
    let m_seg = shoot.dts.len();
    let n_nodes = m_seg + 1;
    let nx = NS * n_nodes;
    let inc_active = p.enforce_inclination && shoot.event != Event::Crash;
    let di_start = if inc_active {
        (p.di_node_fraction * m_seg as f64).ceil() as usize
    } else {
        n_nodes
    };
    let di_nodes: Vec<usize> = if inc_active {
        (di_start.min(m_seg)..m_seg).collect()
    } else {
        Vec::new()
    };
    let limits_pos = [
        m.data.constraints.max_heat_flux > 0.0,
        m.data.constraints.max_load_factor > 0.0,
        m.data.constraints.max_dynamic_pressure > 0.0,
    ];
    let enforced: Vec<usize> = [p.enforce_heat_flux, p.enforce_g_load, p.enforce_pdyn]
        .iter()
        .enumerate()
        .filter_map(|(t, &on)| if on && limits_pos[t] { Some(t) } else { None })
        .collect();
    let n_types = enforced.len();
    let n_path = n_types * n_nodes;

    let i_u0 = nx;
    let i_np0 = i_u0 + m_seg;
    let i_nhl = i_np0 + n_path;
    let i_seps = i_nhl + 1;
    let i_sinc = i_seps + 1;
    let i_di0 = i_sinc + 1;
    let n = i_di0 + di_nodes.len();

    // ── Equalities: dx_0 = 0; dx_{k+1} = A_k dx_k + B_k du_k ──
    let mut eq = Rows::new();
    for j in 0..NS {
        eq.add(&[(j, 1.0)], 0.0);
    }
    for k in 0..m_seg {
        for i in 0..NS {
            let mut entries: Vec<(usize, f64)> = vec![(NS * (k + 1) + i, 1.0)];
            for (j, &aij) in a_mats[k][i].iter().enumerate() {
                if aij != 0.0 {
                    entries.push((NS * k + j, -aij));
                }
            }
            if b_mats[k][i] != 0.0 {
                entries.push((i_u0 + k, -b_mats[k][i]));
            }
            eq.add(&entries, 0.0);
        }
    }

    // ── Inequalities ──
    let mut ineq = Rows::new();
    let sigma_max = p.sigma_max_deg.to_radians();
    for k in 0..m_seg {
        let u_bar = shoot.u_nodes[k];
        ineq.add(&[(i_u0 + k, 1.0)], max_rate - u_bar);
        ineq.add(&[(i_u0 + k, -1.0)], max_rate + u_bar);
        ineq.add(&[(i_u0 + k, 1.0)], trust * U_SCALE);
        ineq.add(&[(i_u0 + k, -1.0)], trust * U_SCALE);
    }
    for k in 1..n_nodes {
        let sig_bar = shoot.x_nodes[k][ISIGMA];
        ineq.add(&[(NS * k + ISIGMA, 1.0)], sigma_max - sig_bar);
        ineq.add(&[(NS * k + ISIGMA, -1.0)], sigma_max + sig_bar);
    }

    let limits = &m.data.constraints;
    let limit_vals = [
        limits.max_heat_flux,
        limits.max_load_factor / G0, // path_quantities returns g's
        limits.max_dynamic_pressure,
    ];
    for (k, x) in shoot.x_nodes.iter().enumerate() {
        if n_types == 0 {
            break;
        }
        let (hf, gl, pd) = path_quantities(x, m);
        let raw = [hf, gl, pd];
        for (s_i, &t) in enforced.iter().enumerate() {
            let val = raw[t] / limit_vals[t];
            let grad = fd_state_gradient(
                |xx| {
                    let (a, b, c) = path_quantities(xx, m);
                    [a, b, c][t] / limit_vals[t]
                },
                x,
            );
            let mut entries: Vec<(usize, f64)> = Vec::with_capacity(NS + 1);
            for (j, &g) in grad.iter().enumerate() {
                if g != 0.0 {
                    entries.push((NS * k + j, g));
                }
            }
            entries.push((i_np0 + n_types * k + s_i, -1.0));
            ineq.add(&entries, 1.0 - val);
            ineq.add(&[(i_np0 + n_types * k + s_i, -1.0)], 0.0);
        }
    }

    // Terminal heat load (normalized by the limit; skipped when unconfigured)
    let q_max = limits.max_heat_load;
    if q_max > 0.0 {
        let q_bar = shoot.x_nodes[m_seg][IQ];
        ineq.add(
            &[(NS * m_seg + IQ, 1.0 / q_max), (i_nhl, -1.0)],
            (q_max - q_bar) / q_max,
        );
    }
    ineq.add(&[(i_nhl, -1.0)], 0.0);

    // Terminal eps (energy-based apoapsis targeting), MJ/kg
    let x_n = &shoot.x_nodes[m_seg];
    let eps_bar = eps_apoapsis(x_n, m) / EPS_SCALE;
    let grad_eps = fd_state_gradient(|xx| eps_apoapsis(xx, m) / EPS_SCALE, x_n);
    let ge: Vec<(usize, f64)> = grad_eps
        .iter()
        .enumerate()
        .filter(|(_, g)| **g != 0.0)
        .map(|(j, &g)| (NS * m_seg + j, g))
        .collect();
    let mut row = ge.clone();
    row.push((i_seps, -1.0));
    ineq.add(&row, -eps_bar);
    let mut row_neg: Vec<(usize, f64)> = ge.iter().map(|&(c, v)| (c, -v)).collect();
    row_neg.push((i_seps, -1.0));
    ineq.add(&row_neg, eps_bar);
    ineq.add(&[(i_seps, -1.0)], 0.0);

    // Terminal + intermediate inclination (cos i, frame-free)
    if inc_active {
        let ci_t = m.data.target_orbit.inclination.cos();
        let f_ci = |xx: &State8| cos_inclination(xx, m.planet);
        let ci_bar_n = f_ci(x_n) - ci_t;
        let grad_ci = fd_state_gradient(f_ci, x_n);
        let gi: Vec<(usize, f64)> = grad_ci
            .iter()
            .enumerate()
            .filter(|(_, g)| **g != 0.0)
            .map(|(j, &g)| (NS * m_seg + j, g))
            .collect();
        let mut row = gi.clone();
        row.push((i_sinc, -1.0));
        ineq.add(&row, -ci_bar_n);
        let mut row_neg: Vec<(usize, f64)> = gi.iter().map(|&(c, v)| (c, -v)).collect();
        row_neg.push((i_sinc, -1.0));
        ineq.add(&row_neg, ci_bar_n);
        let deadband = deadband_cos(m, p);
        for (idx, &k) in di_nodes.iter().enumerate() {
            let xk = &shoot.x_nodes[k];
            let ci_bar_k = f_ci(xk) - ci_t;
            let grad_k = fd_state_gradient(f_ci, xk);
            let gk: Vec<(usize, f64)> = grad_k
                .iter()
                .enumerate()
                .filter(|(_, g)| **g != 0.0)
                .map(|(j, &g)| (NS * k + j, g))
                .collect();
            let mut row = gk.clone();
            row.push((i_di0 + idx, -1.0));
            ineq.add(&row, deadband - ci_bar_k);
            let mut row_neg: Vec<(usize, f64)> = gk.iter().map(|&(c, v)| (c, -v)).collect();
            row_neg.push((i_di0 + idx, -1.0));
            ineq.add(&row_neg, deadband + ci_bar_k);
            ineq.add(&[(i_di0 + idx, -1.0)], 0.0);
        }
    }
    ineq.add(&[(i_sinc, -1.0)], 0.0);

    // ── Column scaling ──
    let mut col_scale = vec![1.0_f64; n];
    for k in 0..n_nodes {
        for j in 0..NS {
            col_scale[NS * k + j] = STATE_SCALE[j];
        }
    }
    for k in 0..m_seg {
        col_scale[i_u0 + k] = U_SCALE;
    }

    // ── Objective (P upper-triangular triplets + q) ──
    let mut p_i: Vec<usize> = Vec::new();
    let mut p_j: Vec<usize> = Vec::new();
    let mut p_v: Vec<f64> = Vec::new();
    let mut q_vec = vec![0.0_f64; n];
    for k in 0..m_seg {
        let dt_k = shoot.dts[k];
        p_i.push(i_u0 + k);
        p_j.push(i_u0 + k);
        p_v.push(2.0 * p.alpha1 * dt_k);
        q_vec[i_u0 + k] += 2.0 * p.alpha1 * dt_k * shoot.u_nodes[k];
    }
    q_vec[i_seps] = p.alpha3;
    q_vec[i_sinc] = if inc_active { p.alpha2 } else { 0.0 };
    for s in q_vec.iter_mut().take(i_nhl).skip(i_np0) {
        *s = p.alpha5;
    }
    q_vec[i_nhl] = p.alpha5;
    for s in q_vec.iter_mut().take(i_di0 + di_nodes.len()).skip(i_di0) {
        *s = p.lambda_di;
    }

    // ── Column-scale + stack for Clarabel: [eq (Zero); ineq (Nonneg)] ──
    let n_eq = eq.b.len();
    let n_in = ineq.b.len();
    let mut a_i = eq.i;
    let mut a_j = eq.j;
    let mut a_v = eq.v;
    // Row-scale dynamics/init rows by the target-state scale.
    let mut b_all: Vec<f64> = Vec::with_capacity(n_eq + n_in);
    let row_scale_eq: Vec<f64> = (0..n_eq).map(|r| 1.0 / STATE_SCALE[r % NS]).collect();
    for (idx, &r) in a_i.iter().enumerate() {
        a_v[idx] *= row_scale_eq[r] * col_scale[a_j[idx]];
    }
    for (r, &bv) in eq.b.iter().enumerate() {
        b_all.push(bv * row_scale_eq[r]);
    }
    for (idx, &r) in ineq.i.iter().enumerate() {
        a_i.push(n_eq + r);
        a_j.push(ineq.j[idx]);
        a_v.push(ineq.v[idx] * col_scale[ineq.j[idx]]);
    }
    b_all.extend_from_slice(&ineq.b);

    let p_vals: Vec<f64> = p_v
        .iter()
        .enumerate()
        .map(|(idx, &v)| v * col_scale[p_i[idx]] * col_scale[p_j[idx]])
        .collect();
    let q_scaled: Vec<f64> = q_vec
        .iter()
        .enumerate()
        .map(|(idx, &v)| v * col_scale[idx])
        .collect();

    let p_mat: CscMatrix<f64> = CscMatrix::new_from_triplets(n, n, p_i, p_j, p_vals);
    let a_mat: CscMatrix<f64> = CscMatrix::new_from_triplets(n_eq + n_in, n, a_i, a_j, a_v);
    let cones = [
        SupportedConeT::ZeroConeT(n_eq),
        SupportedConeT::NonnegativeConeT(n_in),
    ];
    let settings = DefaultSettings {
        verbose: false,
        ..DefaultSettings::default()
    };
    let mut solver =
        DefaultSolver::new(&p_mat, &q_scaled, &a_mat, &b_all, &cones, settings).ok()?;
    solver.solve();
    let ok = matches!(
        solver.solution.status,
        SolverStatus::Solved | SolverStatus::AlmostSolved
    );
    if !ok {
        return None;
    }
    let du: Vec<f64> = (0..m_seg)
        .map(|k| solver.solution.x[i_u0 + k] * col_scale[i_u0 + k])
        .collect();
    Some((du, solver.info.iterations as usize))
}

/// One full SCP replan from `x0` (sigma0 = x0[ISIGMA]). `u_init = None` seeds
/// from the constant-bank grid (cold start); `Some` warm-starts (guidance loop).
fn scp_replan(
    x0: &State8,
    m: &Model,
    p: &CpagParams,
    u_init: Option<Vec<f64>>,
    max_rate: f64,
    max_iters: usize,
) -> ReplanResult {
    let n_seg = (p.horizon_max / p.seg_dt).ceil() as usize;
    let warm = u_init.is_some();
    let mut u = match u_init {
        Some(mut prof) => {
            prof.resize(n_seg, 0.0);
            for v in prof.iter_mut() {
                *v = v.clamp(-max_rate, max_rate);
            }
            prof
        }
        None => vec![0.0; n_seg],
    };

    let mut shoot = shoot_profile(x0, &u, m, p);
    let mut cur = merit(&shoot, m, p);

    // Constant-bank grid seeding on cold starts (C0's local-optimum fix) AND on
    // warm replans whose held plan now CRASHES: a stale plan under a density
    // surprise needs the recovery bracket, and the warm iteration budget can't
    // climb the crash tier one trust step at a time (observed failure: shallow
    // entries plan full lift-down early, the atmosphere arrives 30% dense, and
    // 4-iteration warm replans ride -180 deg into the ground).
    let escalate = warm && shoot.event == Event::Crash;
    if !warm || escalate {
        let sign = if x0[ISIGMA] != 0.0 {
            x0[ISIGMA].signum()
        } else {
            1.0
        };
        for &deg in &SEED_BANKS_DEG {
            let u_cand =
                roll_to_bank_profile(x0[ISIGMA], sign * deg.to_radians(), n_seg, p, max_rate);
            let shoot_cand = shoot_profile(x0, &u_cand, m, p);
            let cand = merit(&shoot_cand, m, p);
            if cand.j < cur.j {
                u = u_cand;
                shoot = shoot_cand;
                cur = cand;
            }
        }
    }
    let max_iters = if escalate {
        max_iters.max(p.max_iters)
    } else {
        max_iters
    };

    let mut trust = p.trust_init;
    let mut iters = 0usize;
    let mut merit_hist: Vec<f64> = Vec::new();
    for _ in 0..max_iters {
        iters += 1;
        let (a_mats, b_mats) = linearize(&shoot, m, p);
        let m_seg = shoot.dts.len();
        let mut accepted = false;
        for _ in 0..12 {
            let Some((du, _solver_iters)) =
                solve_subproblem(&shoot, &a_mats, &b_mats, m, p, trust, max_rate)
            else {
                trust = (trust * 0.5).max(p.trust_min);
                continue;
            };
            let mut u_new = u.clone();
            let mut dx_inf = 0.0_f64;
            for k in 0..m_seg {
                u_new[k] = (u[k] + du[k]).clamp(-max_rate, max_rate);
                dx_inf = dx_inf.max((du[k] / U_SCALE).abs());
            }
            for v in u_new.iter_mut().skip(m_seg) {
                *v = 0.0;
            }
            let shoot_new = shoot_profile(x0, &u_new, m, p);
            let new = merit(&shoot_new, m, p);
            if new.j > cur.j * (1.0 + 1e-3) + 1e-6 {
                if trust <= p.trust_min * 1.001 {
                    break; // at the floor a retry is a deterministic repeat
                }
                trust = (trust * 0.4).max(p.trust_min);
                continue;
            }
            trust = (trust * 1.5).min(p.trust_max);
            u = u_new;
            shoot = shoot_new;
            cur = new;
            merit_hist.push(cur.j);
            accepted = dx_inf.is_finite();
            break;
        }
        if !accepted {
            break;
        }
        if let Some(&last) = merit_hist.last()
            && last < p.alpha3 * (p.tol_apo / 1e7)
        {
            break; // residual far inside tolerance
        }
        // Merit-stagnation convergence (re-shoot re-timing sets a noise floor).
        let hl = merit_hist.len();
        if hl >= 3
            && (merit_hist[hl - 1] - merit_hist[hl - 3]).abs()
                <= 0.005 * merit_hist[hl - 3].max(1.0)
        {
            break;
        }
    }

    ReplanResult {
        u_profile: u,
        iters,
        eps_mj: cur.eps_mj,
    }
}

// ── Guidance entry ───────────────────────────────────────────────────────────

/// Compute the CPAG signed bank command.
///
/// FNPAG's throttle pattern with the PROFILE as the held object: between
/// replans the commanded bank plays back the stored bank-rate plan; each
/// replan warm-starts from the remaining plan. Returns a SIGNED bank (rad) —
/// CPAG bypasses exit, lateral, and thermal-limiter guidance.
pub fn cpag_bank(
    nav: &NavigationOutput,
    state: &mut CpagState,
    data: &SimData,
    planet: &PlanetConfig,
    sim_time: f64,
) -> f64 {
    let params = &data.guidance.cpag;

    // Outside the sensible atmosphere: hold the plan (FNPAG's guard).
    let (altitude, _) = geodetic_from_spherical(
        nav.position_estimated[0],
        nav.position_estimated[1],
        nav.position_estimated[2],
        planet,
    );
    let rho_onboard = data
        .atmosphere_onboard
        .density_at(altitude, &data.atmosphere);
    let in_atmosphere = rho_onboard >= 1e-10;

    let due = sim_time - state.last_replan_time >= params.replan_period;
    if in_atmosphere && (!state.initialized || due) {
        // Estimated atmospheric dispersion factor (FNPAG's exact pattern).
        let density_factor = {
            let f = nav.density_guidance / rho_onboard;
            if f.is_finite() && f > 0.0 { f } else { 1.0 }
        };
        let model = Model::new(data, planet, density_factor);

        let sigma_now = state.sigma_at(sim_time, params);
        let x0: State8 = [
            nav.position_estimated[0],
            nav.position_estimated[1],
            nav.position_estimated[2],
            nav.velocity_estimated[0],
            nav.velocity_estimated[1],
            nav.velocity_estimated[2],
            sigma_now,
            nav.heat_load_fraction * data.constraints.max_heat_load,
        ];

        // Warm start: resample the remaining plan onto the new grid.
        let u_init = if state.initialized {
            let n_seg = (params.horizon_max / params.seg_dt).ceil() as usize;
            let elapsed = sim_time - state.anchor_time;
            let prof: Vec<f64> = (0..n_seg)
                .map(|k| {
                    let idx = ((elapsed + k as f64 * params.seg_dt) / params.seg_dt) as usize;
                    state.u_profile.get(idx).copied().unwrap_or(0.0)
                })
                .collect();
            Some(prof)
        } else {
            None
        };

        let max_rate = data.capsule.max_bank_rate;
        let max_iters = if state.initialized {
            params.max_iters_warm
        } else {
            params.max_iters
        };
        let result = scp_replan(&x0, &model, params, u_init, max_rate, max_iters);

        state.u_profile = result.u_profile;
        state.sigma_anchor = sigma_now;
        state.anchor_time = sim_time;
        state.last_replan_time = sim_time;
        state.initialized = true;
        state.last_iters = result.iters;
        state.last_eps_mj = result.eps_mj;
    }

    state.sigma_at(sim_time, params)
}

#[cfg(test)]
#[path = "cpag_tests.rs"]
mod cpag_tests;
