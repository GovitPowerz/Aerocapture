//! Per-run initialization and Monte Carlo dispersion application.
//!
//! Matches Fortran inimsr.f.

use crate::config::SimInput;
use crate::data::{EntryConditions, SimData, SphericalState};

/// Per-simulation-run state after applying dispersions.
#[derive(Debug, Clone)]
pub struct RunState {
    pub entry: EntryConditions,
    pub cx_bias: f64,       // drag coefficient bias (fractional)
    pub cz_bias: f64,       // lift coefficient bias (fractional)
    pub density_bias: f64,  // atmosphere density bias (fractional)
    pub mass_bias: f64,     // mass bias (fractional)
    pub incidence_bias: f64, // incidence error (radians)
}

/// Raw dispersions read from the lottery file.
///
/// These values are drawn from Gaussian distributions and stored in
/// the lottery file. They are directly added to the nominal conditions.
///
/// Matches Fortran inimsr.f read format:
///   i, xaleat, daltit, dlongi, dlatit, dvites, dazimu, dpente,
///   ddensi, dcxeng, dczeng,
///   dnalti, dnlati, dnlong, dnvite, dnazim, dnpent,
///   dndrag, dalfae, dmvehi
#[derive(Debug, Clone, Default)]
pub struct LotteryDraw {
    pub daltit: f64,  // altitude dispersion (meters, added to radius)
    pub dlongi: f64,  // longitude dispersion (radians)
    pub dlatit: f64,  // latitude dispersion (radians)
    pub dvites: f64,  // velocity dispersion (m/s)
    pub dazimu: f64,  // azimuth dispersion (radians)
    pub dpente: f64,  // FPA dispersion (radians)
    pub ddensi: f64,  // density dispersion factor
    pub dcxeng: f64,  // drag coeff dispersion factor
    pub dczeng: f64,  // lift coeff dispersion factor
    pub dalfae: f64,  // incidence dispersion (radians)
    pub dmvehi: f64,  // mass dispersion factor
}

/// Read lottery dispersions for a given simulation index.
///
/// Matches Fortran inimsr.f lottery file reading.
/// For single runs: reads the numsim-th line (1-based).
/// For Monte Carlo: reads sequentially.
pub fn read_lottery(
    path: &str,
    sim_index: i32,  // 0-based
    visualize_sim: i32,  // numsim from config (1-based)
    is_monte_carlo: bool,
) -> LotteryDraw {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return LotteryDraw::default(),
    };

    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return LotteryDraw::default();
    }

    // Determine which line to read
    let line_idx = if is_monte_carlo {
        // Monte Carlo: read sim_index-th line (0-based)
        sim_index as usize
    } else {
        // Single run: skip (numsim-1) lines, read numsim-th
        // Fortran: numsim is 1-based, so line index = numsim - 1
        (visualize_sim - 1).max(0) as usize
    };

    if line_idx >= lines.len() {
        return LotteryDraw::default();
    }

    let line = lines[line_idx];
    // Parse: normalize D-notation and split
    let tokens: Vec<f64> = line
        .split_whitespace()
        .filter_map(|t| {
            let norm = t.replace('D', "E").replace('d', "e");
            norm.parse::<f64>().ok()
        })
        .collect();

    // tokens[0] = i (sim number), tokens[1] = xaleat (random seed)
    // tokens[2..] = dispersions
    if tokens.len() < 20 {
        return LotteryDraw::default();
    }

    LotteryDraw {
        daltit: tokens[2],
        dlongi: tokens[3],
        dlatit: tokens[4],
        dvites: tokens[5],
        dazimu: tokens[6],
        dpente: tokens[7],
        ddensi: tokens[8],
        dcxeng: tokens[9],
        dczeng: tokens[10],
        // tokens[11..17] = navigation dispersions (dnalti..dnpent, dndrag)
        dalfae: tokens[18],
        dmvehi: tokens[19],
    }
}

/// Initialize a simulation run.
///
/// Reads lottery dispersions and applies them to entry conditions.
/// Matches Fortran inimsr.f.
pub fn init_run(
    sim_data: &SimData,
    config: &SimInput,
    sim_index: i32,
    _rng_seed: f64,
) -> RunState {
    // Read lottery file dispersions
    let lottery_path = config.data_path("loterie", &config.suffixes.lottery);
    let is_mc = config.n_sims > 1;
    let draw = read_lottery(&lottery_path, sim_index, config.replay_sim, is_mc);

    // Apply initial condition dispersions (matches inimsr.f lines 232-237)
    let mut entry = sim_data.entry;
    // positr(1) = daltit + positz(1)  — daltit added to radius directly
    // But entry.state.altitude is geodetic altitude, and positz(1) is radius.
    // The Fortran adds daltit to the radius (positz(1) = radius from geodes).
    // Since r = altitude + req approximately, adding to altitude has the same effect.
    entry.state.altitude += draw.daltit;
    entry.state.longitude += draw.dlongi;
    entry.state.latitude += draw.dlatit;
    entry.state.velocity += draw.dvites;
    entry.state.flight_path += draw.dpente;
    entry.state.azimuth += draw.dazimu;

    // Aero/atmo dispersions (matches inimsr.f lines 196-198)
    // disatm = ddensi, dxdrag = dcxeng, dxlift = dczeng
    let density_bias = draw.ddensi;
    let cx_bias = draw.dcxeng;
    let cz_bias = draw.dczeng;
    let mass_bias = draw.dmvehi;
    let incidence_bias = draw.dalfae;

    RunState {
        entry,
        cx_bias,
        cz_bias,
        density_bias,
        mass_bias,
        incidence_bias,
    }
}
