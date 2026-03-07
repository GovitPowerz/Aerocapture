//! Per-run initialization and Monte Carlo dispersion application.
//!
//! Two modes:
//! - **Legacy**: reads pre-computed draws from lottery files (Fortran inimsr.f compat)
//! - **Domain-based**: generates draws at runtime from seeded RNG (new system)

use crate::config::SimInput;
use crate::data::dispersions::DispersionDraw;
use crate::data::{EntryConditions, SimData};
use crate::gnc::navigation::estimator::NavigationBiases;

/// Per-simulation-run state after applying dispersions.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct RunState {
    pub entry: EntryConditions,
    pub cx_bias: f64,        // drag coefficient bias (fractional)
    pub cz_bias: f64,        // lift coefficient bias (fractional)
    pub density_bias: f64,   // atmosphere density bias (fractional)
    pub mass_bias: f64,      // mass bias (fractional)
    pub incidence_bias: f64, // incidence error (radians)
    pub nav_biases: NavigationBiases,
}

/// Raw dispersions read from the lottery file.
///
/// These values are pre-multiplied (sigma × random draw) and stored in
/// the lottery file. They are directly added to the nominal conditions.
///
/// Matches Fortran inimsr.f read format:
///   i, xaleat, daltit, dlongi, dlatit, dvites, dazimu, dpente,
///   ddensi, dcxeng, dczeng,
///   dnalti, dnlati, dnlong, dnvite, dnazim, dnpent,
///   dndrag, dalfae, dmvehi
#[derive(Debug, Clone, Default)]
pub struct LotteryDraw {
    pub daltit: f64, // altitude dispersion (meters)
    pub dlongi: f64, // longitude dispersion (radians)
    pub dlatit: f64, // latitude dispersion (radians)
    pub dvites: f64, // velocity dispersion (m/s)
    pub dazimu: f64, // azimuth dispersion (radians)
    pub dpente: f64, // FPA dispersion (radians)
    pub ddensi: f64, // density dispersion factor
    pub dcxeng: f64, // drag coeff dispersion factor
    pub dczeng: f64, // lift coeff dispersion factor
    // Navigation biases
    pub dnalti: f64, // nav altitude bias (meters)
    pub dnlati: f64, // nav latitude bias (radians)
    pub dnlong: f64, // nav longitude bias (radians)
    pub dnvite: f64, // nav velocity bias (m/s)
    pub dnazim: f64, // nav azimuth bias (radians)
    pub dnpent: f64, // nav FPA bias (radians)
    pub dndrag: f64, // nav drag accel bias (m/s²)
    pub dalfae: f64, // incidence dispersion (radians)
    pub dmvehi: f64, // mass dispersion factor
}

/// Read lottery dispersions for a given simulation index.
///
/// Matches Fortran inimsr.f lottery file reading.
/// For single runs: reads the numsim-th line (1-based).
/// For Monte Carlo: reads sequentially.
pub fn read_lottery(
    path: &str,
    sim_index: i32,     // 0-based
    visualize_sim: i32, // numsim from config (1-based)
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
        sim_index as usize
    } else {
        (visualize_sim - 1).max(0) as usize
    };

    if line_idx >= lines.len() {
        return LotteryDraw::default();
    }

    let line = lines[line_idx];
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
        dnalti: tokens[11],
        dnlati: tokens[12],
        dnlong: tokens[13],
        dnvite: tokens[14],
        dnazim: tokens[15],
        dnpent: tokens[16],
        dndrag: tokens[17],
        dalfae: tokens[18],
        dmvehi: tokens[19],
    }
}

/// Initialize a simulation run using domain-based RNG draws.
pub fn init_run_from_draw(sim_data: &SimData, draw: &DispersionDraw) -> RunState {
    let mut entry = sim_data.entry;
    entry.state.altitude += draw.altitude;
    entry.state.longitude += draw.longitude;
    entry.state.latitude += draw.latitude;
    entry.state.velocity += draw.velocity;
    entry.state.flight_path += draw.flight_path;
    entry.state.azimuth += draw.azimuth;

    RunState {
        entry,
        cx_bias: draw.drag_coeff,
        cz_bias: draw.lift_coeff,
        density_bias: draw.density,
        mass_bias: draw.mass,
        incidence_bias: draw.incidence,
        nav_biases: NavigationBiases {
            pos: [draw.nav_altitude, draw.nav_longitude, draw.nav_latitude],
            vel: [draw.nav_velocity, draw.nav_flight_path, draw.nav_azimuth],
            drag: draw.nav_drag_accel,
        },
    }
}

/// Initialize a simulation run using legacy lottery file draws.
///
/// Matches Fortran inimsr.f.
pub fn init_run(sim_data: &SimData, config: &SimInput, sim_index: i32, _rng_seed: f64) -> RunState {
    let lottery_path = config.data_path("loterie", &config.suffixes.lottery);
    let is_mc = config.n_sims > 1;
    let draw = read_lottery(&lottery_path, sim_index, config.replay_sim, is_mc);

    let mut entry = sim_data.entry;
    entry.state.altitude += draw.daltit;
    entry.state.longitude += draw.dlongi;
    entry.state.latitude += draw.dlatit;
    entry.state.velocity += draw.dvites;
    entry.state.flight_path += draw.dpente;
    entry.state.azimuth += draw.dazimu;

    RunState {
        entry,
        cx_bias: draw.dcxeng,
        cz_bias: draw.dczeng,
        density_bias: draw.ddensi,
        mass_bias: draw.dmvehi,
        incidence_bias: draw.dalfae,
        nav_biases: NavigationBiases {
            pos: [draw.dnalti, draw.dnlong, draw.dnlati],
            vel: [draw.dnvite, draw.dnpent, draw.dnazim],
            drag: draw.dndrag,
        },
    }
}
