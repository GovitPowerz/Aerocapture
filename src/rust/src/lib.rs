pub mod config;
pub mod data;
pub mod gnc;
pub mod integration;
pub mod orbit;
pub mod physics;
pub mod simulation;

/// Public output from a single simulation run, for use by PyO3 and tests.
#[derive(Debug, Clone)]
pub struct RunOutput {
    /// Per-timestep state: [alt_km, lon_deg, lat_deg, vel_m_s, fpa_deg, heading_deg, flux_placeholder, time, energy_MJkg, pdyn_kPa, bank_deg, incl_deg]
    pub trajectory: Vec<[f64; 12]>,
    /// Full 52-column final record (same layout as CSV file output)
    pub final_record: [f64; 52],
    /// True if orbit is bound (ecc < 1 && energy < 0) and not a pending crash (ifinal != 4).
    pub captured: bool,
    /// Dispersion draws for this simulation (24 fields from DispersionDraw::to_array)
    pub dispersions: [f64; data::dispersions::DISPERSION_DRAW_LEN],
}
