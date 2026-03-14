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
    /// Per-timestep state from photo output: [alt_km, lon_deg, lat_deg, vel_m_s, fpa_deg, heading_deg, flux, time]
    pub trajectory: Vec<[f64; 8]>,
    /// Full 52-column final record (same layout as CSV file output)
    pub final_record: [f64; 52],
    /// True if orbit is bound (ecc < 1 && energy < 0)
    pub captured: bool,
}
