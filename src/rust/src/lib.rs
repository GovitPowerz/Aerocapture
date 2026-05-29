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
    /// Per-timestep state: [alt_km, lon_deg, lat_deg, vel_m_s, fpa_deg, heading_deg,
    /// heat_flux_kw_m2, time_s, energy_mj_kg, pdyn_kpa, bank_angle_deg, inclination_deg,
    /// g_load_g, nav_density_ratio, truth_density_kg_m3, heat_load_kj_m2, density_perturbation]
    pub trajectory: Vec<[f64; 17]>,
    /// Full 52-column final record (same layout as CSV file output)
    pub final_record: [f64; 52],
    /// True if orbit is bound (ecc < 1 && energy < 0) and not a pending crash (ifinal != 4).
    pub captured: bool,
    /// Dispersion draws for this simulation (26 fields from DispersionDraw::to_array)
    pub dispersions: [f64; data::dispersions::DISPERSION_DRAW_LEN],
    /// When the runner was invoked with `collect_supervised = true`, holds
    /// per-tick (nn_input, pre_shaper_signed_bank, prev_realized_bank, sim_time,
    /// energy_mj_kg) tuples. The third element is the previous-tick pilot-realized
    /// bank, consistent with the nn_input row at that step; the last two carry the
    /// tick's sim time and orbital energy. Empty otherwise.
    pub supervised_trace: Vec<(Vec<f64>, f64, f64, f64, f64)>,
}
