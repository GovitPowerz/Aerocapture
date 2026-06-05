//! Named index constants for the 52-element final-record array.
//!
//! Single source of truth for `fr[N]` assignments in `finalize.rs` and
//! `fr[N]` reads in `aerocapture-py` (`results.rs`, `env.rs`).
//!
//! Indices that remain at their zero-initialised default (unused/reserved)
//! are listed as comments only; they do not get a const to avoid inventing
//! meanings that do not exist in the codebase.

// ── Exit state ──────────────────────────────────────────────────────────────
pub const FR_ALT_KM: usize = 0;
pub const FR_LON_DEG: usize = 1;
pub const FR_LAT_DEG: usize = 2;
pub const FR_VEL_MS: usize = 3;
pub const FR_FPA_DEG: usize = 4;
pub const FR_HDG_DEG: usize = 5;
pub const FR_RADIAL_VEL_MS: usize = 6;

// ── Orbital mechanics ────────────────────────────────────────────────────────
pub const FR_ENERGY_MJKG: usize = 7;
pub const FR_SMA_KM: usize = 8;
pub const FR_ECC: usize = 9;
pub const FR_INCL_DEG: usize = 10;
pub const FR_RAAN_DEG: usize = 11;
pub const FR_ARG_PERI_DEG: usize = 12;
pub const FR_TRUE_ANOM_DEG: usize = 13;
pub const FR_PERIAPSIS_ALT_KM: usize = 14;
pub const FR_APOAPSIS_ALT_KM: usize = 15;

// ── Peak flight constraints ──────────────────────────────────────────────────
pub const FR_HEAT_FLUX_KW_M2: usize = 16;
pub const FR_G_LOAD: usize = 17;
pub const FR_DYN_PRESSURE_KPA: usize = 18;

// ── Altitudes and times of peak constraint values ────────────────────────────
pub const FR_ALT_MAX_FLUX_KM: usize = 19;
pub const FR_ALT_MAX_LOAD_KM: usize = 20;
pub const FR_ALT_MAX_PDYN_KM: usize = 21;
pub const FR_TIME_MAX_FLUX_S: usize = 22;
pub const FR_TIME_MAX_LOAD_S: usize = 23;
pub const FR_TIME_MAX_PDYN_S: usize = 24;

// ── Trajectory events ────────────────────────────────────────────────────────
pub const FR_BOUNCE_ALT_KM: usize = 25;
pub const FR_BOUNCE_TIME_S: usize = 26;
pub const FR_SIM_TIME_S: usize = 27;

// ── Integrated thermal state + orbit errors ──────────────────────────────────
// State[6] is cumulative heat load in kJ/m²; stored as MJ/m² here (÷1e6).
pub const FR_HEAT_LOAD_MJM2: usize = 28;
pub const FR_PERIAPSIS_ERR_KM: usize = 29;
pub const FR_APOAPSIS_ERR_KM: usize = 30;

// ── Terminal classification ──────────────────────────────────────────────────
pub const FR_IFINAL: usize = 31;
// Indices 32-36: reserved / zero (unused in current codebase).

// ── Delta-V components ───────────────────────────────────────────────────────
pub const FR_DV1_MS: usize = 37;
pub const FR_DV2_MS: usize = 38;
pub const FR_DV3_MS: usize = 39;
pub const FR_DV_PLANE_MS: usize = 40; // |dv1| + |dv2|
pub const FR_DV_TOTAL_MS: usize = 41;
// Indices 42-44: reserved / zero (unused in current codebase).

// ── Guidance telemetry ────────────────────────────────────────────────────────
pub const FR_CUMULATIVE_BANK_DEG: usize = 45;
pub const FR_INCL_ERR_DEG: usize = 46;
// Index 47: reserved / zero.
pub const FR_N_REVERSALS: usize = 48;
// Indices 49-51: reserved / zero.

/// Width of the final-record array.
pub const FINAL_RECORD_LEN: usize = 52;
