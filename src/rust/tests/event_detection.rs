//! Integration tests and proptests for DOPRI45 event detection.

mod common;

use aerocapture::config::SimInput;
use aerocapture::data::SimData;
use aerocapture::simulation::runner::run_for_api;

fn load_config(name: &str) -> (SimInput, SimData) {
    let repo = common::repo_root();
    std::env::set_current_dir(&repo).expect("set cwd");
    let path = repo.join("configs").join(name);
    let (si, tc) = SimInput::from_toml_file(&path)
        .unwrap_or_else(|e| panic!("Failed to load {}: {}", path.display(), e));
    let sd = SimData::from_toml(&tc, &si)
        .unwrap_or_else(|e| panic!("Failed to build SimData for {}: {}", path.display(), e));
    (si, sd)
}

/// Bounce time from event detection must fall strictly between tick boundaries.
/// If it were a tick-boundary detection, bounce_time % 1.0 would be exactly 0.
#[test]
fn bounce_time_is_sub_tick() {
    let (config, data) = load_config("test/test_ref_adaptive.toml");
    let results = run_for_api(&config, &data, false, None).expect("run simulation");
    let bounce_time = results[0].final_record[26];

    // bounce_time must be positive and finite
    assert!(
        bounce_time > 0.0 && bounce_time.is_finite(),
        "invalid bounce_time: {}",
        bounce_time
    );

    // With event detection the bounce is located between ticks (dt=1.0 s outer cadence),
    // so remainder must not be essentially 0 or essentially 1.
    let remainder = bounce_time % 1.0;
    assert!(
        remainder > 0.01 && remainder < 0.99,
        "bounce_time={} has remainder={} -- looks like a tick-boundary detection, not sub-tick",
        bounce_time,
        remainder,
    );
}

/// Adaptive DOPRI45 with event detection must still produce a captured trajectory.
#[test]
fn adaptive_with_events_still_captures() {
    let (config, data) = load_config("test/test_ref_adaptive.toml");
    let results = run_for_api(&config, &data, false, None).expect("run simulation");
    let r = &results[0];

    assert!(
        r.captured,
        "adaptive mode with event detection must capture"
    );
    assert!(
        r.final_record[9] < 1.0,
        "eccentricity must be < 1.0 for a captured orbit, got {}",
        r.final_record[9],
    );
}

/// Fixed Gill RK4 must not be affected by event detection framework.
/// It uses tick-boundary detection only; that path must remain intact.
#[test]
fn fixed_rk4_unchanged() {
    let (config, data) = load_config("test/test_ref_orig.toml");
    let results = run_for_api(&config, &data, false, None).expect("run simulation");
    let r = &results[0];

    let ifinal = r.final_record[31] as i32;
    assert_eq!(ifinal, 3, "fixed Gill RK4 must produce ifinal=3 (captured)");
    assert!(
        r.captured,
        "captured flag must be true for Gill RK4 reference run"
    );
}

/// With trajectories enabled, event rows are injected into the photo stream.
/// Times must be monotonically non-decreasing, and bounce_time must appear
/// as a trajectory row.
#[test]
fn trajectory_includes_event_rows() {
    let (config, data) = load_config("test/test_ref_adaptive.toml");
    let results = run_for_api(&config, &data, true, None).expect("run simulation");
    let r = &results[0];

    let traj = &r.trajectory;
    assert!(
        !traj.is_empty(),
        "trajectory must not be empty when include_trajectories=true"
    );

    // Times (column 7) must be monotonically non-decreasing
    for i in 1..traj.len() {
        assert!(
            traj[i][7] >= traj[i - 1][7],
            "trajectory times not monotone at index {}: t[{}]={} > t[{}]={}",
            i,
            i,
            traj[i][7],
            i - 1,
            traj[i - 1][7],
        );
    }

    // bounce_time from final_record must appear as a trajectory row time
    let bounce_time = r.final_record[26];
    if bounce_time > 0.0 && bounce_time < 1e29 {
        let tol = 0.01; // 10 ms -- Brent converges well within this
        let found = traj.iter().any(|row| (row[7] - bounce_time).abs() < tol);
        assert!(
            found,
            "bounce_time={} not found in trajectory times (tolerance={}, first few: {:?})",
            bounce_time,
            tol,
            traj[..traj.len().min(5)]
                .iter()
                .map(|r| r[7])
                .collect::<Vec<_>>(),
        );
    }
}

/// Adaptive DOPRI45 bounce_alt must use the geodetic altitude convention, matching
/// the fixed-RK4 path and the trajectory row altitude at the bounce event.
///
/// Before this fix, the adaptive path stored `state[0] - equatorial_radius`
/// (geometric), while the event trajectory row uses `geodetic_from_spherical`.
/// On Mars at typical bounce latitudes (~10-30 deg), the discrepancy is ~1-4 km.
///
/// This test fails against the old geometric value and passes after unification.
#[test]
fn adaptive_bounce_alt_matches_geodetic_trajectory_row() {
    let (config, data) = load_config("test/test_ref_adaptive.toml");
    let results = run_for_api(&config, &data, true, None).expect("run simulation");
    let r = &results[0];

    let bounce_alt_km = r.final_record[25]; // km, from state.bounce_alt / 1e3
    let bounce_time = r.final_record[26];

    assert!(
        bounce_alt_km > 0.0 && bounce_alt_km.is_finite(),
        "bounce_alt must be positive and finite, got {}",
        bounce_alt_km
    );

    // The event trajectory row at bounce_time has altitude computed via
    // geodetic_from_spherical (see build_event_photo_values).
    // Find the trajectory row closest in time to bounce_time.
    let traj = &r.trajectory;
    let tol_time = 0.01; // 10 ms tolerance for Brent convergence
    let bounce_row = traj
        .iter()
        .find(|row| (row[7] - bounce_time).abs() < tol_time)
        .unwrap_or_else(|| {
            panic!(
                "no trajectory row found within {}s of bounce_time={}",
                tol_time, bounce_time
            )
        });

    let traj_alt_km = bounce_row[0]; // geodetic altitude (km) from build_event_photo_values

    // The recorded bounce_alt must match the geodetic trajectory row altitude.
    // On Mars at typical bounce latitudes (~15-30 deg), the geodetic vs geometric
    // difference is ~2-4 km, so a 0.1 km tolerance is tight enough to distinguish
    // the conventions while absorbing any floating-point rounding in the conversion.
    assert!(
        (bounce_alt_km - traj_alt_km).abs() < 0.1,
        "bounce_alt ({:.4} km) must match geodetic trajectory row altitude ({:.4} km); \
         difference {:.4} km suggests the geometric convention is still in use",
        bounce_alt_km,
        traj_alt_km,
        (bounce_alt_km - traj_alt_km).abs(),
    );
}

// ── Proptests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod proptest_event_detection {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            cases: 20,
            ..Default::default()
        })]

        /// For a range of MC seeds, bounce_time, bounce_alt, and sim_time must all
        /// be finite, and sim_time must be positive.
        #[test]
        fn event_bounce_values_finite(seed in 0u64..20u64) {
            let repo = common::repo_root();
            std::env::set_current_dir(&repo).expect("set cwd");
            let path = repo.join("configs").join("test/test_ref_adaptive.toml");
            let (si, tc) = SimInput::from_toml_file(&path).unwrap();
            let mut sd = SimData::from_toml(&tc, &si).unwrap();
            // Vary seed so we exercise different MC draws across runs
            if let Some(ref mut dc) = sd.dispersion_config {
                dc.seed = seed;
            }

            let results = run_for_api(&si, &sd, false, None).expect("run must not error");
            let r = &results[0];

            let bounce_time = r.final_record[26];
            let bounce_alt  = r.final_record[25];
            let sim_time    = r.final_record[27];

            prop_assert!(bounce_time.is_finite(), "bounce_time not finite: {}", bounce_time);
            prop_assert!(bounce_alt.is_finite(),  "bounce_alt not finite: {}",  bounce_alt);
            prop_assert!(sim_time.is_finite(),    "sim_time not finite: {}",    sim_time);
            prop_assert!(sim_time > 0.0,          "sim_time must be positive: {}", sim_time);
        }
    }
}
