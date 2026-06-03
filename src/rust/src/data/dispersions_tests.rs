use super::*;

fn medium_config(seed: u64) -> DispersionConfig {
    DispersionConfig {
        seed,
        sampling: SamplingMethod::Random,
        initial_state: Some(InitialStateSigmas::from_level(DispersionLevel::Medium)),
        atmosphere: Some(AtmosphereSigmas::from_level(DispersionLevel::Medium)),
        aerodynamics: Some(AerodynamicsSigmas::from_level(DispersionLevel::Medium)),
        navigation: Some(NavigationSigmas::from_level(DispersionLevel::Medium)),
        mass: Some(MassSigmas::from_level(DispersionLevel::Medium)),
        vehicle: Some(VehicleSigmas::from_level(DispersionLevel::Medium)),
        pilot: Some(PilotSigmas::from_level(DispersionLevel::Medium)),
        nav_filter: Some(NavFilterSigmas::from_level(DispersionLevel::Medium)),
        wind: None,
        density_perturbation: None,
    }
}

#[test]
fn test_generate_draws_reproducible() {
    let draws_a = medium_config(42).generate_draws(10);
    let draws_b = medium_config(42).generate_draws(10);
    for (a, b) in draws_a.iter().zip(draws_b.iter()) {
        assert_eq!(a.altitude, b.altitude);
        assert_eq!(a.velocity, b.velocity);
        assert_eq!(a.density, b.density);
        assert_eq!(a.drag_coeff, b.drag_coeff);
        assert_eq!(a.nav_altitude, b.nav_altitude);
        assert_eq!(a.mass, b.mass);
        assert_eq!(a.ref_area, b.ref_area);
        assert_eq!(a.pilot_tau, b.pilot_tau);
        assert_eq!(a.filter_gain, b.filter_gain);
    }
}

#[test]
fn test_generate_draws_different_seeds() {
    let draws_a = medium_config(42).generate_draws(5);
    let draws_b = medium_config(99).generate_draws(5);
    // With different seeds, at least one draw should differ
    let any_differ = draws_a
        .iter()
        .zip(draws_b.iter())
        .any(|(a, b)| a.velocity != b.velocity);
    assert!(any_differ, "Different seeds should produce different draws");
}

#[test]
fn test_generate_draws_count() {
    for n in [0, 1, 5, 100] {
        let draws = medium_config(42).generate_draws(n);
        assert_eq!(draws.len(), n);
    }
}

#[test]
fn test_all_none_gives_zeros() {
    let config = DispersionConfig {
        seed: 42,
        sampling: SamplingMethod::Random,
        initial_state: None,
        atmosphere: None,
        aerodynamics: None,
        navigation: None,
        mass: None,
        vehicle: None,
        pilot: None,
        nav_filter: None,
        wind: None,
        density_perturbation: None,
    };
    let draws = config.generate_draws(10);
    for d in &draws {
        assert_eq!(d.altitude, 0.0);
        assert_eq!(d.longitude, 0.0);
        assert_eq!(d.velocity, 0.0);
        assert_eq!(d.density, 0.0);
        assert_eq!(d.drag_coeff, 0.0);
        assert_eq!(d.nav_altitude, 0.0);
        assert_eq!(d.mass, 0.0);
        assert_eq!(d.ref_area, 0.0);
        assert_eq!(d.max_bank_rate, 0.0);
        assert_eq!(d.pilot_tau, 0.0);
        assert_eq!(d.pilot_damping, 0.0);
        assert_eq!(d.pilot_frequency, 0.0);
        assert_eq!(d.filter_gain, 0.0);
        assert_eq!(
            d.wind_scale, 1.0,
            "wind_scale default should be 1.0 (identity)"
        );
        assert_eq!(d.wind_direction_bias, 0.0);
    }
}

#[test]
fn test_sigma_presets_nonzero() {
    for level in [
        DispersionLevel::Low,
        DispersionLevel::Medium,
        DispersionLevel::High,
    ] {
        let s = InitialStateSigmas::from_level(level);
        assert!(
            s.velocity > 0.0,
            "velocity sigma should be > 0 for {:?}",
            level
        );

        let a = AtmosphereSigmas::from_level(level);
        assert!(a.density > 0.0);

        let n = NavigationSigmas::from_level(level);
        assert!(n.altitude > 0.0);

        let v = VehicleSigmas::from_level(level);
        assert!(v.ref_area > 0.0);
        assert!(v.max_bank_rate > 0.0);

        let p = PilotSigmas::from_level(level);
        assert!(p.time_constant > 0.0);

        let nf = NavFilterSigmas::from_level(level);
        assert!(nf.filter_gain > 0.0);
    }

    let s_off = InitialStateSigmas::from_level(DispersionLevel::Off);
    assert_eq!(s_off.altitude, 0.0);
    assert_eq!(s_off.velocity, 0.0);

    let v_off = VehicleSigmas::from_level(DispersionLevel::Off);
    assert_eq!(v_off.ref_area, 0.0);
    assert_eq!(v_off.max_bank_rate, 0.0);
}

#[test]
fn test_wind_config_off() {
    let cfg = WindDispersionConfig::from_level(DispersionLevel::Off);
    assert_eq!(cfg.scale_min, 1.0);
    assert_eq!(cfg.scale_max, 1.0);
    assert_eq!(cfg.direction_bias_deg, 0.0);
}

#[test]
fn test_wind_config_medium() {
    let cfg = WindDispersionConfig::from_level(DispersionLevel::Medium);
    assert_eq!(cfg.scale_min, 0.5);
    assert_eq!(cfg.scale_max, 1.5);
    assert_eq!(cfg.direction_bias_deg, 10.0);
}

#[test]
fn test_wind_config_high() {
    let cfg = WindDispersionConfig::from_level(DispersionLevel::High);
    assert_eq!(cfg.scale_min, 0.2);
    assert_eq!(cfg.scale_max, 2.0);
    assert_eq!(cfg.direction_bias_deg, 20.0);
}

#[test]
fn test_wind_config_custom_defaults_to_medium() {
    let cfg = WindDispersionConfig::from_level(DispersionLevel::Custom);
    let med = WindDispersionConfig::from_level(DispersionLevel::Medium);
    assert_eq!(cfg.scale_min, med.scale_min);
    assert_eq!(cfg.scale_max, med.scale_max);
    assert_eq!(cfg.direction_bias_deg, med.direction_bias_deg);
}

#[test]
fn test_uniform_fields_bounded() {
    let config = DispersionConfig {
        seed: 12345,
        sampling: SamplingMethod::Random,
        initial_state: None,
        atmosphere: Some(AtmosphereSigmas { density: 50.0 }),
        aerodynamics: Some(AerodynamicsSigmas {
            drag: 5.0,
            lift: 10.0,
            incidence: 1.0,
        }),
        navigation: None,
        mass: Some(MassSigmas { mass: 1.0 }),
        vehicle: Some(VehicleSigmas {
            ref_area: 2.0,
            max_bank_rate: 10.0,
        }),
        pilot: Some(PilotSigmas {
            time_constant: 10.0,
            damping: 10.0,
            frequency: 10.0,
        }),
        nav_filter: None,
        wind: None,
        density_perturbation: None,
    };
    let draws = config.generate_draws(1000);
    for d in &draws {
        // Uniform[-1,1] * sigma/100, so |value| <= sigma/100
        assert!(
            d.density.abs() <= 0.50 + 1e-10,
            "density out of bounds: {}",
            d.density
        );
        assert!(
            d.drag_coeff.abs() <= 0.05 + 1e-10,
            "drag out of bounds: {}",
            d.drag_coeff
        );
        assert!(
            d.lift_coeff.abs() <= 0.10 + 1e-10,
            "lift out of bounds: {}",
            d.lift_coeff
        );
        assert!(
            d.incidence.abs() <= 1.0 * DEG2RAD + 1e-10,
            "incidence out of bounds: {}",
            d.incidence
        );
        assert!(
            d.mass.abs() <= 0.01 + 1e-10,
            "mass out of bounds: {}",
            d.mass
        );
        assert!(
            d.ref_area.abs() <= 0.02 + 1e-10,
            "ref_area out of bounds: {}",
            d.ref_area
        );
        assert!(
            d.max_bank_rate.abs() <= 0.10 + 1e-10,
            "max_bank_rate out of bounds: {}",
            d.max_bank_rate
        );
        assert!(
            d.pilot_tau.abs() <= 0.10 + 1e-10,
            "pilot_tau out of bounds: {}",
            d.pilot_tau
        );
        assert!(
            d.pilot_damping.abs() <= 0.10 + 1e-10,
            "pilot_damping out of bounds: {}",
            d.pilot_damping
        );
        assert!(
            d.pilot_frequency.abs() <= 0.10 + 1e-10,
            "pilot_frequency out of bounds: {}",
            d.pilot_frequency
        );
    }
}

#[test]
fn test_filter_gain_gaussian_range() {
    let config = DispersionConfig {
        seed: 54321,
        sampling: SamplingMethod::Random,
        initial_state: None,
        atmosphere: None,
        aerodynamics: None,
        navigation: None,
        mass: None,
        vehicle: None,
        pilot: None,
        nav_filter: Some(NavFilterSigmas { filter_gain: 0.10 }),
        wind: None,
        density_perturbation: None,
    };
    let draws = config.generate_draws(1000);
    // Gaussian: most draws within ±3sigma = ±0.30
    let within_3sigma = draws.iter().filter(|d| d.filter_gain.abs() <= 0.30).count();
    assert!(
        within_3sigma > 990,
        "Expected >99% within 3-sigma, got {}/1000",
        within_3sigma
    );
    // At least some should be nonzero
    let any_nonzero = draws.iter().any(|d| d.filter_gain.abs() > 0.001);
    assert!(any_nonzero, "Filter gain draws should not all be zero");
}

#[test]
fn dispersion_draw_to_array_roundtrip() {
    let draw = DispersionDraw {
        altitude: 1.0,
        longitude: 2.0,
        latitude: 3.0,
        velocity: 4.0,
        flight_path: 5.0,
        azimuth: 6.0,
        density: 7.0,
        drag_coeff: 8.0,
        lift_coeff: 9.0,
        incidence: 10.0,
        nav_altitude: 11.0,
        nav_longitude: 12.0,
        nav_latitude: 13.0,
        nav_velocity: 14.0,
        nav_flight_path: 15.0,
        nav_azimuth: 16.0,
        nav_drag_accel: 17.0,
        mass: 18.0,
        ref_area: 19.0,
        max_bank_rate: 20.0,
        pilot_tau: 21.0,
        pilot_damping: 22.0,
        pilot_frequency: 23.0,
        filter_gain: 24.0,
        wind_scale: 25.0,
        wind_direction_bias: 26.0,
    };
    let arr = draw.to_array();
    assert_eq!(arr.len(), 26);
    for (i, &val) in arr.iter().enumerate() {
        assert_eq!(val, (i + 1) as f64);
    }
}

#[test]
fn dispersion_draw_default_to_array_len() {
    let arr = DispersionDraw::default().to_array();
    assert_eq!(arr.len(), 26);
    // All zeros except wind_scale which defaults to 1.0
    assert!(arr[..24].iter().all(|&v| v == 0.0));
    assert_eq!(arr[24], 1.0, "wind_scale default is 1.0");
    assert_eq!(arr[25], 0.0, "wind_direction_bias default is 0.0");
}

#[test]
fn test_dispersion_level_parsing() {
    assert_eq!(
        DispersionLevel::from_str("off").unwrap(),
        DispersionLevel::Off
    );
    assert_eq!(
        DispersionLevel::from_str("low").unwrap(),
        DispersionLevel::Low
    );
    assert_eq!(
        DispersionLevel::from_str("medium").unwrap(),
        DispersionLevel::Medium
    );
    assert_eq!(
        DispersionLevel::from_str("high").unwrap(),
        DispersionLevel::High
    );
    assert_eq!(
        DispersionLevel::from_str("custom").unwrap(),
        DispersionLevel::Custom
    );
    assert!(DispersionLevel::from_str("invalid").is_err());
}

#[test]
fn test_density_perturbation_config_off() {
    let cfg = DensityPerturbationConfig::from_level(DispersionLevel::Off);
    assert_eq!(cfg.sigma, 0.0);
    assert_eq!(cfg.tau, 0.0);
}

#[test]
fn test_density_perturbation_config_low() {
    let cfg = DensityPerturbationConfig::from_level(DispersionLevel::Low);
    assert_eq!(cfg.tau, 120.0);
    assert_eq!(cfg.sigma, 0.05);
}

#[test]
fn test_density_perturbation_config_medium() {
    let cfg = DensityPerturbationConfig::from_level(DispersionLevel::Medium);
    assert_eq!(cfg.tau, 60.0);
    assert_eq!(cfg.sigma, 0.10);
}

#[test]
fn test_density_perturbation_config_high() {
    let cfg = DensityPerturbationConfig::from_level(DispersionLevel::High);
    assert_eq!(cfg.tau, 30.0);
    assert_eq!(cfg.sigma, 0.20);
}

#[test]
fn test_density_perturbation_config_custom_defaults_to_medium() {
    let cfg = DensityPerturbationConfig::from_level(DispersionLevel::Custom);
    assert_eq!(cfg.tau, 60.0);
    assert_eq!(cfg.sigma, 0.10);
}

#[test]
fn test_density_perturbation_is_disabled() {
    assert!(
        DensityPerturbationConfig::from_level(DispersionLevel::Off).is_disabled(),
        "Off preset should be disabled"
    );
    assert!(
        !DensityPerturbationConfig::from_level(DispersionLevel::Medium).is_disabled(),
        "Medium preset should not be disabled"
    );
}

#[test]
fn test_step_density_perturbation_disabled_sigma_zero() {
    assert_eq!(step_density_perturbation(0.5, 0.1, 60.0, 0.0, 1.0), 0.0);
}

#[test]
fn test_step_density_perturbation_disabled_tau_zero() {
    assert_eq!(step_density_perturbation(0.5, 0.1, 0.0, 0.10, 1.0), 0.0);
}

#[test]
fn test_step_density_perturbation_deterministic() {
    let a = step_density_perturbation(0.0, 0.1, 60.0, 0.10, 0.5);
    let b = step_density_perturbation(0.0, 0.1, 60.0, 0.10, 0.5);
    assert_eq!(a, b);
}

#[test]
fn test_step_density_perturbation_decay() {
    // With zero noise (normal_sample=0), the state should decay toward 0
    let x = step_density_perturbation(1.0, 0.1, 60.0, 0.10, 0.0);
    assert!(x < 1.0, "state should decay: got {}", x);
    assert!(
        x > 0.0,
        "state should remain positive with no noise: got {}",
        x
    );
}

#[test]
fn test_step_density_perturbation_statistical_properties() {
    // Run many steps from x=0 and check steady-state variance ~ sigma^2
    let tau = 60.0;
    let sigma = 0.10;
    let dt = 0.1;
    let n_steps = 100_000;

    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let normal = Normal::new(0.0, 1.0).unwrap();

    let mut x = 0.0;
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let burn_in = 10_000; // let it reach steady state

    for i in 0..n_steps {
        let z = normal.sample(&mut rng);
        x = step_density_perturbation(x, dt, tau, sigma, z);
        if i >= burn_in {
            sum += x;
            sum_sq += x * x;
        }
    }

    let n = (n_steps - burn_in) as f64;
    let mean = sum / n;
    let variance = sum_sq / n - mean * mean;

    // Mean should be ~0
    assert!(mean.abs() < 0.02, "mean should be ~0, got {}", mean);
    // Variance should be ~sigma^2 = 0.01
    assert!(
        (variance - sigma * sigma).abs() < 0.002,
        "variance should be ~{}, got {}",
        sigma * sigma,
        variance
    );
}

#[test]
fn test_sampling_method_parsing() {
    assert_eq!(
        SamplingMethod::from_str("random").unwrap(),
        SamplingMethod::Random
    );
    assert_eq!(
        SamplingMethod::from_str("lhs").unwrap(),
        SamplingMethod::Lhs
    );
    assert_eq!(
        SamplingMethod::from_str("sobol").unwrap(),
        SamplingMethod::Sobol
    );
    // case-insensitive
    assert_eq!(
        SamplingMethod::from_str("LHS").unwrap(),
        SamplingMethod::Lhs
    );
    assert_eq!(
        SamplingMethod::from_str("Random").unwrap(),
        SamplingMethod::Random
    );
    assert_eq!(
        SamplingMethod::from_str("SOBOL").unwrap(),
        SamplingMethod::Sobol
    );
    // unknown string errors
    assert!(SamplingMethod::from_str("invalid").is_err());
    assert!(SamplingMethod::from_str("").is_err());
}

#[test]
fn test_sampling_method_default_is_random() {
    assert_eq!(SamplingMethod::default(), SamplingMethod::Random);
}

mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn step_always_finite(
            x in -10.0f64..10.0,
            dt in 0.001f64..10.0,
            tau in 0.01f64..1000.0,
            sigma in 0.0f64..1.0,
            z in -5.0f64..5.0,
        ) {
            let result = step_density_perturbation(x, dt, tau, sigma, z);
            prop_assert!(result.is_finite(), "got {}", result);
        }

        #[test]
        fn all_sampling_methods_produce_finite_draws(
            seed in 0u64..10_000,
            n_sims in 1usize..200,
            method_idx in 0u32..3,
        ) {
            let method = match method_idx {
                0 => SamplingMethod::Random,
                1 => SamplingMethod::Lhs,
                _ => SamplingMethod::Sobol,
            };
            let config = DispersionConfig {
                seed,
                sampling: method,
                initial_state: Some(InitialStateSigmas::from_level(DispersionLevel::Medium)),
                atmosphere: Some(AtmosphereSigmas::from_level(DispersionLevel::Medium)),
                aerodynamics: Some(AerodynamicsSigmas::from_level(DispersionLevel::Medium)),
                navigation: Some(NavigationSigmas::from_level(DispersionLevel::Medium)),
                mass: Some(MassSigmas::from_level(DispersionLevel::Medium)),
                vehicle: Some(VehicleSigmas::from_level(DispersionLevel::Medium)),
                pilot: Some(PilotSigmas::from_level(DispersionLevel::Medium)),
                nav_filter: Some(NavFilterSigmas::from_level(DispersionLevel::Medium)),
                wind: None,
                density_perturbation: None,
            };
            let draws = config.generate_draws(n_sims);
            prop_assert_eq!(draws.len(), n_sims);
            for draw in &draws {
                let arr = draw.to_array();
                for &val in &arr {
                    prop_assert!(val.is_finite(), "non-finite draw value: {}", val);
                }
            }
        }
    }
}

// ── Task 2: norm_ppf + DimTransform tests ──────────────────────────────

#[test]
fn test_norm_ppf_known_values() {
    let tol = 1e-6;
    assert!((norm_ppf(0.5) - 0.0).abs() < tol, "p=0.5 -> 0");
    assert!((norm_ppf(0.841344746) - 1.0).abs() < tol, "p=0.841 -> 1");
    assert!(
        (norm_ppf(0.158655254) - (-1.0)).abs() < tol,
        "p=0.159 -> -1"
    );
    assert!((norm_ppf(0.977249868) - 2.0).abs() < tol, "p=0.977 -> 2");
    assert!(
        (norm_ppf(0.022750132) - (-2.0)).abs() < tol,
        "p=0.023 -> -2"
    );
    assert!((norm_ppf(0.998650102) - 3.0).abs() < tol, "p~0.99865 -> 3");
}

#[test]
fn test_norm_ppf_symmetry() {
    let tol = 1e-12;
    for p in [0.01, 0.1, 0.25, 0.4] {
        let sum = norm_ppf(p) + norm_ppf(1.0 - p);
        assert!(sum.abs() < tol, "symmetry failed at p={}: sum={}", p, sum);
    }
}

#[test]
fn test_dim_transform_gaussian() {
    let tx = DimTransform::Gaussian { sigma: 2.0 };
    // u=0.5 -> norm_ppf(0.5)=0.0 -> 0.0*2.0=0.0
    assert!((tx.apply(0.5) - 0.0).abs() < 1e-12);
    // u=0.841344746 -> ~1.0 * 2.0 = ~2.0
    assert!((tx.apply(0.841344746) - 2.0).abs() < 1e-5);
}

#[test]
fn test_dim_transform_uniform() {
    let tx = DimTransform::Uniform { half_width: 5.0 };
    // u=0.5 -> (2*0.5-1)*5 = 0.0
    assert_eq!(tx.apply(0.5), 0.0);
    // u=1.0 -> (2*1.0-1)*5 = 5.0
    assert_eq!(tx.apply(1.0), 5.0);
    // u=0.0 -> (2*0.0-1)*5 = -5.0
    assert_eq!(tx.apply(0.0), -5.0);
}

#[test]
fn test_dim_transform_uniform_range() {
    let tx = DimTransform::UniformRange { min: 0.5, max: 1.5 };
    // u=0.0 -> 0.5 + 0.0*1.0 = 0.5
    assert_eq!(tx.apply(0.0), 0.5);
    // u=1.0 -> 0.5 + 1.0*1.0 = 1.5
    assert_eq!(tx.apply(1.0), 1.5);
    // u=0.5 -> 1.0
    assert_eq!(tx.apply(0.5), 1.0);
}

#[test]
fn test_dim_transform_fixed() {
    let tx = DimTransform::Fixed(42.0);
    assert_eq!(tx.apply(0.0), 42.0);
    assert_eq!(tx.apply(0.5), 42.0);
    assert_eq!(tx.apply(1.0), 42.0);
}

#[test]
fn test_build_dim_transforms_medium_config() {
    let cfg = medium_config(42);
    let txs = cfg.build_dim_transforms();
    // dim 0 (altitude) should be Gaussian
    assert!(
        matches!(txs[0], DimTransform::Gaussian { .. }),
        "dim 0 should be Gaussian"
    );
    // dim 6 (density) should be Uniform
    assert!(
        matches!(txs[6], DimTransform::Uniform { .. }),
        "dim 6 should be Uniform"
    );
    // wind=None -> dim 24 = Fixed(1.0), dim 25 = Fixed(0.0)
    assert_eq!(
        txs[24],
        DimTransform::Fixed(1.0),
        "dim 24 wind=None should be Fixed(1.0)"
    );
    assert_eq!(
        txs[25],
        DimTransform::Fixed(0.0),
        "dim 25 wind=None should be Fixed(0.0)"
    );
}

// ── Task 3: LHS tests ──────────────────────────────────────────────────

#[test]
fn test_lhs_stratification() {
    let cfg = medium_config(42);
    let n = 100usize;
    let samples = cfg.generate_lhs_unit_samples(n);
    assert_eq!(samples.len(), n);
    // Each stratum [k/n, (k+1)/n) must contain exactly one sample per dimension
    for d in 0..DISPERSION_DRAW_LEN {
        let mut stratum_counts = vec![0u32; n];
        for row in &samples {
            let v = row[d];
            assert!(
                (0.0..1.0).contains(&v),
                "dim {} value {} out of [0,1)",
                d,
                v
            );
            let k = (v * n as f64) as usize;
            stratum_counts[k] += 1;
        }
        for (k, &count) in stratum_counts.iter().enumerate() {
            assert_eq!(
                count, 1,
                "dim {} stratum {} has {} samples (expected 1)",
                d, k, count
            );
        }
    }
}

#[test]
fn test_lhs_deterministic() {
    let a = medium_config(7).generate_lhs_unit_samples(50);
    let b = medium_config(7).generate_lhs_unit_samples(50);
    for (row_a, row_b) in a.iter().zip(b.iter()) {
        for (va, vb) in row_a.iter().zip(row_b.iter()) {
            assert_eq!(va, vb);
        }
    }
}

// ── Task 4: Sobol tests ────────────────────────────────────────────────

#[test]
fn test_sobol_bounds() {
    let cfg = medium_config(0);
    let samples = cfg.generate_sobol_unit_samples(1000);
    assert_eq!(samples.len(), 1000);
    for row in &samples {
        for (d, &v) in row.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&v),
                "dim {} value {} out of [0,1]",
                d,
                v
            );
        }
    }
}

#[test]
fn test_sobol_deterministic() {
    let a = medium_config(123).generate_sobol_unit_samples(100);
    let b = medium_config(123).generate_sobol_unit_samples(100);
    for (row_a, row_b) in a.iter().zip(b.iter()) {
        for (va, vb) in row_a.iter().zip(row_b.iter()) {
            assert_eq!(va, vb);
        }
    }
}

#[test]
fn test_sobol_different_seeds() {
    let a = medium_config(1).generate_sobol_unit_samples(50);
    let b = medium_config(2).generate_sobol_unit_samples(50);
    let any_differ = a
        .iter()
        .zip(b.iter())
        .any(|(ra, rb)| ra.iter().zip(rb.iter()).any(|(va, vb)| va != vb));
    assert!(
        any_differ,
        "different seeds should produce different Sobol samples"
    );
}

// ── Task 5: generate_draws() dispatch + from_array() tests ────────────

#[test]
fn test_from_array_roundtrip() {
    let draw = DispersionDraw {
        altitude: 1.0,
        longitude: 2.0,
        latitude: 3.0,
        velocity: 4.0,
        flight_path: 5.0,
        azimuth: 6.0,
        density: 7.0,
        drag_coeff: 8.0,
        lift_coeff: 9.0,
        incidence: 10.0,
        nav_altitude: 11.0,
        nav_longitude: 12.0,
        nav_latitude: 13.0,
        nav_velocity: 14.0,
        nav_flight_path: 15.0,
        nav_azimuth: 16.0,
        nav_drag_accel: 17.0,
        mass: 18.0,
        ref_area: 19.0,
        max_bank_rate: 20.0,
        pilot_tau: 21.0,
        pilot_damping: 22.0,
        pilot_frequency: 23.0,
        filter_gain: 24.0,
        wind_scale: 25.0,
        wind_direction_bias: 26.0,
    };
    let arr = draw.to_array();
    let roundtrip = DispersionDraw::from_array(arr);
    let arr2 = roundtrip.to_array();
    assert_eq!(arr, arr2);
}

#[test]
fn test_generate_draws_lhs_produces_valid_draws() {
    let mut config = medium_config(42);
    config.sampling = SamplingMethod::Lhs;
    let draws = config.generate_draws(100);
    assert_eq!(draws.len(), 100);
    for draw in &draws {
        assert!(draw.altitude.is_finite());
        assert!(draw.velocity.is_finite());
        assert!(draw.density.is_finite());
        assert!(draw.wind_scale.is_finite());
    }
}

#[test]
fn test_generate_draws_sobol_produces_valid_draws() {
    let mut config = medium_config(42);
    config.sampling = SamplingMethod::Sobol;
    let draws = config.generate_draws(100);
    assert_eq!(draws.len(), 100);
    for draw in &draws {
        assert!(draw.altitude.is_finite());
        assert!(draw.velocity.is_finite());
        assert!(draw.density.is_finite());
        assert!(draw.wind_scale.is_finite());
    }
}

#[test]
#[should_panic(expected = "Sobol sampling limited to 65536")]
fn test_sobol_rejects_too_many_sims() {
    let mut config = medium_config(42);
    config.sampling = SamplingMethod::Sobol;
    config.generate_draws(70_000);
}
