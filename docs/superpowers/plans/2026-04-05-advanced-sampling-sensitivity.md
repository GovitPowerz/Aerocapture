# Advanced Sampling & Sensitivity Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add LHS + Sobol quasi-random sampling to the Rust MC engine and Morris/Sobol sensitivity analysis to the Python toolchain.

**Architecture:** Sampling methods live in Rust `dispersions.rs` (TOML-selected, backward-compatible `"random"` default). Sensitivity analysis is a Python-only post-processing module (`sensitivity.py`) that drives sims via a new `run_with_draws()` PyO3 API accepting pre-computed (N,26) draw matrices. Charts integrate into the existing `charts.py` / `report.py` / `report.typ` pipeline.

**Tech Stack:** Rust (`sobol_burley` 0.5 crate), Python (`SALib` >= 1.5), existing PyO3/numpy/matplotlib stack.

**Spec:** `docs/superpowers/specs/2026-04-05-advanced-sampling-sensitivity-design.md`

---

### Task 1: Rust Dependencies + SamplingMethod Enum + TOML Parsing

**Files:**
- Modify: `src/rust/Cargo.toml:10-17`
- Modify: `src/rust/src/data/dispersions.rs:395-409`
- Modify: `src/rust/src/config.rs:817-830`
- Modify: `src/rust/src/data/mod.rs:893-905`
- Test: inline `#[cfg(test)]` in `dispersions.rs`

- [ ] **Step 1: Add sobol_burley dependency**

In `src/rust/Cargo.toml`, add to `[dependencies]`:

```toml
sobol_burley = "0.5"
```

- [ ] **Step 2: Add SamplingMethod enum to dispersions.rs**

After the `DispersionLevel` enum (around line 39), add:

```rust
/// Sampling strategy for generating dispersion draws.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum SamplingMethod {
    /// Standard pseudo-random (existing behavior).
    #[default]
    Random,
    /// Latin Hypercube Sampling (stratified).
    Lhs,
    /// Owen-scrambled Sobol quasi-random sequence.
    Sobol,
}

impl SamplingMethod {
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "random" => Ok(Self::Random),
            "lhs" => Ok(Self::Lhs),
            "sobol" => Ok(Self::Sobol),
            other => Err(format!("Unknown sampling method: '{}' (expected random/lhs/sobol)", other)),
        }
    }
}
```

- [ ] **Step 3: Add sampling field to DispersionConfig**

In `dispersions.rs`, add `sampling` to `DispersionConfig`:

```rust
pub struct DispersionConfig {
    pub seed: u64,
    pub sampling: SamplingMethod,
    pub initial_state: Option<InitialStateSigmas>,
    // ... rest unchanged
}
```

- [ ] **Step 4: Add sampling field to TomlMonteCarlo**

In `config.rs`, add to `TomlMonteCarlo`:

```rust
pub struct TomlMonteCarlo {
    pub seed: u64,
    #[serde(default)]
    pub sampling: Option<String>,
    pub initial_state: Option<TomlMcDomain>,
    // ... rest unchanged
}
```

- [ ] **Step 5: Parse sampling in build_dispersion_config()**

In `src/rust/src/data/mod.rs`, in `build_dispersion_config()`, parse the sampling field and pass it through to `DispersionConfig`. Before the final `Ok(DispersionConfig { ... })` block:

```rust
    let sampling = if let Some(ref s) = mc.sampling {
        SamplingMethod::from_str(s).map_err(|e| DataError(e))?
    } else {
        SamplingMethod::default()
    };

    Ok(DispersionConfig {
        seed: mc.seed,
        sampling,
        initial_state,
        // ... rest unchanged
    })
```

- [ ] **Step 6: Update medium_config test helper**

In the `#[cfg(test)]` module of `dispersions.rs`, update the `medium_config()` helper to include the new field:

```rust
    fn medium_config(seed: u64) -> DispersionConfig {
        DispersionConfig {
            seed,
            sampling: SamplingMethod::Random,
            initial_state: Some(InitialStateSigmas::from_level(DispersionLevel::Medium)),
            // ... rest unchanged
        }
    }
```

- [ ] **Step 7: Write test for SamplingMethod parsing**

Add to the test module in `dispersions.rs`:

```rust
    #[test]
    fn test_sampling_method_parsing() {
        assert_eq!(SamplingMethod::from_str("random").unwrap(), SamplingMethod::Random);
        assert_eq!(SamplingMethod::from_str("lhs").unwrap(), SamplingMethod::Lhs);
        assert_eq!(SamplingMethod::from_str("sobol").unwrap(), SamplingMethod::Sobol);
        assert_eq!(SamplingMethod::from_str("LHS").unwrap(), SamplingMethod::Lhs);
        assert!(SamplingMethod::from_str("invalid").is_err());
    }
```

- [ ] **Step 8: Run tests to verify**

Run: `cd src/rust && cargo test -- dispersions`
Expected: all existing tests pass, new test passes.

- [ ] **Step 9: Commit**

```bash
git add src/rust/Cargo.toml src/rust/src/data/dispersions.rs src/rust/src/config.rs src/rust/src/data/mod.rs
git commit -m "feat(rust): add SamplingMethod enum + TOML parsing + sobol_burley dep"
```

---

### Task 2: Inverse Normal CDF + Dimension Transform Types

**Files:**
- Modify: `src/rust/src/data/dispersions.rs`
- Test: inline `#[cfg(test)]` in `dispersions.rs`

- [ ] **Step 1: Write test for norm_ppf**

Add to the test module in `dispersions.rs`:

```rust
    #[test]
    fn test_norm_ppf_known_values() {
        // Standard normal quantiles: P(Z <= z) = p
        let cases = [
            (0.5, 0.0),
            (0.841344746, 1.0),
            (0.158655254, -1.0),
            (0.977249868, 2.0),
            (0.022750132, -2.0),
            (0.999_866_39, 3.0),
        ];
        for (p, expected_z) in cases {
            let z = norm_ppf(p);
            assert!(
                (z - expected_z).abs() < 1e-6,
                "norm_ppf({}) = {}, expected {}",
                p, z, expected_z,
            );
        }
    }

    #[test]
    fn test_norm_ppf_symmetry() {
        for &p in &[0.01, 0.1, 0.25, 0.4] {
            let low = norm_ppf(p);
            let high = norm_ppf(1.0 - p);
            assert!(
                (low + high).abs() < 1e-9,
                "norm_ppf({}) + norm_ppf({}) = {}, expected 0",
                p, 1.0 - p, low + high,
            );
        }
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd src/rust && cargo test -- norm_ppf`
Expected: FAIL with `cannot find function norm_ppf`

- [ ] **Step 3: Implement norm_ppf**

Add to `dispersions.rs`, after the imports, before the structs:

```rust
/// Inverse standard normal CDF (quantile function).
///
/// Peter Acklam's rational approximation, accurate to ~1.15e-9.
/// Input: p in (0, 1). Output: z such that P(Z <= z) = p for Z ~ N(0,1).
fn norm_ppf(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_690e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239e0,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838e0,
        -2.549_732_539_343_734e0,
        4.374_664_141_464_968e0,
        2.938_163_982_698_783e0,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996e0,
        3.754_408_661_907_416e0,
    ];

    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd src/rust && cargo test -- norm_ppf`
Expected: PASS

- [ ] **Step 5: Write test for DimTransform**

Add to the test module:

```rust
    #[test]
    fn test_dim_transform_gaussian() {
        let t = DimTransform::Gaussian { sigma: 2.0 };
        // u=0.5 -> norm_ppf(0.5)=0.0 -> 0.0 * 2.0 = 0.0
        assert!((t.apply(0.5)).abs() < 1e-9);
        // u=0.841... -> norm_ppf ~= 1.0 -> 1.0 * 2.0 = 2.0
        let val = t.apply(0.841344746);
        assert!((val - 2.0).abs() < 1e-5, "got {}", val);
    }

    #[test]
    fn test_dim_transform_uniform() {
        let t = DimTransform::Uniform { half_width: 0.5 };
        // u=0.0 -> (2*0 - 1) * 0.5 = -0.5
        assert!((t.apply(0.0) - (-0.5)).abs() < 1e-9);
        // u=1.0 -> (2*1 - 1) * 0.5 = 0.5
        assert!((t.apply(1.0) - 0.5).abs() < 1e-9);
        // u=0.5 -> (2*0.5 - 1) * 0.5 = 0.0
        assert!((t.apply(0.5)).abs() < 1e-9);
    }

    #[test]
    fn test_dim_transform_uniform_range() {
        let t = DimTransform::UniformRange { min: 0.5, max: 1.5 };
        assert!((t.apply(0.0) - 0.5).abs() < 1e-9);
        assert!((t.apply(1.0) - 1.5).abs() < 1e-9);
        assert!((t.apply(0.5) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_dim_transform_fixed() {
        let t = DimTransform::Fixed(1.0);
        assert_eq!(t.apply(0.0), 1.0);
        assert_eq!(t.apply(0.5), 1.0);
        assert_eq!(t.apply(1.0), 1.0);
    }
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cd src/rust && cargo test -- dim_transform`
Expected: FAIL with `cannot find type DimTransform`

- [ ] **Step 7: Implement DimTransform**

Add to `dispersions.rs`, after `norm_ppf()`:

```rust
/// Transform specification for a single dispersion dimension.
/// Maps a unit-uniform [0,1] sample to the target distribution.
#[derive(Debug, Clone, Copy)]
enum DimTransform {
    /// Inverse normal CDF, then scale by sigma.
    Gaussian { sigma: f64 },
    /// Linear map: (2u - 1) * half_width.
    Uniform { half_width: f64 },
    /// Linear map: min + u * (max - min).
    UniformRange { min: f64, max: f64 },
    /// Constant value (dimension is off).
    Fixed(f64),
}

impl DimTransform {
    fn apply(&self, u: f64) -> f64 {
        match self {
            Self::Gaussian { sigma } => norm_ppf(u) * sigma,
            Self::Uniform { half_width } => (2.0 * u - 1.0) * half_width,
            Self::UniformRange { min, max } => min + u * (max - min),
            Self::Fixed(val) => *val,
        }
    }
}
```

- [ ] **Step 8: Implement build_dim_transforms()**

Add to the `impl DispersionConfig` block:

```rust
    /// Build the 26-element transform array for LHS/Sobol two-stage draw generation.
    ///
    /// Each entry maps a [0,1] unit sample to the target distribution for that dimension.
    /// Order matches `DispersionDraw` field order and `to_array()`.
    fn build_dim_transforms(&self) -> [DimTransform; DISPERSION_DRAW_LEN] {
        let f = DimTransform::Fixed(0.0);
        let f1 = DimTransform::Fixed(1.0);

        // Initial state (Gaussian, 6 dims)
        let (d0, d1, d2, d3, d4, d5) = if let Some(ref s) = self.initial_state {
            (
                DimTransform::Gaussian { sigma: s.altitude * 1e3 },
                DimTransform::Gaussian { sigma: s.longitude * DEG2RAD },
                DimTransform::Gaussian { sigma: s.latitude * DEG2RAD },
                DimTransform::Gaussian { sigma: s.velocity },
                DimTransform::Gaussian { sigma: s.flight_path * DEG2RAD },
                DimTransform::Gaussian { sigma: s.azimuth * DEG2RAD },
            )
        } else {
            (f, f, f, f, f, f)
        };

        // Atmosphere (Uniform, 1 dim)
        let d6 = if let Some(ref s) = self.atmosphere {
            DimTransform::Uniform { half_width: s.density / 100.0 }
        } else {
            f
        };

        // Aerodynamics (Uniform, 3 dims)
        let (d7, d8, d9) = if let Some(ref s) = self.aerodynamics {
            (
                DimTransform::Uniform { half_width: s.drag / 100.0 },
                DimTransform::Uniform { half_width: s.lift / 100.0 },
                DimTransform::Uniform { half_width: s.incidence * DEG2RAD },
            )
        } else {
            (f, f, f)
        };

        // Navigation (Gaussian, 7 dims)
        let (d10, d11, d12, d13, d14, d15, d16) = if let Some(ref s) = self.navigation {
            (
                DimTransform::Gaussian { sigma: s.altitude * 1e3 },
                DimTransform::Gaussian { sigma: s.longitude * DEG2RAD },
                DimTransform::Gaussian { sigma: s.latitude * DEG2RAD },
                DimTransform::Gaussian { sigma: s.velocity },
                DimTransform::Gaussian { sigma: s.flight_path * DEG2RAD },
                DimTransform::Gaussian { sigma: s.azimuth * DEG2RAD },
                DimTransform::Gaussian { sigma: s.drag_accel },
            )
        } else {
            (f, f, f, f, f, f, f)
        };

        // Mass (Uniform, 1 dim)
        let d17 = if let Some(ref s) = self.mass {
            DimTransform::Uniform { half_width: s.mass / 100.0 }
        } else {
            f
        };

        // Vehicle (Uniform, 2 dims)
        let (d18, d19) = if let Some(ref s) = self.vehicle {
            (
                DimTransform::Uniform { half_width: s.ref_area / 100.0 },
                DimTransform::Uniform { half_width: s.max_bank_rate / 100.0 },
            )
        } else {
            (f, f)
        };

        // Pilot (Uniform, 3 dims)
        let (d20, d21, d22) = if let Some(ref s) = self.pilot {
            (
                DimTransform::Uniform { half_width: s.time_constant / 100.0 },
                DimTransform::Uniform { half_width: s.damping / 100.0 },
                DimTransform::Uniform { half_width: s.frequency / 100.0 },
            )
        } else {
            (f, f, f)
        };

        // Nav filter (Gaussian, 1 dim)
        let d23 = if let Some(ref s) = self.nav_filter {
            DimTransform::Gaussian { sigma: s.filter_gain }
        } else {
            f
        };

        // Wind (2 dims: scale = UniformRange, direction = Uniform)
        let (d24, d25) = if let Some(ref w) = self.wind {
            (
                DimTransform::UniformRange { min: w.scale_min, max: w.scale_max },
                DimTransform::Uniform { half_width: w.direction_bias_deg * DEG2RAD },
            )
        } else {
            (f1, f) // wind_scale defaults to 1.0
        };

        [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23, d24, d25]
    }
```

- [ ] **Step 9: Write test for build_dim_transforms**

Add to the test module:

```rust
    #[test]
    fn test_build_dim_transforms_medium_config() {
        let config = medium_config(42);
        let transforms = config.build_dim_transforms();
        assert_eq!(transforms.len(), DISPERSION_DRAW_LEN);
        // Initial state dims should be Gaussian
        assert!(matches!(transforms[0], DimTransform::Gaussian { .. }));
        // Atmosphere should be Uniform
        assert!(matches!(transforms[6], DimTransform::Uniform { .. }));
        // Wind dims should be Fixed (wind=None in medium_config)
        assert!(matches!(transforms[24], DimTransform::Fixed(v) if v == 1.0));
        assert!(matches!(transforms[25], DimTransform::Fixed(v) if v == 0.0));
    }
```

- [ ] **Step 10: Run all tests**

Run: `cd src/rust && cargo test -- dispersions`
Expected: all PASS

- [ ] **Step 11: Commit**

```bash
git add src/rust/src/data/dispersions.rs
git commit -m "feat(rust): add norm_ppf, DimTransform, build_dim_transforms for two-stage sampling"
```

---

### Task 3: LHS Implementation

**Files:**
- Modify: `src/rust/src/data/dispersions.rs`
- Test: inline `#[cfg(test)]`

- [ ] **Step 1: Write test for LHS stratification**

Add to the test module:

```rust
    #[test]
    fn test_lhs_stratification() {
        let config = medium_config(42);
        let n = 100;
        let samples = config.generate_lhs_unit_samples(n);
        assert_eq!(samples.len(), n);
        assert_eq!(samples[0].len(), DISPERSION_DRAW_LEN);

        // Stratification: for each dimension, each stratum [k/N, (k+1)/N]
        // should contain exactly one sample.
        for dim in 0..DISPERSION_DRAW_LEN {
            let mut strata_hit = vec![false; n];
            for row in &samples {
                let u = row[dim];
                assert!((0.0..1.0).contains(&u), "u={} out of [0,1)", u);
                let stratum = (u * n as f64) as usize;
                let stratum = stratum.min(n - 1); // clamp edge case
                assert!(!strata_hit[stratum], "dim {} stratum {} hit twice", dim, stratum);
                strata_hit[stratum] = true;
            }
            assert!(strata_hit.iter().all(|&h| h), "dim {} missing strata", dim);
        }
    }

    #[test]
    fn test_lhs_deterministic() {
        let config = medium_config(42);
        let a = config.generate_lhs_unit_samples(50);
        let b = config.generate_lhs_unit_samples(50);
        for (ra, rb) in a.iter().zip(b.iter()) {
            for (va, vb) in ra.iter().zip(rb.iter()) {
                assert_eq!(va, vb);
            }
        }
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd src/rust && cargo test -- test_lhs`
Expected: FAIL with `no method named generate_lhs_unit_samples`

- [ ] **Step 3: Implement LHS**

Add to the `impl DispersionConfig` block:

```rust
    /// Generate N Latin Hypercube samples in [0,1]^26.
    ///
    /// For each dimension, the [0,1] interval is divided into N equal strata.
    /// A Fisher-Yates shuffle assigns one sample per stratum, with uniform
    /// jitter within each stratum.
    fn generate_lhs_unit_samples(&self, n: usize) -> Vec<[f64; DISPERSION_DRAW_LEN]> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        let n_f = n as f64;

        // For each dimension, create a shuffled permutation of [0..n]
        let mut perms = [[0usize; 0]; 0]; // placeholder
        let mut perm_vecs: Vec<Vec<usize>> = (0..DISPERSION_DRAW_LEN)
            .map(|_| {
                let mut perm: Vec<usize> = (0..n).collect();
                // Fisher-Yates shuffle
                for i in (1..n).rev() {
                    let j = rng.random_range(0..=i);
                    perm.swap(i, j);
                }
                perm
            })
            .collect();

        (0..n)
            .map(|i| {
                let mut sample = [0.0f64; DISPERSION_DRAW_LEN];
                for d in 0..DISPERSION_DRAW_LEN {
                    let stratum = perm_vecs[d][i] as f64;
                    let jitter: f64 = rng.random();
                    sample[d] = (stratum + jitter) / n_f;
                }
                sample
            })
            .collect()
    }
```

Requires `use rand::Rng;` at top of file. In `rand 0.10`, the methods are `rng.random_range(0..=i)` for range and `rng.random::<f64>()` for unit float.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd src/rust && cargo test -- test_lhs`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rust/src/data/dispersions.rs
git commit -m "feat(rust): implement Latin Hypercube Sampling unit sample generation"
```

---

### Task 4: Sobol Implementation

**Files:**
- Modify: `src/rust/src/data/dispersions.rs`
- Test: inline `#[cfg(test)]`

- [ ] **Step 1: Write test for Sobol bounds and determinism**

Add to the test module:

```rust
    #[test]
    fn test_sobol_bounds() {
        let config = medium_config(42);
        let samples = config.generate_sobol_unit_samples(1000);
        assert_eq!(samples.len(), 1000);
        for row in &samples {
            assert_eq!(row.len(), DISPERSION_DRAW_LEN);
            for &u in row {
                assert!((0.0..=1.0).contains(&u), "u={} out of [0,1]", u);
            }
        }
    }

    #[test]
    fn test_sobol_deterministic() {
        let config = medium_config(42);
        let a = config.generate_sobol_unit_samples(100);
        let b = config.generate_sobol_unit_samples(100);
        for (ra, rb) in a.iter().zip(b.iter()) {
            for (va, vb) in ra.iter().zip(rb.iter()) {
                assert_eq!(va, vb);
            }
        }
    }

    #[test]
    fn test_sobol_different_seeds() {
        let a = medium_config(42).generate_sobol_unit_samples(10);
        let b = medium_config(99).generate_sobol_unit_samples(10);
        let any_differ = a.iter().zip(b.iter()).any(|(ra, rb)| ra[0] != rb[0]);
        assert!(any_differ, "Different seeds should produce different Sobol sequences");
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd src/rust && cargo test -- test_sobol`
Expected: FAIL with `no method named generate_sobol_unit_samples`

- [ ] **Step 3: Implement Sobol**

Add to the `impl DispersionConfig` block:

```rust
    /// Generate N Owen-scrambled Sobol quasi-random samples in [0,1]^26.
    ///
    /// Uses `sobol_burley` crate. Maximum 65536 samples (2^16 limit).
    fn generate_sobol_unit_samples(&self, n: usize) -> Vec<[f64; DISPERSION_DRAW_LEN]> {
        let seed = self.seed as u32;
        (0..n)
            .map(|i| {
                let mut sample = [0.0f64; DISPERSION_DRAW_LEN];
                for d in 0..DISPERSION_DRAW_LEN {
                    sample[d] = sobol_burley::sample(i as u32, d as u32, seed) as f64;
                }
                sample
            })
            .collect()
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd src/rust && cargo test -- test_sobol`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rust/src/data/dispersions.rs
git commit -m "feat(rust): implement Sobol quasi-random unit sample generation via sobol_burley"
```

---

### Task 5: Wire Sampling Methods into generate_draws()

**Files:**
- Modify: `src/rust/src/data/dispersions.rs`
- Test: inline `#[cfg(test)]`

- [ ] **Step 1: Write test for LHS draw generation end-to-end**

```rust
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd src/rust && cargo test -- test_generate_draws_lhs`
Expected: FAIL (LHS path not wired yet)

- [ ] **Step 3: Add from_array to DispersionDraw**

Add to the `impl DispersionDraw` block:

```rust
    /// Construct from a flat array (inverse of `to_array()`).
    pub fn from_array(a: [f64; DISPERSION_DRAW_LEN]) -> Self {
        Self {
            altitude: a[0],
            longitude: a[1],
            latitude: a[2],
            velocity: a[3],
            flight_path: a[4],
            azimuth: a[5],
            density: a[6],
            drag_coeff: a[7],
            lift_coeff: a[8],
            incidence: a[9],
            nav_altitude: a[10],
            nav_longitude: a[11],
            nav_latitude: a[12],
            nav_velocity: a[13],
            nav_flight_path: a[14],
            nav_azimuth: a[15],
            nav_drag_accel: a[16],
            mass: a[17],
            ref_area: a[18],
            max_bank_rate: a[19],
            pilot_tau: a[20],
            pilot_damping: a[21],
            pilot_frequency: a[22],
            filter_gain: a[23],
            wind_scale: a[24],
            wind_direction_bias: a[25],
        }
    }
```

- [ ] **Step 4: Add draws_from_unit_samples helper**

Add to the `impl DispersionConfig` block, before `generate_draws`:

```rust
    /// Transform unit samples [0,1]^26 into DispersionDraws using the dimension transforms.
    fn draws_from_unit_samples(&self, unit_samples: &[[f64; DISPERSION_DRAW_LEN]]) -> Vec<DispersionDraw> {
        let transforms = self.build_dim_transforms();
        unit_samples
            .iter()
            .map(|row| {
                let arr: [f64; DISPERSION_DRAW_LEN] = std::array::from_fn(|d| transforms[d].apply(row[d]));
                DispersionDraw::from_array(arr)
            })
            .collect()
    }
```

- [ ] **Step 5: Update generate_draws() with dispatch**

Replace the `generate_draws` method body to dispatch based on sampling method. Keep existing code for `Random`, delegate to new methods for `Lhs`/`Sobol`:

```rust
    pub fn generate_draws(&self, n_sims: usize) -> Vec<DispersionDraw> {
        match self.sampling {
            SamplingMethod::Random => self.generate_draws_random(n_sims),
            SamplingMethod::Lhs => {
                let unit_samples = self.generate_lhs_unit_samples(n_sims);
                self.draws_from_unit_samples(&unit_samples)
            }
            SamplingMethod::Sobol => {
                assert!(
                    n_sims <= 65_536,
                    "Sobol sampling limited to 65536 samples, got {}",
                    n_sims,
                );
                let unit_samples = self.generate_sobol_unit_samples(n_sims);
                self.draws_from_unit_samples(&unit_samples)
            }
        }
    }
```

Rename the existing `generate_draws` body to `generate_draws_random`:

```rust
    /// Original pseudo-random draw generation (backward-compatible).
    fn generate_draws_random(&self, n_sims: usize) -> Vec<DispersionDraw> {
        // ... exact existing code from the current generate_draws body ...
    }
```

- [ ] **Step 6: Run all tests**

Run: `cd src/rust && cargo test -- dispersions`
Expected: all PASS including new LHS/Sobol end-to-end tests and existing Random tests unchanged.

- [ ] **Step 7: Add proptest for all sampling methods**

Add to the proptests submodule:

```rust
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
            let mut config = DispersionConfig {
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
```

- [ ] **Step 8: Run proptests**

Run: `cd src/rust && cargo test -- proptests`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add src/rust/src/data/dispersions.rs
git commit -m "feat(rust): wire LHS/Sobol into generate_draws with dispatch + from_array + proptest"
```

---

### Task 6: PyO3 run_with_draws() API

**Files:**
- Modify: `src/rust/src/simulation/runner.rs`
- Modify: `src/rust/aerocapture-py/src/lib.rs`
- Modify: `src/rust/aerocapture-py/src/batch.rs`
- Test: `tests/test_pyo3.py`

- [ ] **Step 1: Add run_for_api_with_draws to runner.rs**

In `src/rust/src/simulation/runner.rs`, add after `run_for_api()`:

```rust
/// Run simulations with externally supplied dispersion draws (no file I/O).
///
/// Bypasses `generate_draws()` entirely. Each draw in `external_draws` produces
/// one simulation. Used by the PyO3 sensitivity analysis path.
pub fn run_for_api_with_draws(
    config: &SimInput,
    data: &SimData,
    external_draws: Vec<crate::data::dispersions::DispersionDraw>,
    include_trajectories: bool,
    wall_timeout: Option<Duration>,
) -> Result<Vec<crate::RunOutput>, SimError> {
    use crate::data::dispersions::DISPERSION_DRAW_LEN;

    let n_sims = external_draws.len();
    let run_states: Vec<(init::RunState, [f64; DISPERSION_DRAW_LEN])> = external_draws
        .iter()
        .map(|draw| (init::init_run_from_draw(data, draw), draw.to_array()))
        .collect();

    let results: Vec<SimResult> = if n_sims > 1 {
        run_states
            .par_iter()
            .enumerate()
            .map(|(idx, (run_state, disp_array))| {
                let mut result =
                    run_single(config, data, run_state, idx as i32, include_trajectories, wall_timeout)?;
                result.dispersions = *disp_array;
                Ok(result)
            })
            .collect::<Result<Vec<_>, _>>()?
    } else if n_sims == 1 {
        let (run_state, disp_array) = &run_states[0];
        let mut result = run_single(config, data, run_state, 0, include_trajectories, wall_timeout)?;
        result.dispersions = *disp_array;
        vec![result]
    } else {
        return Ok(Vec::new());
    };

    Ok(results
        .into_iter()
        .map(|r| {
            let energy = r.final_line[7];
            let ecc = r.final_line[9];
            let trajectory = if include_trajectories {
                r.photo_lines
                    .iter()
                    .map(|p| {
                        [
                            p[1], p[2], p[3], p[4], p[5], p[6], p[24], p[0],
                            p[18] / 1e6, p[19] / 1e3, p[14], p[9], p[25],
                            p[26], p[27], p[28], p[29],
                        ]
                    })
                    .collect()
            } else {
                Vec::new()
            };
            let ifinal_val = r.final_line[31] as i32;
            crate::RunOutput {
                trajectory,
                final_record: r.final_line,
                captured: ifinal_val == 3 && ecc < 1.0 && energy < 0.0,
                dispersions: r.dispersions,
            }
        })
        .collect())
}
```

- [ ] **Step 2: Add run_with_external_draws to batch.rs**

In `src/rust/aerocapture-py/src/batch.rs`, add:

```rust
/// Run simulations with pre-computed dispersion draws.
///
/// Each row in `draws` is a 26-element array mapping to DispersionDraw fields.
pub fn run_with_external_draws(
    toml_path: &Path,
    overrides: Vec<(String, OverrideValue)>,
    draws: Vec<[f64; 26]>,
    include_trajectories: bool,
    wall_timeout: Option<Duration>,
) -> Result<Vec<RunOutput>, String> {
    use aerocapture::data::dispersions::{DispersionDraw, DISPERSION_DRAW_LEN};

    let (config, data) = crate::config::load_and_override(toml_path, &overrides)?;

    let dispersion_draws: Vec<DispersionDraw> = draws
        .into_iter()
        .map(|arr| DispersionDraw::from_array(arr))
        .collect();

    aerocapture::simulation::runner::run_for_api_with_draws(
        &config,
        &data,
        dispersion_draws,
        include_trajectories,
        wall_timeout,
    )
    .map_err(|e| format!("Simulation error: {}", e))
}
```

Add necessary imports at the top of batch.rs:

```rust
use aerocapture::RunOutput;
```

- [ ] **Step 3: Add run_with_draws PyO3 function to lib.rs**

In `src/rust/aerocapture-py/src/lib.rs`, add the function:

```rust
#[pyfunction]
#[pyo3(signature = (toml_path, draws, overrides=None, include_trajectories=false, sim_timeout_secs=None))]
fn run_with_draws(
    toml_path: &str,
    draws: numpy::PyReadonlyArray2<'_, f64>,
    overrides: Option<&Bound<'_, PyDict>>,
    include_trajectories: bool,
    sim_timeout_secs: Option<f64>,
) -> PyResult<BatchResults> {
    let overrides_vec = match overrides {
        Some(d) => extract_overrides(d)?,
        None => Vec::new(),
    };
    let wall_timeout = sim_timeout_secs.map(Duration::from_secs_f64);

    let draws_array = draws.as_array();
    let n_rows = draws_array.nrows();
    let n_cols = draws_array.ncols();
    if n_cols != 26 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("draws must have 26 columns, got {}", n_cols),
        ));
    }

    let draws_vec: Vec<[f64; 26]> = (0..n_rows)
        .map(|i| {
            let mut arr = [0.0f64; 26];
            for j in 0..26 {
                arr[j] = draws_array[[i, j]];
            }
            arr
        })
        .collect();

    let outputs = batch::run_with_external_draws(
        Path::new(toml_path),
        overrides_vec,
        draws_vec,
        include_trajectories,
        wall_timeout,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    Ok(BatchResults::from_outputs(outputs, include_trajectories))
}
```

Register in the module:

```rust
m.add_function(wrap_pyfunction!(run_with_draws, m)?)?;
```

- [ ] **Step 4: Build PyO3 bindings**

Run from repo root: `uv run maturin develop --release --manifest-path src/rust/aerocapture-py/Cargo.toml`
Expected: builds successfully

- [ ] **Step 5: Write Python test for run_with_draws**

Add to `tests/test_pyo3.py`:

```python
class TestRunWithDraws:
    def test_run_with_draws_returns_batch_results(self) -> None:
        # Create 5 zero-dispersion draws (nominal conditions)
        draws = np.zeros((5, 26), dtype=np.float64)
        draws[:, 24] = 1.0  # wind_scale = 1.0 (identity)
        result = aero.run_with_draws(GOLDEN_TOML, draws)
        assert len(result) == 5
        assert result.final_records.shape == (5, 52)

    def test_run_with_draws_wrong_columns(self) -> None:
        draws = np.zeros((5, 10), dtype=np.float64)
        with pytest.raises(ValueError, match="26 columns"):
            aero.run_with_draws(GOLDEN_TOML, draws)

    def test_run_with_draws_dispersions_roundtrip(self) -> None:
        # The dispersions output should match the input draws
        draws = np.zeros((3, 26), dtype=np.float64)
        draws[:, 24] = 1.0  # wind_scale
        draws[0, 3] = 5.0   # velocity offset
        draws[1, 6] = 0.1   # density bias 10%
        result = aero.run_with_draws(GOLDEN_TOML, draws)
        np.testing.assert_allclose(result.dispersions, draws, atol=1e-12)
```

- [ ] **Step 6: Run PyO3 tests**

Run: `uv run pytest tests/test_pyo3.py::TestRunWithDraws -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/rust/src/simulation/runner.rs src/rust/aerocapture-py/src/lib.rs src/rust/aerocapture-py/src/batch.rs tests/test_pyo3.py
git commit -m "feat(pyo3): add run_with_draws API for externally supplied dispersion draws"
```

---

### Task 7: Rust Integration Test for All Sampling Methods

**Files:**
- Modify: `src/rust/tests/e2e.rs` (or new file)
- Test: cargo test --test

- [ ] **Step 1: Write integration test**

Add to `src/rust/tests/e2e.rs`:

```rust
#[test]
fn lhs_sampling_completes() {
    let output = run_sim_with_override(
        "nominal/msr_aller_reference.toml",
        &[("monte_carlo.sampling", "lhs")],
    );
    assert!(output.status.success(), "LHS sim failed: {}", String::from_utf8_lossy(&output.stderr));
}

#[test]
fn sobol_sampling_completes() {
    let output = run_sim_with_override(
        "nominal/msr_aller_reference.toml",
        &[("monte_carlo.sampling", "sobol")],
    );
    assert!(output.status.success(), "Sobol sim failed: {}", String::from_utf8_lossy(&output.stderr));
}
```

If `run_sim_with_override` doesn't exist, use the CLI binary with a TOML that has `sampling = "lhs"`. Alternatively, test via `run_for_api()` in a Rust integration test. Adapt to the existing test infrastructure in e2e.rs.

- [ ] **Step 2: Run integration tests**

Run: `cd src/rust && cargo test --test e2e`
Expected: PASS

- [ ] **Step 3: Run full Rust test suite**

Run: `cd src/rust && cargo test`
Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add src/rust/tests/
git commit -m "test(rust): add integration tests for LHS and Sobol sampling methods"
```

---

### Task 8: SALib Dependency + Problem Definition Builder

**Files:**
- Modify: `pyproject.toml`
- Create: `src/python/aerocapture/training/sensitivity.py`
- Test: `tests/test_sensitivity.py`

- [ ] **Step 1: Add SALib dependency**

In `pyproject.toml`, add to `dependencies`:

```toml
    "SALib>=1.5",
```

- [ ] **Step 2: Install dependencies**

Run: `uv sync`
Expected: SALib installed successfully

- [ ] **Step 3: Write test for problem builder**

Create `tests/test_sensitivity.py`:

```python
"""Tests for aerocapture.training.sensitivity module."""

from __future__ import annotations

import pytest


class TestBuildProblem:
    def test_build_problem_returns_salib_dict(self) -> None:
        from aerocapture.training.sensitivity import DISPERSION_COLUMNS, build_problem

        mc_config = {
            "seed": 42,
            "initial_state": {"level": "medium"},
            "atmosphere": {"level": "medium"},
            "aerodynamics": {"level": "medium"},
            "navigation": {"level": "medium"},
            "mass": {"level": "medium"},
            "vehicle": {"level": "medium"},
            "pilot": {"level": "medium"},
            "nav_filter": {"level": "medium"},
        }
        problem = build_problem(mc_config)
        assert problem["num_vars"] == 26
        assert problem["names"] == DISPERSION_COLUMNS
        assert len(problem["bounds"]) == 26
        assert len(problem["dists"]) == 26

    def test_build_problem_distribution_types(self) -> None:
        from aerocapture.training.sensitivity import build_problem

        mc_config = {
            "seed": 42,
            "initial_state": {"level": "medium"},
            "atmosphere": {"level": "medium"},
            "aerodynamics": {"level": "medium"},
            "navigation": {"level": "medium"},
            "mass": {"level": "medium"},
            "vehicle": {"level": "medium"},
            "pilot": {"level": "medium"},
            "nav_filter": {"level": "medium"},
        }
        problem = build_problem(mc_config)
        # Initial state dims (0-5) should be Gaussian
        assert problem["dists"][0] == "norm"
        # Atmosphere dim (6) should be Uniform
        assert problem["dists"][6] == "unif"
        # Wind dims (24-25) with no wind config -> fixed (zero-width uniform)
        assert problem["dists"][24] == "unif"

    def test_dispersion_columns_length(self) -> None:
        from aerocapture.training.sensitivity import DISPERSION_COLUMNS

        assert len(DISPERSION_COLUMNS) == 26
```

- [ ] **Step 4: Run test to verify it fails**

Run: `uv run pytest tests/test_sensitivity.py -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError`

- [ ] **Step 5: Create sensitivity.py with constants and build_problem**

Create `src/python/aerocapture/training/sensitivity.py`:

```python
"""Variance-based sensitivity analysis for Monte Carlo dispersions.

Two-stage workflow:
1. Morris screening (cheap) -- identifies influential parameters
2. Sobol decomposition (expensive) -- quantifies variance contributions

Uses SALib for sampling design and index computation.
Drives simulations via aerocapture_rs.run_with_draws().
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Column order contract (matches DispersionDraw field order in Rust)
# ---------------------------------------------------------------------------
DISPERSION_COLUMNS: list[str] = [
    "altitude",
    "longitude",
    "latitude",
    "velocity",
    "flight_path",
    "azimuth",
    "density",
    "drag_coeff",
    "lift_coeff",
    "incidence",
    "nav_altitude",
    "nav_longitude",
    "nav_latitude",
    "nav_velocity",
    "nav_flight_path",
    "nav_azimuth",
    "nav_drag_accel",
    "mass",
    "ref_area",
    "max_bank_rate",
    "pilot_tau",
    "pilot_damping",
    "pilot_frequency",
    "filter_gain",
    "wind_scale",
    "wind_direction_bias",
]

# Distribution type per dimension: True = Gaussian, False = Uniform
_GAUSSIAN_DIMS: set[int] = {0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 23}

# ---------------------------------------------------------------------------
# Sigma lookup tables (mirror Rust dispersions.rs level presets)
# ---------------------------------------------------------------------------
_INITIAL_STATE_SIGMAS: dict[str, dict[str, float]] = {
    "off": {"altitude": 0.0, "longitude": 0.0, "latitude": 0.0, "velocity": 0.0, "flight_path": 0.0, "azimuth": 0.0},
    "low": {"altitude": 0.0, "longitude": 0.0, "latitude": 0.0, "velocity": 0.13, "flight_path": 0.001, "azimuth": 0.0},
    "medium": {"altitude": 0.1, "longitude": 0.005, "latitude": 0.005, "velocity": 1.0, "flight_path": 0.005, "azimuth": 0.01},
    "high": {"altitude": 0.5, "longitude": 0.01, "latitude": 0.01, "velocity": 2.0, "flight_path": 0.01, "azimuth": 0.02},
}

_ATMOSPHERE_SIGMAS: dict[str, dict[str, float]] = {
    "off": {"density": 0.0},
    "low": {"density": 20.0},
    "medium": {"density": 50.0},
    "high": {"density": 100.0},
}

_AERODYNAMICS_SIGMAS: dict[str, dict[str, float]] = {
    "off": {"drag": 0.0, "lift": 0.0, "incidence": 0.0},
    "low": {"drag": 3.0, "lift": 3.0, "incidence": 0.5},
    "medium": {"drag": 5.0, "lift": 5.0, "incidence": 1.0},
    "high": {"drag": 10.0, "lift": 10.0, "incidence": 2.0},
}

_NAVIGATION_SIGMAS: dict[str, dict[str, float]] = {
    "off": {"altitude": 0.0, "longitude": 0.0, "latitude": 0.0, "velocity": 0.0, "flight_path": 0.0, "azimuth": 0.0, "drag_accel": 0.0},
    "low": {"altitude": 0.3, "longitude": 0.003, "latitude": 0.003, "velocity": 0.13, "flight_path": 0.003, "azimuth": 0.003, "drag_accel": 0.001},
    "medium": {"altitude": 0.667, "longitude": 0.005, "latitude": 0.005, "velocity": 1.0, "flight_path": 0.005, "azimuth": 0.005, "drag_accel": 0.005},
    "high": {"altitude": 1.0, "longitude": 0.01, "latitude": 0.01, "velocity": 2.0, "flight_path": 0.01, "azimuth": 0.01, "drag_accel": 0.01},
}

_MASS_SIGMAS: dict[str, dict[str, float]] = {
    "off": {"mass": 0.0},
    "low": {"mass": 0.5},
    "medium": {"mass": 1.0},
    "high": {"mass": 2.0},
}

_VEHICLE_SIGMAS: dict[str, dict[str, float]] = {
    "off": {"ref_area": 0.0, "max_bank_rate": 0.0},
    "low": {"ref_area": 1.0, "max_bank_rate": 5.0},
    "medium": {"ref_area": 2.0, "max_bank_rate": 10.0},
    "high": {"ref_area": 5.0, "max_bank_rate": 20.0},
}

_PILOT_SIGMAS: dict[str, dict[str, float]] = {
    "off": {"time_constant": 0.0, "damping": 0.0, "frequency": 0.0},
    "low": {"time_constant": 5.0, "damping": 5.0, "frequency": 5.0},
    "medium": {"time_constant": 10.0, "damping": 10.0, "frequency": 10.0},
    "high": {"time_constant": 20.0, "damping": 20.0, "frequency": 20.0},
}

_NAV_FILTER_SIGMAS: dict[str, dict[str, float]] = {
    "off": {"filter_gain": 0.0},
    "low": {"filter_gain": 0.05},
    "medium": {"filter_gain": 0.10},
    "high": {"filter_gain": 0.15},
}

_WIND_DEFAULTS: dict[str, dict[str, float]] = {
    "off": {"scale_min": 1.0, "scale_max": 1.0, "direction_bias_deg": 0.0},
    "low": {"scale_min": 0.7, "scale_max": 1.3, "direction_bias_deg": 15.0},
    "medium": {"scale_min": 0.5, "scale_max": 1.5, "direction_bias_deg": 30.0},
    "high": {"scale_min": 0.2, "scale_max": 2.0, "direction_bias_deg": 45.0},
}

_DEG2RAD = np.pi / 180.0


def build_problem(mc_config: dict[str, Any]) -> dict[str, Any]:
    """Build a SALib problem dict from the [monte_carlo] TOML config.

    Maps dispersion domains to SALib bounds and distribution types.
    Gaussian dims use 'norm' with bounds=[mean, std].
    Uniform dims use 'unif' with bounds=[low, high].
    """
    bounds: list[list[float]] = []
    dists: list[str] = []

    def _get_level(domain: str) -> str:
        d = mc_config.get(domain)
        if d is None:
            return "off"
        return d.get("level", "medium")

    # Initial state (Gaussian, 6 dims) -- sigma in [km, deg, deg, m/s, deg, deg]
    level = _get_level("initial_state")
    s = _INITIAL_STATE_SIGMAS.get(level, _INITIAL_STATE_SIGMAS["medium"])
    for key in ["altitude", "longitude", "latitude", "velocity", "flight_path", "azimuth"]:
        sigma = s[key]
        if key == "altitude":
            sigma *= 1e3  # km -> m
        elif key != "velocity":
            sigma *= _DEG2RAD  # deg -> rad
        bounds.append([0.0, sigma])
        dists.append("norm")

    # Atmosphere (Uniform, 1 dim) -- sigma in %
    level = _get_level("atmosphere")
    s = _ATMOSPHERE_SIGMAS.get(level, _ATMOSPHERE_SIGMAS["medium"])
    hw = s["density"] / 100.0
    bounds.append([-hw, hw])
    dists.append("unif")

    # Aerodynamics (Uniform, 3 dims) -- sigma in %, %, deg
    level = _get_level("aerodynamics")
    s = _AERODYNAMICS_SIGMAS.get(level, _AERODYNAMICS_SIGMAS["medium"])
    bounds.append([-s["drag"] / 100.0, s["drag"] / 100.0])
    dists.append("unif")
    bounds.append([-s["lift"] / 100.0, s["lift"] / 100.0])
    dists.append("unif")
    bounds.append([-s["incidence"] * _DEG2RAD, s["incidence"] * _DEG2RAD])
    dists.append("unif")

    # Navigation (Gaussian, 7 dims) -- same units as initial_state + drag_accel in m/s^2
    level = _get_level("navigation")
    s = _NAVIGATION_SIGMAS.get(level, _NAVIGATION_SIGMAS["medium"])
    for key in ["altitude", "longitude", "latitude", "velocity", "flight_path", "azimuth", "drag_accel"]:
        sigma = s[key]
        if key == "altitude":
            sigma *= 1e3
        elif key not in ("velocity", "drag_accel"):
            sigma *= _DEG2RAD
        bounds.append([0.0, sigma])
        dists.append("norm")

    # Mass (Uniform, 1 dim) -- sigma in %
    level = _get_level("mass")
    s = _MASS_SIGMAS.get(level, _MASS_SIGMAS["medium"])
    hw = s["mass"] / 100.0
    bounds.append([-hw, hw])
    dists.append("unif")

    # Vehicle (Uniform, 2 dims) -- sigma in %
    level = _get_level("vehicle")
    s = _VEHICLE_SIGMAS.get(level, _VEHICLE_SIGMAS["medium"])
    bounds.append([-s["ref_area"] / 100.0, s["ref_area"] / 100.0])
    dists.append("unif")
    bounds.append([-s["max_bank_rate"] / 100.0, s["max_bank_rate"] / 100.0])
    dists.append("unif")

    # Pilot (Uniform, 3 dims) -- sigma in %
    level = _get_level("pilot")
    s = _PILOT_SIGMAS.get(level, _PILOT_SIGMAS["medium"])
    bounds.append([-s["time_constant"] / 100.0, s["time_constant"] / 100.0])
    dists.append("unif")
    bounds.append([-s["damping"] / 100.0, s["damping"] / 100.0])
    dists.append("unif")
    bounds.append([-s["frequency"] / 100.0, s["frequency"] / 100.0])
    dists.append("unif")

    # Nav filter (Gaussian, 1 dim)
    level = _get_level("nav_filter")
    s = _NAV_FILTER_SIGMAS.get(level, _NAV_FILTER_SIGMAS["medium"])
    bounds.append([0.0, s["filter_gain"]])
    dists.append("norm")

    # Wind (2 dims: scale = Uniform range, direction = Uniform symmetric)
    level = _get_level("wind")
    w = _WIND_DEFAULTS.get(level, _WIND_DEFAULTS["off"])
    wind_cfg = mc_config.get("wind")
    if wind_cfg is not None and "level" not in wind_cfg:
        w = _WIND_DEFAULTS["medium"]  # backward compat default
    bounds.append([w["scale_min"], w["scale_max"]])
    dists.append("unif")
    hw = w["direction_bias_deg"] * _DEG2RAD
    bounds.append([-hw, hw])
    dists.append("unif")

    return {
        "num_vars": 26,
        "names": list(DISPERSION_COLUMNS),
        "bounds": bounds,
        "dists": dists,
    }
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_sensitivity.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/python/aerocapture/training/sensitivity.py tests/test_sensitivity.py
git commit -m "feat(python): add SALib dependency + sensitivity problem builder"
```

---

### Task 9: Sensitivity Analysis Pipelines (Morris + Sobol)

**Files:**
- Modify: `src/python/aerocapture/training/sensitivity.py`
- Test: `tests/test_sensitivity.py`

- [ ] **Step 1: Write test for run_morris**

Add to `tests/test_sensitivity.py`:

```python
aero = pytest.importorskip("aerocapture_rs")

GOLDEN_TOML = "configs/test/test_ref_orig.toml"


class TestMorrisPipeline:
    def test_run_morris_returns_indices(self) -> None:
        from aerocapture.training.sensitivity import run_morris

        result = run_morris(GOLDEN_TOML, n=10)
        assert "mu_star" in result
        assert "sigma" in result
        assert "names" in result
        assert len(result["mu_star"]) == 26
        assert len(result["sigma"]) == 26
```

- [ ] **Step 2: Write test for run_sobol**

```python
class TestSobolPipeline:
    def test_run_sobol_returns_indices(self) -> None:
        from aerocapture.training.sensitivity import run_sobol

        result = run_sobol(GOLDEN_TOML, n=64, param_indices=list(range(26)))
        assert "S1" in result
        assert "ST" in result
        assert "S1_conf" in result
        assert "ST_conf" in result
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_sensitivity.py::TestMorrisPipeline -v`
Expected: FAIL with `cannot import name 'run_morris'`

- [ ] **Step 4: Implement run_morris**

Add to `sensitivity.py`:

```python
def _evaluate_draws(toml_path: str, draws: npt.NDArray[np.float64], overrides: dict[str, Any] | None = None, sim_timeout_secs: float | None = None) -> npt.NDArray[np.float64]:
    """Run simulations for the given draw matrix, return DV costs."""
    import aerocapture_rs  # type: ignore[import-not-found, import-untyped]

    result = aerocapture_rs.run_with_draws(
        toml_path,
        draws,
        overrides=overrides,
        sim_timeout_secs=sim_timeout_secs,
    )
    final_records = result.final_records
    # DV total is column 41
    return final_records[:, 41].copy()


def run_morris(
    toml_path: str,
    n: int = 1000,
    *,
    overrides: dict[str, Any] | None = None,
    mc_config: dict[str, Any] | None = None,
    sim_timeout_secs: float | None = None,
) -> dict[str, Any]:
    """Run Morris screening on all 26 dispersion parameters.

    Returns dict with 'mu_star', 'sigma', 'names', and 'mu_star_conf'.
    """
    from SALib.analyze.morris import analyze as morris_analyze
    from SALib.sample.morris import sample as morris_sample

    from aerocapture.training.toml_utils import load_toml_with_bases

    if mc_config is None:
        config = load_toml_with_bases(toml_path)
        mc_config = config.get("monte_carlo", {})

    problem = build_problem(mc_config)
    X = morris_sample(problem, N=n, num_levels=4, seed=42)

    Y = _evaluate_draws(toml_path, X, overrides=overrides, sim_timeout_secs=sim_timeout_secs)

    Si = morris_analyze(problem, X, Y, num_levels=4)

    return {
        "mu_star": Si["mu_star"].tolist(),
        "sigma": Si["sigma"].tolist(),
        "mu_star_conf": Si["mu_star_conf"].tolist(),
        "names": problem["names"],
    }
```

- [ ] **Step 5: Implement run_sobol**

Add to `sensitivity.py`:

```python
def run_sobol(
    toml_path: str,
    n: int = 1024,
    *,
    param_indices: list[int] | None = None,
    overrides: dict[str, Any] | None = None,
    mc_config: dict[str, Any] | None = None,
    calc_second_order: bool = True,
    sim_timeout_secs: float | None = None,
) -> dict[str, Any]:
    """Run Sobol sensitivity analysis on selected parameters.

    If param_indices is provided, only those dimensions vary; others are fixed at 0
    (wind_scale at 1.0). Returns dict with 'S1', 'ST', 'S1_conf', 'ST_conf',
    'S2' (if calc_second_order), 'names', 'param_indices'.
    """
    from SALib.analyze.sobol import analyze as sobol_analyze
    from SALib.sample.sobol import sample as sobol_sample

    from aerocapture.training.toml_utils import load_toml_with_bases

    if mc_config is None:
        config = load_toml_with_bases(toml_path)
        mc_config = config.get("monte_carlo", {})

    full_problem = build_problem(mc_config)

    if param_indices is None:
        param_indices = list(range(26))

    # Build reduced problem for the selected parameters
    sub_problem = {
        "num_vars": len(param_indices),
        "names": [full_problem["names"][i] for i in param_indices],
        "bounds": [full_problem["bounds"][i] for i in param_indices],
        "dists": [full_problem["dists"][i] for i in param_indices],
    }

    X_sub = sobol_sample(sub_problem, N=n, calc_second_order=calc_second_order, scramble=True, seed=42)

    # Expand to full 26-dim draw matrix (fixed dims at nominal)
    n_rows = X_sub.shape[0]
    X_full = np.zeros((n_rows, 26), dtype=np.float64)
    X_full[:, 24] = 1.0  # wind_scale default
    for col_idx, dim_idx in enumerate(param_indices):
        X_full[:, dim_idx] = X_sub[:, col_idx]

    Y = _evaluate_draws(toml_path, X_full, overrides=overrides, sim_timeout_secs=sim_timeout_secs)

    Si = sobol_analyze(sub_problem, Y, calc_second_order=calc_second_order)

    result: dict[str, Any] = {
        "S1": Si["S1"].tolist(),
        "ST": Si["ST"].tolist(),
        "S1_conf": Si["S1_conf"].tolist(),
        "ST_conf": Si["ST_conf"].tolist(),
        "names": sub_problem["names"],
        "param_indices": param_indices,
    }
    if calc_second_order:
        result["S2"] = Si["S2"].tolist()
        result["S2_conf"] = Si["S2_conf"].tolist()

    return result
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_sensitivity.py -v`
Expected: PASS (may be slow due to actual sim runs -- use small n)

- [ ] **Step 7: Commit**

```bash
git add src/python/aerocapture/training/sensitivity.py tests/test_sensitivity.py
git commit -m "feat(python): implement Morris screening + Sobol decomposition pipelines"
```

---

### Task 10: Sensitivity CLI

**Files:**
- Modify: `src/python/aerocapture/training/sensitivity.py`
- Test: manual CLI invocation

- [ ] **Step 1: Add CLI main function**

Add to the bottom of `sensitivity.py`:

```python
def run_full_analysis(
    toml_path: str,
    *,
    morris_n: int = 1000,
    sobol_n: int = 1024,
    top_k: int = 10,
    morris_only: bool = False,
    sobol_only: bool = False,
    output_dir: Path | None = None,
    overrides: dict[str, Any] | None = None,
    sim_timeout_secs: float | None = None,
) -> dict[str, Any]:
    """Run full sensitivity analysis pipeline and save results."""
    from aerocapture.training.toml_utils import load_toml_with_bases

    config = load_toml_with_bases(toml_path)
    mc_config = config.get("monte_carlo", {})
    guidance_type = config.get("guidance", {}).get("type", "unknown")

    if output_dir is None:
        output_dir = Path("training_output") / guidance_type / "sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {"toml_path": toml_path, "guidance_type": guidance_type}

    # Stage 1: Morris screening
    if not sobol_only:
        print(f"Running Morris screening (N={morris_n}, {morris_n * 27} sims)...")
        morris_result = run_morris(toml_path, n=morris_n, mc_config=mc_config, overrides=overrides, sim_timeout_secs=sim_timeout_secs)
        results["morris"] = morris_result

        # Rank by mu_star
        ranked = sorted(range(26), key=lambda i: morris_result["mu_star"][i], reverse=True)
        results["morris_ranking"] = [morris_result["names"][i] for i in ranked]
        print(f"Top {top_k} influential parameters: {results['morris_ranking'][:top_k]}")

    # Stage 2: Sobol decomposition
    if not morris_only:
        if sobol_only:
            param_indices = list(range(26))
            print(f"Running Sobol analysis on all 26 parameters (N={sobol_n})...")
        else:
            # Use top-k from Morris
            ranked_indices = sorted(range(26), key=lambda i: results["morris"]["mu_star"][i], reverse=True)
            param_indices = sorted(ranked_indices[:top_k])
            print(f"Running Sobol analysis on top {top_k} parameters (N={sobol_n})...")

        sobol_result = run_sobol(
            toml_path,
            n=sobol_n,
            param_indices=param_indices,
            mc_config=mc_config,
            overrides=overrides,
            sim_timeout_secs=sim_timeout_secs,
        )
        results["sobol"] = sobol_result

    # Save results
    results_path = output_dir / "sensitivity_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {results_path}")

    return results


def main() -> None:
    """CLI entry point for sensitivity analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="Variance-based sensitivity analysis for MC dispersions")
    parser.add_argument("toml", type=str, help="Path to training TOML config")
    parser.add_argument("--morris-n", type=int, default=1000, help="Morris sample size (default: 1000)")
    parser.add_argument("--sobol-n", type=int, default=1024, help="Sobol base sample size (default: 1024)")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top parameters for Sobol (default: 10)")
    parser.add_argument("--morris-only", action="store_true", help="Run Morris screening only")
    parser.add_argument("--sobol-only", action="store_true", help="Run Sobol on all 26 params (no Morris screening)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: training_output/<scheme>/sensitivity/)")
    parser.add_argument("--sim-timeout", type=float, default=None, help="Wall-clock timeout per sim in seconds")
    args = parser.parse_args()

    run_full_analysis(
        args.toml,
        morris_n=args.morris_n,
        sobol_n=args.sobol_n,
        top_k=args.top_k,
        morris_only=args.morris_only,
        sobol_only=args.sobol_only,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        sim_timeout_secs=args.sim_timeout,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify CLI help works**

Run: `uv run python -m aerocapture.training.sensitivity --help`
Expected: help text with all arguments

- [ ] **Step 3: Commit**

```bash
git add src/python/aerocapture/training/sensitivity.py
git commit -m "feat(python): add sensitivity analysis CLI entry point"
```

---

### Task 11: Sensitivity Charts

**Files:**
- Modify: `src/python/aerocapture/training/charts.py`
- Test: `tests/test_charts.py`

- [ ] **Step 1: Write tests for new chart functions**

Add to `tests/test_charts.py`:

```python
from aerocapture.training.charts import (
    chart_morris_scatter,
    chart_sobol_bars,
    chart_sobol_heatmap,
    chart_sobol_convergence,
)


class TestSensitivityCharts:
    def test_morris_scatter_produces_svg(self, tmp_svg: Path) -> None:
        morris_data = {
            "names": [f"param_{i}" for i in range(5)],
            "mu_star": [10.0, 5.0, 3.0, 1.0, 0.5],
            "sigma": [8.0, 3.0, 2.0, 0.5, 0.2],
            "mu_star_conf": [1.0, 0.5, 0.3, 0.1, 0.05],
        }
        chart_morris_scatter(morris_data, tmp_svg)
        assert tmp_svg.exists()
        assert tmp_svg.stat().st_size > 0

    def test_sobol_bars_produces_svg(self, tmp_svg: Path) -> None:
        sobol_data = {
            "names": [f"param_{i}" for i in range(5)],
            "S1": [0.4, 0.2, 0.1, 0.05, 0.02],
            "ST": [0.5, 0.3, 0.15, 0.08, 0.04],
            "S1_conf": [0.05, 0.03, 0.02, 0.01, 0.005],
            "ST_conf": [0.06, 0.04, 0.03, 0.015, 0.008],
        }
        chart_sobol_bars(sobol_data, tmp_svg)
        assert tmp_svg.exists()
        assert tmp_svg.stat().st_size > 0

    def test_sobol_heatmap_produces_svg(self, tmp_svg: Path) -> None:
        sobol_data = {
            "names": [f"param_{i}" for i in range(4)],
            "S2": [[0.0, 0.1, 0.05, 0.02], [0.1, 0.0, 0.03, 0.01], [0.05, 0.03, 0.0, 0.005], [0.02, 0.01, 0.005, 0.0]],
        }
        chart_sobol_heatmap(sobol_data, tmp_svg)
        assert tmp_svg.exists()
        assert tmp_svg.stat().st_size > 0

    def test_sobol_convergence_produces_svg(self, tmp_svg: Path) -> None:
        convergence_data = {
            "sample_sizes": [64, 128, 256, 512],
            "S1_series": {"param_0": [0.3, 0.35, 0.38, 0.4], "param_1": [0.15, 0.18, 0.19, 0.2]},
            "ST_series": {"param_0": [0.4, 0.45, 0.48, 0.5], "param_1": [0.25, 0.28, 0.29, 0.3]},
        }
        chart_sobol_convergence(convergence_data, tmp_svg)
        assert tmp_svg.exists()
        assert tmp_svg.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_charts.py::TestSensitivityCharts -v`
Expected: FAIL with `cannot import name`

- [ ] **Step 3: Implement chart_morris_scatter**

Add to `charts.py`:

```python
def chart_morris_scatter(morris_data: dict[str, Any], output: Path) -> None:
    """Morris mu* vs sigma scatter plot with parameter labels."""
    names = morris_data["names"]
    mu_star = np.array(morris_data["mu_star"])
    sigma = np.array(morris_data["sigma"])

    fig, ax = plt.subplots(figsize=FULL_WIDTH)
    ax.scatter(mu_star, sigma, s=50, alpha=0.8, color=COLOR_BEST, zorder=3)
    for i, name in enumerate(names):
        ax.annotate(name, (mu_star[i], sigma[i]), fontsize=7, ha="left", va="bottom", xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel(r"$\mu^*$ (mean absolute elementary effect)")
    ax.set_ylabel(r"$\sigma$ (std of elementary effects)")
    ax.set_title("Morris Screening: Parameter Importance")
    # Diagonal line: sigma = mu_star (nonlinear/interactive threshold)
    lim = max(mu_star.max(), sigma.max()) * 1.1
    ax.plot([0, lim], [0, lim], "--", color="grey", alpha=0.5, linewidth=0.8)
    _save_svg(fig, output)
```

- [ ] **Step 4: Implement chart_sobol_bars**

```python
def chart_sobol_bars(sobol_data: dict[str, Any], output: Path) -> None:
    """Grouped bar chart of Sobol S1 and ST indices with confidence intervals."""
    names = sobol_data["names"]
    s1 = np.array(sobol_data["S1"])
    st = np.array(sobol_data["ST"])
    s1_conf = np.array(sobol_data["S1_conf"])
    st_conf = np.array(sobol_data["ST_conf"])

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.6), 4))
    ax.bar(x - width / 2, s1, width, yerr=s1_conf, label="S1 (first-order)", color=COLOR_BEST, alpha=0.8, capsize=3)
    ax.bar(x + width / 2, st, width, yerr=st_conf, label="ST (total-order)", color=COLOR_MEAN, alpha=0.8, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Sobol Index")
    ax.set_title("Sobol Sensitivity Indices")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    _save_svg(fig, output)
```

- [ ] **Step 5: Implement chart_sobol_heatmap**

```python
def chart_sobol_heatmap(sobol_data: dict[str, Any], output: Path) -> None:
    """Heatmap of Sobol S2 (second-order interaction) indices."""
    names = sobol_data["names"]
    s2 = np.array(sobol_data["S2"])

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.5), max(5, len(names) * 0.5)))
    sns.heatmap(s2, xticklabels=names, yticklabels=names, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title("Sobol S2: Parameter Interactions")
    fig.tight_layout()
    _save_svg(fig, output)
```

- [ ] **Step 6: Implement chart_sobol_convergence**

```python
def chart_sobol_convergence(convergence_data: dict[str, Any], output: Path) -> None:
    """Convergence of Sobol indices vs sample size."""
    sizes = convergence_data["sample_sizes"]
    s1_series = convergence_data["S1_series"]
    st_series = convergence_data["ST_series"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for name, values in s1_series.items():
        ax1.plot(sizes, values, marker="o", markersize=3, label=name, linewidth=1)
    ax1.set_xlabel("Sample size N")
    ax1.set_ylabel("S1")
    ax1.set_title("First-Order Index Convergence")
    ax1.legend(fontsize=7, ncol=2)

    for name, values in st_series.items():
        ax2.plot(sizes, values, marker="o", markersize=3, label=name, linewidth=1)
    ax2.set_xlabel("Sample size N")
    ax2.set_ylabel("ST")
    ax2.set_title("Total-Order Index Convergence")
    ax2.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    _save_svg(fig, output)
```

- [ ] **Step 7: Run tests**

Run: `uv run pytest tests/test_charts.py::TestSensitivityCharts -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/python/aerocapture/training/charts.py tests/test_charts.py
git commit -m "feat(python): add sensitivity analysis chart functions (Morris scatter, Sobol bars/heatmap/convergence)"
```

---

### Task 12: Report Integration + Typst Template

**Files:**
- Modify: `src/python/aerocapture/training/report.py`
- Modify: `src/typst/report.typ`
- Test: `tests/test_training_report.py`

- [ ] **Step 1: Add --sensitivity flag to report.py CLI**

In `report.py`, add to the argparse block:

```python
    parser.add_argument("--sensitivity", action="store_true", help="Include Part 3: Sensitivity Analysis (requires pre-computed data)")
```

- [ ] **Step 2: Add _generate_sensitivity_charts helper**

In `report.py`, add after `_generate_trajectory_charts()`:

```python
def _generate_sensitivity_charts(sensitivity_dir: Path, out_dir: Path) -> bool:
    """Generate Part 3 (sensitivity) SVG charts from pre-computed results. Returns True if data exists."""
    results_path = sensitivity_dir / "sensitivity_results.json"
    if not results_path.exists():
        return False

    results = json.loads(results_path.read_text())

    if "morris" in results:
        charts.chart_morris_scatter(results["morris"], out_dir / "morris_scatter.svg")

    if "sobol" in results:
        charts.chart_sobol_bars(results["sobol"], out_dir / "sobol_bars.svg")
        if "S2" in results["sobol"]:
            charts.chart_sobol_heatmap(results["sobol"], out_dir / "sobol_heatmap.svg")

    return True
```

- [ ] **Step 3: Wire sensitivity charts into generate_report()**

In `generate_report()`, after `_generate_trajectory_charts()` call (around line 551), add:

```python
        # Part 3: Sensitivity Analysis (optional, from pre-computed data)
        has_sensitivity = False
        if sensitivity:
            sensitivity_dir = scheme_dir / "sensitivity"
            has_sensitivity = _generate_sensitivity_charts(sensitivity_dir, tmp_dir)
            if not has_sensitivity:
                print(f"No sensitivity data found in {sensitivity_dir} -- skipping Part 3")
```

Pass the `sensitivity` parameter into `generate_report()` and the metadata:

Add `sensitivity: bool = False` to `generate_report()` signature.

In `_build_metadata()`, add `has_sensitivity` field.

Update the CLI call to pass `sensitivity=args.sensitivity`.

- [ ] **Step 4: Update Typst template**

In `src/typst/report.typ`, after line 78 (end of Part 2 else block), add:

```typst
// Part 3: Sensitivity Analysis (optional)
#if meta.at("has_sensitivity", default: false) {
  pagebreak()
  section-heading("Part 3: Sensitivity Analysis")

  full-width-chart(dir + "/morris_scatter.svg")
  full-width-chart(dir + "/sobol_bars.svg")

  if "sobol_heatmap.svg" in dir {
    full-width-chart(dir + "/sobol_heatmap.svg")
  }
}
```

Use metadata flags (`has_morris`, `has_sobol`, `has_sobol_heatmap`) to control which charts render. Update `_generate_sensitivity_charts` to return a dict of booleans, pass them to `_build_metadata()`:

```typst
#if meta.at("has_sensitivity", default: false) {
  pagebreak()
  section-heading("Part 3: Sensitivity Analysis")

  if meta.at("has_morris", default: false) {
    full-width-chart(dir + "/morris_scatter.svg")
  }
  if meta.at("has_sobol", default: false) {
    full-width-chart(dir + "/sobol_bars.svg")
  }
  if meta.at("has_sobol_heatmap", default: false) {
    full-width-chart(dir + "/sobol_heatmap.svg")
  }
}
```

- [ ] **Step 5: Run linting**

Run: `uv run ruff check src/python/aerocapture/training/report.py src/python/aerocapture/training/sensitivity.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add src/python/aerocapture/training/report.py src/typst/report.typ
git commit -m "feat: integrate sensitivity analysis into PDF report pipeline (Part 3)"
```

---

### Task 13: Final Verification + Smart Commit

**Files:**
- All modified files

- [ ] **Step 1: Run full Rust test suite**

Run: `cd src/rust && cargo test`
Expected: all PASS

- [ ] **Step 2: Run Rust formatting + clippy**

Run: `cd src/rust && cargo fmt --check && cargo clippy --all-targets`
Expected: no warnings, no formatting issues

- [ ] **Step 3: Build PyO3 bindings**

Run from repo root: `uv run maturin develop --release --manifest-path src/rust/aerocapture-py/Cargo.toml`
Expected: builds successfully

- [ ] **Step 4: Run full Python test suite**

Run: `uv run pytest tests/ -v`
Expected: all PASS

- [ ] **Step 5: Run Python linting**

Run: `./lint_code.sh`
Expected: no errors from ruff or mypy

- [ ] **Step 6: Invoke smart-commit skill**

Use the `smart-commit` skill to update CLAUDE.md + README.md with the new sampling methods and sensitivity analysis documentation, then commit the whole branch.
