//! Atmosphere model loader.
//!
//! Format: 3 header lines, then:
//!   naltit — number of altitude points
//!   naltit lines of: altitude(m)  density(kg/m3)
//!   density dispersion profile boundary marker (-1)
//!   4 altitude breakpoints (km) — or fewer if marker is not -1
//!   4 density error breakpoints (%)
//!   rozmod — exponential model reference density (kg/m3)
//!   facech — exponential model scale factor (1/m)
//!   zromod — exponential model reference altitude (m)
//!   cstgam — gas constant ratio (gamma)

use super::{DataError, parse_data_file};

/// Density dispersion profile (altitude-dependent bias envelope)
#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct DensityProfile {
    pub altitudes: Vec<f64>,      // meters (up to 5 breakpoints)
    pub max_dispersion: Vec<f64>, // fractional (converted from %)
    pub slopes: Vec<f64>,         // linear interpolation slopes
    pub intercepts: Vec<f64>,     // linear interpolation intercepts
}

#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct AtmosphereModel {
    pub n_points: usize,
    pub altitudes: Vec<f64>, // meters
    pub densities: Vec<f64>, // kg/m^3
    pub ref_density: f64,    // exponential model rho0 (kg/m3)
    pub scale_factor: f64,   // exponential model H^-1 (1/m)
    pub ref_altitude: f64,   // exponential model z0 (m)
    pub gas_constant: f64,   // gamma (ratio of specific heats)
    pub density_profile: DensityProfile,
}

impl AtmosphereModel {
    pub fn load(path: &str) -> Result<Self, DataError> {
        let rows = parse_data_file(path)?;
        if rows.is_empty() {
            return Err(DataError(format!("Atmosphere file empty: {}", path)));
        }

        let n_points = rows[0][0] as usize;
        if rows.len() < 1 + n_points + 10 {
            return Err(DataError(format!(
                "Atmosphere file too short ({} rows): {}",
                rows.len(),
                path
            )));
        }

        let mut altitudes = Vec::with_capacity(n_points);
        let mut densities = Vec::with_capacity(n_points);
        for i in 0..n_points {
            let row = &rows[1 + i];
            altitudes.push(row[0]);
            densities.push(row[1]);
        }

        // After the density table: marker line (-1), then profile breakpoints
        let mut idx = 1 + n_points;

        // Skip the marker line (contains -1)
        idx += 1;

        // Read 4 altitude breakpoints (km) and 4 dispersion breakpoints (%)
        // 5 total breakpoints per dimension; first altitude and first dispersion are 0
        let mut prof_alts = vec![0.0f64]; // first breakpoint is 0
        for _ in 0..4 {
            if idx >= rows.len() {
                break;
            }
            prof_alts.push(rows[idx][0] * 1e3); // km -> m
            idx += 1;
        }

        let mut prof_disp = vec![0.0f64]; // first dispersion is 0%
        for _ in 0..4 {
            if idx >= rows.len() {
                break;
            }
            prof_disp.push(rows[idx][0] / 100.0); // % -> fraction
            idx += 1;
        }

        // Compute slopes and intercepts for piecewise-linear profile
        let n_prof = prof_alts.len();
        let mut slopes = vec![0.0f64; n_prof];
        let mut intercepts = vec![0.0f64; n_prof];
        for i in 1..n_prof {
            let dalt = prof_alts[i] - prof_alts[i - 1];
            if dalt.abs() > 1e-30 {
                slopes[i] = (prof_disp[i] - prof_disp[i - 1]) / dalt;
                intercepts[i] = prof_disp[i] - slopes[i] * prof_alts[i];
            }
        }

        // Exponential model parameters
        let ref_density = if idx < rows.len() { rows[idx][0] } else { 0.0 };
        idx += 1;
        let scale_factor = if idx < rows.len() { rows[idx][0] } else { 0.0 };
        idx += 1;
        let ref_altitude = if idx < rows.len() { rows[idx][0] } else { 0.0 };
        idx += 1;
        let gas_constant = if idx < rows.len() { rows[idx][0] } else { 1.3 };

        Ok(AtmosphereModel {
            n_points,
            altitudes,
            densities,
            ref_density,
            scale_factor,
            ref_altitude,
            gas_constant,
            density_profile: DensityProfile {
                altitudes: prof_alts,
                max_dispersion: prof_disp,
                slopes,
                intercepts,
            },
        })
    }

    /// Interpolate density at a given altitude (linear interpolation in table)
    pub fn density_at(&self, altitude: f64) -> f64 {
        let n = self.n_points;
        if n == 0 {
            return self.exponential_density(altitude);
        }
        if altitude <= self.altitudes[0] {
            return self.densities[0];
        }
        if altitude >= self.altitudes[n - 1] {
            return self.exponential_density(altitude);
        }
        for i in 1..n {
            if altitude <= self.altitudes[i] {
                let frac = (altitude - self.altitudes[i - 1])
                    / (self.altitudes[i] - self.altitudes[i - 1]);
                return self.densities[i - 1] + frac * (self.densities[i] - self.densities[i - 1]);
            }
        }
        self.exponential_density(altitude)
    }

    /// Exponential atmosphere model: rho = rho0 * exp(-H * (z - z0))
    pub fn exponential_density(&self, altitude: f64) -> f64 {
        self.ref_density * (-self.scale_factor * (altitude - self.ref_altitude)).exp()
    }
}

/// One altitude band of the onboard piecewise exponential model.
#[derive(Debug, Clone)]
pub struct ExponentialSegment {
    pub alt_low: f64,      // meters
    pub alt_high: f64,     // meters
    pub rho_ref: f64,      // kg/m^3 (density at alt_low)
    pub scale_height: f64, // meters
}

/// Onboard atmosphere model — degraded representation of truth.
#[derive(Debug, Clone)]
pub enum OnboardAtmosphereModel {
    /// Use the truth table directly (backward-compatible mode).
    Identical,
    /// Piecewise exponential segments auto-fitted or manually specified.
    PiecewiseExponential { segments: Vec<ExponentialSegment> },
}

impl OnboardAtmosphereModel {
    /// Auto-fit a piecewise exponential model from the truth atmosphere table.
    ///
    /// Divides the truth table altitude range into `n_segments` equal bands.
    /// For each band, samples the truth table at the band endpoints and any
    /// interior table points, then performs a linear regression on ln(rho) vs
    /// altitude to extract scale height H and reference density rho_ref.
    pub fn fit_from_table(truth: &AtmosphereModel, n_segments: usize) -> Self {
        if truth.n_points < 2 || n_segments == 0 {
            return OnboardAtmosphereModel::Identical;
        }

        let alt_min = truth.altitudes[0];
        let alt_max = truth.altitudes[truth.n_points - 1];
        let band_width = (alt_max - alt_min) / n_segments as f64;

        let mut segments = Vec::with_capacity(n_segments);
        for i in 0..n_segments {
            let alt_low = alt_min + i as f64 * band_width;
            let alt_high = alt_min + (i + 1) as f64 * band_width;

            // Sample truth densities within this band (at least 2 points: endpoints)
            let mut samples: Vec<(f64, f64)> = Vec::new();

            // Add band endpoints
            let rho_low = truth.density_at(alt_low);
            if rho_low > 0.0 {
                samples.push((alt_low, rho_low));
            }
            let rho_high = truth.density_at(alt_high);
            if rho_high > 0.0 {
                samples.push((alt_high, rho_high));
            }

            // Add interior table points
            for j in 0..truth.n_points {
                let alt_j = truth.altitudes[j];
                if alt_j > alt_low && alt_j < alt_high {
                    let rho_j = truth.densities[j];
                    if rho_j > 0.0 {
                        samples.push((alt_j, rho_j));
                    }
                }
            }

            // Linear regression on ln(rho) vs altitude
            let (rho_ref, scale_height) = if samples.len() >= 2 {
                fit_exponential(&samples, alt_low)
            } else if let Some(&(_, rho)) = samples.first() {
                (rho, 1.0 / truth.scale_factor)
            } else {
                (truth.ref_density, 1.0 / truth.scale_factor)
            };

            segments.push(ExponentialSegment {
                alt_low,
                alt_high,
                rho_ref,
                scale_height,
            });
        }

        OnboardAtmosphereModel::PiecewiseExponential { segments }
    }

    /// Query onboard density at a given altitude.
    ///
    /// For `Identical`, delegates to the truth table.
    /// For `PiecewiseExponential`, finds the containing segment and evaluates
    /// `rho_ref * exp(-(alt - alt_low) / H)`. Below the first segment uses
    /// the first segment's rho_ref. Above the last segment uses exponential
    /// extrapolation from the last segment.
    pub fn density_at(&self, altitude: f64, truth: &AtmosphereModel) -> f64 {
        match self {
            OnboardAtmosphereModel::Identical => truth.density_at(altitude),
            OnboardAtmosphereModel::PiecewiseExponential { segments } => {
                if segments.is_empty() {
                    return truth.density_at(altitude);
                }
                // Below first segment: clamp to first segment's rho_ref
                if altitude <= segments[0].alt_low {
                    return segments[0].rho_ref;
                }
                // Find containing segment
                for seg in segments {
                    if altitude <= seg.alt_high {
                        return seg.rho_ref * (-(altitude - seg.alt_low) / seg.scale_height).exp();
                    }
                }
                // Above last segment: extrapolate from last segment
                let last = &segments[segments.len() - 1];
                last.rho_ref * (-(altitude - last.alt_low) / last.scale_height).exp()
            }
        }
    }
}

/// Fit rho_ref and scale_height from samples using linear regression on ln(rho).
///
/// Model: ln(rho) = ln(rho_ref) - (alt - alt_low) / H
/// Which is: y = c + m*x where y=ln(rho), x=(alt-alt_low), c=ln(rho_ref), m=-1/H
fn fit_exponential(samples: &[(f64, f64)], alt_low: f64) -> (f64, f64) {
    let n = samples.len() as f64;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xx = 0.0;
    let mut sum_xy = 0.0;

    for &(alt, rho) in samples {
        let x = alt - alt_low;
        let y = rho.ln();
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }

    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-30 {
        let rho_ref = (sum_y / n).exp();
        return (rho_ref, 10_000.0);
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n;

    let rho_ref = intercept.exp();
    let scale_height = if slope < -1e-15 { -1.0 / slope } else { 1e6 };

    (rho_ref, scale_height)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    /// Build a small 3-point atmosphere table for testing.
    fn test_atm() -> AtmosphereModel {
        AtmosphereModel {
            n_points: 3,
            altitudes: vec![10_000.0, 20_000.0, 30_000.0],
            densities: vec![1.0, 0.5, 0.1],
            ref_density: 0.1,
            scale_factor: 1e-4,
            ref_altitude: 30_000.0,
            gas_constant: 1.3,
            density_profile: DensityProfile::default(),
        }
    }

    #[test]
    fn piecewise_exponential_single_segment() {
        let model = OnboardAtmosphereModel::PiecewiseExponential {
            segments: vec![ExponentialSegment {
                alt_low: 0.0,
                alt_high: 50_000.0,
                rho_ref: 0.02,
                scale_height: 10_000.0,
            }],
        };
        let truth = test_atm();
        // At alt_low the density should be rho_ref
        assert_abs_diff_eq!(model.density_at(0.0, &truth), 0.02, epsilon = 1e-10);
        // At one scale height above, density should be rho_ref * exp(-1)
        let expected = 0.02 * (-1.0_f64).exp();
        assert_abs_diff_eq!(
            model.density_at(10_000.0, &truth),
            expected,
            epsilon = 1e-10
        );
    }

    #[test]
    fn piecewise_exponential_two_segments() {
        let model = OnboardAtmosphereModel::PiecewiseExponential {
            segments: vec![
                ExponentialSegment {
                    alt_low: 0.0,
                    alt_high: 20_000.0,
                    rho_ref: 0.02,
                    scale_height: 10_000.0,
                },
                ExponentialSegment {
                    alt_low: 20_000.0,
                    alt_high: 50_000.0,
                    rho_ref: 0.002,
                    scale_height: 8_000.0,
                },
            ],
        };
        let truth = test_atm();
        // In first segment
        let expected_low = 0.02 * (-15_000.0 / 10_000.0_f64).exp();
        assert_abs_diff_eq!(
            model.density_at(15_000.0, &truth),
            expected_low,
            epsilon = 1e-10
        );
        // In second segment
        let expected_high = 0.002 * (-5_000.0 / 8_000.0_f64).exp();
        assert_abs_diff_eq!(
            model.density_at(25_000.0, &truth),
            expected_high,
            epsilon = 1e-10
        );
    }

    #[test]
    fn identical_mode_delegates_to_truth() {
        let truth = test_atm();
        let model = OnboardAtmosphereModel::Identical;
        assert_abs_diff_eq!(
            model.density_at(15_000.0, &truth),
            truth.density_at(15_000.0)
        );
        assert_abs_diff_eq!(
            model.density_at(35_000.0, &truth),
            truth.density_at(35_000.0)
        );
    }

    #[test]
    fn auto_fit_produces_correct_segment_count() {
        let truth = AtmosphereModel {
            n_points: 5,
            altitudes: vec![0.0, 25_000.0, 50_000.0, 75_000.0, 100_000.0],
            densities: vec![0.013, 0.003, 5e-4, 6e-5, 5e-6],
            ref_density: 5e-6,
            scale_factor: 1e-4,
            ref_altitude: 100_000.0,
            gas_constant: 1.3,
            density_profile: DensityProfile::default(),
        };
        let model = OnboardAtmosphereModel::fit_from_table(&truth, 3);
        match &model {
            OnboardAtmosphereModel::PiecewiseExponential { segments } => {
                assert_eq!(segments.len(), 3);
                assert_abs_diff_eq!(segments[0].alt_low, 0.0);
                assert_abs_diff_eq!(segments[2].alt_high, 100_000.0);
                for seg in segments {
                    assert!(seg.scale_height > 0.0, "scale_height must be positive");
                    assert!(seg.rho_ref > 0.0, "rho_ref must be positive");
                }
            }
            _ => panic!("Expected PiecewiseExponential variant"),
        }
    }

    #[test]
    fn auto_fit_approximates_truth_within_tolerance() {
        let truth = AtmosphereModel {
            n_points: 5,
            altitudes: vec![0.0, 25_000.0, 50_000.0, 75_000.0, 100_000.0],
            densities: vec![0.013, 0.003, 5e-4, 6e-5, 5e-6],
            ref_density: 5e-6,
            scale_factor: 1e-4,
            ref_altitude: 100_000.0,
            gas_constant: 1.3,
            density_profile: DensityProfile::default(),
        };
        let model = OnboardAtmosphereModel::fit_from_table(&truth, 5);
        for &alt in &truth.altitudes {
            let rho_truth = truth.density_at(alt);
            let rho_onboard = model.density_at(alt, &truth);
            if rho_truth > 1e-10 {
                let rel_err = (rho_onboard - rho_truth).abs() / rho_truth;
                assert!(
                    rel_err < 0.5,
                    "relative error {:.2}% at alt={} m exceeds 50%",
                    rel_err * 100.0,
                    alt,
                );
            }
        }
    }

    #[test]
    fn auto_fit_mars_like_table_diverges_from_truth() {
        let altitudes: Vec<f64> = (0..14).map(|i| i as f64 * 10_000.0).collect();
        let densities: Vec<f64> = altitudes
            .iter()
            .map(|&alt| 0.013 * (-alt / 9_000.0).exp())
            .collect();
        let n = altitudes.len();

        let truth = AtmosphereModel {
            n_points: n,
            altitudes: altitudes.clone(),
            densities: densities.clone(),
            ref_density: densities[n - 1],
            scale_factor: 1.0 / 9_000.0,
            ref_altitude: altitudes[n - 1],
            gas_constant: 1.3,
            density_profile: DensityProfile::default(),
        };

        let model = OnboardAtmosphereModel::fit_from_table(&truth, 5);

        let mut max_rel_err = 0.0_f64;
        for &alt in &[5_000.0, 15_000.0, 35_000.0, 55_000.0, 95_000.0] {
            let rho_truth = truth.density_at(alt);
            let rho_onboard = model.density_at(alt, &truth);
            if rho_truth > 1e-15 {
                let rel_err = (rho_onboard - rho_truth).abs() / rho_truth;
                max_rel_err = max_rel_err.max(rel_err);
            }
        }

        assert!(
            max_rel_err > 1e-6,
            "onboard model should differ from truth; max_rel_err={}",
            max_rel_err,
        );
        assert!(
            max_rel_err < 1.0,
            "onboard model too far from truth; max_rel_err={}",
            max_rel_err,
        );
    }

    #[test]
    fn piecewise_density_always_positive() {
        let truth = AtmosphereModel {
            n_points: 5,
            altitudes: vec![0.0, 25_000.0, 50_000.0, 75_000.0, 100_000.0],
            densities: vec![0.013, 0.003, 5e-4, 6e-5, 5e-6],
            ref_density: 5e-6,
            scale_factor: 1e-4,
            ref_altitude: 100_000.0,
            gas_constant: 1.3,
            density_profile: DensityProfile::default(),
        };
        let model = OnboardAtmosphereModel::fit_from_table(&truth, 5);

        for alt_km in 0..=150 {
            let alt = alt_km as f64 * 1_000.0;
            let rho = model.density_at(alt, &truth);
            assert!(
                rho > 0.0,
                "density must be positive at alt={} m, got {}",
                alt,
                rho,
            );
            assert!(
                rho.is_finite(),
                "density must be finite at alt={} m, got {}",
                alt,
                rho,
            );
        }
    }
}
