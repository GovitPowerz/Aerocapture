//! Incidence (angle of attack) profile.

#[derive(Debug, Clone)]
pub struct IncidenceProfile {
    pub n_points: usize,
    pub altitudes: Vec<f64>,  // meters
    pub incidences: Vec<f64>, // radians
}

impl IncidenceProfile {
    /// Interpolate commanded incidence at a given altitude
    pub fn incidence_at(&self, altitude: f64) -> f64 {
        if self.n_points == 0 {
            return 0.0;
        }
        if self.n_points == 1 || altitude <= self.altitudes[0] {
            return self.incidences[0];
        }
        if altitude >= self.altitudes[self.n_points - 1] {
            return self.incidences[self.n_points - 1];
        }
        for i in 1..self.n_points {
            if altitude <= self.altitudes[i] {
                let frac = (altitude - self.altitudes[i - 1])
                    / (self.altitudes[i] - self.altitudes[i - 1]);
                return self.incidences[i - 1]
                    + frac * (self.incidences[i] - self.incidences[i - 1]);
            }
        }
        self.incidences[self.n_points - 1]
    }
}
