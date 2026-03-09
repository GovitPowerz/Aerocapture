//! Capsule properties.

use super::TimePeriods;

#[derive(Debug, Clone, Copy)]
pub struct Capsule {
    pub mass: f64,           // kg
    pub reference_area: f64, // m^2
    pub cq: f64,             // heat flux coefficient
    pub max_bank_rate: f64,  // rad/s
    pub periods: TimePeriods,
}
