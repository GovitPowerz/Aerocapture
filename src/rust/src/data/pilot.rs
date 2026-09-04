//! Pilot dynamics model.

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PilotType {
    Perfect,     // natpil = 0
    FirstOrder,  // natpil = 1
    SecondOrder, // natpil = 2
}

impl PilotType {
    /// Parse the `[vehicle.pilot] model` string.
    pub fn parse(model: &str) -> Result<Self, String> {
        match model {
            "perfect" => Ok(Self::Perfect),
            "first_order" => Ok(Self::FirstOrder),
            "second_order" => Ok(Self::SecondOrder),
            other => Err(format!("Unknown pilot model: {}", other)),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PilotModel {
    pub pilot_type: PilotType,
    pub time_constant: f64, // first-order tau (s)
    pub damping: f64,       // second-order zeta
    pub frequency: f64,     // second-order omega (rad/s)
}
