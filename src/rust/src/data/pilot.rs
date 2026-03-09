//! Pilot dynamics model.

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PilotType {
    Perfect,     // natpil = 0
    FirstOrder,  // natpil = 1
    SecondOrder, // natpil = 2
}

#[derive(Debug, Clone, Copy)]
pub struct PilotModel {
    pub pilot_type: PilotType,
    pub time_constant: f64, // first-order tau (s)
    pub damping: f64,       // second-order zeta
    pub frequency: f64,     // second-order omega (rad/s)
}
