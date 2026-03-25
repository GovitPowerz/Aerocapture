# TODO

- [ ] add an animation script of entry corridors and trajectories evolution during training based on checkpoints
- [ ] Update IMPROVEMENTS.md
- [ ] Replace plotly reports with typst -> pdf reports
- [ ] Can you check that when using -fs with nn guidance, the existing eights are ignored, please?
- [ ] Can you check that the simulator exit on apoapsis lower than max atmospheric altitude because I see trajectories that go down to -12MJ orbital energy which doesn't seem possible unless both apoapsis and periapsis are lower than max atmospheric altitude.

## Simulation — High Impact

- [ ] Implement wind model (currently a stub returning zero) — `IMPROVEMENTS.md` §1.4
- [ ] Add Mach-dependent Cx/Cz tables (2D interpolation: AoA x Mach) — §3.1
- [ ] Implement integrated heat load tracking (`sum(q*dt)`) — §4.2
- [ ] Enable exit phase guidance (phase transition logic present but inactive) — §6.3
- [ ] Replace bias-only navigation with EKF/UKF — §5.1

## Simulation — Medium Impact

- [ ] Improve FNPAG predictor fidelity (add J2, actual atmo table) — §6.4
- [ ] Adaptive RK4 step sizing (Dormand-Prince or two-phase scheme) — §10.1
- [ ] Separate truth vs onboard atmosphere models — §1.2

## Training & ML

- [ ] Add alternative optimizers: CMA-ES, PSO, Bayesian optimization — §12.1
- [ ] Explore LSTM / Transformer architectures for guidance (BPTT for recurrent) — §12.2
- [ ] Add neural counterparts for navigation and control — §12.3

## Mission Extensions

- [ ] Add Venus and Titan atmosphere/gravity models — §13.3
- [ ] Multi-pass aerocapture support — §13.1
- [ ] Develop ESR (Earth Sample Return) mission profiles — §13.4
