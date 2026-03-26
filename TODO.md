# TODO

- [ ] add an animation script of entry corridors and trajectories evolution during training based on checkpoints
- [ ] Update IMPROVEMENTS.md

## Simulation — High Impact

- [x] Implement wind model (altitude-dependent zonal/meridional profiles with MC dispersions) — `IMPROVEMENTS.md` §1.4
- [ ] Add Mach-dependent Cx/Cz tables (2D interpolation: AoA x Mach) — §3.1
- [x] Expose integrated heat load tracking in trajectory, CSV, PyO3, and GA cost function — §4.2
- [ ] Enable exit phase guidance (phase transition logic present but inactive) — §6.3
- [x] Replace bias-only navigation with EKF (13-state, IMU + star tracker + drag-derived density) — §5.1

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
