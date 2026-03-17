# TODO

- [ ] fix coridor visualization. I think there is an issue with corridors trajectories computation / serialization. The corridor should build 5 trajectories (orbital energy vs dynamic pressure): nominal best DeltaV, undershoot and overshoot boundary trajectories for capture with apoapsis +-200km, crash limit, hyperbolic exit. Here are detailed definitions: a good way to characterize aerocapture missions is to represent the trajectories as the orbital energy
  versus the dynamic pressure. In this plane, an aerocapture corridor is delimited by two trajectories with a
  constant bank angle:
  • an overshoot trajectory that represents the limit between an exit of the atmosphere on an elliptic orbit
  and an exit of the atmosphere on a hyperbolic one,
  • an undershoot trajectory that represents the limit between the crash of the vehicle on the ground and
  an exit of the atmosphere on an elliptic orbit.
  However, in order to have a more practical representation of the mission objectives, one can slightly modify
  this definition and build a restricted corridor with:
  • an overshoot trajectory with a constant bank angle that leads to an error on the apoapsis at atmosphere
  exit of +δZa,
  14 of 21
  American Institute of Aeronautics and Astronautics
  • an undershoot trajectory with a constant bank angle that leads to an error on the apoapsis at atmo-
  sphere exit of -δZa,
  where δZa depends on the mission (here, we considered δZa = 200 km).
- [ ] add an animation script of entry corridors and trajectories evolution during training based on checkpoints
- [ ] 1e30 for Dv is too much, we should use something like log(Dv) for values higher than a threshold but make it continuous (and derivable) at the threshold (1000 m/s seems reasonable)
- [ ] Update IMPROVEMENTS.md

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
- [ ] Let NN handle roll reversal directly (remove FTC lateral dependency) — §8.3
- [ ] Add neural counterparts for navigation and control — §12.3

## Mission Extensions

- [ ] Add Venus and Titan atmosphere/gravity models — §13.3
- [ ] Multi-pass aerocapture support — §13.1
- [ ] Develop ESR (Earth Sample Return) mission profiles — §13.4
