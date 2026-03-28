# TODO

- [ ] add an animation script of entry corridors and trajectories evolution during training based on checkpoints
- [ ] Make 3 separate graphs for individual burn corrections and add xticklabeld on total DeltV distribution graph
- [ ] Fix guidance schemes training than never finishes
- [ ] Maybe better scale parameters in non nn guidance schemes to improve training

## Simulation — High Impact

- [ ] Add Mach-dependent Cx/Cz tables (2D interpolation: AoA x Mach) — §3.1
- [ ] Enable exit phase guidance (phase transition logic present but inactive) — §6.3

## Simulation — Medium Impact

- [ ] Improve FNPAG predictor fidelity (add J2, actual atmo table) — §6.4
- [ ] Adaptive RK4 step sizing (Dormand-Prince or two-phase scheme) — §10.1
- [x] Separate truth vs onboard atmosphere models — §1.2

## Training & ML

- [ ] Add alternative optimizers: CMA-ES, PSO, Bayesian optimization — §12.1
- [ ] Explore LSTM / Transformer architectures for guidance (BPTT for recurrent) — §12.2
- [ ] Add neural counterparts for navigation and control — §12.3
- [ ] Switch from binary GA to real-valued GA: SBX crossover + polynomial mutation (DEAP built-in), normalize all parameters to [0,1] internally, adaptive mutation rates per parameter — eliminates scale-blind bit-flip problem entirely

## Mission Extensions

- [ ] Add Venus and Titan atmosphere/gravity models — §13.3
- [ ] Multi-pass aerocapture support — §13.1
- [ ] Develop ESR (Earth Sample Return) mission profiles — §13.4
