# TODO

## Visualization

- [ ] Add an animation script of entry corridors and trajectories evolution during training based on checkpoints

## Simulation — High Impact

- [ ] Enable exit phase guidance (phase transition logic present but inactive) — §4.2
- [ ] Add heat rate and heat load as active guidance constraints — §2.1

## Simulation — Medium Impact

- [ ] Improve FNPAG predictor fidelity (add J2, actual atmo table) — §4.3
- [ ] Fix FTC gain discontinuity at altitude table boundary — §4.1
- [ ] Add bank angle rate/acceleration limits to guidance — §4.4
- [ ] Improve roll reversal logic (predictive instead of corridor-based) — §5.1
- [ ] Time-varying density perturbations (Gauss-Markov process) — §1.1
- [ ] Event detection (root-finding for atmo entry/exit) — §7.1

## Navigation

- [ ] Improve density estimation filter (gain saturation, outlier rejection) — §3.1
- [ ] Better drag acceleration extraction (decompose drag/lift) — §3.2

## Monte Carlo & Analysis

- [ ] Advanced MC sampling (LHS, Sobol, importance sampling) — §6.1
- [ ] Sensitivity analysis (Sobol indices, tornado diagrams) — §6.2
- [ ] Output format improvements (HDF5/Parquet, metadata, dispersions in final CSV) — §8.1

## Training & ML

- [ ] Switch to real-valued GA + alternative optimizers (CMA-ES, PSO, RL) — §9.1
- [ ] Explore LSTM / Transformer architectures for guidance — §9.2
- [ ] Add neural counterparts for navigation and control — §9.3

## Mission Extensions

- [ ] Develop ESR (Earth Sample Return) mission profiles — §10.1
