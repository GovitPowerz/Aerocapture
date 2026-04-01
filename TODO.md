# TODO

## Simulation — High Impact

- [ ] Add heat rate and heat load as active guidance constraints — §2.1 in IMPROVEMENTS.md

## Simulation — Medium Impact

- [ ] Improve FNPAG predictor fidelity (add J2, actual atmo table) — §4.3  in IMPROVEMENTS.md
- [ ] Fix FTC gain discontinuity at altitude table boundary — §4.1  in IMPROVEMENTS.md
- [ ] Add bank angle rate/acceleration limits to guidance — §4.4  in IMPROVEMENTS.md
- [ ] Improve roll reversal logic (predictive instead of corridor-based) — §5.1  in IMPROVEMENTS.md
- [ ] Time-varying density perturbations (Gauss-Markov process) — §1.1  in IMPROVEMENTS.md
- [ ] Event detection (root-finding for atmo entry/exit) — §7.1  in IMPROVEMENTS.md

## Navigation

- [ ] Improve density estimation filter (gain saturation, outlier rejection) — §3.1  in IMPROVEMENTS.md
- [ ] Better drag acceleration extraction (decompose drag/lift) — §3.2  in IMPROVEMENTS.md

## Monte Carlo & Analysis

- [ ] Advanced MC sampling (LHS, Sobol, importance sampling) — §6.1  in IMPROVEMENTS.md
- [ ] Sensitivity analysis (Sobol indices, tornado diagrams) — §6.2  in IMPROVEMENTS.md
- [ ] Output format improvements (HDF5/Parquet, metadata, dispersions in final CSV) — §8.1  in IMPROVEMENTS.md

## Training & ML

- [ ] Switch to real-valued GA + alternative optimizers (CMA-ES, PSO, RL) — §9.1  in IMPROVEMENTS.md
- [ ] Explore LSTM / Transformer architectures for guidance — §9.2  in IMPROVEMENTS.md
- [ ] Add neural counterparts for navigation and control — §9.3  in IMPROVEMENTS.md

## Mission Extensions

- [ ] Develop ESR (Earth Sample Return) mission profiles — §10.1  in IMPROVEMENTS.md
