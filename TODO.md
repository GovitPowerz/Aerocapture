# TODO

## Codebase Cleanup

- [ ] Clean up Rust warnings and rename variables properly (no French, no 6-char Fortran legacy names)
- [ ] Remove legacy Fortran codebase (`old_codebase/`)
  - Preserve all mission data as proper TOML configs first
  - Decide what reference data is worth keeping for validation
- [ ] Remove Rust and Python code for legacy `.in` input/output formats
- [ ] Expand test coverage (unit + integration) before and after removals to catch regressions
- [ ] Analyse directory structure (Rust and Python) and suggest improvements
- [ ] Review test quality and coverage gaps

## CI / DevOps

- [ ] Run CI tests only on PRs, not on every push

## Documentation

- [ ] Update `Improvements.md` based on completed work and future ideas

## Simulation Improvements

- [ ] Implement improvements from `Improvements.md` in Rust simulator
- [ ] Revisit roll reversal strategy for NN guidance — explore letting the NN handle it directly
- [ ] Rework cost function design (energy, correction cost, etc.)

## Training & ML

- [ ] Update training algorithms (GA, RL, PSO, ...)
- [ ] Improve output visualisation for training runs
- [ ] Explore LSTM / Transformer architectures for guidance
- [ ] Add neural counterparts for navigation and control (to compare against classical algorithms)
