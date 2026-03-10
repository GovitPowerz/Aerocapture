# TODO

## Codebase Cleanup

- [ ] Clean up Rust warnings and rename variables properly (no French, no 6-char Fortran legacy names)

## Documentation

- [ ] Update `IMPROVEMENTS.md` based on completed work and future ideas

## Simulation Improvements

- [ ] Implement improvements from `IMPROVEMENTS.md` in Rust simulator
- [ ] Revisit roll reversal strategy for NN guidance — explore letting the NN handle it directly
- [ ] Rework cost function design (energy, correction cost, etc.)

## Training & ML

- [ ] Update training algorithms (GA, RL, PSO, ...)
- [ ] Improve output visualisation for training runs
- [ ] Explore LSTM / Transformer architectures for guidance
- [ ] For recurrent architecture, investigate the use of Backprop through time
- [ ] Add neural counterparts for navigation and control (to compare against classical algorithms)
