# Harden & Clean — Design

## Goal

Make the Rust simulator + Python tools fully self-sufficient — no dependency on Fortran code or legacy formats — with enough test coverage to confidently evolve the codebase.

## Strategy

Gradual removal: build the safety net first, then remove legacy code, then do the big rename as a final polish pass.

## Phases (in order)

### Phase 1 — Rust golden reference outputs

Capture current Rust simulator outputs (photo, final, fort.* files) as golden reference files for each supported configuration (FTC guided, reference bank angle, each planet). These replace Fortran as the "source of truth" for regression testing. Store under `tests/reference_data/rust_golden/`.

### Phase 2 — Domain test coverage

Fill gaps in guidance scheme coverage (all 6 schemes need at least one integration test), edge cases (bounce detection, phase transitions, dispersions on/off), and multi-planet support. Audit existing 93 tests for quality.

### Phase 3 — Migrate mission data to TOML

Ensure every mission variant in `old_codebase/donnees/` has a corresponding TOML config in `configs/`. Discuss which variants are worth keeping vs. which are historical dead weight.

### Phase 4 — Convert suffix-mode configs to consolidated TOML + remove suffix mode

~~Remove legacy `.in` format support~~ — already done, Rust has no stdin parsing.
The real dependency is "suffix mode" TOML configs that reference `old_codebase/donnees/` files.
Convert all configs to consolidated (inline data), move external data to `data/`, then remove
the suffix mode code path from Rust. Also update Python hardcoded `old_codebase/` paths.

### Phase 5 — Remove `old_codebase/`

Delete the Fortran source, Makefiles, and `donnees/` directory. At this point, all valuable data lives in `configs/` and `tests/reference_data/`, and all tests pass without Fortran.

### Phase 6 — Full variable rename

Systematic rename of all French/Fortran-legacy variable names to clear English across the Rust codebase. Done last because test coverage is maximal and the codebase is stable.

### Phase 7 — CI update

Change GitHub Actions to trigger on PRs to `main` + `workflow_dispatch` only (remove push triggers).

## Out of scope (deferred to "Extend Capabilities" phase)

- `IMPROVEMENTS.md` creation
- New guidance schemes or cost functions
- Training algorithm updates
- LSTM/Transformer exploration
- Neural navigation/control
