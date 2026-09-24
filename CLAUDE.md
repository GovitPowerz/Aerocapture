# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Aerocapture is a trajectory simulation and guidance-training tool for aerocapture maneuvers (primarily Mars Sample Return): a Rust simulator (physics, navigation, seven guidance
schemes, the neural-network runtime, Monte Carlo dispersions) driven through one PyO3 seam by a Python package that trains any scheme's parameters with population search (pymoo GA /
CMA-ES / DE / PSO / QPSO / islands), evaluates on reserved seed pools, and renders reports. The Rust simulator was validated against a legacy reference implementation to bit-level
precision (FTC guided trajectories matched across all 725 timesteps; 22/24 photo columns exact, the remaining 2 were uninitialized-variable artifacts in the reference);
`docs/validation.md` adds the independent evidence (vacuum conservation, a cross-check against the open-source AMAT tool, a reproduced published Mars corridor). The research
result is the paper (`articles/paper/paper.pdf`): stateful neural guidance sized on the far tail of the correction delta-v.

The GNC chain per tick is Navigation (bias mode or 13-state EKF, with capture/exit phase management) -> Guidance (FTC, NN, Equilibrium Glide, Energy Controller, PredGuid, FNPAG,
Piecewise Constant; the unsigned-magnitude schemes switch to a shared exit-phase controller after the trajectory nadir) -> Thermal Limiter -> Lateral guidance (roll reversal) ->
Command shaping -> Pilot dynamics. Signed-bank schemes (Piecewise Constant; NN in `full_neural` mode) bypass lateral, exit and thermal guidance. The NN sees a 35-element candidate
input vector selected by an `input_mask`; the contract is exported from Rust as `aerocapture_rs.candidate_inputs()`.

Reading order for a human: `README.md` -> `docs/ARCHITECTURE.md` -> `CONTEXT.md` (vocabulary) -> `docs/adr/` (decisions) -> the module reference below -> `docs/design/` (dated
designs).

## Module reference

Module detail lives next to the module it describes. Update those files, not this one, when module behaviour changes; this file keeps only the always-loaded interface (commands,
lessons, conventions).

- `src/rust/README.md` — the simulator: module map (config, data, physics, GNC, integration, orbit, simulation), entry points, termination outcomes / virtual DV, data files, tests.
- `src/rust/aerocapture-py/README.md` — the PyO3 seam `aerocapture_rs`: the five tiers, every entry point's signature and contract.
- `src/rust/src/data/neural/README.md` — the NN runtime: the 35-input contract and normalization, the bank decoders, the model JSON, the ten layer types and their flat order, the
  PyTorch mirror (`training/torch_mirror/`), extensibility recipes, gates.
- `configs/README.md` — TOML configuration: base inheritance, strict validation, every section and key, training configs by scheme.
- `src/python/aerocapture/training/README.md` — the training package: commands, the loop (seed strategies, validation gate, final selection, resume), every module by role, the
  reference trajectory and training order, warm-start, paper tooling and experiments.
- `src/python/aerocapture/training/rl/README.md` — the shelved RL trainer (PPO / SAC, `BatchedSimulation`, reward structure, the paper's Section 5 baseline).
- `experiments/paper/README.md` — the paper's campaign runners; `docs/design/README.md` — the dated design index.
- `docs/validation.md` — physics validation: the AMAT cross-check (oracle `experiments/external_validation/amat_oracle.py`, frozen outputs, our side
  `aerocapture.physics_crosscheck`), its tolerances and findings, the published corridor; background in `docs/research/2026-09-23-amat-capabilities.md`.

## Build & Development Commands

```bash
# ── Rust Simulator ──
cd src/rust
cargo build --release              # Build optimized binary
# Run from repo root:
./src/rust/target/release/aerocapture configs/test/test_ref_orig.toml

# ── PyO3 Bindings (always from the repo root) ──
uv run maturin develop --release --manifest-path src/rust/aerocapture-py/Cargo.toml

# ── Python Analysis ──
uv sync                            # Install dependencies (Python >=3.14)
uv sync --group dev                # Include dev tools (pytest, ruff, mypy, maturin)
pytest tests                       # Run all tests
pytest tests/test_foo.py::test_bar -v

# ── Utility Scripts (from repo root) ──
./build.sh                         # Build Rust binary + PyO3 bindings (-c to clean artifacts)
./setup_env.sh                     # Create fresh .venv + install deps
./lint_code.sh                     # Run ruff (imports, format, lint; also over articles/paper/scripts) + mypy
./check_all.sh                     # Rust: test + fmt --check + clippy + release build
make -C articles/paper             # Paper figures from the committed bundle (no Rust, no run logs); `paper` = fetch-logs -> results -> confirmatory-marginal -> figures -> provenance -> pdf
make -C articles/paper check       # Bundle SHA256SUMS + results.json schema + per-draw confirmatory extract current + figures/results.json unchanged in git (CI runs `-B figures` then this on every PR)
./upgrade_dependencies.sh          # uv sync --upgrade
./train_all.sh                     # Train all 19 registered schemes with optimized GA/PSO/PPO settings
./train_all.sh eqglide             # Train a single scheme (aliases: pc, eq, ec, pg, nn, gru, gru_ppo, rl, scaledpi, delta, etc.)

# ── Training / comparison / report (details: src/python/aerocapture/training/README.md) ──
uv run python -m aerocapture.training.train configs/training/msr_aller_eqglide_train.toml --n-gen 2500 --n-pop 60
uv run python -m aerocapture.training.compare_guidance --n-sims 500 --schemes ftc fnpag neural_network
uv run python -m aerocapture.training.report training_output/equilibrium_glide/ --toml configs/training/msr_aller_eqglide_train.toml
```

Gates before declaring a change done: `./lint_code.sh` (read ruff's and mypy's own output, the script has no `set -e`), `./check_all.sh`, `uv run pytest tests -q -m "not slow"`;
the slow PyO3 suite (`tests/test_pyo3.py`, `tests/test_run_grid.py`) after any change that touches the seam or the version; the slow `tests/test_external_validation.py` after
any physics, aerodynamics, atmosphere or integrator change (tolerance gate against frozen AMAT outputs, and it fails when `docs/validation.md` quotes stale tables). Numbers must not move: the six guidance goldens
(`tests/reference_data/rust_golden/`), `tests/test_pyo3.py::test_pyo3_matches_subprocess`, `tests/test_run_grid.py`; name any additional bit-identity gate a change touches.

## Key Lessons & Pitfalls

### Historical: Density Filter Gain Clamping

*Context: explains why the density filter code has careful gain-clamping logic in `estimator.rs`.*

The original codebase had a memory corruption bug that turned the density filter gain from 0.8 to 56.0, causing 55x error amplification per step. The Rust code clamps lambda to [0.01, 0.99] as a
safety net. Additionally, the legacy bias-mode filter now has rate-of-change limiting (configurable `density_gain_max_delta`, default 0.1) and gain saturation bounds [0.1, 10.0] (named
`DENSITY_FACTOR_MIN`/`DENSITY_FACTOR_MAX`; the EKF centered-state clamp is derived as `MIN-1.0`/`MAX-1.0`) matching the EKF density correction factor range. The saturation clamp runs UNCONDITIONALLY
every tick (not only when the filter updates), so an out-of-range gain is always corrected. Both navigation modes use lift-corrected drag extraction: the density inversion denominator uses
`Cx*cos(alpha) + Cz*sin(alpha)` instead of just `Cx`, correcting a ~4% error at typical AoA=10 deg, and the denominator must be POSITIVE — a lift-dominated negative is rejected (yields a held
estimate) rather than `.abs()`'d into a non-physical negative density. The bias filter additionally skips guard-tripped steps (`density_estimated == 0`) to match the EKF path.

### Historical: Regression Test Tolerances

*Context: explains why regression tests tolerate 2/24 column mismatches.*

Two output columns in the reference implementation used uninitialized variables, producing non-deterministic values. The Rust validation excludes these columns.

### Energy Computation

Energy must use **absolute (inertial) velocity**, not relative velocity. The Rust `total_energy()` converts relative->absolute via `to_absolute_cartesian` before computing E = V_abs^2/2 - mu/r.

### GA Parameter Routing

`param_spaces.py` uses prefixed names to route params to TOML sections: `nav.` -> `[navigation]`, `lateral.` -> `[guidance.lateral]`, `exit.` -> `[guidance.ftc]`, `thermal.` ->
`[guidance.thermal_limiter]`, unprefixed -> `[guidance.<scheme>]`. The path mapping is `param_spaces.route_param_path`; the full deploy rule (routing +
`shaping.* => guidance.command_shaping.enabled = true` + `ParamSpec.is_integer` coercion + `ref_bank` skip) is `deploy_overrides.overrides_from_params`, which `evaluate.write_guidance_toml`,
`reference.nominal_flight_overrides`, the warm-start supervisor batch, `animate`, and (via `resolve_eval_toml` inside `cell_eval`, the one deploy-side evaluation path) `report.py`, `demo.py`,
`compare_guidance` and the paper scripts all consume -- five hand-pasted copies of the enable-flag rule and two private prefix maps (one routing only `lateral.`) preceded it. `problem.py::_build_grid_overrides` (the `run_grid` hot
path) routes with `route_param_path` + its own `is_integer` set and deliberately does not add the enable flag (a section created by a `max_bank_acceleration` override is enabled by default in Rust),
keeping the grid overrides byte-identical. NN training bypasses `write_guidance_toml()` entirely -- navigation-level TOML overrides must be set in the NN training config directly.

### Navigation-Level Config

Density filter params (`density_filter_gain`, `density_gain_max_delta`) live in `[navigation]` TOML section (`TomlNavigation` in config.rs), not in `[guidance.ftc]`. They affect all guidance schemes
via `estimator.rs`. When adding new navigation-level tunable params, put them in `[navigation]` from the start, add to `_NAV_PARAMS` in `param_spaces.py` with `nav.` prefix.

### Silent Config Wiring (existence checks prove nothing)

For months every ref-tracking scheme trained against the legacy `data/reference_trajectory/msr_aller.dat` while `train.py` checked that `training_output/<mission>/ref_trajectory.dat` existed and
printed "Using reference trajectory: ..." — nothing ever injected the path into the config. The lesson: a guard that only checks a file EXISTS does not verify the sim LOADS it; assert on the
resolved config value (`check_ref_trajectory_wiring`). Related trap in the same family: gains and reference co-adapt strongly — FTC's GA optimum gets 133 m/s p50 DV on the reference it trained with
and 0.3% capture rate on a different one, so swapping the reference is always a retrain-everything event. The same lesson applied to keys: serde ignored unknown keys in every section until #105 (a
committed config carried two keys Rust never read for five months); section-level `deny_unknown_fields` + the recursive `every_committed_config_parses_and_validates` gate + the
`toml_keys_reachable_in_sim_data` table are the test. Neither gate sees a key relayed into a runtime field nothing reads: #128 found four LIVE GA genes (3 FTC, 1 EC) that trained as a pure
noise walk for months; the dead-code lint on allow-free runtime structs is that gate, and the model JSON / `[cost_function]` channels now deny unknown keys too.

### Reference Trajectory Design (open-loop optimum != good reference)

Three lessons from wiring the generated reference in (2026-06-11). (1) UNITS: the ref file contract is MJ/kg in column 0 (the Rust loader multiplies by 1e6); the writer shipped J/kg for months —
invisible while the file was unused, catastrophic when wired in (FTC val RMS 6.9e9 vs 2.7e6). The loader now hard-errors on J/kg-looking files. (2) COVERAGE DOMINATES: a reference must reach the
target orbit energy on its nominal (legacy overshoots by ~0.7 MJ/kg). GA-optimal open-loop profiles under-capture and leave the table short of the energies trackers fly through; FTC val RMS tracked
the shortfall almost monotonically (0 short -> 2.7e6, 0.43 short -> 9.7e6, 1.65 short -> 1.5e11). Constant-bank energy-matched nominals (`make_reference.py`) reproduce the legacy methodology and train
FTC to parity. (3) RUN VARIANCE IS LARGE: two identical-config FTC trainings on the same reference table spanned 2.78e6-4.08e6 val RMS — single-run comparisons between references/knobs are not
conclusive; quote seed-repeat error bars before attributing improvements.

### Golden File Regeneration

Physics changes (density estimation, gravity, aerodynamics) and virtual-DV formula changes invalidate guidance regression golden files in `tests/reference_data/rust_golden/`. Regenerate by running the
updated binary on each test config and replacing the CSV files. The 6 golden files cover: eqglide, energy_ctrl, pred_guid, fnpag, ftc, neural. Virtual-DV changes affect only the `dv_total_m_s` column
on non-capture rows (ifinal != 3) -- captures are bit-identical.

### Resume: cross-gen training-cost incomparability

On resume under adaptive/rotating seeds, the checkpointed `best_overall_cost` (training RMS at the gen it was promoted, under seed list A) is NOT comparable to the resumed population's cost snapshot
(under the most recent seed list B). `SingleAlgoTrainer.__init__`'s initial-best-init block is gated on `best_overall_individual is None` (fresh start only) -- it must never swap the
checkpointed best via a `<` comparison, because that would silently promote an un-validated individual and corrupt the re-validation + `best_model.json` write. Regression test:
`tests/test_train_interrupt.py::TestResumePreservesCheckpointedBest`.

### pymoo CMA-ES self-terminates -- the single-algo loop must guard `next()` with `has_next()`

CMA-ES is the only optimizer wrapping an external engine (pycma) that has its OWN internal termination; GA/DE/PSO/QPSO use `NoTermination` and never self-stop. When pycma's convergence/restart
criteria fire (e.g. `restart_strategy = "ipop"` exhausts its restarts on the noisy adaptive-seed objective), pymoo's `CMAES._advance` catches the generator's `StopIteration`, sets `next_X = None`, and
flags termination. Calling `algorithm.next()` again crashes in `norm.backward(np.array(None))` -- `np.atleast_2d(np.array(None))` is shape `(1,1)`, so the population width collapses to 1 and the
26-long (= `n_var`) bounds mask raises `IndexError: boolean index did not match indexed array along axis 1`. The single-algorithm `train()` loop therefore guards every iteration with a CMA-ES-gated
`if is_cmaes and not algorithm.has_next(): break` (the `is_cmaes` instance check, computed once before the loop, also gates the pre-next re-eval skip), so a converged CMA-ES ends cleanly and the
post-loop final selection / eval / report still run on the converged pop. Consequence for eval-budget-matched optimizer benchmarks (e.g. `experiments/paper/03_optimizer_dimensionality.sh`): CMA-ES may
stop well before `n_gen` -- raise `restarts` / use `bipop`, or footnote the asymmetry. The islands path is unaffected (islands are hardcoded PSO/GA/DE). Regression:
`tests/test_optimizer.py::TestCmaesInternalTermination`.

### Noise regimes: per-draw is the default, legacy reproduces quoted numbers

`[monte_carlo] noise_seeding = "per_draw"` (ADR-0006) is the default: each dispersion draw gets its own OU-density / EKF-noise stream. `"legacy"` freezes ONE noise path across every
n_sims=1 config and exists only to reproduce numbers quoted under it: the goldens, every `configs/test/*.toml`, the paper's main-body tables and every script that re-flies a shared-path cell
pin it through `deploy_overrides.LEGACY_NOISE_REGIME`. A new evaluation script must state which regime it runs and not mix the two in one table.

## Conventions

- **Rust**: Edition 2024, nalgebra for linear algebra, release profile with LTO. Every aero/physics expression is pinned by goldens and bit-identity gates: never reassociate or `mul_add`.
- **Python**: Python >=3.14, Ruff (line-length 160, target py314), uv package manager, pytest, mypy strict mode. Dev tools in `[dependency-groups]` (not `[project.optional-dependencies]`). Training
  deps (pymoo, scipy) are core dependencies; **pymoo is pinned `>=0.6,<0.6.2`** (0.6.2 routes internals through the compiled moocore 0.3.1, which silently SIGABRTs the interpreter — exit 134, no
  traceback, faulthandler mute. Two known trips: `igd()` for points wider than 32 dims, fed n_var-wide design-space points by pymoo's default single-objective termination, so every >32-param training
  problem dies — repro `moocore.igd(rand(1, 64), ref=rand(1, 64))`; and `gd_common`'s stack buffer overflow at n_pop ≈ 512 — smoke tests at n_pop=8 pass while real runs die after gen 0. The June
  2026 campaign trained on 0.6.1.6; lift only after both repros stop aborting). Shell runners must survive macOS bash 3.2:
  expanding an EMPTY array under `set -u` is an unbound-variable error — use `${arr[@]+"${arr[@]}"}`. Every module CI's pure-Python job imports must soft-import `aerocapture_rs`
  (`tests/test_soft_import.py`).
- **Testing (Python)**: pytest, hypothesis (property-based). Golden reference files under `tests/reference_data/`. Shared fixtures in `tests/conftest.py` (session-scoped Rust build) and
  `tests/fixtures/factories.py` (config/chromosome factories). `pytest --collect-only -q` is the inventory; every file under `tests/` runs in CI (no allowlist), so a new test file runs there.
- **Testing (Rust)**: Three-tier pyramid — unit tests (inline `#[cfg(test)]` modules with proptest property tests), integration tests (`src/rust/tests/`), E2E subprocess tests. Shared test
  infrastructure in `tests/common/` (fixtures.rs, assertions.rs). Dev-dependencies: `approx`, `rstest`, `proptest`, `tempfile`. Run with `cargo test` or `./check_all.sh`.
- **CI**: GitHub Actions (`.github/workflows/ci.yml`) - Rust (fmt, `clippy --workspace`, `test --workspace`: both crates), Python lint (ruff lint + ruff format over
  `src/python tests experiments articles/paper/scripts`, mypy over `src/python tests experiments` -- the `lint_code.sh` scope; the paper scripts are untyped), ONE Python test job that builds the CLI
  binary and the PyO3 extension, installs Typst 0.15.1 (so the report compile gate in `tests/test_report_render.py` runs instead of skipping) and runs every file under `tests/`, fast and
  slow (an import step before pytest proves the extension is present), and a pure-Python `paper` job (`make -C articles/paper -B figures` + `check` + `pdf` to /tmp, pinned Typst 0.15.1) that
  proves the 18 figures are byte-identical to git; `paper-results` (workflow_dispatch only) fetches the 195 MB run logs, runs `make paper` end to end and requires `results.json` unchanged.
  Runs on every push to `main`, every PR to `main`, and manual dispatch.
- **Docs**: module behaviour is documented in the per-package README next to the code (the pointer block above); this file carries lessons and conventions only. Design docs are dated files
  under `docs/design/` (indexed in its README); decisions are ADRs under `docs/adr/`; the roadmap is `TODO.md`. `DEVELOPMENT.md` is the human-facing agent policy; it quotes the
  `.claude/settings.json` deny list verbatim (tracked since #111, `tests/test_development_md.py` fails on drift), so a deny-list edit updates both.
- **Validation**: 22/24 photo columns bit-identical to the reference implementation across 725 timesteps, plus the independent evidence in `docs/validation.md`. "Validation"
  also names the training seed pool: say physics validation for this sense (`CONTEXT.md`).

## Tone

Be a **quirky friendly but critical peer reviewer**. Think of yourself as a quirky senior developer doing a code review: helpful, but holding me to high standards. Always **Challenge inefficiencies**:
if I'm doing something the hard way, call it out.

## Agent skills

### Issue tracker

Issues live in this repo's GitHub Issues, operated via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary: the five canonical role names are the label strings (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`); categories are `bug` / `enhancement`. See
`docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` at the repo root + ADRs in `docs/adr/`. See `docs/agents/domain.md`.
