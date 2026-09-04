# Design documents

One dated design per feature, in the order they were written. The **status** column is the
document's own `Status` field at the time of writing (design / approved / draft ...); nearly
everything here shipped -- `CLAUDE.md` describes the current behaviour, and `docs/adr/` records
the decisions that still constrain new work. A design with no status line predates the field.

## NN architectures & inputs

| date | design | status |
|---|---|---|
| 2026-03-12 | [Neural Network Weight Initialization for GA Training](2026-03-12-nn-weight-initialization-design.md) | Approved |
| 2026-03-30 | [Exit Phase Guidance Design](2026-03-30-exit-phase-guidance-design.md) | Approved |
| 2026-04-02 | [NN Expanded Inputs & Full-Envelope Guidance](2026-04-02-nn-expanded-inputs-design.md) | Draft |
| 2026-04-13 | [NN Input Expansion and Pruning Design](2026-04-13-nn-input-expansion-pruning-design.md) | — |
| 2026-04-15 | [Reinforcement Learning for Neural Network Guidance](2026-04-15-rl-nn-guidance-design.md) | Design (brainstormed 2026-04-15) |
| 2026-04-16 | [RL Reward Redesign: Phase-Aware Per-Step Rewards + Normalization](2026-04-16-rl-reward-redesign.md) | Approved |
| 2026-04-17 | [Phase 1 -- PSO-GRU MVP](2026-04-17-phase-1-gru-mvp-design.md) | Design approved, ready for implementation planning |
| 2026-04-17 | [Phase 0 -- Stateful Neural Network Runtime Infrastructure](2026-04-17-stateful-nn-runtime-infrastructure-design.md) | Design approved, ready for implementation planning |
| 2026-04-18 | [Phase 1.5 -- PPO-GRU with Truncated BPTT](2026-04-18-phase-1-5-ppo-gru-bptt-design.md) | Design approved, ready for implementation planning |
| 2026-04-18 | [Phase 2a -- LSTM MVP (PSO + PPO-BPTT) with Activation-Aware Init](2026-04-18-phase-2a-lstm-mvp-design.md) | Design approved, ready for implementation planning |
| 2026-04-20 | [Phase 2b -- Window-MLP (PSO-only)](2026-04-20-phase-2b-window-mlp-design.md) | Design approved, ready for implementation planning |
| 2026-04-22 | [Phase 3a -- Transformer MVP (PSO-only)](2026-04-22-phase-3a-transformer-mvp-design.md) | Design approved, ready for implementation planning |
| 2026-04-24 | [Phase 4a -- Mamba Selective SSM MVP (PSO-only)](2026-04-24-phase-4a-mamba-ssm-mvp-design.md) | Design approved, ready for implementation planning |
| 2026-05-07 | [NN-vs-FTC Parity Bundle (3-fix design)](2026-05-07-nn-ftc-parity-bundle-design.md) | — |
| 2026-05-22 | [Warm-Start for All NN Architectures (FTC+friends → BPTT → GA fine-tune)](2026-05-22-warm-start-all-archs-design.md) | — |
| 2026-05-29 | [NN bank decoders: `scaled_pi` + `delta`, with `(sin,cos)` angle inputs](2026-05-29-nn-bank-decoders-design.md) | — |
| 2026-05-29 | [NN input behavior report](2026-05-29-nn-input-report-design.md) | — |
| 2026-05-29 | [NN input rescaling (asinh signed-log) + periapsis altitude](2026-05-29-nn-input-rescale-design.md) | — |
| 2026-05-29 | [Three-way `scaffolding` knob for NN training](2026-05-29-scaffolding-three-way-knob-design.md) | Design approved, pending implementation plan |
| 2026-06-01 | [NN Input Vector v2: Renormalize + pdyn_error + Live Correction-DV](2026-06-01-nn-input-vector-v2-design.md) | Design approved, pending implementation plan |
| 2026-06-01 | [NN Input Pipeline v3: Unified Normalization Schema + Smooth Always-Defined DV](2026-06-01-nn-normalization-and-smooth-dv-design.md) | Design approved, pending implementation plan |
| 2026-06-08 | [Aerocapture Neural-Guidance Article — Design](2026-06-08-aerocapture-nn-article-design.md) | Design approved (pending spec review) |
| 2026-06-08 | [DV-inferred reward for dense PPO](2026-06-08-rl-ppo-dv-reward-design.md) | — |
| 2026-07-07 | [CfC + xLSTM probes -- design](2026-07-07-cfc-xlstm-probes-design.md) | — |
| 2026-07-07 | [Mamba-3 ablation spike -- design](2026-07-07-mamba3-ablation-design.md) | — |
| 2026-07-10 | [Quantization study appendix (Mamba-962) -- design](2026-07-10-quantization-study-appendix-design.md) | — |

## Optimization & training loop

| date | design | status |
|---|---|---|
| 2026-03-12 | [Design: Rotating Dispersion Seeds for GA Training](2026-03-12-rotating-dispersion-seeds-design.md) | — |
| 2026-03-14 | [Adaptive Seed Pool & Graceful Keyboard Interrupt](2026-03-14-adaptive-seeds-graceful-interrupt-design.md) | Draft |
| 2026-03-15 | [Cost Function Rework: Delta-V Primary with Constraint Penalties](2026-03-15-cost-function-rework-design.md) | — |
| 2026-03-20 | [Unified Cost Function, Pending Crash Detection & Sentinel Corridor](2026-03-20-unified-cost-function-design.md) | Proposed |
| 2026-03-28 | [Training Fixes & DV Chart Improvements](2026-03-28-training-fixes-design.md) | — |
| 2026-04-03 | [Adaptive Seed Pool: Stress Tests + Keep-Hardest Eviction](2026-04-03-adaptive-seed-stress-tests-design.md) | — |
| 2026-04-10 | [Training Strategy: Optimal GA Settings for All Guidance Schemes](2026-04-10-training-strategy-design.md) | — |
| 2026-04-12 | [Design: Epoch Seed Rotation + Validation Gate](2026-04-12-epoch-rotation-validation-gate-design.md) | — |
| 2026-04-12 | [Real-Valued Optimization Framework with pymoo](2026-04-12-pymoo-optimizer-framework-design.md) | Design |
| 2026-04-14 | [Curated-CDF Adaptive Seed Framework](2026-04-14-curated-cdf-seed-framework-design.md) | design |
| 2026-04-14 | [Explicit `seed_strategy` Configuration](2026-04-14-explicit-seed-strategy-design.md) | design |
| 2026-04-14 | [Gap-Closure Seed Pool Eviction Design](2026-04-14-gap-closure-eviction-design.md) | — |
| 2026-04-21 | [Re-validate best individual on training resume](2026-04-21-validate-on-resume-design.md) | — |
| 2026-05-28 | [Three-Island PSO/GA/DE with Episodic Migration — Design](2026-05-28-island-model-pso-ga-de-design.md) | Approved (brainstorming complete, awaiting implementation plan) |
| 2026-06-02 | [Resume Enhancements: Re-validation, Population Growth, Cost-Transform Reset](2026-06-02-resume-enhancements-design.md) | — |
| 2026-06-10 | [End-of-Training Final Selection — Design](2026-06-10-final-selection-design.md) | Approved (brainstorming complete, awaiting implementation plan) |
| 2026-06-10 | [QPSO Optimizer — Design](2026-06-10-qpso-optimizer-design.md) | Approved (brainstorming complete, awaiting implementation plan) |
| 2026-06-29 | [Worst-case objective-shaping is regime-matched (centered training under adversarial dispersions)](2026-06-29-objective-centering-regime-matched-design.md) | — |

## Simulation & GNC

| date | design | status |
|---|---|---|
| 2026-03-10 | [Design: Full Sweep Rust Variable Rename](2026-03-10-rust-variable-rename-design.md) | Complete |
| 2026-03-19 | [Piecewise-Constant Bank Guidance & Corridor Optimization](2026-03-19-piecewise-constant-corridor-design.md) | Approved |
| 2026-03-24 | [Shortest-Path Bank Angle Control](2026-03-24-shortest-path-bank-angle-design.md) | — |
| 2026-03-26 | [Simulation Credibility Improvements — Design Spec](2026-03-26-simulation-credibility-improvements-design.md) | — |
| 2026-03-27 | [Adaptive Integration: Dormand-Prince 4(5)](2026-03-27-adaptive-dopri45-integration-design.md) | Approved |
| 2026-03-27 | [Separate Truth vs Onboard Atmosphere Models](2026-03-27-separate-onboard-atmosphere-design.md) | — |
| 2026-03-28 | [Roll Reversal for Unsigned-Magnitude Guidance Schemes](2026-03-28-roll-reversal-design.md) | — |
| 2026-03-29 | [Higher-Order Gravity Harmonics (J3, J4) Design](2026-03-29-gravity-harmonics-design.md) | Approved |
| 2026-04-01 | [Thermal Safety Limiter for Guidance](2026-04-01-thermal-limiter-design.md) | — |
| 2026-04-01 | [Time-Varying Density Perturbations](2026-04-01-time-varying-density-perturbations-design.md) | Approved |
| 2026-04-02 | [Density Estimation Improvements Design](2026-04-02-density-estimation-improvements-design.md) | — |
| 2026-04-03 | [Bank Angle Rate and Acceleration Command Shaping](2026-04-03-bank-angle-rate-shaping-design.md) | Design approved |
| 2026-04-03 | [FTC Gain Analytical Model Design](2026-04-03-ftc-gain-analytical-model-design.md) | — |
| 2026-04-04 | [FNPAG 3D Predictor Upgrade](2026-04-04-fnpag-3d-predictor-design.md) | Proposed |
| 2026-04-05 | [Advanced Sampling Methods & Sensitivity Analysis](2026-04-05-advanced-sampling-sensitivity-design.md) | — |
| 2026-04-05 | [Predictive Roll Reversal (First-Order Inclination Projection)](2026-04-05-predictive-roll-reversal-design.md) | — |
| 2026-04-07 | [Event Detection for DOPRI45 Adaptive Integrator](2026-04-07-event-detection-design.md) | — |

## Reports & visualization

| date | design | status |
|---|---|---|
| 2026-03-11 | [Training Visualization Design](2026-03-11-training-visualization-design.md) | — |
| 2026-03-13 | [Final Evaluation Report Design](2026-03-13-final-evaluation-report-design.md) | — |
| 2026-03-16 | [Training Report Improvements](2026-03-16-training-report-improvements-design.md) | — |
| 2026-03-17 | [Corridor Visualization Redesign](2026-03-17-corridor-visualization-design.md) | — |
| 2026-03-17 | [Final Report Improvements — Design Spec](2026-03-17-final-report-improvements-design.md) | — |
| 2026-03-23 | [Cap DV at 5000 m/s + Log Scale in Final Report](2026-03-23-dv-plot-cap-log-scale-design.md) | — |
| 2026-03-24 | [Replace Plotly Reports with Typst PDF Reports](2026-03-24-plotly-to-typst-pdf-reports-design.md) | Approved |
| 2026-03-30 | [Training Animation Design](2026-03-30-training-animation-design.md) | Approved |
| 2026-04-11 | [Output & Analysis Improvements Design](2026-04-11-output-analysis-improvements-design.md) | Approved |
| 2026-06-10 | [Single-Algorithm TUI Dashboard — Design](2026-06-10-tui-dashboard-design.md) | Approved (brainstorming complete with browser mockups, awaiting implementation plan) |
| 2026-07-03 | [Per-scheme mission-report appendix](2026-07-03-per-scheme-appendix-design.md) | design (awaiting review) |
| 2026-07-06 | [Reachable-corridor visualization](2026-07-06-reachable-corridor-viz-design.md) | design (awaiting review) |

## Infrastructure & paper

| date | design | status |
|---|---|---|
| 2026-03-10 | [Directory Restructure Design](2026-03-10-directory-restructure-design.md) | — |
| 2026-03-10 | [Test Coverage Expansion Design](2026-03-10-test-coverage-expansion-design.md) | — |
| 2026-03-14 | [PyO3 Rust-Python Interface for Aerocapture](2026-03-14-pyo3-interface-design.md) | Draft |
| 2026-03-16 | [TOML Base Inheritance Design](2026-03-16-toml-base-inheritance-design.md) | — |
| 2026-03-20 | [Remove `--guidance` CLI Flag](2026-03-20-remove-guidance-cli-flag-design.md) | Approved |
| 2026-06-04 | [Python + PyO3 Review Remediation — Design](2026-06-04-python-pyo3-review-fixes-design.md) | — |
| 2026-06-12 | [Paper Experiments Reorganization — Design](2026-06-12-paper-experiments-reorg-design.md) | Approved (user, 2026-06-12) |
