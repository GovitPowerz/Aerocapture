# ADR-0006: Per-scenario noise seeding is the default regime; the shared noise path is reproduction-only

**Status:** accepted · **Date:** 2026-09-16 (issue #108) · supersedes the default clause of ADR-0003

## Context

ADR-0003 (2026-08-27) shipped `[monte_carlo] noise_seeding = "per_draw"` behind a flag and kept
`"legacy"` as the default so that every committed number, golden and paper table stayed
bit-identical through the arxiv-v3 freeze. That was the right call for the freeze and the wrong
resting state: the simulator's ordinary behaviour remained the conditioning defect the paper's
Appendix E discloses (every n_sims = 1 evaluation sharing ONE realization of the time-varying
density noise, which the networks exploit 2-4x more than the classical laws), the demo flew the
shared-path champion and needed `--per-draw` to show the honest number, and the README, TODO and
the paper's abstract and conclusion led with the shared-path result (CVaR99.9 123.3 m/s) and
reached the corrected one (163.2 +- 1.3 m/s) last. A reader saw a caveat, not a correction.

## Decision

`NoiseSeeding::PerDraw` is the default: an absent `noise_seeding` key means every distinct
dispersion draw gets its own noise realization (identical draw, identical stream). `"legacy"`
keeps its name and its bit-exact behaviour and must be set explicitly; it exists to reproduce
numbers quoted under the shared path (the paper's main body through arxiv-v3, and every appendix
cell trained before this decision). The corrected per-scenario fine-tune
(`models/demo/ft_mamba_962/`, the Appendix E champion) is the demo's model; the shared-path
champion stays as `models/demo/mamba_962_legacy/` behind `--legacy`. Front-facing numbers (README,
TODO, the paper's abstract and conclusion) lead with the per-scenario result; the shared-path
result is quoted under a heading that names it as the historical result and its correction.

## Consequences

- Every config whose committed output must not move pins `noise_seeding = "legacy"`: the six
  guidance goldens and the seven undispersed test configs (`configs/test/*.toml`, regime-neutral in
  practice since they run without the OU perturbation, pinned to state it; a seed-only
  `[monte_carlo]` table is byte-neutral for an undispersed run) and every evaluation script that
  re-flies a cell trained under the shared path: `articles/paper/scripts/*`, `experiments/fnpag_ab/`,
  `param_sweep --eval`, `quantize`, the architecture-probe drivers. They spread one constant,
  `deploy_overrides.LEGACY_NOISE_REGIME`, into their overrides. `confirmatory_eval.py` records the
  regime in its JSON and refuses to mix regimes in one file.
- Training configs inherit the new default. Re-running a paper campaign script now trains under
  per-scenario noise: that is a new experiment, not a reproduction. The committed bundle
  (`articles/paper/data/runs/`) is the reproduction of the shared-path campaign; the per-scenario
  cells live under `experiments/ou_marginal/` and `configs/training/ou_marginal/`.
- `report.py` prints the regime it flies; a report regenerated for a shared-path cell without a
  pin quotes per-scenario numbers, and says so.
- ADR-0003's other consequence stands: frozen and per-draw numbers are never compared in one table
  without naming the regime.
- The paper's next arXiv version leads with the per-scenario result in the abstract and the
  conclusion (`articles/paper/CHANGES.md`); its committed PDF is recompiled only when that version
  pass is complete, so the source and the PDF differ until then.
