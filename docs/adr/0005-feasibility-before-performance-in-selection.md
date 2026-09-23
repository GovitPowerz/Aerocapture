# ADR-0005: A candidate must be feasible on the validation pool before its performance can promote it

**Status:** accepted · **Date:** 2026-09-16 (gate implemented 2026-07-16 on the CPAG branch, commit `0f6d531`; extracted by PR #116, issue #109)

## Context

The training objective penalizes constraint exceedances softly (`cost.py`: heat flux, g-load,
heat load as normalized penalties on top of the correction DV). Soft penalties do not enforce
feasibility at the optimum: the paper's LSTM cell, the lowest-validation-RMS run of its campaign,
violates the integrated heat-load limit on 13.7% of 10^6 confirmatory scenarios
(`articles/paper/data/confirmatory_eval.json`, `lstm_p1082_long`), and its per-draw fine-tune
inherits the defect (`experiments/ou_marginal/RESULTS.md`). Promotion was val-RMS only: the
in-training validation gate (`evaluate.run_validation_gate`), the islands `validate_each`, the
pre-loop initial-champion validation and the end-of-training final selection
(`final_select.select_final_individual`, ADR-0002) all compared RMS and nothing else, so an
infeasible winner deployed silently. The CPAG C2 campaign showed the failure mode live: the
ungated pre-loop path anchored a 5.75%-heat-load-violating gen-0 argmin as champion, and once
gated the optimizer produced 21+ better-RMS infeasible candidates against 3 feasible promotions
(`docs/plans/2026-07-19-cpag-c2-results.md` on `feature/cpag-c1-rust-mvp`; summary on main:
`docs/design/2026-09-23-cpag-shelved.md`).

## Decision

Selection is lexicographic. (1) A candidate is **feasible** when, on the validation pool, every
constraint with a configured limit (`[flight.constraints]` via `read_cost_kwargs`) has a
violation rate at or below `[optimizer] max_violation_rate` (default `0.0`, strict).
(2) Among feasible candidates, the existing rule applies: strictly lower validation RMS
promotes. (3) The final-eval and confirmatory pools stay report-only (ADR-0002).

The check runs at every promotion site: the validation gate, the islands per-island gate, the
pre-loop initial / resumed champion validation, the islands resume re-validation, and final
selection (fresh candidates evaluated once via `evaluate_population_records_per_seed`, so the
records that carry the constraint columns cost no second MC pass). An infeasible fresh candidate
never displaces a champion. A resumed champion that fails the check is kept as
`last_validated_individual` but its RMS does not anchor `best_val_cost` (reset to `inf`); at final
selection it competes as an ordinary candidate rather than as a trusted champion. With no trusted
champion and no feasible candidate, final selection deploys the best-RMS infeasible candidate
(that resumed champion included) with a loud warning and `winner_feasible = false` in
`final_selection.json`, rather than leaving downstream consumers without an artifact. The retro
CLI (`python -m aerocapture.training.final_select`) re-validates checkpoint champions under the
same rule before trusting them, so a pre-rule directory can be re-selected honestly.

`constraint_violation_rates(final_records, cost_kwargs)` in `evaluate.py` is the single
implementation; rates and the `feasible` flag are written to the validation JSONL records and
the final-selection sidecar.

## Consequences

- Deployed champions are feasible on the validation pool at the configured ceiling; the paper's
  LSTM asterisk cannot recur unnoticed.
- The ceiling is a sample-size question: on a 1000-sim validation pool `0.0` is the right default;
  on a 400-sim pool one tail draw would block every promotion, which is why the CPAG C2 campaign
  ran at `0.01`. State the ceiling with the pool size when quoting a run.
- Cells trained before this rule were never gated. Re-checked offline on the bundled final-eval
  pools (n = 1000, ceiling 0): 114 of 122 pass; `headline/lstm_p1082` (15.6% heat load) and
  `training_n_sims/adaptive_2` (0.4% heat load) fail on heat load, and six cells (three classical
  baselines among them) exceed the heat-flux limit on 0.1-1.1% of draws. Quoted tables carry the
  violation column so a reader sees which numbers rest on an infeasible policy.
- Headline tables report capture, violation rate, the tail statistic with its uncertainty, the
  worst observed outcome and `n` together; CVaR over captured scenarios alone hides rare
  failures and constraint exceedances.
