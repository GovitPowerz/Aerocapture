# Trainer-seam bit-equivalence gate

The `trainer.py` refactor (one per-generation loop contract, two adapters:
`SingleAlgoTrainer` and `IslandsTrainer`) is required to reproduce the legacy
training loops bit-exactly. This gate is the evidence: three short trainings
covering the three code paths, run before and after the refactor, compared
record by record.

| gate | path exercised | config |
|------|----------------|--------|
| a | single-algo GA, fixed seeds, classical scheme | `gate_a_eqglide_fixed.toml` |
| b | single-algo GA, adaptive seeds, NN scheme (curation, validation gate, scaffolding, NN deploy) | `gate_b_mamba_adaptive.toml` |
| c | 3-island PSO/GA/DE with migration and per-island validation | `gate_c_islands.toml` |

Procedure (outputs go to the gitignored `.scratch/trainer_seam_gate/<tag>/`):

```bash
git checkout <pre-refactor-commit>
experiments/trainer_seam_gate/run_gate.sh baseline
git checkout <post-refactor-commit>
experiments/trainer_seam_gate/run_gate.sh post
uv run python experiments/trainer_seam_gate/diff_gate.py baseline post
```

`diff_gate.py` compares every JSONL training record (volatile keys such as
timestamps, wall times, and tag-bearing paths stripped), every array of the
final checkpoint npz, and `final_selection.json`; exit 0 means bit-equivalent.
It trains, so it is a script, not a pytest. Gate configs base-inherit the
real training configs and write their NN artifacts to gate-local deploy paths
so they never clobber a real cell.

The gate passed for the seam refactor on `feature/front-door` (2026-08-27):
all three runs identical pre/post. Two prerequisites it surfaced, now in
`train.py`: pymoo's per-algorithm RNG must be seeded from the training RNG
(operators drew from an unseeded stream, so full-run reproducibility never held
even under `seed_strategy = "fixed"`), and `config_hash` embeds a memory
address and must be treated as volatile.

Re-run for the feasibility gate (`feature/feasibility-gate`, 2026-09-16, ADR-0005):
baseline = `main` source, post = the gated source, same lockfiles. Gate b is
identical modulo the two new validation-record keys (`feasible`,
`violation_rates`). Gates a and c diverge only where the gate rejected an
infeasible promotion on the 20-sim gate pool at the strict default ceiling:
in a, every validated candidate exceeds the heat-flux limit on 2/20 draws, so
nothing promotes and the run ends on the no-champion fallback (same deployed
individual as the baseline champion, `winner_feasible = false`); in c, the PSO
island's gen-0 argmin exceeds the heat-load limit on 1/20 draws, is rejected,
and every later record is identical. That is the intended behaviour change,
not a seam regression.
