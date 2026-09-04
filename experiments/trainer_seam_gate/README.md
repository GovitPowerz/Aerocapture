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
