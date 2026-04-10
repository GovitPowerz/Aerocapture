# TODO

## Seed Pool: Adversarial Curriculum Collapse

The keep-hardest eviction strategy in `seed_pool.py` creates a difficulty ratchet --
the pool fills with only the hardest MC scenarios, causing over-optimization for edge
cases at the expense of average performance. Combined with stress tests (every 5 gens:
probe 200 fresh seeds, inject the 20 hardest), this is textbook adversarial curriculum
collapse.

**Fix options (pick one or combine):**
- Revert to gap-closure eviction (preserves difficulty spectrum coverage)
- Cap difficulty bias (always keep the 20% easiest seeds)
- Reduce stress test injection (fewer seeds, or stop at pool capacity)
- Separate evaluation from training (fixed hold-out set for real metrics)

## Stale Comment in param_spaces.py

Line 53-54 says `_NAV_PARAMS` routes to `[guidance.ftc]` but `evaluate.py` correctly
routes to `[navigation]`. Comment needs updating.

---

## Backlog

- [ ] Output format improvements (HDF5/Parquet, metadata, dispersions in final CSV)
- [ ] Switch to real-valued GA + alternative optimizers (CMA-ES, PSO, RL)
- [ ] Explore LSTM / Transformer architectures for guidance
- [ ] Add neural counterparts for navigation and control
- [ ] Develop ESR (Earth Sample Return) mission profiles
