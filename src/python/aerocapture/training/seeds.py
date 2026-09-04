"""Reserved Monte Carlo seed pools.

Every purpose (training, validation, final eval, RL training, warm-start collection,
NN input report, calibration, sweep scoring, headline requote, stress, probes,
confirmatory sizing) draws its seeds from its own RNG stream, keyed by an offset
registered here, so no two pools share seeds by construction. Add a new pool by
registering its offset in this file; never hard-code a seed base elsewhere.
"""

from __future__ import annotations

import numpy as np

# Reserved seed offsets — guarantees training, validation, final eval, RL
# training, and supervised warm-start collection never share the same RNG stream.
VALIDATION_SEED_OFFSET = 1_000_000
FINAL_EVAL_SEED_OFFSET = 2_000_000
RL_TRAINING_SEED_OFFSET = 3_000_000
WARM_START_SEED_OFFSET = 4_000_000
NN_INPUT_REPORT_SEED_OFFSET = 5_000_000
CALIBRATION_SEED_OFFSET = 6_000_000
# Architecture parameter-budget sweep scoring pool (param_sweep.py --eval): every
# sweep cell is re-scored on this ONE pool so the Pareto curve is directly comparable.
SWEEP_EVAL_SEED_OFFSET = 7_000_000
# Fresh pool for the paper's headline re-quote: the headline config is SELECTED
# by sweeps scored on the 2M final-eval pool, so quoting that pool is
# selection-on-test; the abstract number comes from this untouched stream.
HEADLINE_REQUOTE_SEED_OFFSET = 8_000_000
# Off-nominal robustness stress pool: deployed policies evaluated on a HARDER
# MC regime (atmosphere/density/nav at level=high), disjoint from every training
# and eval stream so the stress test is reproducible and uncontaminated.
STRESS_EVAL_SEED_OFFSET = 9_000_000
# Shared reserved eval pool for the architecture probe scripts (mamba3 2x2,
# cfc-vs-gru, lstm-vs-slstm-vs-mlstm): all probes score on ONE pool so their
# reports are directly comparable. Disjoint from every training/validation/
# final/other-eval stream above.
PROBE_EVAL_SEED_OFFSET = 10_000_000
# Legacy alias (mamba3_962_compare.py imports this name).
MAMBA3_EVAL_SEED_OFFSET = PROBE_EVAL_SEED_OFFSET


def make_reserved_seeds(base_mc_seed: int, offset: int, n: int) -> list[int]:
    """Generate a deterministic, reproducible list of MC seeds from a reserved RNG stream.

    Given the same (base_mc_seed, offset, n), always returns the same seeds.
    Different offsets produce independent streams -- disjointness between pools
    is therefore probabilistic (collision odds ~n^2/2^31 per pool pair), not
    guaranteed by construction; train.py additionally excludes the reserved
    validation/final-eval pools from rotating/adaptive training draws.
    """
    seeds: list[int] = np.random.default_rng(base_mc_seed + offset).integers(0, 2**31, size=n).tolist()
    return seeds


# Confirmatory sizing-pool RNG stream (the seeds themselves live in [2^31, 2^32), a
# different mechanism: see make_confirmatory_pools). NOTE: articles/paper/scripts/
# collect_corridor.py uses 10_000_000 as a raw `monte_carlo.seed` base for its bank
# draws -- not a make_reserved_seeds stream, so it does not collide with PROBE_EVAL.
CONFIRM_EVAL_SEED_OFFSET = 20_000_000


def make_confirmatory_pools(base_mc_seed: int, n_replicates: int = 10, n: int = 100_000) -> list[list[int]]:
    """Frozen confirmatory sizing pools: n_replicates independent pools of n seeds each.

    Seeds are drawn WITHOUT duplicates from [2**31, 2**32) -- structurally disjoint
    from every historical pool and training/curation draw (all generated in
    [0, 2**31) via make_reserved_seeds or the trainer's seed draws), which makes the
    pool selection-disjoint by construction rather than by birthday-bound argument.
    Deterministic in base_mc_seed.
    """
    rng = np.random.default_rng(base_mc_seed + CONFIRM_EVAL_SEED_OFFSET)
    total = n_replicates * n
    seeds: set[int] = set()
    while len(seeds) < total:
        seeds.update(rng.integers(2**31, 2**32, size=total - len(seeds)).tolist())
    ordered = rng.permutation(np.array(sorted(seeds), dtype=np.int64))
    return [ordered[i * n : (i + 1) * n].tolist() for i in range(n_replicates)]
