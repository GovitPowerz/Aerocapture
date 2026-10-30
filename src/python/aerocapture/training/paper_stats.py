"""Run-log accounting for the paper: the simulation count a training run actually spent.

The statistics (capture, the DV tail, CVaR, paired comparisons) are `sizing_tail`.
"""

from __future__ import annotations


def actual_sims(
    records: list[dict],
    training_n_sims: int,
    validation_n_sims: int = 1000,
    curation_sample_size: int = 1000,
    curation_top_k: int = 1,
) -> dict[str, int]:
    """Reconstruct the run's ACTUAL simulation count from its JSONL records.

    training   = n_gen x n_pop x training_n_sims (selection-driving evals)
    validation = validation-gate fires x validation_n_sims
    curation   = distinct curation events x top_k x sample_size (probe sims)
    reeval     = parent re-evals on seed-change gens (n_pop x n_sims each,
                 approximated by the curation-event count; rotating-strategy
                 runs re-draw every gen, so their re-evals are already part of
                 the per-gen training term)
    """
    n_gen = len(records)
    n_pop = max((len(r.get("all_costs", [])) for r in records), default=0)
    validations = sum(1 for r in records if r.get("validation"))
    curation_gens = {r["pool_metrics"]["last_curation_gen"] for r in records if r.get("pool_metrics", {}).get("last_curation_gen") is not None}
    n_curations = len(curation_gens)
    out = {
        "n_gen": n_gen,
        "n_pop": n_pop,
        "training": n_gen * n_pop * training_n_sims,
        "validation": validations * validation_n_sims,
        "curation": n_curations * curation_top_k * curation_sample_size,
        "reeval": n_curations * n_pop * training_n_sims,
    }
    out["total"] = out["training"] + out["validation"] + out["curation"] + out["reeval"]
    return out
