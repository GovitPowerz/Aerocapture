"""Run-log accounting for the paper (`paper_stats.actual_sims`)."""

from aerocapture.training.paper_stats import actual_sims


def test_actual_sims_formula() -> None:
    # 3 gens, n_pop=2, n_sims=10; validation fired twice; one curation event;
    # seeds changed once after the curation -> one parent re-eval.
    records = [
        {"all_costs": [1, 2], "pool_metrics": {"last_curation_gen": 0}},
        {"all_costs": [1, 2], "validation": {"rms_cost": 5.0}, "pool_metrics": {"last_curation_gen": 1}},
        {"all_costs": [1, 2], "validation": {"rms_cost": 4.0}, "pool_metrics": {"last_curation_gen": 1}},
    ]
    s = actual_sims(records, training_n_sims=10, validation_n_sims=1000, curation_sample_size=1000, curation_top_k=1)
    assert s["training"] == 3 * 2 * 10
    assert s["validation"] == 2 * 1000
    assert s["curation"] == 2 * 1 * 1000  # two distinct last_curation_gen values
    assert s["reeval"] == 2 * 2 * 10  # parent re-eval on each curation-change gen
    assert s["total"] == s["training"] + s["validation"] + s["curation"] + s["reeval"]
