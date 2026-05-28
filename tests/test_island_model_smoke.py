"""End-to-end smoke test for the 3-island PSO/GA/DE trainer.

Runs 5 generations with k_period=1 (forces migration every gen after gen 0)
on a reduced 16->8->2 dense architecture.  Verifies:
  - per-island JSONL records are produced (3 per gen, one per island name)
  - migration events fire (migration_log is non-empty)
  - a winner is selected and best_model.json is written
  - best_model.json loads and runs forward via the Rust runtime

Marked @slow because it runs ~3 * n_pop * training_n_sims * n_gen MC sims
(3 islands x 8 individuals x 2 seeds x 5 gens = 240 sims, plus a few
validation and final-eval passes).

Runs in the python-pyo3 CI job (bindings required).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

aerocapture_rs = pytest.importorskip("aerocapture_rs")


@pytest.mark.slow
def test_islands_smoke_5_gens(tmp_path: Path) -> None:
    """5-gen end-to-end: per-island JSONL, migration fires, winner loads via Rust."""
    from aerocapture.training.config import NetworkConfig, SimConfig, TrainingConfig
    from aerocapture.training.optimizer import IslandSettings, OptimizerConfig
    from aerocapture.training.train import train

    save_dir = tmp_path / "neural_network_islands_smoke"

    # input_size must match the input_mask length AND the TOML's [network] input_mask.
    # msr_aller_islands_train.toml has input_mask = [0..24] (25 inputs), so we use
    # 25 inputs here to avoid the Rust "input_mask length != layer_sizes[0]" error.
    architecture = [
        {"type": "dense", "input_size": 25, "output_size": 8, "activation": "swish"},
        {"type": "dense", "input_size": 8, "output_size": 2, "activation": "linear"},
    ]

    nn_cfg = NetworkConfig(
        architecture=architecture,
        input_mask=list(range(25)),
    )
    sim_cfg = SimConfig(
        executable="src/rust/target/release/aerocapture",
        nn_param_file=str(save_dir / "best_model.json"),
        toml_config="configs/training/msr_aller_islands_train.toml",
        n_sims=2,
    )
    optimizer = OptimizerConfig(
        algorithm="islands",
        n_pop=8,
        n_gen=5,
        seed_strategy="fixed",
        training_n_sims=2,
        validation_n_sims=4,
        seed_pool_interval=1000,  # never triggers periodic curation in 5 gens
        islands=IslandSettings(
            enabled=True,
            k_period=1,  # migrate every gen (after gen 0)
            k_top=2,
            pso_inject_velocity_scale=0.05,
        ),
    )
    cfg = TrainingConfig(
        network=nn_cfg,
        optimizer=optimizer,
        sim=sim_cfg,
        save_dir=str(save_dir),
        guidance_type="neural_network",
    )

    result = train(cfg, seed=42, cwd=".", verbose=False, no_tui=True, from_scratch=True)

    assert result is not None
    assert not result.get("interrupted", False), "training was interrupted"

    # Winner must be selected (all 3 islands should promote at least once in 5 gens).
    winner = result.get("winner")
    assert winner is not None, "No winner selected — no island promoted a validated best in 5 gens. Increase n_gen or validation_n_sims."
    assert winner["island"] in {"pso", "ga", "de"}

    # Migration must have fired: k_period=1 fires at gens 1, 2, 3, 4.
    migration_log = result.get("migration_log", [])
    assert len(migration_log) > 0, "migration_log is empty — migrate() never fired even with k_period=1"
    # Spot-check a migration event has the expected fields.
    ev = migration_log[0]
    assert hasattr(ev, "gen") and hasattr(ev, "src_island") and hasattr(ev, "dst_island")
    assert ev.src_island in {"pso", "ga", "de"}
    assert ev.dst_island in {"pso", "ga", "de"}
    assert ev.src_island != ev.dst_island  # full-mesh: never self-migration

    # Verify best_model.json was written.
    best_model = save_dir / "best_model.json"
    assert best_model.exists(), f"best_model.json missing under {save_dir}"
    raw = json.loads(best_model.read_text())
    assert raw["format_version"] == 2
    assert [e["type"] for e in raw["architecture"]] == ["dense", "dense"]

    # Verify per-island JSONL records (3 per gen).
    jsonl_files = list(save_dir.glob("run_*.jsonl"))
    assert len(jsonl_files) == 1, f"Expected 1 JSONL file, found {len(jsonl_files)}: {jsonl_files}"
    records = [json.loads(line) for line in jsonl_files[0].read_text().splitlines() if line.strip()]

    island_records = [r for r in records if "island_name" in r]
    island_names_seen = {r["island_name"] for r in island_records}
    assert island_names_seen == {"pso", "ga", "de"}, f"Expected JSONL records for all 3 islands, got: {island_names_seen}"

    # Each of gens 1..4 must have exactly 3 records (one per island).
    records_by_gen: dict[int, list[dict]] = {}
    for r in island_records:
        records_by_gen.setdefault(r["generation"], []).append(r)
    for g in range(1, 5):
        n = len(records_by_gen.get(g, []))
        assert n == 3, f"Expected 3 island records at gen {g}, got {n}"

    # Rust forward pass on the produced JSON.
    output = aerocapture_rs.nn_forward(str(best_model), [0.0] * 25)
    assert isinstance(output, (list, tuple))
    assert len(output) == 2
    assert all(isinstance(v, float) for v in output)
