"""RL report generator: Part 1 (RL convergence) + Parts 2/3 reused from the GA report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aerocapture.training import charts
from aerocapture.training import report as ga_report
from aerocapture.training.cell_eval import evaluate_cell
from aerocapture.training.report_render import render_pdf, staged_assets
from aerocapture.training.seeds import FINAL_EVAL_SEED_OFFSET
from aerocapture.training.toml_utils import load_toml_with_bases


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def generate_report(output_dir: Path, toml_path: Path) -> Path | None:
    """Generate a PDF report for an RL training run.

    Loads rl_training_*.jsonl, runs final MC evaluation on reserved seeds,
    generates Part 1 (RL convergence) + Parts 2/3 (shared with GA report),
    and compiles via Typst.

    Returns path to report.pdf, or None if Typst is unavailable.
    """
    jsonl_files = sorted(output_dir.glob("rl_training_*.jsonl"))
    if not jsonl_files:
        print(f"No rl_training_*.jsonl found in {output_dir}")
        return None

    jsonl_path = jsonl_files[-1]  # most recent
    records = _load_jsonl(jsonl_path)

    with staged_assets() as tmp_dir:
        # Part 1: RL convergence charts
        _chart_rl_return_curve(records, tmp_dir / "rl_return.svg")
        _chart_rl_dv_curve(records, tmp_dir / "rl_dv.svg")
        _chart_rl_entropy(records, tmp_dir / "rl_entropy.svg")
        _chart_rl_value_loss(records, tmp_dir / "rl_value_loss.svg")
        _chart_rl_capture_rate(records, tmp_dir / "rl_capture.svg")
        _chart_rl_validation_waterfall(records, tmp_dir / "rl_val.svg")

        # Final eval on reserved seeds
        has_trajectories = False
        has_final_eval = False
        n_sims = 1000
        sensitivity_flags: dict[str, bool] = {"has_sensitivity": False, "has_morris": False, "has_sobol": False, "has_sobol_heatmap": False}

        try:
            import aerocapture_rs  # type: ignore[import-not-found, import-untyped]  # noqa: F401

            results = evaluate_cell(None, toml_path, pool=(FINAL_EVAL_SEED_OFFSET, n_sims), model=output_dir / "best_model.json", include_trajectories=True)

            final_records = results.final_records
            trajectories = results.trajectories
            dispersions = results.dispersions
            assert trajectories is not None

            # Write Parquet
            try:
                from aerocapture.training.parquet_output import write_parquet

                resolved_config = load_toml_with_bases(toml_path)
                write_parquet(output_dir / "final_eval.parquet", final_records, dispersions, resolved_config, toml_path=str(toml_path))
            except ImportError:
                pass
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: Parquet write failed: {exc}")

            # Part 2: Mission performance charts
            cost_kwargs = ga_report.read_cost_kwargs(toml_path)
            ga_report.render_mission_performance_charts(
                final_records,
                trajectories,
                dispersions,
                tmp_dir,
                toml_path=toml_path,
                scheme_dir=output_dir,
                cost_kwargs=cost_kwargs,
            )
            has_trajectories = True
            has_final_eval = True

            # Part 3: Sensitivity (if available)
            sensitivity_flags = ga_report.maybe_render_sensitivity_charts(output_dir, tmp_dir)

        except ImportError:
            print("aerocapture_rs not available -- skipping final evaluation and Part 2/3")

        # Write metadata.json
        n_updates = len(records)
        resolved_cfg = load_toml_with_bases(toml_path) if toml_path is not None else {}
        rl_algo = resolved_cfg.get("rl", {}).get("algorithm", "ppo")
        metadata: dict[str, Any] = {
            "scheme": "neural_network_rl",
            "algorithm": rl_algo.upper(),
            "mission": "RL Training",
            "date": _today(),
            "n_updates": str(n_updates),
            "final_eval_n_sims": str(n_sims),
            "has_trajectories": has_trajectories,
            "has_final_eval": has_final_eval,
        }
        metadata.update(sensitivity_flags)
        (tmp_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        return render_pdf("report_rl", tmp_dir, output_dir / "report.pdf")


def _today() -> str:
    from datetime import UTC, datetime

    return datetime.now(tz=UTC).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Part 1: RL convergence chart functions
# ---------------------------------------------------------------------------


def _chart_rl_return_curve(records: list[dict[str, Any]], out: Path) -> None:
    steps = [r["env_steps"] for r in records]
    mean = [r.get("episodic_return_mean", float("nan")) for r in records]
    charts.save_line_chart(
        steps,
        mean,
        xlabel="env steps",
        ylabel="episodic return (mean)",
        title="RL: episodic return vs env steps",
        output_path=out,
    )


def _chart_rl_dv_curve(records: list[dict[str, Any]], out: Path) -> None:
    steps = [r["env_steps"] for r in records]
    dv = [r.get("episodic_dv_m_s_mean", float("nan")) for r in records]
    charts.save_line_chart(steps, dv, xlabel="env steps", ylabel="mean DV (m/s)", title="RL: DV vs env steps", output_path=out)


def _chart_rl_entropy(records: list[dict[str, Any]], out: Path) -> None:
    steps = [r["env_steps"] for r in records]
    ent = [r.get("entropy", float("nan")) for r in records]
    charts.save_line_chart(steps, ent, xlabel="env steps", ylabel="policy entropy", title="RL: entropy", output_path=out)


def _chart_rl_value_loss(records: list[dict[str, Any]], out: Path) -> None:
    steps = [r["env_steps"] for r in records]
    vl = [r.get("value_loss", float("nan")) for r in records]
    charts.save_line_chart(steps, vl, xlabel="env steps", ylabel="value loss", title="RL: value loss", output_path=out)


def _chart_rl_capture_rate(records: list[dict[str, Any]], out: Path) -> None:
    steps = [r["env_steps"] for r in records]
    cr = [r.get("episodic_capture_rate", float("nan")) for r in records]
    charts.save_line_chart(steps, cr, xlabel="env steps", ylabel="capture rate", title="RL: capture rate", output_path=out)


def _chart_rl_validation_waterfall(records: list[dict[str, Any]], out: Path) -> None:
    attempts = [r for r in records if r.get("val_attempted")]
    if not attempts:
        # Emit empty SVG so Typst include does not fail.
        out.write_text("<svg xmlns='http://www.w3.org/2000/svg' width='400' height='200'/>")
        return
    steps = [r["env_steps"] for r in attempts]
    val = [r.get("val_rms_cost", float("nan")) for r in attempts]
    charts.save_line_chart(steps, val, xlabel="env steps", ylabel="validation RMS cost", title="RL: validation", output_path=out)
