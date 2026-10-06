"""Collect the quotable artifacts of every completed paper run into the committed bundle.

Walks the campaign output trees and, for each run dir holding a final_eval.parquet,
copies {best_model.json, best_params.json, final_eval.parquet, final_selection.json}
and gzips the run_*.jsonl fragments (one per launch of a resumed run) as one run.jsonl.gz into
articles/paper/data/runs/<study>/<cell>/.
Idempotent: a destination file is rewritten only when the source is newer.
"""

import argparse
import gzip
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "articles/paper/data/runs"
TRAINING = REPO / "training_output"

ARTIFACTS = ("best_model.json", "best_params.json", "final_eval.parquet", "final_selection.json", "fresh_pool_requote.json", "ablation_results.json")
CLASSICAL = ("piecewise_constant", "ftc", "equilibrium_glide", "energy_controller", "pred_guid", "fnpag")
# Preserved PRE-FIX legacy dirs the paper footnote-quotes (RL, warm-start/joint,
# QAT/pruning). Bundled under legacy/ so those table rows also reproduce from
# the committed bundle, not just from a local checkout.
LEGACY = (
    "neural_network_rl",
    "neural_network_gru_ppo",
    "paper_opt_warmstart",
    "best_neural_network_joint",
    "neural_network_joint",
    "neural_gru_joint",
    "neural_network_atan2",
    "neural_network_atan2_qat4",
    "neural_network_atan2_qat8",
    "neural_network_pruned",
    "neural_network_pruned_dv",
    "neural_network_pruned_dv2",
    "neural_network_pruned_dv3",
    "neural_network_scaledpi_pso",
    "neural_network_scaledpi_pso_pruned",
    "neural_network_scaledpi_pso_pruned_dv",
    "neural_network_scaledpi_pso_pruned_dv3",
    "neural_network_delta_pso",
    "neural_network_delta_pso_pruned",
    "neural_network_delta_pso_pruned_dv3",
)

# Per-scenario (per_draw, ADR-0006) cells under training_output/ou_marginal/: every run dir
# holding a final_eval.parquet (report.py on the 2M pool) is bundled as ou_marginal/<path>: the
# #173 headline-allocation cells (hl_*, the dense fine-tune seeds), the fine-tuned champions the
# Section 5 RL baseline is protocol-matched to (issue #101; PPO cells train under
# experiments/paper/18_rl_baseline.sh into training_output/paper/rl/ and are bundled as
# rl/<cell> by the paper/ walk above), the classical retunes (#172, classical/<cell> and the
# #188 classical_ungated/<cell>). The 60 x 10 scratch repeats have no parquet and stay out;
# the #192 ceiling legs (hs_*) are a sensitivity study, quoted by heat_load_slope.py. The bundled
# cells' run logs travel like every other study's (issue #179) and join the v4 Release asset: the 512 x 2
# logs are ~240 MB each raw, so run strip_run_logs.py on the new files before `make sums`.
OU_MARGINAL_SKIP = ("hs_",)
# Off-campaign studies under training_output/paper/ (scripts 13-16) whose cells were never
# bundled: their quoted numbers live in the FROZEN data files (sigma_extras.json, ...), not in
# results.json. Walking them into runs/ made every local collect diverge from the committed
# bundle (check_results_schema: 27 "only-in-bundle" keys) until the copies were deleted by hand.
OFF_CAMPAIGN = ("objective_centering", "robustness_retrain", "sigma_extras", "state_controls")


# Manual headline / parameter-efficiency runs (NOT campaign cells; trained by
# hand at the deployment allocation n_sims=2/20000 gens). dense_p515 is the
# deployed headline; dense_p972 is the GA-dimensionality data point.
HEADLINE = {
    "mamba_p962_long": "headline/mamba_p962",  # THE sizing headline (10c verdict)
    "dense_p515_ga_paper_best": "headline/dense_p515",  # efficiency reference
    "dense_p972_ga_paper_best": "headline/dense_p972",  # GA-dimensionality point
    "lstm_p1082_long": "headline/lstm_p1082",  # co-leader (close 2nd on the tail)
    "gru_p1014_long": "headline/gru_p1014",  # 3rd recurrent (confirms the pattern)
}


def _run_dirs() -> list[tuple[Path, Path]]:
    """(source run dir, bundle destination) pairs for every completed run."""
    pairs: list[tuple[Path, Path]] = []
    paper = TRAINING / "paper"
    if paper.is_dir():
        for parquet in sorted(paper.rglob("final_eval.parquet")):
            src = parquet.parent
            if src.relative_to(paper).parts[0] in OFF_CAMPAIGN:
                continue
            pairs.append((src, OUT / src.relative_to(paper)))
    for name, dest in HEADLINE.items():
        src = TRAINING / name
        if (src / "final_eval.parquet").exists():
            pairs.append((src, OUT / dest))
    for scheme in CLASSICAL:
        src = TRAINING / scheme
        if (src / "final_eval.parquet").exists():
            pairs.append((src, OUT / "classical_baselines" / scheme))
    for src in sorted(TRAINING.glob("sweep_*")):
        if (src / "final_eval.parquet").exists():
            pairs.append((src, OUT / "architecture_sweep" / src.name))
    for name in LEGACY:
        src = TRAINING / name
        if (src / "final_eval.parquet").exists():
            pairs.append((src, OUT / "legacy" / name))
    ou_marginal = TRAINING / "ou_marginal"
    if ou_marginal.is_dir():
        for parquet in sorted(ou_marginal.rglob("final_eval.parquet")):
            src = parquet.parent
            if src.relative_to(ou_marginal).parts[0].startswith(OU_MARGINAL_SKIP):
                continue
            pairs.append((src, OUT / "ou_marginal" / src.relative_to(ou_marginal)))
    return pairs


def _check_stale_parquet(src: Path) -> str | None:
    """A best_model.json (a classical cell's optimized_<scheme>.toml) newer than
    final_eval.parquet means the dir was re-selected (e.g. retro final_select)
    without re-running report.py -- the parquet quotes the PREVIOUS winner.
    Bundling it would commit inconsistent paper numbers."""
    parquet = src / "final_eval.parquet"
    for model in (src / "best_model.json", *src.glob("optimized_*.toml")):
        if model.exists() and model.stat().st_mtime > parquet.stat().st_mtime + 1:
            return f"STALE: {model.name} newer than final_eval.parquet in {src} -- re-run report.py on this dir before collecting"
    return None


def _copy_if_newer(src: Path, dst: Path) -> bool:
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _gzip_run_log(run_dir: Path, dst: Path) -> bool:
    """Gzip the run's whole log: a resumed run writes one run_*.jsonl per launch, and each
    fragment supersedes the generations it re-logs (the resume generation is in both)."""
    logs = sorted(run_dir.glob("run_*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not logs:
        return False
    if dst.exists() and dst.stat().st_mtime >= logs[-1].stat().st_mtime:
        return False
    lines: list[tuple[int, bytes]] = []
    for log in logs:
        new = [(json.loads(line)["generation"], line) for line in log.read_bytes().splitlines(keepends=True) if line.strip()]
        if new:
            lines = [(g, line) for g, line in lines if g < new[0][0]] + new
    dst.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(dst, "wb", compresslevel=9) as f_out:
        f_out.writelines(line for _, line in lines)
    shutil.copystat(logs[-1], dst)
    return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="list what would be collected without writing")
    args = parser.parse_args(argv)

    pairs = _run_dirs()
    if not pairs:
        sys.exit(f"No completed runs found under {TRAINING}")
    n_new, stale = 0, []
    for src, dst in pairs:
        warning = _check_stale_parquet(src)
        if warning:
            stale.append(warning)
            print(f"  {src.relative_to(TRAINING)}  [SKIPPED -- stale parquet]")
            continue
        copied: list[str] = []
        if not args.dry_run:
            copied += [a for a in ARTIFACTS if (src / a).exists() and _copy_if_newer(src / a, dst / a)]
            # Skip the ~19 MB/cell training log for the architecture-sweep Pareto cells:
            # fig_pareto reads final_eval.parquet (+ the manifest param counts), NOT
            # convergence curves -- only the headline cells need their logs (the plateau
            # figure). This keeps the 24-cell sweep at ~16 MB instead of ~420 MB.
            # rl: the RL cells' logs are the trainer's flat `rl_training_*.jsonl`, which
            # aggregate_results.actual_sims and _best_val_rms cannot read (they expect the
            # population `validation` records).
            if dst.relative_to(OUT).parts[0] not in ("architecture_sweep", "rl") and _gzip_run_log(src, dst / "run.jsonl.gz"):
                copied.append("run.jsonl.gz")
        status = "would collect" if args.dry_run else (f"updated {', '.join(copied)}" if copied else "up to date")
        print(f"  {src.relative_to(TRAINING)} -> {dst.relative_to(REPO)}  [{status}]")
        n_new += bool(copied)
    for w in stale:
        print(f"\nWARNING: {w}")
    print(f"\n{len(pairs) - len(stale)} runs in bundle, {n_new} updated. Remember to `git add articles/paper/data/runs`.")


if __name__ == "__main__":
    main()
