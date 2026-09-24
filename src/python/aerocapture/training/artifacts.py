"""Deployed artifact writers: best_model.json / best_params.json, the optimized
TOML (+ joint-reference table), and the warm-start sidecars.

Shared by the checkpoint writer, both trainer adapters, the train CLI and the
`final_select` CLI. A leaf: no trainer or train import.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from aerocapture.training.config import TrainingConfig
from aerocapture.training.encoding import _decode_nn_weights, decode_normalized
from aerocapture.training.evaluate import write_nn_json
from aerocapture.training.param_spaces import ParamSpec
from aerocapture.training.training_config import _resolve_config_normalization

if TYPE_CHECKING:
    from aerocapture.training.problem import PerSeedEvaluator


def _emit_warm_start_artifacts(
    config: TrainingConfig,
    base_mc_seed: int,
    problem: PerSeedEvaluator,
    val_seeds: list[int] | None,
    warm_chromo: npt.NDArray[np.float64],
    warm_weight_specs: list[ParamSpec],
    verbose: bool,
) -> None:
    """Best-effort warm-start sidecars: gen-0 validation baseline, supervisor-vs-NN trajectory comparison, warm-start report PDF.

    Each block is independently best-effort (never blocks training).
    """
    # Gen-0 validation baseline: evaluate the bare warm-started
    # chromosome on the RESERVED VALIDATION seed pool (same seeds
    # the validation gate uses) so the persisted rms/mean/p95
    # metrics are directly comparable to the `Gen N validation:`
    # line later printed by the validation gate. Best-effort:
    # failure here must not block training.
    from aerocapture.training._warm_start_baseline import write_gen0_baseline
    from aerocapture.training.report import compute_eval_summary, format_eval_summary

    try:
        if val_seeds is None:
            raise RuntimeError("val_seeds not initialized; gen-0 baseline requires the validation pool")
        # Single MC pass on val_seeds returning both per-seed costs
        # AND the (n, 52) final_records so we can derive DV / apo /
        # peri / heat-flux statistics without re-running.
        baseline_costs, baseline_records = problem.evaluate_individual_records_per_seed(warm_chromo, val_seeds)
        eval_summary = compute_eval_summary(baseline_records, len(val_seeds), problem.cost_kwargs)
        baseline_path = write_gen0_baseline(
            save_dir=Path(config.save_dir),
            costs=baseline_costs,
            capture_rate=eval_summary["capture_rate"],
            n_sims=len(val_seeds),
        )
        # Persist the structured eval summary so the warm-start
        # report PDF can embed it (and CLI re-render works).
        (Path(config.save_dir) / "warm_start_eval_summary.json").write_text(json.dumps(eval_summary, indent=2))
        # User-facing block: mirrors the end-of-training final-eval
        # summary so users can compare like-for-like.
        if verbose:
            print()
            for line in format_eval_summary(eval_summary, indent="    "):
                print(f"  {line}" if not line.startswith(" ") else line)
            baseline = json.loads(baseline_path.read_text())
            print(f"  [warm_start] gen-0 baseline cost (val seeds): rms={baseline['rms_cost']:.4e} mean={baseline['mean_cost']:.4e}")
    except Exception as e:
        # Best-effort: failure here must not block training, but the
        # error is always logged so it does not silently mask real
        # bugs in problem.evaluate_individual_records_per_seed /
        # write_gen0_baseline / compute_eval_summary.
        print(f"  [warm_start] WARNING: gen-0 baseline write failed: {type(e).__name__}: {e}")

    # Trajectory comparison: supervisor vs warm-started NN on both
    # training and validation pools. Runs ~2*(n_warm_seeds +
    # validation_n_sims) MC sims (e.g. ~12k for n_warm_seeds=5000,
    # validation_n_sims=1000) and renders 20 SVGs into the report
    # dir. Best-effort: failure here must not block training, but
    # the warm-start PDF will just omit the comparison section.
    try:
        from aerocapture.training.warm_start_compare import render_trajectory_comparison

        render_trajectory_comparison(
            cfg=config,
            base_mc_seed=base_mc_seed,
            warm_chromo=warm_chromo,
            nn_weight_specs=warm_weight_specs,
        )
    except Exception as e:
        print(f"  [warm_start] WARNING: trajectory comparison failed: {type(e).__name__}: {e}")

    # Intermediate warm-start report: charts + Typst PDF summarizing
    # supervised MSE convergence, supervisor selection, search-space
    # bounds, the gen-0 validation baseline + eval summary, and the
    # supervisor-vs-NN trajectory comparison panels (if rendered).
    # Best-effort.
    try:
        from aerocapture.training.warm_start_report import render_report

        render_report(Path(config.save_dir))
    except Exception as e:
        print(f"  [warm_start] WARNING: report rendering failed: {type(e).__name__}: {e}")


def write_best_artifacts(
    best_individual: npt.NDArray[np.float64],
    config: TrainingConfig,
    param_specs: list[ParamSpec],
    save_dir: Path,
    cwd: str | Path | None = None,
    deploy_to_cwd: bool = False,
) -> None:
    """Write best_model.json (NN) / best_params.json from a normalized chromosome.

    Always writes into save_dir. When `deploy_to_cwd` and `cwd` is not None,
    additionally writes the NN model to `cwd / config.sim.nn_param_file`
    (the deploy-path copy save_checkpoint historically maintained).
    """
    if config.guidance_type == "neural_network":
        from aerocapture.training.param_spaces import active_scaffolding_specs

        _pack = active_scaffolding_specs(config.network.scaffolding)
        n_scaff = len(_pack)
        n_weights = len(param_specs) - n_scaff
        weights = _decode_nn_weights(
            best_individual[:n_weights],
            param_specs[:n_weights],
        )
        cfg_norm = _resolve_config_normalization(config, cwd)
        write_nn_json(
            weights,
            config.network,
            save_dir / "best_model.json",
            input_mask=config.network.input_mask,
            output_param=config.network.output_parameterization,
            normalization=cfg_norm,
        )
        if deploy_to_cwd and cwd is not None:
            nn_path = Path(cwd) / config.sim.nn_param_file
            write_nn_json(
                weights,
                config.network,
                nn_path,
                input_mask=config.network.input_mask,
                output_param=config.network.output_parameterization,
                normalization=cfg_norm,
            )
        if n_scaff > 0:
            scaff_params = decode_normalized(
                best_individual[n_weights:],
                list(_pack),
            )
            for s in _pack:
                if s.is_integer and s.name in scaff_params:
                    scaff_params[s.name] = int(round(scaff_params[s.name]))
            with open(save_dir / "best_params.json", "w") as fp:
                json.dump(scaff_params, fp, indent=2)
    else:
        params = decode_normalized(best_individual, param_specs)
        with open(save_dir / "best_params.json", "w") as fp:
            json.dump(params, fp, indent=2)


def deploy_optimized_artifacts(
    params: dict[str, float],
    config: TrainingConfig,
    toml_data: dict,
    save_dir: Path,
    base_toml: Path,
    verbose: bool = True,
) -> None:
    """Write optimized_<scheme>.toml for a non-NN winner and, for joint-ref runs,
    regenerate the winner's reference table and wire the TOML at it.

    Shared by main()'s end-of-training deploy and the final_select CLI — the CLI
    must redeploy both artifacts, else a re-selected ref_bank evaluates against
    the previous winner's table (gains and reference co-adapt strongly).
    """
    from aerocapture.training.evaluate import write_guidance_toml  # noqa: PLC0415

    # ref_bank is not a guidance TOML key — it deploys as the regenerated
    # reference table below (Rust rejects it as an unknown [guidance.<scheme>] key).
    opt_toml = save_dir / f"optimized_{config.guidance_type}.toml"
    write_guidance_toml(base_toml, config.guidance_type, {k: v for k, v in params.items() if k != "ref_bank"}, opt_toml)
    if verbose:
        print(f"  Optimized TOML: {opt_toml}")

    if "ref_bank" in params:
        import tomllib  # noqa: PLC0415

        from aerocapture.training import reference as _reference  # noqa: PLC0415
        from aerocapture.training.toml_utils import write_toml  # noqa: PLC0415

        [tmp_tbl] = _reference.generate_constant_bank_tables(
            str(base_toml), [params["ref_bank"]], toml_data.get("monte_carlo", {}), save_dir, config.sim.sim_timeout_secs
        )
        if tmp_tbl.stat().st_size == 0:
            raise RuntimeError(
                f"joint-reference deploy: the winner's constant-bank nominal (ref_bank={params['ref_bank']:.2f} deg) "
                f"produced no trajectory — refusing to deploy an empty reference table "
                f"(the Rust loader silently yields 0 points and interpolates 0.0)"
            )
        deploy_ref = save_dir / "ref_trajectory.dat"
        tmp_tbl.replace(deploy_ref)
        with open(opt_toml, "rb") as _f:
            _opt = tomllib.load(_f)
        _opt.setdefault("data", {})["reference_trajectory"] = str(deploy_ref)
        write_toml(_opt, opt_toml)
        if verbose:
            print(f"  Joint reference (ref_bank {params['ref_bank']:.2f} deg) deployed to {deploy_ref}")
