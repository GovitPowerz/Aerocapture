"""Guidance parameter training: the CLI and the resolve -> build -> run entry point.

`train()` resolves the config, builds the pymoo problem, lets the chosen adapter
(`trainer.SingleAlgoTrainer` / `trainer.IslandsTrainer`) build itself from
`(config, problem, save_dir)`, and runs the per-generation loop contract
(`trainer.run_loop`). Everything reusable lives in leaf modules
(`training_config`, `checkpoint`, `artifacts`, `initial_population`,
`corridor`, `optimizer`, `seeds`); nothing imports this module as a library.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from aerocapture.training.artifacts import deploy_optimized_artifacts
from aerocapture.training.config import TrainingConfig
from aerocapture.training.corridor import CorridorAccumulator
from aerocapture.training.cost import build_cost_kwargs
from aerocapture.training.display import create_display
from aerocapture.training.encoding import _decode_nn_weights, decode_normalized
from aerocapture.training.evaluate import write_nn_json
from aerocapture.training.logger import TrainingLogger
from aerocapture.training.optimizer import _VALID_SEED_STRATEGIES
from aerocapture.training.problem import AerocaptureProblem
from aerocapture.training.reference import nominal_flight_overrides, piecewise_commanded_cos_bank, ref_trajectory_array
from aerocapture.training.seeds import base_mc_seed_from_toml
from aerocapture.training.toml_utils import load_toml_with_bases
from aerocapture.training.trainer import IslandsTrainer, SingleAlgoTrainer, run_loop
from aerocapture.training.training_config import (
    _resolve_config_normalization,
    _setup_param_specs,
    build_training_config_from_toml,
    check_ref_trajectory_wiring,
)


def train(
    config: TrainingConfig | None = None,
    seed: int | None = None,
    cwd: str | Path | None = None,
    verbose: bool = True,
    checkpoint_interval: int = 10,
    resume_dir: str | Path | None = None,
    no_tui: bool = False,
    corridor_acc: CorridorAccumulator | None = None,
    from_scratch: bool = False,
) -> dict:
    """Run the full optimization training pipeline: resolve -> build -> run.

    Args:
        config: Training configuration. Uses defaults if None.
        seed: Random seed for reproducibility.
        cwd: Working directory for simulations.
        verbose: Print progress.
        checkpoint_interval: Save checkpoint every N generations.
        resume_dir: Directory to resume training from (loads latest checkpoint;
            the islands path probes `save_dir` itself and ignores this).
        no_tui: Disable Rich TUI (use plain-text output).
        corridor_acc: Optional CorridorAccumulator for piecewise_constant training.
        from_scratch: Ignore existing checkpoints and start fresh.

    Returns:
        Dictionary with training results:
            - 'best_cost': Best cost found
            - 'best_individual': Best individual (normalized [0,1] vector)
            - 'cost_history': Cost per generation
            - 'corridor_acc': CorridorAccumulator (if piecewise_constant)
    """
    if config is None:
        config = TrainingConfig()

    if config.optimizer.seed_strategy not in _VALID_SEED_STRATEGIES:
        msg = (
            f"config.optimizer.seed_strategy must be one of {_VALID_SEED_STRATEGIES}, "
            f"got {config.optimizer.seed_strategy!r}. Did the TOML [optimizer] section "
            f"set it, or did you pass a TrainingConfig without overriding the default?"
        )
        raise ValueError(msg)

    rng = np.random.default_rng(seed)

    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load TOML config once (used for cost function params, curator config)
    _toml: dict = {}
    cost_kwargs: dict[str, Any] = {}
    toml_abs_path = ""
    if config.sim.toml_config:
        toml_path = Path(cwd or config.sim.exec_dir) / config.sim.toml_config
        _toml = load_toml_with_bases(toml_path)
        cost_kwargs = build_cost_kwargs(_toml)
        toml_abs_path = str(toml_path.resolve())

    param_specs, _n_params = _setup_param_specs(config, _toml, verbose)

    # Compute config hash for experiment grouping
    config_hash = hashlib.sha256(repr(config).encode()).hexdigest()[:12]

    problem = AerocaptureProblem(
        param_specs=param_specs,
        toml_path=toml_abs_path,
        seeds=[base_mc_seed_from_toml(_toml)],
        cost_kwargs=cost_kwargs,
        scheme=config.guidance_type,
        sim_timeout=config.sim.sim_timeout_secs,
        nn_config=config.network if config.guidance_type == "neural_network" else None,
    )

    # The trainer seam: one loop contract, two adapters (see trainer.py). Each
    # adapter owns its setup (resume, initial population, pymoo seeding).
    trainer_cls: type[SingleAlgoTrainer] | type[IslandsTrainer] = IslandsTrainer if config.optimizer.algorithm == "islands" else SingleAlgoTrainer
    trainer = trainer_cls.from_config(
        config,
        problem,
        save_dir,
        toml=_toml,
        cwd=cwd,
        rng=rng,
        resume_dir=resume_dir,
        from_scratch=from_scratch,
        corridor_acc=corridor_acc,
        verbose=verbose,
        checkpoint_interval=checkpoint_interval,
    )

    # Built after the adapter so a resume bump of `n_gen` reaches the TUI's total.
    display = create_display(
        scheme=config.guidance_type,
        n_runs=1,
        n_generations=config.optimizer.n_gen,
        enabled=not no_tui and verbose,
        algorithm=config.optimizer.algorithm,
        seed_strategy=config.optimizer.seed_strategy,
        training_n_sims=config.optimizer.training_n_sims,
    )
    display.set_start_gen(trainer.start_gen)

    logger = TrainingLogger(
        scheme=config.guidance_type,
        run=0,
        output_dir=save_dir,
        config_hash=config_hash,
        cost_transform=str(problem.cost_kwargs.get("cost_transform", "linear")),
    )

    return run_loop(trainer, config=config, problem=problem, rng=rng, logger=logger, display=display)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train guidance parameters via pymoo optimization")
    parser.add_argument("toml", type=str, help="TOML training config path (must contain [guidance] type)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-gen", type=int, default=None, help="Number of generations (additional when resuming; default: from TOML [optimizer])")
    parser.add_argument("--n-pop", type=int, default=None, help="Population size (default: from TOML [optimizer])")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint directory to resume from (auto-detected if omitted and checkpoint exists)")
    parser.add_argument("-fs", "--from-scratch", action="store_true", help="Wipe existing training output and start fresh (deletes checkpoints, logs, reports)")
    parser.add_argument("--no-tui", action="store_true", help="Disable Rich TUI (use plain-text output)")
    parser.add_argument("--skip-report", "--skip-final-report", action="store_true", dest="skip_report", help="Skip PDF report generation at end of training")
    parser.add_argument("--final-n-sims", type=int, default=1000, help="Number of MC sims for final re-evaluation (default: 1000)")
    parser.add_argument("--sim-timeout", type=float, default=None, help="Wall-clock timeout per simulation in seconds (default: no limit)")
    parser.add_argument("--algorithm", type=str, default=None, help="Optimization algorithm: ga, cma_es, de, pso, qpso (default: from TOML [optimizer])")
    parser.add_argument("--output-dir", type=str, default=None, help="Override the training output directory (default: derived from the scheme)")
    parser.add_argument("--seed-strategy", type=str, default=None, choices=["fixed", "rotating", "adaptive"], help="Override [optimizer] seed_strategy")
    parser.add_argument("--training-n-sims", type=int, default=None, help="Override [optimizer] training_n_sims (sims per individual per generation)")
    args = parser.parse_args()

    cfg, _toml_data = build_training_config_from_toml(args.toml)

    # CLI overrides -- only when explicitly provided (not None / default False)
    if args.n_gen is not None:
        cfg.optimizer.n_gen = args.n_gen
    if args.n_pop is not None:
        cfg.optimizer.n_pop = args.n_pop
    if args.algorithm is not None:
        cfg.optimizer.algorithm = args.algorithm
    if args.seed_strategy is not None:
        cfg.optimizer.seed_strategy = args.seed_strategy
    if args.training_n_sims is not None:
        if args.training_n_sims < 1:
            raise SystemExit(f"--training-n-sims must be >= 1, got {args.training_n_sims}")
        cfg.optimizer.training_n_sims = args.training_n_sims
    cfg.sim.sim_timeout_secs = args.sim_timeout
    if cfg.network.architecture is not None:
        cfg.network.__post_init__()  # re-validate once all fields are set
    cfg.sim.final_file = "output/final.train_nn_temp"
    cfg.sim.exec_dir = "."
    cwd = "."

    # Save dir per (variant × algorithm). For NN schemes the scheme name is
    # encoded in `[data] neural_network` (e.g. "training_output/neural_network_gru_pso/best_model.json"),
    # so we derive save_dir from its parent -- that's the single source of truth
    # and it lines up exactly with what compare_guidance / deploy paths expect.
    # For non-NN schemes (ftc, eqglide, piecewise_constant, etc.), guidance_type
    # already IS the scheme name, so training_output/{guidance_type} is correct.
    if cfg.guidance_type == "neural_network":
        nn_parent = Path(cfg.sim.nn_param_file).parent
        if not str(nn_parent).startswith("training_output/"):
            print(
                f"ERROR: [data] neural_network = '{cfg.sim.nn_param_file}' must live "
                f"under 'training_output/' so checkpoints and report artifacts land alongside the "
                f"deploy JSON. Fix the TOML to point at e.g. 'training_output/neural_network_<variant>/best_model.json'."
            )
            raise SystemExit(1)
        cfg.save_dir = str(nn_parent)
    else:
        cfg.save_dir = f"training_output/{cfg.guidance_type}"

    if args.output_dir:
        cfg.save_dir = args.output_dir

    if args.resume:
        cfg.save_dir = args.resume

    # Derive mission name from the first missions/ base reachable through the
    # base chain (recursive — nested leaf configs inherit it indirectly).
    from aerocapture.training.toml_utils import find_mission_name

    base_toml_path = Path(cwd) / args.toml
    mission_name = find_mission_name(base_toml_path) or Path(args.toml).stem
    # Mission artifacts (corridor, reference trajectory) live at the canonical
    # training_output/<mission>/ regardless of where --output-dir / --resume
    # relocate save_dir — deriving this from Path(save_dir).parent broke the
    # ref-trajectory check for any output dir outside training_output/.
    corr_dir = Path(cwd) / "training_output" / mission_name

    if args.from_scratch:
        if args.resume:
            print("ERROR: --from-scratch and --resume are mutually exclusive")
            raise SystemExit(1)
        save_path = Path(cfg.save_dir)
        if save_path.exists():
            import shutil

            shutil.rmtree(save_path)
            print(f"Wiped existing output: {save_path}")

        # For piecewise_constant, also wipe corridor/ref trajectory in the mission directory
        if cfg.guidance_type == "piecewise_constant":
            for stale in ("corridor_boundaries.npz", "ref_trajectory.dat"):
                stale_path = corr_dir / stale
                if stale_path.exists():
                    stale_path.unlink()
                    print(f"  Removed stale {stale_path}")

    # Auto-resume: if no --resume and no -fs, check for existing checkpoint.
    # Single-algorithm runs write paired checkpoint_g*.{json,npz}; the
    # islands path writes .npz-only checkpoints. Glob both so islands runs
    # auto-resume just like single-algo runs do.
    resume_dir = args.resume
    if resume_dir is None and not args.from_scratch:
        save_path = Path(cfg.save_dir)
        if list(save_path.glob("checkpoint_*.json")) or list(save_path.glob("checkpoint_g*.npz")):
            resume_dir = cfg.save_dir

    corr_dir.mkdir(parents=True, exist_ok=True)

    # Check for reference trajectory requirement
    from aerocapture.training.param_spaces import REQUIRES_REF_TRAJECTORY

    if cfg.guidance_type in REQUIRES_REF_TRAJECTORY:
        ref_traj_path = corr_dir / "ref_trajectory.dat"
        if not ref_traj_path.exists():
            print(f"\nERROR: No reference trajectory found for mission '{mission_name}'.")
            print("Run piecewise_constant training first:")
            print("  uv run python -m aerocapture.training.train configs/training/msr_aller_piecewise_constant_train.toml")
            sys.exit(1)
        check_ref_trajectory_wiring(_toml_data, ref_traj_path)
        print(f"  Using reference trajectory: {ref_traj_path}")

    # Architecture summary (NN schemes only).
    if cfg.guidance_type == "neural_network":
        from aerocapture.training.config import describe_architecture

        print(describe_architecture(cfg.network))

    # Initialize corridor accumulator for piecewise_constant training
    corridor_acc_init: CorridorAccumulator | None = None
    if cfg.guidance_type == "piecewise_constant":
        _pc_toml = _toml_data
        pc_section = _pc_toml.get("guidance", {}).get("piecewise_constant", {})
        energy_min = float(pc_section.get("energy_min", -6.0))
        energy_max = float(pc_section.get("energy_max", 5.0))
        corr_section = _pc_toml.get("corridor", {})
        delta_za_r = float(corr_section.get("delta_za_restricted", 200.0))
        delta_za_low = float(corr_section.get("delta_za_restricted_low", -delta_za_r))
        delta_za_high = float(corr_section.get("delta_za_restricted_high", delta_za_r))
        corridor_acc_init = CorridorAccumulator(energy_min, energy_max, delta_za_restricted=delta_za_r, delta_za_low=delta_za_low, delta_za_high=delta_za_high)

    result = train(cfg, seed=args.seed, cwd=cwd, resume_dir=resume_dir, no_tui=args.no_tui, corridor_acc=corridor_acc_init, from_scratch=args.from_scratch)
    print(f"\nFinal best training cost (RMS over {cfg.optimizer.training_n_sims} seeds): {result['best_cost']:.4e}")

    param_specs = result["param_specs"]

    # Update corridor_acc from train() result (may have been restored from checkpoint)
    corridor_acc_final = result.get("corridor_acc")

    # Save corridor data and reference trajectory for piecewise_constant
    if cfg.guidance_type == "piecewise_constant" and corridor_acc_final is not None and result["best_individual"] is not None:
        import aerocapture_rs as _aero_pc  # type: ignore[import-not-found, import-untyped]

        from aerocapture.training.corridor import save_corridor as _save_corr
        from aerocapture.training.param_spaces import GUIDANCE_TOML_SECTIONS as _GTS

        best_params = decode_normalized(result["best_individual"], param_specs)
        _pc_section = _GTS[cfg.guidance_type]
        best_ovr = nominal_flight_overrides(best_params, _pc_section, _toml_data.get("monte_carlo", {}))

        assert cfg.sim.toml_config is not None
        _pc_toml_path = str((Path(cwd) / cfg.sim.toml_config).resolve())
        best_batch = _aero_pc.run_batch(
            toml_path=_pc_toml_path,
            overrides_list=[best_ovr],
            include_trajectories=True,
            sim_timeout_secs=cfg.sim.sim_timeout_secs,
        )
        nom_traj = np.asarray(best_batch.trajectories[0]) if best_batch.trajectories else np.empty((0, 17))
        nom_dv_total = float(best_batch.final_records[0, 41]) if best_batch.final_records.shape[0] > 0 else 0.0

        # `reference_only = true` ([guidance.piecewise_constant]) marks a run
        # whose sole product is ref_trajectory.dat (e.g. the 1-segment
        # constant-bank reference generator) — it must not clobber the richer
        # corridor_boundaries.npz of the full piecewise baseline run.
        pc_cfg = _toml_data.get("guidance", {}).get("piecewise_constant", {})
        reference_only = bool(pc_cfg.get("reference_only", False))
        if not reference_only:
            # Save corridor_boundaries.npz from accumulated envelopes
            corr_data = corridor_acc_final.to_corridor_data(nominal=nom_traj)
            corr_data["nominal_dv"] = np.array([nom_dv_total])
            corr_npz = corr_dir / "corridor_boundaries.npz"
            _save_corr(corr_data, corr_npz)

        # Generate ref_trajectory.dat (7-column format). cos_bank carries the
        # COMMANDED segment profile (clean steps), not the realized bank whose
        # shaper sweeps through 0 deg whipsaw the trackers' feedforward.
        if nom_traj.ndim == 2 and nom_traj.shape[0] > 0:
            bank_angles = [v for _, v in sorted((int(k.split("_")[-1]), v) for k, v in best_params.items() if k.startswith("bank_angle_"))]
            commanded_cos = piecewise_commanded_cos_bank(
                nom_traj[:, 8],
                bank_angles,
                energy_min_mj=float(pc_cfg.get("energy_min", -6.0)),
                energy_max_mj=float(pc_cfg.get("energy_max", 5.0)),
            )
            ref_data = ref_trajectory_array(nom_traj, cos_bank=commanded_cos)
            ref_path = corr_dir / "ref_trajectory.dat"
            np.savetxt(str(ref_path), ref_data, fmt="  %.16E")
            print(f"  Reference trajectory saved to {ref_path} ({ref_data.shape[0]} points, commanded-cos feedforward)")

    # Save best result and run final evaluation
    if result["best_individual"] is not None:
        if cfg.guidance_type == "neural_network":
            from aerocapture.training.param_spaces import active_scaffolding_specs

            _pack = active_scaffolding_specs(cfg.network.scaffolding)
            n_scaff = len(_pack)
            n_weights = len(param_specs) - n_scaff
            weights = _decode_nn_weights(result["best_individual"][:n_weights], param_specs[:n_weights])
            nn_path = Path(cwd) / cfg.sim.nn_param_file
            write_nn_json(
                weights,
                cfg.network,
                nn_path,
                input_mask=cfg.network.input_mask,
                output_param=cfg.network.output_parameterization,
                normalization=_resolve_config_normalization(cfg, cwd),
            )
            print(f"Best weights saved to {nn_path}")
            if n_scaff > 0:
                scaff_params = decode_normalized(result["best_individual"][n_weights:], list(_pack))
                for s in _pack:
                    if s.is_integer and s.name in scaff_params:
                        scaff_params[s.name] = int(round(scaff_params[s.name]))
                params_path = Path(cfg.save_dir) / "best_params.json"
                with open(params_path, "w") as fp:
                    json.dump(scaff_params, fp, indent=2)
                print(f"Best scaffolding params saved to {params_path}")
        else:
            params = decode_normalized(result["best_individual"], param_specs)
            params_path = Path(cfg.save_dir) / "best_params.json"
            with open(params_path, "w") as fp:
                json.dump(params, fp, indent=2)
            print(f"Best params saved to {params_path}")
            print(f"  Params: {params}")

            assert cfg.sim.toml_config is not None
            deploy_optimized_artifacts(params, cfg, _toml_data, Path(cfg.save_dir), Path(cwd) / cfg.sim.toml_config)

        # Report Generation. NEVER on an interrupted run: the report writes
        # final_eval.parquet, which is the campaign runners' skip-if-done marker
        # and the paper's quoted artifact -- a Ctrl+C'd 4-gen run would
        # self-certify as a completed cell (resume to completion first).
        if result.get("interrupted"):
            print("Run interrupted -- skipping final report/final_eval.parquet (resume to completion to produce them)")
        elif not args.skip_report:
            from aerocapture.training.report import generate_report

            toml_path_report = Path(args.toml)
            generate_report(Path(cfg.save_dir), toml_path_report, n_sims_override=args.final_n_sims, sim_timeout_secs=cfg.sim.sim_timeout_secs)
