"""TOML -> TrainingConfig resolution and the chromosome (ParamSpec) layout.

The one TOML -> `TrainingConfig` chokepoint (`build_training_config_from_toml`,
shared by the train CLI and the `final_select` CLI), `_setup_param_specs` (the
optimizer's gene list), and the small config readers the artifact writers and
the CLI share. A leaf: imports no loop, checkpoint or artifact module.
"""

from __future__ import annotations

import sys
from pathlib import Path

from aerocapture.training.config import TrainingConfig, WarmStartConfig
from aerocapture.training.encoding import nn_param_specs_from_architecture, nn_param_specs_from_v2
from aerocapture.training.evaluate import _aero_rs
from aerocapture.training.optimizer import OptimizerConfig
from aerocapture.training.param_spaces import ParamSpec
from aerocapture.training.toml_utils import load_toml_with_bases, reject_unknown_keys

_DEFAULT_PIECEWISE_N_SEGMENTS = 10

# Python-owned root sections read here with `.get`; build_training_config_from_toml
# rejects unknown keys so a typo cannot train at the default silently (#128 D).
CHECKPOINTS_KEYS = frozenset({"keep_last"})
REFERENCE_KEYS = frozenset({"joint_bank", "bank_low", "bank_high"})
CORRIDOR_KEYS = frozenset({"delta_za_restricted", "delta_za_restricted_low", "delta_za_restricted_high"})


def _resolve_config_normalization(config: TrainingConfig, cwd: str | Path | None) -> list[dict] | None:
    """Read the config's `[network.normalization]` override (35 dicts) or None.

    Threaded into the deployed `best_model.json` write so the model is
    self-describing under a normalization override (otherwise the PSO write path
    embeds DEFAULT_NORMALIZATION and the deployed model ships the wrong scales).
    """
    if not config.sim.toml_config:
        return None
    toml_path = Path(cwd or config.sim.exec_dir) / config.sim.toml_config
    try:
        toml_data = load_toml_with_bases(toml_path)
    except (OSError, ValueError):  # fmt: skip
        return None
    norm = toml_data.get("network", {}).get("normalization")
    return norm if isinstance(norm, list) else None


def _resolve_piecewise_n_segments(toml: dict) -> int:
    """Mirror of Rust TomlPiecewiseConstantParams::resolve_bank_angles_deg.

    Order of precedence: explicit n_segments > bank_angles array length >
    highest bank_angle_N key index + 1 > default (10).
    """
    pc = toml.get("guidance", {}).get("piecewise_constant", {})
    if "n_segments" in pc:
        n = int(pc["n_segments"])
        if n < 1:
            raise ValueError(f"[guidance.piecewise_constant] n_segments must be >= 1, got {n}")
        return n
    if "bank_angles" in pc:
        return len(pc["bank_angles"])
    max_idx = max(
        (int(k.removeprefix("bank_angle_")) for k in pc if k.startswith("bank_angle_") and k.removeprefix("bank_angle_").isdigit()),
        default=-1,
    )
    if max_idx >= 0:
        return max_idx + 1
    return _DEFAULT_PIECEWISE_N_SEGMENTS


def check_ref_trajectory_wiring(toml_data: dict, ref_traj_path: Path) -> None:
    """Hard-fail when a ref-tracking scheme's resolved config doesn't point at the
    mission's optimized reference. Guards against the silent-legacy-ref bug: the
    existence check alone proves nothing about what the sim actually loads.
    """
    configured = toml_data.get("data", {}).get("reference_trajectory")
    if configured is None or Path(configured).resolve() != Path(ref_traj_path).resolve():
        print(f"\nERROR: this scheme tracks the optimized reference trajectory at {ref_traj_path},")
        print(f"but the resolved config's data.reference_trajectory is {configured!r}.")
        print(f'Add `reference_trajectory = "{ref_traj_path}"` to the [data] section of the training TOML.')
        sys.exit(1)


def _setup_param_specs(config: TrainingConfig, _toml: dict, verbose: bool) -> list[ParamSpec]:
    """Build the optimizer ParamSpec list (NN v2/v1 / piecewise / scheme table + scaffolding tail)."""
    # Build parameter specifications
    from aerocapture.training.param_spaces import PARAM_SPACES

    if config.guidance_type == "neural_network":
        if config.network.architecture is not None:
            from pydantic import TypeAdapter

            from aerocapture.training.torch_mirror.schemas import LayerSpec

            specs_adapter = TypeAdapter(list[LayerSpec])
            validated = specs_adapter.validate_python(config.network.architecture)
            # bound_multiplier=2.0 matches create_nn_initial_population's Phase 1
            # convention AND build_initial_population_for_v2 below. Keeping them
            # aligned avoids ~49% boundary-saturation on the initial PSO population.
            # When warm-start is on, use the wider [warm_start] bound_multiplier
            # (default 4.0) so the search space envelops the warm-started chromosome's
            # post-supervised-training drift past Xavier bounds.
            warm_start_active = bool(config.network.warm_start_from) or config.warm_start.enabled
            bound_mult = config.warm_start.bound_multiplier if warm_start_active else 2.0
            param_specs = nn_param_specs_from_v2(validated, bound_multiplier=bound_mult)
        else:
            param_specs = nn_param_specs_from_architecture(
                config.network.layer_sizes,
                config.network.activations,
            )

        if config.network.scaffolding != "off":
            if config.network.architecture is None:
                msg = "scaffolding != 'off' requires v2 [[network.architecture]]; v1 layer_sizes/activations is not supported. Convert your config."
                raise ValueError(msg)
            from aerocapture.training.param_spaces import active_scaffolding_specs

            param_specs = [*param_specs, *active_scaffolding_specs(config.network.scaffolding)]
            if verbose:
                if config.network.scaffolding == "live":
                    print("scaffolding optimization: LIVE — 3 params (nav density filter ×2, command shaping); no FTC dependency")
                else:  # full
                    print("scaffolding optimization: FULL — 17 params, seeded from training_output/ftc/best_params.json")
        else:
            if verbose and config.network.architecture is not None:
                print("scaffolding optimization: OFF — NN weights only")
    elif config.guidance_type == "piecewise_constant":
        from aerocapture.training.param_spaces import make_piecewise_constant_specs

        n_segments = _resolve_piecewise_n_segments(_toml)
        param_specs = make_piecewise_constant_specs(n_segments)
        if verbose:
            pc = _toml.get("guidance", {}).get("piecewise_constant", {})
            e_min = float(pc.get("energy_min", -6.0))
            e_max = float(pc.get("energy_max", 5.0))
            seg_width = (e_max - e_min) / n_segments if n_segments > 0 else float("nan")
            initial = pc.get("bank_angles")
            init_label = f"seeded from TOML bank_angles ({len(initial)} values)" if initial is not None else "GA-initialized (no TOML bank_angles)"
            n_shaping = len(param_specs) - n_segments
            print(
                f"piecewise_constant: {n_segments} segments over E in [{e_min:.2f}, {e_max:.2f}] MJ/kg "
                f"(width {seg_width:.3f} MJ/kg), {init_label}; "
                f"chromosome = {n_segments} bank + {n_shaping} shaping = {len(param_specs)} params"
            )
    else:
        param_specs = PARAM_SPACES[config.guidance_type]

    ref_cfg = _toml.get("reference", {})
    if ref_cfg.get("joint_bank", False):
        from aerocapture.training.param_spaces import JOINT_REF_BANK_SCHEMES  # noqa: PLC0415

        if config.guidance_type not in JOINT_REF_BANK_SCHEMES:
            print(f"ERROR: [reference] joint_bank requires a table-reading scheme ({sorted(JOINT_REF_BANK_SCHEMES)}), got '{config.guidance_type}'")
            sys.exit(1)
        bank_low = float(ref_cfg.get("bank_low", 55.0))
        bank_high = float(ref_cfg.get("bank_high", 80.0))
        param_specs = [*param_specs, ParamSpec("ref_bank", bank_low, bank_high, 68.0)]
        if verbose:
            print(f"joint reference: ref_bank gene in [{bank_low:.1f}, {bank_high:.1f}] deg (per-individual constant-bank reference tables)")

    return param_specs


def build_training_config_from_toml(toml_path: str) -> tuple[TrainingConfig, dict]:
    """TOML -> TrainingConfig (the TOML-derived part of main()'s bootstrap).

    Applies NO CLI overrides: callers overlay n_gen/n_pop/algorithm/sim_timeout
    on the returned config themselves. Raises SystemExit on invalid configs
    (missing/unknown guidance type, bad [checkpoints], warm-start contract
    violations) -- identical messages to the historical main() behavior. Note:
    [optimizer] / [warm_start] PARSE errors (OptimizerConfig.from_dict /
    WarmStartConfig.from_dict) raise ValueError, not SystemExit.
    """
    cfg = TrainingConfig()

    # Load TOML first -- optimizer config comes from TOML, CLI overrides on top
    _toml_data = load_toml_with_bases(Path(toml_path))

    # Parse optimizer config from TOML (uses OptimizerConfig defaults for missing keys)
    cfg.optimizer = OptimizerConfig.from_dict(_toml_data.get("optimizer", {}))

    guidance_type = _toml_data.get("guidance", {}).get("type")
    if guidance_type is None:
        print("ERROR: TOML config must contain [guidance] type = '<scheme>'")
        print("  Valid schemes: neural_network, equilibrium_glide, energy_controller, pred_guid, fnpag, ftc, piecewise_constant")
        raise SystemExit(1)

    from aerocapture.training.param_spaces import PARAM_SPACES

    _valid_types = set(PARAM_SPACES.keys()) | {"neural_network"}
    if guidance_type not in _valid_types:
        print(f"ERROR: Unknown guidance type '{guidance_type}' in TOML")
        print(f"  Valid schemes: {', '.join(sorted(_valid_types))}")
        raise SystemExit(1)

    cfg.guidance_type = guidance_type
    cfg.sim.toml_config = toml_path
    cfg.sim.executable = "src/rust/target/release/aerocapture"
    cfg.sim.nn_param_file = _toml_data.get("data", {}).get("neural_network", "data/neural_network/nn_model.json")
    # Override NN architecture from TOML [network] section if present
    _net = _toml_data.get("network", {})
    if "architecture" in _net:
        # v2 heterogeneous arch (list of per-layer dicts, e.g. dense + gru + dense).
        cfg.network.architecture = list(_net["architecture"])
    if "layer_sizes" in _net:
        cfg.network.layer_sizes = _net["layer_sizes"]
    if "activations" in _net:
        cfg.network.activations = _net["activations"]
    if "input_mask" in _net:
        cfg.network.input_mask = _net["input_mask"]
    if "qat_bits" in _net:
        cfg.network.qat_bits = int(_net["qat_bits"])
    if "qat_granularity" in _net:
        cfg.network.qat_granularity = str(_net["qat_granularity"])
    if "qat_tensor_policy" in _net:
        cfg.network.qat_tensor_policy = str(_net["qat_tensor_policy"])
    _gnn = _toml_data.get("guidance", {}).get("neural_network", {})
    if "scaffolding" in _gnn:
        cfg.network.scaffolding = str(_gnn["scaffolding"])
    if "output_parameterization" in _gnn:
        cfg.network.output_parameterization = str(_gnn["output_parameterization"])
    if "scaled_pi_n" in _gnn:
        cfg.network.scaled_pi_n = float(_gnn["scaled_pi_n"])
    if "delta_max" in _gnn:
        cfg.network.delta_max = float(_gnn["delta_max"])
    if "warm_start_from" in _gnn:
        cfg.network.warm_start_from = str(_gnn["warm_start_from"])
    if cfg.network.warm_start_from is not None:
        warm_path = Path(cfg.network.warm_start_from)
        if not warm_path.exists():
            print(f"ERROR: warm_start_from='{warm_path}' does not exist")
            raise SystemExit(1)
    if "warm_start" in _toml_data:
        cfg.warm_start = WarmStartConfig.from_dict(_toml_data["warm_start"])

    # The field assignments above bypass NetworkConfig.__post_init__, leaving
    # per-entry normalization undone (e.g. Mamba dt_rank resolution). train()
    # used to survive only because init_v2_population mutates the entries as a
    # side effect; callers that never build a population (final_select CLI)
    # crashed run_grid with "missing field dt_rank". Re-run it explicitly.
    cfg.network.__post_init__()

    # `[checkpoints]` block: optional disk-retention policy. `keep_last = N`
    # auto-prunes older `checkpoint_g*.{json,npz}` pairs after each save,
    # keeping only the N most recent. The JSONL log + best_* artifacts are
    # untouched.
    # The Python-owned root sections validate here, the one TOML -> TrainingConfig
    # chokepoint (train CLI and final_select): a misspelled key raises instead
    # of reading the default silently (#128 D).
    reject_unknown_keys("checkpoints", _toml_data.get("checkpoints", {}), CHECKPOINTS_KEYS)
    reject_unknown_keys("corridor", _toml_data.get("corridor", {}), CORRIDOR_KEYS)
    reject_unknown_keys("reference", _toml_data.get("reference", {}), REFERENCE_KEYS)
    if "checkpoints" in _toml_data:
        _ckpt = _toml_data["checkpoints"]
        if "keep_last" in _ckpt:
            kl_raw = _ckpt["keep_last"]
            if kl_raw is not None and not isinstance(kl_raw, int):
                print(f"ERROR: [checkpoints] keep_last must be an int or null, got {type(kl_raw).__name__}")
                raise SystemExit(1)
            if isinstance(kl_raw, int) and kl_raw < 1:
                print(f"ERROR: [checkpoints] keep_last must be >= 1, got {kl_raw}")
                raise SystemExit(1)
            cfg.checkpoints.keep_last = kl_raw

    # Warm-start contract: supervised targets are the post-lateral, pre-shaper
    # SIGNED bank command (tick.rs captures `guidance_out.pre_shaper_signed`).
    # warm_start.py collapses the sign to magnitude only when mode = "magnitude_only".
    # Two matched setups:
    #   - mode = "magnitude_only" + output_parameterization = "acos_tanh":
    #     single-output tanh head -> acos in [0, pi], runtime decoder .abs()'s
    #     the NN output and lateral guidance re-selects the sign.
    #   - mode = "full_neural" + output_parameterization in
    #     {"atan2_signed", "scaled_pi", "delta"}: signed bank with no runtime
    #     lateral/thermal/shaping interception. atan2_signed uses a two-output
    #     atan2 head; scaled_pi/delta use a single-output tanh head decoded via
    #     wrap_to_pi(n*pi*tanh) and wrap_to_pi(prev_realized + delta_max*tanh)
    #     respectively (both hard-required to be full_neural by the Rust runtime).
    # acos_tanh + full_neural is REJECTED here because the Rust runtime
    # (src/rust/src/config.rs::validate_output_parameterization) hard-errors
    # at config load: "output_parameterization=acos_tanh is only legal with
    # mode=magnitude_only". Catching it before warm-start compute saves the
    # ~10 minutes of supervised collection + BPTT pretrain that would
    # otherwise be wasted on a config Rust will reject at gen-0.
    warm_start_active = bool(cfg.network.warm_start_from) or cfg.warm_start.enabled
    if warm_start_active:
        nn_mode = str(_gnn.get("mode", "full_neural"))
        out_param = cfg.network.output_parameterization or "atan2_signed"
        if nn_mode == "full_neural" and out_param == "acos_tanh":
            print(
                "ERROR: output_parameterization='acos_tanh' requires mode='magnitude_only' "
                "(Rust runtime enforces this at config load). Either set mode='magnitude_only' "
                "or switch to output_parameterization='atan2_signed' for full_neural."
            )
            raise SystemExit(1)
        matched = (nn_mode == "magnitude_only" and out_param == "acos_tanh") or (
            nn_mode == "full_neural" and out_param in ("atan2_signed", "scaled_pi", "delta")
        )
        if not matched:
            print(
                f"  [warm_start] WARNING: (mode='{nn_mode}', output_parameterization='{out_param}') "
                f"is not a matched pair. The matched setups are "
                f"(magnitude_only, acos_tanh) and (full_neural, {{atan2_signed, scaled_pi, delta}}). "
                f"Training will still run, but the supervised target and runtime decoder may be suboptimal."
            )

    # Rust-side config rules (the same `config::validate` pass every SimData
    # build runs, no table IO): fail here, before any seed pool or warm-start
    # compute, instead of at the gen-0 run_grid. Soft-import guarded: a machine
    # without the extension keeps the Python mirrors above.
    if _aero_rs is not None:
        try:
            _aero_rs.validate_config(toml_path)
        except ValueError as exc:
            print(f"ERROR: invalid config (Rust validation): {exc}")
            raise SystemExit(1) from exc

    return cfg, _toml_data
