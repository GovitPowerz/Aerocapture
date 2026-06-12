"""Architecture parameter-budget sweep -> Pareto curve (performance vs weights).

Motivation: a single fixed-weight comparison (the ~1000-weight atan2 variants)
answers only "best at fixed deployment cost". It does NOT show whether a richer
architecture (GRU/LSTM/Transformer/Mamba) ever overtakes the plain dense net
given more budget -- the transformer's 4*d_model^2 term and the LSTM's 4-gate
overhead are crippled under 1000 weights. This driver sweeps each architecture
across several param budgets, trains each point, re-evaluates every deployed
model on ONE fixed reserved seed pool (identical scenarios for all points), and
plots cost vs params with the Pareto frontier. The whole curve dissolves the
"is fixed-weight fair?" argument: you read off both "who wins at budget B" and
"does anyone overtake dense with more budget".

Every sweep config base-inherits configs/training/msr_aller_nn_atan2_train.toml,
so the 17-input mask, calibrated normalization, full_neural/atan2_signed,
scaffolding=live, command shaping and navigation are all shared -- only the
[[network.architecture]] stack and the deploy path differ.

CLI (run from repo root):
    python -m aerocapture.training.param_sweep --generate
    python -m aerocapture.training.param_sweep --train --n-gen 1500 --n-pop 64
    python -m aerocapture.training.param_sweep --eval --plot --n-sims 1000
    python -m aerocapture.training.param_sweep --all --n-gen 1500 --n-sims 1000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from aerocapture.training.config import NetworkConfig

INPUT_DIM = 17  # length of the inherited atan2 input_mask
ARCHS = ("dense", "gru", "lstm", "mamba", "transformer", "window")
DEFAULT_BUDGETS = (500, 1000, 2000, 4000)
SWEEP_EVAL_SEED_OFFSET = 7_000_000  # disjoint from validation(1M)/final(2M)/rl(3M)/warm(4M)/report(5M)/calib(6M)

BASE_CONFIG = "msr_aller_nn_atan2_train.toml"
SWEEP_CONFIG_DIR = Path("configs/training/sweep")
TRAINING_OUTPUT = Path("training_output")


# ─────────────────────────── architecture sizing ───────────────────────────
# Each family exposes ONE capacity knob (core hidden width / d_model / d_state).
# We enumerate the knob, compute the EXACT param count via NetworkConfig (the
# same counter Rust mirrors), and later pick the candidate nearest each budget.
# Secondary dims follow a fixed, defensible ratio so the shape stays sane.


def _arch_params(arch: list[dict[str, Any]]) -> int:
    return NetworkConfig(architecture=arch, input_mask=list(range(INPUT_DIM))).n_base_coef


def _dense_stack(hidden: int) -> list[dict[str, Any]]:
    h2 = max(4, round(hidden / 2))
    return [
        {"type": "dense", "input_size": INPUT_DIM, "output_size": hidden, "activation": "swish"},
        {"type": "dense", "input_size": hidden, "output_size": h2, "activation": "swish"},
        {"type": "dense", "input_size": h2, "output_size": 2, "activation": "asinh"},
    ]


def _gru_stack(h: int) -> list[dict[str, Any]]:
    return [
        {"type": "dense", "input_size": INPUT_DIM, "output_size": h, "activation": "swish"},
        {"type": "gru", "input_size": h, "hidden_size": h},
        {"type": "dense", "input_size": h, "output_size": 2, "activation": "asinh"},
    ]


def _lstm_stack(h: int) -> list[dict[str, Any]]:
    return [
        {"type": "dense", "input_size": INPUT_DIM, "output_size": h, "activation": "swish"},
        {"type": "lstm", "input_size": h, "hidden_size": h},
        {"type": "dense", "input_size": h, "output_size": 2, "activation": "asinh"},
    ]


def _mamba_stack(d_inner: int) -> list[dict[str, Any]]:
    d_state = max(8, round(d_inner * 0.75))
    return [
        {"type": "dense", "input_size": INPUT_DIM, "output_size": d_inner, "activation": "swish"},
        {"type": "mamba", "input_size": d_inner, "d_state": d_state},
        {"type": "dense", "input_size": d_inner, "output_size": 2, "activation": "asinh"},
    ]


def _transformer_stack(d_model: int) -> list[dict[str, Any]]:
    n_heads = 4 if d_model % 4 == 0 else (2 if d_model % 2 == 0 else 1)
    return [
        {"type": "dense", "input_size": INPUT_DIM, "output_size": d_model, "activation": "swish"},
        {"type": "transformer", "d_model": d_model, "n_heads": n_heads, "d_ffn": 2 * d_model, "n_seq": 32},
        {"type": "dense", "input_size": d_model, "output_size": 2, "activation": "asinh"},
    ]


def _window_stack(h1: int) -> list[dict[str, Any]]:
    n_steps = 4
    flat = INPUT_DIM * n_steps
    h2 = max(4, round(h1 * 0.6))
    return [
        {"type": "window", "input_size": INPUT_DIM, "n_steps": n_steps},
        {"type": "dense", "input_size": flat, "output_size": h1, "activation": "swish"},
        {"type": "dense", "input_size": h1, "output_size": h2, "activation": "swish"},
        {"type": "dense", "input_size": h2, "output_size": 2, "activation": "asinh"},
    ]


# (builder, knob range) per family. Ranges chosen to span ~300..6000 params.
_FAMILIES = {
    "dense": (_dense_stack, range(8, 110)),
    "gru": (_gru_stack, range(6, 56)),
    "lstm": (_lstm_stack, range(5, 48)),
    "mamba": (_mamba_stack, range(8, 60)),
    "transformer": (_transformer_stack, range(8, 56, 2)),  # even d_model only (n_heads divisibility)
    "window": (_window_stack, range(8, 90)),
}


def candidates(arch: str) -> list[tuple[int, list[dict[str, Any]]]]:
    """All (param_count, architecture) candidates for a family, sorted by params."""
    builder, knobs = _FAMILIES[arch]
    seen: dict[int, list[dict[str, Any]]] = {}
    for k in knobs:
        a = builder(k)
        p = _arch_params(a)
        seen.setdefault(p, a)  # first (smallest knob) wins on param-count ties
    return sorted(seen.items())


def select_for_budgets(arch: str, budgets: tuple[int, ...]) -> list[tuple[int, list[dict[str, Any]]]]:
    """Pick the candidate nearest each budget; dedupe so each point is distinct."""
    cands = candidates(arch)
    picked: dict[int, list[dict[str, Any]]] = {}
    for b in budgets:
        p, a = min(cands, key=lambda pa: abs(pa[0] - b))
        picked[p] = a
    return sorted(picked.items())


# ─────────────────────────── config generation ───────────────────────────


def _dirname(arch: str, params: int) -> str:
    return f"sweep_{arch}_p{params}"


def _arch_toml_block(arch: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for layer in arch:
        lines.append("[[network.architecture]]")
        for key in ("type", "input_size", "output_size", "activation", "hidden_size", "d_state", "d_model", "n_heads", "d_ffn", "n_seq", "n_steps"):
            if key not in layer:
                continue
            v = layer[key]
            lines.append(f'{key} = "{v}"' if isinstance(v, str) else f"{key} = {v}")
        lines.append("")
    return "\n".join(lines)


def _config_text(arch_name: str, params: int, arch: list[dict[str, Any]]) -> str:
    dirname = _dirname(arch_name, params)
    return (
        f"# AUTO-GENERATED by param_sweep.py -- DO NOT EDIT BY HAND.\n"
        f"# {arch_name} variant, {params} trainable weights (Pareto sweep point).\n"
        f"# Base-inherits the tuned atan2 pipeline (17-input mask, calibrated\n"
        f"# normalization, full_neural / atan2_signed, scaffolding = live, command\n"
        f"# shaping + navigation); overrides only the architecture and deploy path.\n"
        f'base = ["../{BASE_CONFIG}"]\n\n'
        f"[data]\n"
        f'neural_network = "{(TRAINING_OUTPUT / dirname / "best_model.json").as_posix()}"\n'
        f'results_suffix = ".sweep_{arch_name}_p{params}"\n\n'
        f"{_arch_toml_block(arch)}"
    )


def generate(archs: tuple[str, ...], budgets: tuple[int, ...]) -> list[dict[str, Any]]:
    SWEEP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for arch_name in archs:
        for params, arch in select_for_budgets(arch_name, budgets):
            cfg_path = SWEEP_CONFIG_DIR / f"{arch_name}_p{params}.toml"
            cfg_path.write_text(_config_text(arch_name, params, arch))
            entry = {
                "arch": arch_name,
                "params": params,
                "config": cfg_path.as_posix(),
                "output_dir": (TRAINING_OUTPUT / _dirname(arch_name, params)).as_posix(),
                "architecture": arch,
            }
            manifest.append(entry)
            print(f"  {arch_name:12s} {params:5d}p -> {cfg_path}")
    SWEEP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    (SWEEP_CONFIG_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {len(manifest)} configs + manifest to {SWEEP_CONFIG_DIR}/")
    return manifest


def _load_manifest() -> list[dict[str, Any]]:
    path = SWEEP_CONFIG_DIR / "manifest.json"
    if not path.exists():
        sys.exit(f"No manifest at {path}; run --generate first.")
    manifest: list[dict[str, Any]] = json.loads(path.read_text())
    return manifest


# ─────────────────────────── training orchestration ───────────────────────────


def train(manifest: list[dict[str, Any]], n_gen: int, n_pop: int, algorithm: str | None, sim_timeout: float | None, force: bool) -> None:
    for i, entry in enumerate(manifest, 1):
        out_model = Path(entry["output_dir"]) / "best_model.json"
        if out_model.exists() and not force:
            print(f"[{i}/{len(manifest)}] skip {entry['arch']} {entry['params']}p (best_model.json exists; --force to retrain)")
            continue
        cmd = [sys.executable, "-m", "aerocapture.training.train", entry["config"], "--n-gen", str(n_gen), "--n-pop", str(n_pop), "--no-tui", "--skip-report"]
        if algorithm:
            cmd += ["--algorithm", algorithm]
        if sim_timeout is not None:
            cmd += ["--sim-timeout", str(sim_timeout)]
        print(f"[{i}/{len(manifest)}] train {entry['arch']} {entry['params']}p: {' '.join(cmd)}")
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"  WARNING: training exited {rc} for {entry['config']}")


# ─────────────────────────── evaluation ───────────────────────────


def _entry_overrides(entry: dict[str, Any], model: Path, seeds: list[int]) -> list[dict[str, Any]]:
    """Per-seed run_batch overrides: deployed NN + co-trained scaffolding.

    Sweep configs inherit `scaffolding = "live"`, so each point's best_params.json
    carries co-trained nav/shaping values that MUST be applied at eval (the same
    convention report.py / compare_guidance.py follow) — without them every model
    is scored against TOML-default scaffolding it was never trained with.
    run_batch == one sim per override; force n_sims=1 (configs inherit n_sims=1000).
    """
    from aerocapture.training.report import _load_nn_scaffolding_overrides

    out_dir = Path(entry["output_dir"])
    scaff = _load_nn_scaffolding_overrides(out_dir, out_dir / f"optimized_{out_dir.name}.toml")
    base: dict[str, Any] = {"simulation.n_sims": 1, "data.neural_network": str(model), **scaff}
    return [{**base, "monte_carlo.seed": int(s)} for s in seeds]


def evaluate(manifest: list[dict[str, Any]], n_sims: int, base_seed: int, sim_timeout: float | None) -> list[dict[str, Any]]:
    import aerocapture_rs

    from aerocapture.training.evaluate import make_reserved_seeds
    from aerocapture.training.report import compute_eval_summary, read_cost_kwargs

    seeds = make_reserved_seeds(base_seed, SWEEP_EVAL_SEED_OFFSET, n_sims)
    results: list[dict[str, Any]] = []
    for entry in manifest:
        model = Path(entry["output_dir"]) / "best_model.json"
        if not model.exists():
            print(f"  skip {entry['arch']} {entry['params']}p (untrained: no {model})")
            continue
        cost_kwargs = read_cost_kwargs(Path(entry["config"]))
        overrides_list = _entry_overrides(entry, model, seeds)
        batch = aerocapture_rs.run_batch(entry["config"], overrides_list, n_threads=None, include_trajectories=False, sim_timeout_secs=sim_timeout)
        final = np.array(batch.final_records, dtype=np.float64)
        summary = compute_eval_summary(final, n_sims=n_sims, cost_kwargs=cost_kwargs)
        row = {
            "arch": entry["arch"],
            "params": entry["params"],
            "rms_cost": summary["cost"]["rms"],
            "cost_p50": summary["cost"]["p50"],
            "capture_rate": summary["capture_rate"],
            "dv_p50": (summary["captured"]["dv"]["p50"] if summary["captured"] else None),
            "dv_p95": (summary["captured"]["dv"]["p95"] if summary["captured"] else None),
        }
        results.append(row)
        dv = f"{row['dv_p50']:.1f}" if row["dv_p50"] is not None else "  n/a"
        print(f"  {row['arch']:12s} {row['params']:5d}p  rms={row['rms_cost']:10.2f}  cap={row['capture_rate']:5.1%}  dvP50={dv}")
    out = SWEEP_CONFIG_DIR / "pareto_results.json"
    out.write_text(json.dumps({"n_sims": n_sims, "base_seed": base_seed, "rows": results}, indent=2))
    print(f"\nWrote {len(results)} eval rows to {out}")
    return results


# ─────────────────────────── Pareto plot ───────────────────────────


def _pareto_front(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Lower-left envelope: minimize both params (x) and cost (y)."""
    front: list[tuple[float, float]] = []
    best_cost = float("inf")
    for x, y in sorted(points):  # ascending params
        if y < best_cost:
            front.append((x, y))
            best_cost = y
    return front


_METRIC_LABELS = {
    "rms_cost": "RMS cost (training objective, lower better)",
    "cost_p50": "median per-sim cost (lower better)",
    "dv_p50": "captured DV p50 [m/s] (survivors only)",
    "dv_p95": "captured DV p95 [m/s] (survivors only)",
}


def plot(results: list[dict[str, Any]] | None = None, metric: str = "rms_cost") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if results is None:
        path = SWEEP_CONFIG_DIR / "pareto_results.json"
        if not path.exists():
            sys.exit(f"No eval results at {path}; run --eval first.")
        results = json.loads(path.read_text())["rows"]
    # DV metrics are defined over survivors only -- drop points with no metric.
    results = [r for r in (results or []) if r.get(metric) is not None]
    if not results:
        sys.exit(f"No evaluated points with metric {metric!r} to plot (train + eval first).")

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    for ci, arch in enumerate(sorted({r["arch"] for r in results})):
        pts = sorted((r["params"], r[metric]) for r in results if r["arch"] == arch)
        xs, ys = zip(*pts, strict=True)
        ax.plot(xs, ys, "-o", color=cmap(ci), label=arch, alpha=0.85, markersize=6)

    front = _pareto_front([(r["params"], r[metric]) for r in results])
    fx, fy = zip(*front, strict=True)
    ax.plot(fx, fy, "--", color="black", linewidth=2, label="Pareto frontier", zorder=1)

    ax.set_xscale("log")
    ax.set_yscale("log")  # robust to cost_transform magnitude (e.g. "cubed")
    ax.set_xlabel("trainable weights")
    ax.set_ylabel(_METRIC_LABELS.get(metric, metric))
    ax.set_title("Architecture Pareto: performance vs parameter budget")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    out = SWEEP_CONFIG_DIR / f"pareto_{metric}.svg"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


# ─────────────────────────── CLI ───────────────────────────


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Architecture parameter-budget sweep -> Pareto curve.")
    p.add_argument("--generate", action="store_true", help="write sweep configs + manifest")
    p.add_argument("--train", action="store_true", help="train each sweep point (subprocess train.py)")
    p.add_argument("--eval", action="store_true", help="re-evaluate trained models on the fixed seed pool")
    p.add_argument("--plot", action="store_true", help="render pareto.svg from eval results")
    p.add_argument("--all", action="store_true", help="generate + train + eval + plot")
    p.add_argument("--archs", nargs="+", default=list(ARCHS), choices=ARCHS, help="architectures to sweep")
    p.add_argument("--budgets", nargs="+", type=int, default=list(DEFAULT_BUDGETS), help="param budgets")
    p.add_argument("--n-gen", type=int, default=1500, help="generations per training point")
    p.add_argument("--n-pop", type=int, default=64, help="population size per training point")
    p.add_argument("--algorithm", default=None, help="optimizer override (else TOML default)")
    p.add_argument("--n-sims", type=int, default=1000, help="eval seed-pool size")
    p.add_argument("--base-seed", type=int, default=0, help="base mc seed for the eval pool")
    p.add_argument("--sim-timeout", type=float, default=30.0, help="per-sim wall-clock timeout (s)")
    p.add_argument("--metric", default="rms_cost", choices=list(_METRIC_LABELS), help="Pareto y-axis metric")
    p.add_argument("--force", action="store_true", help="retrain even if best_model.json exists")
    args = p.parse_args(argv)

    do_gen, do_train, do_eval, do_plot = args.generate, args.train, args.eval, args.plot
    if args.all:
        do_gen = do_train = do_eval = do_plot = True
    if not any((do_gen, do_train, do_eval, do_plot)):
        p.error("specify at least one of --generate/--train/--eval/--plot/--all")

    archs, budgets = tuple(args.archs), tuple(args.budgets)
    manifest = generate(archs, budgets) if do_gen else _load_manifest()
    if do_train:
        train(manifest, args.n_gen, args.n_pop, args.algorithm, args.sim_timeout, args.force)
    results = evaluate(manifest, args.n_sims, args.base_seed, args.sim_timeout) if do_eval else None
    if do_plot:
        plot(results, metric=args.metric)


if __name__ == "__main__":
    main()
