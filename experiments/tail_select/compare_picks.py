"""#234: compare the retro picks of the six headline runs and, once flown, their far tails.

`run_selection.sh` writes three picks per run under training_output/tail_select/<run>/<arm>/
(arms: rms_validation, cvar95_validation, cvar95_tail_select). This script:

- `--manifest`: byte-compares every pick's best_model.json with the deployed model
  (training_output/ou_marginal/<run>/best_model.json; the rms_validation arm must reproduce
  it) and writes tail_select_cells.txt, the `label|toml|model_dir` manifest of the picks that
  differ from the RMS pick, for experiments/ou_marginal/confirmatory_marginal.py. The
  tail-select-pool pick is `tail_select/<run>`; the validation-pool pick, when it differs
  from both, is `tail_select/<run>_val`.
- `--table` (default): the pre-registered read. Per run: the pick each arm made, its CVaR95 on
  the pool that chose it, the deployed seed's and the tail pick's pooled CVaR99.9 on the
  10 x 100,000 confirmatory pool (confirmatory_marginal.json), capture and heat-load
  exceedances beside them; the six-run paired delta (tail pick minus RMS pick, zero where the
  picks coincide) with its mean, sd, 95% t-interval, and the same over the runs whose choice
  set had more than one feasible member. Writes tail_select.json and prints the README table.

Usage (repo root): uv run python experiments/tail_select/compare_picks.py [--manifest | --table]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from scipy.stats import t as student_t

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PICKS = REPO / "training_output/tail_select"
DEPLOYED = REPO / "training_output/ou_marginal"
CONFIRMATORY = REPO / "experiments/ou_marginal/confirmatory_marginal.json"
MANIFEST = HERE / "tail_select_cells.txt"
OUT = HERE / "tail_select.json"
RUNS = ("hl_mamba_p962", "hl_mamba_p962_s2", "hl_mamba_p962_s3", "hl_dense_p515", "hl_dense_p515_s2", "hl_dense_p515_s3")
ARMS = ("rms_validation", "cvar95_validation", "cvar95_tail_select")
PRIMARY = "cvar95_tail_select"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pick(run: str, arm: str) -> dict[str, Any]:
    """One arm's pick: provenance, the sidecar's tail block, choice-set size, model sha256."""
    d = PICKS / run / arm
    side = json.loads((d / "final_selection.json").read_text())
    cands = side["candidate_rms"]
    out: dict[str, Any] = {
        "provenance": side["winner"]["provenance"],
        "val_rms": side["winner"]["val_rms"],
        "feasible": side["winner"]["feasible"],
        "n_feasible": sum(1 for c in cands if c.get("feasible")),
        "n_candidates": len(cands),
        "sha256": sha256(d / "best_model.json"),
    }
    if side.get("criterion") == "cvar95":
        out["pool"] = side["tail_pool"]
        out["tail"] = side["winner"]["tail"]
        out["n_scored_on_pool"] = sum(1 for c in cands if "tail" in c)
        champ = next((c for c in cands if c["provenance"] == "champion"), None)
        out["champion_tail"] = champ.get("tail") if champ else None
    return out


def collect() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for run in RUNS:
        deployed = sha256(DEPLOYED / run / "best_model.json")
        arms = {arm: pick(run, arm) for arm in ARMS if (PICKS / run / arm / "final_selection.json").exists()}
        for p in arms.values():
            p["same_as_deployed"] = p["sha256"] == deployed
            p["same_as_rms_pick"] = "rms_validation" in arms and p["sha256"] == arms["rms_validation"]["sha256"]
        rows[run] = {"deployed_sha256": deployed, "arms": arms}
    return rows


def write_manifest(rows: dict[str, dict[str, Any]]) -> list[tuple[str, str, str]]:
    cells: list[tuple[str, str, str]] = []
    for run, r in rows.items():
        arms = r["arms"]
        if not arms.get("rms_validation", {}).get("same_as_deployed", False):
            raise SystemExit(f"{run}: the rms_validation arm does not reproduce the deployed model (or is missing); the pairing has no reference")
        toml = f"configs/training/ou_marginal/{run}.toml"
        primary = arms.get(PRIMARY)
        if primary and not primary["same_as_rms_pick"]:
            cells.append((f"tail_select/{run}", toml, f"training_output/tail_select/{run}/{PRIMARY}"))
        val = arms.get("cvar95_validation")
        if val and not val["same_as_rms_pick"] and not (primary and val["sha256"] == primary["sha256"]):
            cells.append((f"tail_select/{run}_val", toml, f"training_output/tail_select/{run}/cvar95_validation"))
    header = [
        "# #234 tail-aware picks that differ from the RMS pick, label|toml|model_dir (written by compare_picks.py --manifest). Run:",
        "#   uv run python -u experiments/ou_marginal/confirmatory_marginal.py --manifest experiments/tail_select/tail_select_cells.txt",
        "# `tail_select/<run>`: the 5000-seed tail-select-pool pick (primary arm); `_val`: the validation-pool pick when it differs from both.",
    ]
    MANIFEST.write_text("\n".join(header + [f"{a}|{b}|{c}" for a, b, c in cells]) + "\n")
    return cells


def confirmatory_row(label: str, by_label: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    c = by_label.get(label)
    if c is None:
        return None
    p = c["pooled"]
    return {
        "label": label,
        "capture_pct": round(100 * p["n_captured"] / p["n"], 4),
        "n_non_captured": p["n"] - p["n_captured"],
        "heat_load_viol_n": p.get("heat_load_viol_n"),
        "cvar95": p["cvar95"],
        "cvar999": p["cvar999"],
        "cvar999_se": c["replicate_stats"]["cvar999"]["se"],
        "max": p["max"],
        "model_sha256": c.get("model_sha256"),
    }


def t_interval(deltas: list[float]) -> dict[str, Any]:
    n = len(deltas)
    mean = statistics.mean(deltas)
    sd = statistics.stdev(deltas) if n > 1 else float("nan")
    half = float(student_t.ppf(0.975, n - 1)) * sd / math.sqrt(n) if n > 1 else float("nan")
    signs = ["0" if d == 0 else ("-" if d < 0 else "+") for d in deltas]
    return {"n": n, "mean": round(mean, 2), "sd": round(sd, 2), "ci95": [round(mean - half, 2), round(mean + half, 2)], "signs": signs}


def table(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_label = {c["label"]: c for c in json.loads(CONFIRMATORY.read_text())["cells"]}
    per_run: list[dict[str, Any]] = []
    deltas: list[float] = []
    decidable: list[float] = []
    for run, r in rows.items():
        arms = r["arms"]
        primary = arms[PRIMARY]
        rms_row = confirmatory_row(f"ou_marginal/{run}", by_label)
        if rms_row is None or rms_row["model_sha256"] != r["deployed_sha256"]:
            raise SystemExit(f"{run}: the deployed seed's confirmatory row is missing or was scored on another model")
        tail_row = rms_row if primary["same_as_rms_pick"] else confirmatory_row(f"tail_select/{run}", by_label)
        entry: dict[str, Any] = {
            "run": run,
            "n_feasible_last_gen": arms["rms_validation"]["n_feasible"],
            "rms_pick": arms["rms_validation"]["provenance"],
            "tail_pick": primary["provenance"],
            "tail_pick_differs": not primary["same_as_rms_pick"],
            "val_pick": arms["cvar95_validation"]["provenance"],
            "val_pick_differs": not arms["cvar95_validation"]["same_as_rms_pick"],
            "pool_cvar95_tail_pick": primary["tail"]["dv_cvar95"] if primary.get("tail") else None,
            "pool_cvar95_rms_pick": primary["champion_tail"]["dv_cvar95"] if primary.get("champion_tail") else None,
            "pool_n_captured_tail_pick": primary["tail"]["n_captured"] if primary.get("tail") else None,
            "rms_confirmatory": rms_row,
            "tail_confirmatory": tail_row,
        }
        if tail_row is not None:
            delta = 0.0 if primary["same_as_rms_pick"] else round(tail_row["cvar999"] - rms_row["cvar999"], 2)
            entry["delta_cvar999"] = delta
            entry["delta_capture_pts"] = round(tail_row["capture_pct"] - rms_row["capture_pct"], 4)
            entry["delta_heat_load_viol_n"] = (tail_row["heat_load_viol_n"] or 0) - (rms_row["heat_load_viol_n"] or 0)
            deltas.append(delta)
            if arms["rms_validation"]["n_feasible"] > 0:
                decidable.append(delta)
        else:
            entry["delta_cvar999"] = None
        val = arms["cvar95_validation"]
        if not val["same_as_rms_pick"]:
            label = f"tail_select/{run}" if val["sha256"] == primary["sha256"] else f"tail_select/{run}_val"
            entry["val_confirmatory"] = confirmatory_row(label, by_label)
        per_run.append(entry)
    flown = len(deltas) == len(rows)
    result: dict[str, Any] = {
        "issue": 234,
        "pool": "CONFIRMATORY 10 x 100000 per_draw (confirmatory_marginal.json rows)",
        "rule": (
            "six-run paired delta of pooled CVaR99.9, tail pick (cvar95_tail_select) minus RMS pick; 0 where the picks coincide; "
            "win = mean < 0 with the 95% t-interval excluding 0, no capture loss, no more heat-load exceedances"
        ),
        "per_run": per_run,
        "six_run": t_interval(deltas) if flown else None,
        "decidable_runs": t_interval(decidable) if flown and len(decidable) > 1 else None,
    }
    if flown:
        six = result["six_run"]
        no_regression = all(e["delta_capture_pts"] >= 0 and e["delta_heat_load_viol_n"] <= 0 for e in per_run)
        if six["mean"] < 0 and six["ci95"][1] < 0 and no_regression:
            result["verdict"] = "win"
        else:
            result["verdict"] = "suggestive" if six["mean"] < 0 and no_regression else "negative"
    return result


def markdown(result: dict[str, Any]) -> str:
    lines = [
        "| run | feasible last-gen | RMS pick | tail pick (5000 pool) | pool CVaR95 RMS / tail | capture RMS / tail "
        "| heat load > limit RMS / tail (per 10^6) | CVaR99.9 RMS / tail (+- s.e.) | delta CVaR99.9 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for e in result["per_run"]:
        r, tl = e["rms_confirmatory"], e["tail_confirmatory"]
        tail = f"{tl['cvar999']:.1f} +- {tl['cvar999_se']:.1f}" if tl else "not flown"
        cap = f"{r['capture_pct']:.4f}% / " + (f"{tl['capture_pct']:.4f}%" if tl else "-")
        viol = f"{r['heat_load_viol_n']} / " + (str(tl["heat_load_viol_n"]) if tl else "-")
        pool = f"{e['pool_cvar95_rms_pick']} / {e['pool_cvar95_tail_pick']}"
        delta = f"{e['delta_cvar999']:+.1f}" if e["delta_cvar999"] is not None else "-"
        tail_pick = e["tail_pick"] + ("" if e["tail_pick_differs"] else " (same model)")
        rms_ci = f"{r['cvar999']:.1f} +- {r['cvar999_se']:.1f}"
        lines.append(f"| `{e['run']}` | {e['n_feasible_last_gen']} | {e['rms_pick']} | {tail_pick} | {pool} | {cap} | {viol} | {rms_ci} / {tail} | {delta} |")
    if result.get("six_run"):
        six, dec = result["six_run"], result.get("decidable_runs")
        lines.append("")
        ci = f"[{six['ci95'][0]:+.1f}, {six['ci95'][1]:+.1f}]"
        lines.append(f"Six-run paired delta: mean {six['mean']:+.1f}, sd {six['sd']:.1f}, 95% t-interval {ci}, signs {' '.join(six['signs'])}.")
        if dec:
            ci = f"[{dec['ci95'][0]:+.1f}, {dec['ci95'][1]:+.1f}]"
            lines.append(f"Runs with more than one feasible member ({dec['n']}): mean {dec['mean']:+.1f}, sd {dec['sd']:.1f}, 95% t-interval {ci}.")
        lines.append(f"Verdict under the pre-registered rule: **{result['verdict']}**.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="store_true", help="write tail_select_cells.txt (the picks to fly) and stop")
    args = parser.parse_args()
    rows = collect()
    if args.manifest:
        cells = write_manifest(rows)
        for run, r in rows.items():
            flags = {arm: ("= rms" if p["same_as_rms_pick"] else "DIFFERS") for arm, p in r["arms"].items()}
            print(f"{run}: {flags}")
        print(f"{len(cells)} cell(s) to fly -> {MANIFEST.relative_to(REPO)}")
        return
    result = table(rows)
    OUT.write_text(json.dumps({"picks": rows, **result}, indent=1) + "\n")
    print(markdown(result))


if __name__ == "__main__":
    main()
