"""Write (or --check) articles/paper/data/quant_v4/confirmatory_marginal.json.

The 10^6 per-scenario rows of the v4 quantization finalists (issue #178: the deployed champion, the
PTQ verdict cell, the QAT fine-tune and the QAT scratch arm) and the paired replicate delta of each
finalist against the champion, the first row of experiments/ou_marginal/quant_cells_v4.txt. Source:
experiments/ou_marginal/confirmatory_marginal.json, written by confirmatory_marginal.py (hours per
cell). Same pre-registered pools for every row, so the deltas are paired per replicate: mean, s.e.
and the t(df = n - 1) 95% interval over the replicates, as confirmatory_eval.py quotes its pairs.
Both modes are pending (a message, exit 0, nothing written) while the campaign has not scored every
finalist and no copy is committed, so `make paper` and `make check` pass before the campaign; a
committed copy must be what the source yields. Pure stdlib.
"""

import json
import math
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments/ou_marginal"))

from confirmatory_marginal import read_manifest  # noqa: E402

SRC = REPO / "experiments/ou_marginal/confirmatory_marginal.json"
OUT = REPO / "articles/paper/data/quant_v4/confirmatory_marginal.json"
MANIFEST = REPO / "experiments/ou_marginal/quant_cells_v4.txt"
# The exact violation counts ride along: the 2-decimal viol_pct reads fewer than 50 per 10^6 as 0.00.
POOLED = ("n", "n_captured", "cvar95", "cvar999", "p999", "max", "viol_pct", "viol_n", "heat_flux_viol_n", "g_load_viol_n", "heat_load_viol_n")
PAIRED = ("cvar95", "cvar999", "p999", "max")
# t(0.975, df) for the replicate counts in use; 10 replicates is the pre-registered pool.
T95 = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 9: 2.262}


def _agg(values: list[float]) -> dict:
    n = len(values)
    if n - 1 not in T95:
        sys.exit(f"{n} replicates: no t(0.975) entry for df = {n - 1} in extract_quant_v4.T95 (the pre-registered pool has 10)")
    mean = statistics.mean(values)
    se = statistics.stdev(values) / math.sqrt(n)
    t = T95[n - 1]
    return {"mean": round(mean, 2), "se": round(se, 2), "ci95": [round(mean - t * se, 2), round(mean + t * se, 2)]}


def _missing(by_label: dict, labels: tuple[str, ...]) -> list[str]:
    return [label for label in labels if label not in by_label]


def _pending(src: dict) -> bool:
    """True while the campaign has not scored every finalist and no copy is committed: nothing to write or check."""
    by_label = {c["label"]: c for c in src.get("cells", [])}
    labels = tuple(label for label, _, _ in read_manifest(MANIFEST))
    if _missing(by_label, labels) and not OUT.exists():
        print(f"{OUT.relative_to(REPO)}: pending (the #178 campaign has not scored every finalist yet)")
        return True
    return False


def build(src: dict | None = None) -> dict:
    if src is None:
        src = json.loads(SRC.read_text())
    if src.get("noise_seeding") != "per_draw":
        sys.exit(f"{SRC.relative_to(REPO)} is not a per_draw confirmatory")
    by_label = {c["label"]: c for c in src["cells"]}
    labels = tuple(label for label, _, _ in read_manifest(MANIFEST))
    if missing := _missing(by_label, labels):
        sys.exit(f"{SRC.relative_to(REPO)} lacks the quoted cell(s) {', '.join(missing)}: run experiments/paper/17_quantization.sh v4 finalists")
    reference, finalists = labels[0], labels[1:]
    ref_reps = by_label[reference]["replicates"]
    paired = {}
    for label in finalists:
        reps = by_label[label]["replicates"]
        if [r["replicate"] for r in reps] != [r["replicate"] for r in ref_reps]:
            sys.exit(f"{label}'s replicates are not {reference}'s pools in the same order: nothing to pair")
        paired[label] = {f"delta_{k}": _agg([a[k] - b[k] for a, b in zip(reps, ref_reps, strict=True)]) for k in PAIRED}
    return {
        "source": str(SRC.relative_to(REPO)),
        "manifest": str(MANIFEST.relative_to(REPO)),
        "noise_seeding": src["noise_seeding"],
        "freeze_commit": src["freeze_commit"],
        "n_replicates": src["n_replicates"],
        "n_per_replicate": src["n_per_replicate"],
        "reference": reference,
        "cells": [
            {
                "label": label,
                "toml": by_label[label]["toml"],
                "eval_commit": by_label[label].get("eval_commit"),
                # The ptq4_verdict row flies the champion's TOML: the model hash is what tells the two apart.
                "model_sha256": by_label[label].get("model_sha256"),
                "pooled": {k: by_label[label]["pooled"].get(k) for k in POOLED},
                "replicate_stats": {k: {"se": by_label[label]["replicate_stats"][k]["se"]} for k in ("cvar95", "cvar999")},
            }
            for label in labels
        ],
        "paired": paired,
    }


def _text(src: dict) -> str:
    return json.dumps(build(src), indent=1) + "\n"


def check() -> None:
    src = json.loads(SRC.read_text())  # 8 MB with every replicate: parsed once per invocation
    if _pending(src):
        return
    if not OUT.exists() or OUT.read_text() != _text(src):
        sys.exit(
            f"{OUT.relative_to(REPO)} is not what {SRC.relative_to(REPO)} yields: "
            "run `make -C articles/paper quant-v4`, `git add` it (`sums` lists tracked files only), "
            "`make -C articles/paper sums provenance`, commit, then recompile the PDF"
        )
    print(f"{OUT.relative_to(REPO)}: current")


def write() -> None:
    src = json.loads(SRC.read_text())
    if _pending(src):
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(_text(src))
    print(f"wrote {OUT.relative_to(REPO)}", file=sys.stderr)


def main() -> None:
    if sys.argv[1:] not in ([], ["--check"]):
        sys.exit(f"usage: {Path(__file__).name} [--check]")
    check() if sys.argv[1:] else write()


if __name__ == "__main__":
    main()
