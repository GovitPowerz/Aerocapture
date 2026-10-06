"""Write (or --check) articles/paper/data/confirmatory_marginal.json.

The per-scenario-noise (per_draw) 10^6 confirmatory the paper's performance tables read through
results.typ (issues #137, #174). Its source, experiments/ou_marginal/confirmatory_marginal.json, is
written by confirmatory_marginal.py next to it (hours per cell); this extract keeps every cell of
the scorer's manifests (confirmatory_cells.txt, the arxiv-v3 rows; confirmatory_cells_v4.txt, every
v4 row: issue #179) and only the fields the paper quotes, so the bundle carries them under
data/SHA256SUMS and the provenance digest. The manifests are the one cell list: a cell added to a
manifest is scored, extracted and quotable from the same line. `make check` runs `--check`, which
fails when the committed extract is not what the source yields. Pure stdlib.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "experiments/ou_marginal/confirmatory_marginal.json"
OUT = REPO / "articles/paper/data/confirmatory_marginal.json"
MANIFEST_REL = ("experiments/ou_marginal/confirmatory_cells.txt", "experiments/ou_marginal/confirmatory_cells_v4.txt")
MANIFESTS = tuple(REPO / m for m in MANIFEST_REL)
# heat_load_viol_n (the #173 outcome's "scenarios over the ceiling per 10^6") postdates the #137 rows
# (fnpag, mamba_p962_long), which carry null for it.
POOLED = ("n", "n_captured", "cvar95", "cvar999", "max", "viol_pct", "heat_load_viol_n")


def manifest_labels(paths: tuple[Path, ...] = MANIFESTS) -> tuple[str, ...]:
    """The 'label|toml[|model_dir]' rows of the scorer's manifests, first field, in order, deduplicated
    (the line grammar of confirmatory_marginal.read_manifest: `#` starts a comment anywhere on a line)."""
    labels: dict[str, None] = {}
    for path in paths:
        for raw in path.read_text().splitlines():
            line = raw.split("#", 1)[0].strip()
            if line:
                labels[line.split("|", 1)[0].strip()] = None
    return tuple(labels)


def build() -> dict:
    src = json.loads(SRC.read_text())
    if src["noise_seeding"] != "per_draw":
        sys.exit(f"{SRC.relative_to(REPO)} is not a per_draw confirmatory")
    by_label = {c["label"]: c for c in src["cells"]}
    cells = manifest_labels()
    missing = [label for label in cells if label not in by_label]
    if missing:
        sys.exit(
            f"{SRC.relative_to(REPO)} lacks the quoted cell(s) {', '.join(missing)}: "
            "run experiments/ou_marginal/confirmatory_marginal.py --manifest experiments/ou_marginal/confirmatory_cells_v4.txt"
        )
    return {
        "source": str(SRC.relative_to(REPO)),
        "manifests": list(MANIFEST_REL),
        "noise_seeding": src["noise_seeding"],
        "freeze_commit": src["freeze_commit"],
        "n_replicates": src["n_replicates"],
        "n_per_replicate": src["n_per_replicate"],
        "cells": [
            {
                "label": label,
                "toml": by_label[label]["toml"],
                "pooled": {k: by_label[label]["pooled"].get(k) for k in POOLED},
                "replicate_stats": {"cvar999": {"se": by_label[label]["replicate_stats"]["cvar999"]["se"]}},
            }
            for label in cells
        ],
    }


def main() -> None:
    if sys.argv[1:] not in ([], ["--check"]):
        sys.exit(f"usage: {Path(__file__).name} [--check]")
    text = json.dumps(build(), indent=1) + "\n"
    if sys.argv[1:] == ["--check"]:
        if not OUT.exists() or OUT.read_text() != text:
            sys.exit(
                f"{OUT.relative_to(REPO)} is not what {SRC.relative_to(REPO)} yields: "
                "run `make -C articles/paper confirmatory-marginal sums provenance`, commit, then recompile the PDF"
            )
        print(f"{OUT.relative_to(REPO)}: current")
        return
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(REPO)}", file=sys.stderr)


if __name__ == "__main__":
    main()
