"""Write (or --check) articles/paper/data/confirmatory_marginal.json.

The per-scenario-noise (per_draw) far-tail confirmatory that the paper's headline quotes and
Appendix E's far-tail table read through results.typ (issue #137). Its source,
experiments/ou_marginal/confirmatory_marginal.json, is written by confirmatory_marginal.py next
to it (hours per cell); this extract keeps only the cells and fields the paper quotes, so the
bundle carries them under data/SHA256SUMS and the provenance digest. `make check` runs
`--check`, which fails when the committed extract is not what the source yields. Pure stdlib.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "experiments/ou_marginal/confirmatory_marginal.json"
OUT = REPO / "articles/paper/data/confirmatory_marginal.json"
# The deployed fine-tune seed and its two repeats, the dense fine-tune, the shared-path champion, FNPAG.
CELLS = (
    "ou_marginal/ft_mamba_p962",
    "ou_marginal/ft_mamba_p962_s2",
    "ou_marginal/ft_mamba_p962_s3",
    "ou_marginal/ft_dense_p515",
    "mamba_p962_long",
    "fnpag",
)
POOLED = ("n", "n_captured", "cvar95", "cvar999", "max")


def build() -> dict:
    src = json.loads(SRC.read_text())
    assert src["noise_seeding"] == "per_draw", f"{SRC.relative_to(REPO)} is not a per_draw confirmatory"
    by_label = {c["label"]: c for c in src["cells"]}
    return {
        "source": str(SRC.relative_to(REPO)),
        "noise_seeding": src["noise_seeding"],
        "freeze_commit": src["freeze_commit"],
        "n_replicates": src["n_replicates"],
        "n_per_replicate": src["n_per_replicate"],
        "cells": [
            {
                "label": label,
                "toml": by_label[label]["toml"],
                "pooled": {k: by_label[label]["pooled"][k] for k in POOLED},
                "replicate_stats": {"cvar999": {"se": by_label[label]["replicate_stats"]["cvar999"]["se"]}},
            }
            for label in CELLS
        ],
    }


def main() -> None:
    text = json.dumps(build(), indent=1) + "\n"
    if sys.argv[1:] == ["--check"]:
        if not OUT.exists() or OUT.read_text() != text:
            sys.exit(f"{OUT.relative_to(REPO)} is not what {SRC.relative_to(REPO)} yields: run `make -C articles/paper confirmatory-marginal` and commit it")
        print(f"{OUT.relative_to(REPO)}: current")
        return
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(REPO)}", file=sys.stderr)


if __name__ == "__main__":
    main()
