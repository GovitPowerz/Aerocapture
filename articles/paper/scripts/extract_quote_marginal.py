"""Write (or --check) articles/paper/data/quote_marginal.json.

The n = 1000 paired per-scenario quotes the paper reads through results.typ (issues #157, #175).
Its source, experiments/ou_marginal/quote_results.json, is written by quote_marginal.py next to it
(every OU-marginal cell flown under both regimes on one shared seed pool: "frozen" pins the shared
noise path, "marginal" gives scenario i its own path through a per-seed simulation.random_seed
override); this extract keeps every cell the scorer wrote (issue #179: the v4 rows are the whole
campaign, so the cell list is the source's) and only the fields the paper quotes, plus the protocol
record the source writes (`regimes`, `seed_pool`, copied verbatim so results.typ's regime assert
tests the source, never this script), so the bundle carries them under data/SHA256SUMS and the
provenance digest. `make check` runs `--check`, which fails when the committed extract is not what
the source yields. Pure stdlib.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "experiments/ou_marginal/quote_results.json"
OUT = REPO / "articles/paper/data/quote_marginal.json"
# The protocol record quote_marginal.py writes next to its cells (issue #166); a source without
# it predates the record and must be re-run, never patched here.
PROTOCOL = ("regimes", "seed_pool")
FIELDS = ("capture_pct", "dv_cvar95", "heat_load_viol_pct")


def build() -> dict:
    src = json.loads(SRC.read_text())
    if any(k not in src for k in PROTOCOL):
        sys.exit(f"{SRC.relative_to(REPO)} carries no protocol record ({', '.join(PROTOCOL)}): re-run experiments/ou_marginal/quote_marginal.py")
    if not src["cells"]:
        sys.exit(f"{SRC.relative_to(REPO)} has no cells: run experiments/ou_marginal/quote_marginal.py")
    return {
        "source": str(SRC.relative_to(REPO)),
        "n_sims": src["n_sims"],
        **{k: src[k] for k in PROTOCOL},
        "cells": {key: {f: src["cells"][key][f] for f in FIELDS} for key in sorted(src["cells"])},
    }


def main() -> None:
    if sys.argv[1:] not in ([], ["--check"]):
        sys.exit(f"usage: {Path(__file__).name} [--check]")
    text = json.dumps(build(), indent=1) + "\n"
    if sys.argv[1:] == ["--check"]:
        if not OUT.exists() or OUT.read_text() != text:
            sys.exit(
                f"{OUT.relative_to(REPO)} is not what {SRC.relative_to(REPO)} yields: "
                "run `make -C articles/paper quote-marginal sums provenance`, commit, then recompile the PDF"
            )
        print(f"{OUT.relative_to(REPO)}: current")
        return
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(REPO)}", file=sys.stderr)


if __name__ == "__main__":
    main()
