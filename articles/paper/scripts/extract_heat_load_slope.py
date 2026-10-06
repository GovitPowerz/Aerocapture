"""Write (or --check) articles/paper/data/heat_load_slope.json.

The heat-load ceiling sensitivity of the v4 headline cells (issue #192): the two seed-1
champions fine-tuned under three ceilings, each deployed policy scored on the paired n = 1000
per-scenario pool against its own ceiling and against the v4 limit. Its source,
experiments/ou_marginal/heat_load_slope.json, is written by heat_load_slope.py next to it; the
paper reads the copy through results.typ, so the bundle carries it under data/SHA256SUMS and the
provenance digest. The copy is verbatim (the file is small and every row is quotable); the
protocol record (`regime`, `seed_pool`, `v4_max_heat_load`, `override`) travels with it so
results.typ's regime assert tests the scorer, never this script. `make check` runs `--check`,
which fails when the committed copy is not what the source yields. Pure stdlib.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "experiments/ou_marginal/heat_load_slope.json"
OUT = REPO / "articles/paper/data/heat_load_slope.json"
PROTOCOL = ("regime", "seed_pool", "v4_max_heat_load", "override")


def build() -> dict:
    src = json.loads(SRC.read_text())
    if any(k not in src for k in PROTOCOL):
        sys.exit(f"{SRC.relative_to(REPO)} carries no protocol record ({', '.join(PROTOCOL)}): re-run experiments/ou_marginal/heat_load_slope.py")
    if not src["rows"]:
        sys.exit(f"{SRC.relative_to(REPO)} has no rows: run experiments/ou_marginal/heat_load_slope.py")
    return {"source": str(SRC.relative_to(REPO)), **src}


def main() -> None:
    if sys.argv[1:] not in ([], ["--check"]):
        sys.exit(f"usage: {Path(__file__).name} [--check]")
    text = json.dumps(build(), indent=1) + "\n"
    if sys.argv[1:] == ["--check"]:
        if not OUT.exists() or OUT.read_text() != text:
            sys.exit(
                f"{OUT.relative_to(REPO)} is not what {SRC.relative_to(REPO)} yields: "
                "run `make -C articles/paper heat-load-slope sums provenance`, commit, then recompile the PDF"
            )
        print(f"{OUT.relative_to(REPO)}: current")
        return
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(REPO)}", file=sys.stderr)


if __name__ == "__main__":
    main()
