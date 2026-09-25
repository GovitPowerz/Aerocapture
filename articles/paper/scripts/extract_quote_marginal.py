"""Write (or --check) articles/paper/data/quote_marginal.json.

The n = 1000 paired per-scenario quotes of Appendix E (the shared-path versus per-scenario table,
the retraining table and their prose, and the conclusion's shallow-tail quotes) read this file
through results.typ (issue #157). Its source, experiments/ou_marginal/quote_results.json, is
written by quote_marginal.py next to it (every OU-marginal cell flown under both regimes on one
shared seed pool: "frozen" pins the shared noise path, "marginal" gives scenario i its own path
through simulation.random_seed = 1000 + 7 i); this extract keeps only the cells and fields the
paper quotes, so the bundle carries them under data/SHA256SUMS and the provenance digest.
`make check` runs `--check`, which fails when the committed extract is not what the source
yields. Pure stdlib.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "experiments/ou_marginal/quote_results.json"
OUT = REPO / "articles/paper/data/quote_marginal.json"
# quote_marginal.py's protocol, which the source file does not record itself: both regimes pin
# monte_carlo.noise_seeding = legacy; the marginal one gives scenario i its own density-noise path
# through a per-seed override. results.typ asserts this shape at load.
REGIMES = {
    "frozen": {"noise_seeding": "legacy", "per_seed_override": None},
    "marginal": {"noise_seeding": "legacy", "per_seed_override": "simulation.random_seed = 1000 + 7 i"},
}
# The five shared-path-trained champions and the three classical laws Appendix E's regime table
# quotes, under both regimes.
SHARED_PATH = ("mamba_p962", "lstm_p1082", "gru_p1014", "dense_p972", "dense_p515", "fnpag", "pred_guid", "ftc")
# The retraining table: three scratch repeats and one fine-tune per cell, plus the Mamba pilot.
RETRAINED = ("mamba_p962", "gru_p1014", "dense_p972", "lstm_p1082", "dense_p515")
KEYS = (
    *(f"{label}/{regime}" for label in SHARED_PATH for regime in REGIMES),
    *(f"ou_{label}{suffix}/marginal" for label in RETRAINED for suffix in ("", "_s2", "_s3")),
    *(f"ou_ft_{label}/marginal" for label in RETRAINED),
    "ou_pilot_mamba/marginal",
)
FIELDS = ("capture_pct", "dv_cvar95", "heat_load_viol_pct")


def build() -> dict:
    src = json.loads(SRC.read_text())
    missing = [key for key in KEYS if key not in src["cells"]]
    if missing:
        sys.exit(f"{SRC.relative_to(REPO)} lacks the quoted cell(s) {', '.join(missing)}: run experiments/ou_marginal/quote_marginal.py")
    return {
        "source": str(SRC.relative_to(REPO)),
        "n_sims": src["n_sims"],
        "regimes": REGIMES,
        "cells": {key: {f: src["cells"][key][f] for f in FIELDS} for key in KEYS},
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
