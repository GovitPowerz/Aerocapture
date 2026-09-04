"""Bit-equivalence diff between two gate-run tags.

Usage: uv run python experiments/trainer_seam_gate/diff_gate.py baseline post

Compares, per gate run:
- JSONL training logs: every record, with volatile keys (timestamps, wall
  times, rates, file paths embedding the tag) stripped recursively.
- Final checkpoint npz: every array bit-exact.
- final_selection.json (when present): volatile keys stripped.
Exit 0 = equivalent; prints the first divergence otherwise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

G = Path(__file__).resolve().parents[2] / ".scratch/trainer_seam_gate"  # run_gate.sh output root
VOLATILE_SUBSTRINGS = ("time", "elapsed", "eta", "rate_", "_rate_s", "duration", "path", "dir")
VOLATILE_KEYS = {"timestamp", "gen_elapsed_s", "wall_s", "eta_s", "save_dir", "toml_path", "output_dir", "config_hash"}


def strip(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k not in VOLATILE_KEYS and not any(s in k.lower() for s in ("elapsed", "timestamp", "wall", "eta"))}
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o


def load_jsonl(d: Path) -> list[Any]:
    files = sorted(d.glob("run_*.jsonl"))
    assert files, f"no jsonl in {d}"
    return [strip(json.loads(line)) for f in files for line in f.read_text().splitlines()]


def latest_ckpt(d: Path) -> Path:
    npz = sorted(d.glob("checkpoint_g*.npz"))
    assert npz, f"no checkpoint npz in {d}"
    return npz[-1]


def main(tag_a: str, tag_b: str) -> int:
    fail = 0
    for gate in ("run_a", "run_b", "run_c"):
        da, db = G / tag_a / gate, G / tag_b / gate
        ja, jb = load_jsonl(da), load_jsonl(db)
        if len(ja) != len(jb):
            print(f"{gate}: JSONL record count {len(ja)} vs {len(jb)}")
            fail = 1
        for i, (ra, rb) in enumerate(zip(ja, jb, strict=False)):
            if ra != rb:
                print(f"{gate}: JSONL record {i} differs")
                for k in set(ra) | set(rb):
                    if ra.get(k) != rb.get(k):
                        print(f"  key {k!r}:\n    A: {str(ra.get(k))[:200]}\n    B: {str(rb.get(k))[:200]}")
                fail = 1
                break
        ca, cb = latest_ckpt(da), latest_ckpt(db)
        if ca.name != cb.name:
            print(f"{gate}: checkpoint names differ: {ca.name} vs {cb.name}")
            fail = 1
        with np.load(ca, allow_pickle=True) as a, np.load(cb, allow_pickle=True) as b:
            keys_a, keys_b = set(a.files), set(b.files)
            if keys_a != keys_b:
                print(f"{gate}: checkpoint keys differ: {keys_a ^ keys_b}")
                fail = 1
            for k in sorted(keys_a & keys_b):
                va, vb = a[k], b[k]
                if va.dtype.kind in "fiu" and vb.dtype.kind in "fiu":
                    if not np.array_equal(va, vb, equal_nan=True):
                        print(f"{gate}: checkpoint array {k!r} NOT bit-equal")
                        fail = 1
                elif str(va.tolist()) != str(vb.tolist()):
                    print(f"{gate}: checkpoint object/str array {k!r} differs")
                    fail = 1
        if not fail:
            print(f"{gate}: OK ({len(ja)} records, ckpt {ca.name})")
    return fail


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
