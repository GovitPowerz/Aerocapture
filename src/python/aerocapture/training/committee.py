"""Committee builder (#232): assemble N deployed v2 `best_model.json` files into
one committee model, a single `committee` layer whose members are the deployed
stacks nested verbatim, their 2-output heads combined under one rule before the
atan2 decode. Deploy-only: the Rust runtime flies it, the torch mirror refuses it
(`model_io.load_policy_from_json`), no chromosome layout exists for it.

    uv run python -m aerocapture.training.committee --rule circular_mean \\
        --out training_output/committee/hl_mamba3_circular_mean \\
        training_output/ou_marginal/hl_mamba_p962/best_model.json ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

RULES = ("circular_mean", "raw_sum", "medoid", "sign_leader")

# Decoder and input contract every member must share: the committee carries one copy.
_SHARED_KEYS = ("input_mask", "normalization", "output_param", "ablated_input", "scaled_pi_n", "delta_max")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_committee(member_paths: Sequence[Path], rule: str) -> dict[str, Any]:
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
    if not member_paths:
        raise ValueError("a committee needs at least one member")
    members = [json.loads(Path(p).read_text()) for p in member_paths]
    first = members[0]
    for path, m in zip(member_paths, members, strict=True):
        if m.get("format_version") != 2:
            raise ValueError(f"{path}: format_version must be 2, got {m.get('format_version')}")
        if any(layer.get("type") == "committee" for layer in m["architecture"]):
            raise ValueError(f"{path}: a committee member cannot itself be a committee")
        if (m.get("output_param") or "atan2_signed") != "atan2_signed":
            raise ValueError(f"{path}: members must use the atan2_signed decoder, got {m['output_param']!r}")
        if m.get("ablated_input") is not None:
            raise ValueError(f"{path}: an ablated member cannot join a committee")
        for key in _SHARED_KEYS:
            if m.get(key) != first.get(key):
                raise ValueError(f"{path}: {key} differs from {member_paths[0]}")
    return {
        "format_version": 2,
        "architecture": [
            {
                "type": "committee",
                "rule": rule,
                "members": [m["architecture"] for m in members],
                "member_sha256": [sha256_of(Path(p)) for p in member_paths],
            }
        ],
        "weights": {"layer_0": {"members": [m["weights"] for m in members]}},
        "input_mask": first.get("input_mask"),
        "ablated_input": None,
        "ablated_value": 0.0,
        "output_param": "atan2_signed",
        "scaled_pi_n": first.get("scaled_pi_n", 1.0),
        "delta_max": first.get("delta_max", 0.35),
        "normalization": first.get("normalization"),
    }


def write_committee(member_paths: Sequence[Path], rule: str, out_dir: Path) -> Path:
    model = build_committee(member_paths, rule)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "best_model.json"
    out.write_text(json.dumps(model, indent=2) + "\n")
    # Provenance sidecar: the cell is assembled, not trained (no checkpoints, no final
    # selection); the confirmatory driver reads it as the cell's "run finished" marker.
    (out_dir / "committee.json").write_text(
        json.dumps({"rule": rule, "members": [str(p) for p in member_paths], "member_sha256": model["architecture"][0]["member_sha256"]}, indent=2) + "\n"
    )
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Assemble deployed v2 models into one committee best_model.json")
    parser.add_argument("members", nargs="+", type=Path, help="member best_model.json files, in order")
    parser.add_argument("--rule", choices=RULES, required=True)
    parser.add_argument("--out", type=Path, required=True, help="cell directory; writes <out>/best_model.json")
    args = parser.parse_args(argv)
    out = write_committee(args.members, args.rule, args.out)
    print(f"{out} ({len(args.members)} members, {args.rule}) sha256 {sha256_of(out)}")


if __name__ == "__main__":
    main()
