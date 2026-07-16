"""Stage C0 spike driver: python -m aerocapture.cpag [--quick] [--skip-bench]."""

from __future__ import annotations

import argparse
import json
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="CPAG C0: SCP convergence studies + solver benchmark")
    parser.add_argument("--quick", action="store_true", help="reduced dispersed batch (10 sims)")
    parser.add_argument("--skip-bench", action="store_true")
    parser.add_argument("--skip-studies", action="store_true")
    parser.add_argument("--out-dir", default="training_output/cpag_c0")
    args = parser.parse_args()

    if not args.skip_studies:
        from aerocapture.cpag.studies import run_c0_studies  # noqa: PLC0415

        summary = run_c0_studies(n_dispersed=10 if args.quick else 40, out_dir=args.out_dir)
        print("== Nominal entry replan ==")
        print(json.dumps(summary["nominal"][0], indent=1))
        print(f"== Constant-bank sweep ({summary['n_sweep_cases']} states) ==")
        print(json.dumps(summary["sweep_summary"], indent=1))
        print(f"== Dispersed batch ({summary['n_dispersed_cases']} states) ==")
        print(json.dumps(summary["dispersed_summary"], indent=1))

    if not args.skip_bench:
        from aerocapture.cpag.bench import run_bench  # noqa: PLC0415

        results = run_bench(out_dir=args.out_dir)
        print("== Solver benchmark ==")
        for entry in results:
            _print_bench_entry(entry)


def _print_bench_entry(entry: dict[str, Any]) -> None:
    cl = entry["clarabel"]
    line = (
        f"{entry['instance']:<18} n_seg={entry['n_seg']:>3} vars={cl['n_vars']:>5}  "
        f"clarabel {cl['time_ms_p50']:7.2f}ms p95 {cl['time_ms_p95']:7.2f} it {cl['iters_p50']:.0f}"
    )
    osqp_entry = entry.get("osqp")
    if osqp_entry and "time_ms_p50" in osqp_entry:
        line += f" | osqp cold {osqp_entry['time_ms_p50']:8.2f}ms it {osqp_entry['iters_p50']:.0f}"
        warm = osqp_entry.get("warm")
        if warm:
            line += f" warm {warm['time_ms_p50']:8.2f}ms it {warm['iters_p50']:.0f}"
    print(line)


if __name__ == "__main__":
    main()
