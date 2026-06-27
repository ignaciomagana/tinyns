"""Summarize JSON output from validation/run_validation.py."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _std_or_none(values: list[float]) -> float | None:
    if len(values) > 1:
        return float(np.std(values, ddof=1))
    return 0.0 if values else None


def summarize_results(results: list[dict]) -> list[dict]:
    """Group per-run validation results by target and sampler."""

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for result in results:
        groups[(result["target"], result["sampler"])].append(result)

    summaries = []
    for (target, sampler), rows in sorted(groups.items()):
        logz_errors = [
            row["logz_error"]
            for row in rows
            if row.get("logz_error") is not None
        ]
        logzerrs = [row["logzerr"] for row in rows if row.get("logzerr") is not None]
        covered = [
            abs(row["logz_error"]) <= row["logzerr"]
            for row in rows
            if row.get("logz_error") is not None and row.get("logzerr") is not None
        ]
        warnings = sum(len(row.get("warnings", [])) for row in rows)
        failures = sum(int(row.get("replacement_failures") or 0) for row in rows)
        summaries.append(
            {
                "target": target,
                "sampler": sampler,
                "nruns": len(rows),
                "success_fraction": float(
                    np.mean([bool(row.get("success")) for row in rows])
                ),
                "mean_logz_error": _mean_or_none(logz_errors),
                "std_logz_error": _std_or_none(logz_errors),
                "mean_logzerr": _mean_or_none(logzerrs),
                "coverage_fraction": float(np.mean(covered)) if covered else None,
                "mean_ncall": _mean_or_none(
                    [row["ncall"] for row in rows if row.get("ncall") is not None]
                ),
                "mean_posterior_ess": _mean_or_none(
                    [
                        row["posterior_ess"]
                        for row in rows
                        if row.get("posterior_ess") is not None
                    ]
                ),
                "mean_replacement_ncall": _mean_or_none(
                    [
                        row["replacement_mean_ncall"]
                        for row in rows
                        if row.get("replacement_mean_ncall") is not None
                    ]
                ),
                "total_warnings": warnings,
                "total_replacement_failures": failures,
            }
        )
    return summaries


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def print_table(summaries: list[dict]) -> None:
    columns = [
        "target",
        "sampler",
        "nruns",
        "success_fraction",
        "mean_logz_error",
        "std_logz_error",
        "mean_logzerr",
        "coverage_fraction",
        "mean_ncall",
        "mean_posterior_ess",
        "mean_replacement_ncall",
        "total_warnings",
        "total_replacement_failures",
    ]
    widths = {
        col: max(len(col), *(len(_fmt(row[col])) for row in summaries))
        for col in columns
    }
    print("  ".join(col.ljust(widths[col]) for col in columns))
    print("  ".join("-" * widths[col] for col in columns))
    for row in summaries:
        print("  ".join(_fmt(row[col]).ljust(widths[col]) for col in columns))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path")
    args = parser.parse_args()
    payload = json.loads(Path(args.json_path).read_text())
    print_table(summarize_results(payload["results"]))


if __name__ == "__main__":
    main()
