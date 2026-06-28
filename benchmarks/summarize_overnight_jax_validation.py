"""Summarize JSON rows from benchmarks/overnight_jax_validation.py."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def _mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if len(values) == 1 else None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["target"]), str(row["config_name"]))].append(row)
    summaries = []
    for (target, config), group in sorted(grouped.items()):
        logz = [float(row["logz"]) for row in group if row.get("logz") is not None]
        errors = []
        for row in group:
            expected = row.get("expected_logz", row.get("analytic_logz"))
            if expected is not None and row.get("logz") is not None:
                errors.append(abs(float(row["logz"]) - float(expected)))
        summaries.append(
            {
                "target": target,
                "config": config,
                "mean_seconds": _mean(
                    [
                        float(row["seconds"])
                        for row in group
                        if row.get("seconds") is not None
                    ]
                ),
                "mean_logz": _mean(logz),
                "std_logz": _std(logz),
                "mean_abs_logz_error": _mean(errors),
                "failures": sum(
                    (not bool(row.get("success", True)))
                    + int(row.get("replacement_failures", 0) or 0)
                    for row in group
                ),
            }
        )
    return summaries


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def print_table(summaries: list[dict[str, Any]]) -> None:
    print("target config mean_seconds mean_logz std_logz mean_abs_logz_error failures")
    for row in summaries:
        print(
            f"{row['target']} {row['config']} {_fmt(row['mean_seconds'])} "
            f"{_fmt(row['mean_logz'])} {_fmt(row['std_logz'])} "
            f"{_fmt(row['mean_abs_logz_error'])} {row['failures']}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rows = json.loads(Path(args.json_path).read_text())
    if isinstance(rows, dict):
        rows = rows.get("results", [])
    print_table(summarize_rows(rows))


if __name__ == "__main__":
    main()
