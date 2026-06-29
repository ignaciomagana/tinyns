"""Summarize one or more overnight JAX validation JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any


def _mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _median(values: list[float]) -> float | None:
    return None if not values else float(median(values))


def _rms(values: list[float]) -> float | None:
    return None if not values else math.sqrt(
        sum(value * value for value in values) / len(values)
    )


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _infer_label(path: str | Path) -> str:
    stem = Path(path).stem
    prefix = "overnight_jax_validation_"
    return stem.removeprefix(prefix) or stem


def _read_rows(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    rows = payload.get("results", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"{path} does not contain a JSON list or results list")
    return [row for row in rows if isinstance(row, dict)]


def _value_list(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [value for row in rows if (value := _as_float(row.get(key))) is not None]


def load_labeled_rows(
    json_paths: list[str], labels: list[str] | None = None
) -> list[dict[str, Any]]:
    if labels is not None and len(labels) != len(json_paths):
        raise ValueError("--label must be supplied once per JSON file")
    all_rows: list[dict[str, Any]] = []
    for index, path in enumerate(json_paths):
        label = labels[index] if labels is not None else _infer_label(path)
        for row in _read_rows(path):
            copied = dict(row)
            copied["run_label"] = label
            all_rows.append(copied)
    return all_rows


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row.get("run_label", "run")),
                str(row.get("target", "unknown")),
                str(row.get("config_name", row.get("config", "unknown"))),
            )
        ].append(row)

    summaries = []
    for (run_label, target, config_name), group in sorted(grouped.items()):
        n = len(group)
        success_count = sum(1 for row in group if bool(row.get("success", True)))
        replacement_failures_total = sum(
            int(row.get("replacement_failures", 0) or 0) for row in group
        )
        errors: list[float] = []
        pulls: list[float] = []
        for row in group:
            logz = _as_float(row.get("logz"))
            expected = _as_float(row.get("expected_logz", row.get("analytic_logz")))
            if logz is None or expected is None:
                continue
            error = logz - expected
            errors.append(error)
            logzerr = _as_float(row.get("logzerr"))
            if logzerr is not None and logzerr != 0.0:
                pulls.append(error / logzerr)
        summaries.append(
            {
                "run_label": run_label,
                "target": target,
                "config_name": config_name,
                "n": n,
                "success_count": success_count,
                "success_rate": success_count / n if n else 0.0,
                "replacement_failures_total": replacement_failures_total,
                "mean_seconds": _mean(_value_list(group, "seconds")),
                "median_seconds": _median(_value_list(group, "seconds")),
                "min_seconds": min(_value_list(group, "seconds"), default=None),
                "max_seconds": max(_value_list(group, "seconds"), default=None),
                "mean_ncall": _mean(_value_list(group, "ncall")),
                "median_ncall": _median(_value_list(group, "ncall")),
                "mean_niter": _mean(_value_list(group, "niter")),
                "median_niter": _median(_value_list(group, "niter")),
                "mean_final_delta_logz": _mean(_value_list(group, "final_delta_logz")),
                "median_final_delta_logz": _median(
                    _value_list(group, "final_delta_logz")
                ),
                "mean_error": _mean(errors),
                "mean_abs_error": _mean([abs(error) for error in errors]),
                "rms_error": _rms(errors),
                "mean_pull": _mean(pulls),
                "rms_pull": _rms(pulls),
                "max_abs_pull": max([abs(pull) for pull in pulls], default=None),
                "dlogz": _mean(_value_list(group, "dlogz")),
            }
        )
    return summaries


def fastest_passing_by_target(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baselines = {
        row["target"]: row
        for row in summaries
        if row["run_label"] == "no_block"
        and row["config_name"] == "unbounded_isotropic_rwalk"
        and row["median_seconds"] is not None
    }
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        if (
            row["success_rate"] == 1.0
            and row["replacement_failures_total"] == 0
            and row["median_seconds"] is not None
        ):
            by_target[row["target"]].append(row)
    winners = []
    for target, candidates in sorted(by_target.items()):
        winner = min(candidates, key=lambda row: float(row["median_seconds"]))
        baseline_seconds = baselines.get(target, {}).get("median_seconds")
        speedup = None
        if baseline_seconds is not None and winner["median_seconds"]:
            speedup = float(baseline_seconds) / float(winner["median_seconds"])
        winners.append({**winner, "speedup_vs_baseline": speedup})
    return winners


def warnings_for(summaries: list[dict[str, Any]]) -> list[str]:
    warnings = []
    for row in summaries:
        prefix = f"{row['run_label']}/{row['target']}/{row['config_name']}"
        if row["success_rate"] < 1.0:
            warnings.append(f"{prefix}: success_rate={_fmt(row['success_rate'])}")
        if row["replacement_failures_total"] > 0:
            warnings.append(
                f"{prefix}: replacement_failures_total="
                f"{row['replacement_failures_total']}"
            )
        if row["rms_pull"] is not None and row["rms_pull"] > 2.0:
            warnings.append(f"{prefix}: rms_pull={_fmt(row['rms_pull'])}")
        if row["max_abs_pull"] is not None and row["max_abs_pull"] > 5.0:
            warnings.append(f"{prefix}: max_abs_pull={_fmt(row['max_abs_pull'])}")
        if (
            row["dlogz"] is not None
            and row["mean_final_delta_logz"] is not None
            and row["mean_final_delta_logz"] < 0.1 * row["dlogz"]
        ):
            warnings.append(
                f"{prefix}: mean_final_delta_logz={_fmt(row['mean_final_delta_logz'])} "
                f"is much lower than dlogz={_fmt(row['dlogz'])}"
            )
    return warnings


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _print_table(columns: list[str], rows: list[dict[str, Any]]) -> None:
    widths = [len(column) for column in columns]
    rendered = [[_fmt(row.get(column)) for column in columns] for row in rows]
    for cells in rendered:
        widths = [
            max(width, len(cell))
            for width, cell in zip(widths, cells, strict=True)
        ]
    print(
        "  ".join(
            column.ljust(width)
            for column, width in zip(columns, widths, strict=True)
        )
    )
    print("  ".join("-" * width for width in widths))
    for cells in rendered:
        print(
            "  ".join(
                cell.ljust(width)
                for cell, width in zip(cells, widths, strict=True)
            )
        )


def print_summary(summaries: list[dict[str, Any]]) -> None:
    print("Overall by file/target/config")
    _print_table(
        [
            "run_label",
            "target",
            "config_name",
            "n",
            "success_count",
            "success_rate",
            "replacement_failures_total",
            "mean_seconds",
            "median_seconds",
            "mean_ncall",
            "median_ncall",
            "mean_niter",
            "median_niter",
            "mean_final_delta_logz",
            "median_final_delta_logz",
        ],
        summaries,
    )
    print("\nAccuracy on analytic targets")
    _print_table(
        [
            "run_label",
            "target",
            "config_name",
            "mean_error",
            "mean_abs_error",
            "rms_error",
            "mean_pull",
            "rms_pull",
            "max_abs_pull",
        ],
        [row for row in summaries if row["mean_error"] is not None],
    )
    print("\nPer-target fastest passing config")
    _print_table(
        [
            "target",
            "run_label",
            "config_name",
            "median_seconds",
            "mean_seconds",
            "speedup_vs_baseline",
        ],
        fastest_passing_by_target(summaries),
    )
    print("\nWarnings")
    warnings = warnings_for(summaries)
    if warnings:
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("- none")


def write_csv(path: str | Path, summaries: list[dict[str, Any]]) -> None:
    if not summaries:
        Path(path).write_text("")
        return
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_paths", nargs="+", help="overnight validation JSON files")
    parser.add_argument(
        "--label",
        action="append",
        dest="labels",
        help="run label for a JSON file; repeat once per input file",
    )
    parser.add_argument("--csv", dest="csv_path", help="optional aggregate CSV path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summaries = summarize_rows(load_labeled_rows(args.json_paths, args.labels))
    print_summary(summaries)
    if args.csv_path:
        write_csv(args.csv_path, summaries)


if __name__ == "__main__":
    main()
