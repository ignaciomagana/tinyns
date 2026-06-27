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


def _median_or_none(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _recommendation(summary: dict) -> str:
    """Return a simple heuristic validation recommendation.

    This is a debugging aid for repeated validation runs, not a formal
    statistical test.
    """

    if summary["success_fraction"] < 1.0:
        return "failures occurred"
    if (summary.get("frac_abs_z_gt_3") or 0.0) > 0.0:
        return "poor calibration: at least one >3 sigma evidence error"
    if (summary.get("frac_abs_z_gt_2") or 0.0) > 0.25:
        return "possible under-estimated uncertainty or sampler bias"
    if (summary.get("mean_live_weight_fraction") or 0.0) > 0.25:
        return "try smaller dlogz or more live points"
    if (summary.get("mean_max_weight_fraction") or 0.0) > 0.1:
        return "posterior weights concentrated; increase nlive"
    return "ok"


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
        abs_logz_errors = [abs(error) for error in logz_errors]
        abs_z_scores = [
            abs(row["z_score"]) for row in rows if row.get("z_score") is not None
        ]
        covered = [
            abs(row["logz_error"]) <= row["logzerr"]
            for row in rows
            if row.get("logz_error") is not None and row.get("logzerr") is not None
        ]
        warnings = sum(len(row.get("warnings", [])) for row in rows)
        failures = sum(int(row.get("replacement_failures") or 0) for row in rows)
        summary = {
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
            "mean_abs_logz_error": _mean_or_none(abs_logz_errors),
            "median_abs_logz_error": _median_or_none(abs_logz_errors),
            "max_abs_z_score": max(abs_z_scores) if abs_z_scores else None,
            "frac_abs_z_gt_1": (
                float(np.mean([score > 1.0 for score in abs_z_scores]))
                if abs_z_scores
                else None
            ),
            "frac_abs_z_gt_2": (
                float(np.mean([score > 2.0 for score in abs_z_scores]))
                if abs_z_scores
                else None
            ),
            "frac_abs_z_gt_3": (
                float(np.mean([score > 3.0 for score in abs_z_scores]))
                if abs_z_scores
                else None
            ),
            "mean_mean_error_norm": _mean_or_none(
                [
                    row["mean_error_norm"]
                    for row in rows
                    if row.get("mean_error_norm") is not None
                ]
            ),
            "mean_cov_error_frobenius": _mean_or_none(
                [
                    row["cov_error_frobenius"]
                    for row in rows
                    if row.get("cov_error_frobenius") is not None
                ]
            ),
            "mean_live_weight_fraction": _mean_or_none(
                [
                    row["live_weight_fraction"]
                    for row in rows
                    if row.get("live_weight_fraction") is not None
                ]
            ),
            "mean_max_weight_fraction": _mean_or_none(
                [
                    row["max_weight_fraction"]
                    for row in rows
                    if row.get("max_weight_fraction") is not None
                ]
            ),
            "mean_entropy_fraction": _mean_or_none(
                [
                    row["posterior_weight_entropy_fraction"]
                    for row in rows
                    if row.get("posterior_weight_entropy_fraction") is not None
                ]
            ),
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
        summary["recommendation"] = _recommendation(summary)
        summaries.append(summary)
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
        "frac_abs_z_gt_2",
        "max_abs_z_score",
        "mean_ncall",
        "mean_posterior_ess",
        "recommendation",
    ]
    labels = {
        "success_fraction": "success",
        "mean_logz_error": "mean_err",
        "std_logz_error": "std_err",
        "mean_logzerr": "mean_logzerr",
        "coverage_fraction": "covg1",
        "frac_abs_z_gt_2": "frac|z|>2",
        "max_abs_z_score": "max|z|",
        "mean_ncall": "mean_ncall",
        "mean_posterior_ess": "mean_ess",
    }
    widths = {
        col: max(len(labels.get(col, col)), *(len(_fmt(row[col])) for row in summaries))
        for col in columns
    }
    print("  ".join(labels.get(col, col).ljust(widths[col]) for col in columns))
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
