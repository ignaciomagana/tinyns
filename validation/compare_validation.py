"""Compare JSON outputs from repeated validation runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validation.summarize_validation import summarize_results


def _index_summary(summary: list[dict]) -> dict[tuple[str, str], dict]:
    return {(row["target"], row["sampler"]): row for row in summary}


def _delta(candidate: Any, baseline: Any) -> Any:
    if candidate is None or baseline is None:
        return None
    return candidate - baseline


def _ratio(candidate: Any, baseline: Any) -> float | None:
    if candidate is None or baseline in (None, 0):
        return None
    return candidate / baseline


def _is_lower(candidate: Any, baseline: Any) -> bool:
    return candidate is not None and baseline is not None and candidate < baseline


def _is_lower_or_equal(candidate: Any, baseline: Any) -> bool:
    return candidate is not None and baseline is not None and candidate <= baseline


def _is_greater(candidate: Any, baseline: Any) -> bool:
    return candidate is not None and baseline is not None and candidate > baseline


def _verdict(row: dict) -> str:
    """Return a simple comparison heuristic.

    This verdict is a compact debugging aid for validation experiments. It is
    not a statistical test and should not replace inspecting the raw summary
    metrics for the target and sampler.
    """

    if _is_lower(row["candidate_success"], row["baseline_success"]):
        return "worse: more failures"

    better_z_outliers = _is_lower(
        row["candidate_frac_abs_z_gt_2"], row["baseline_frac_abs_z_gt_2"]
    )
    no_worse_mean_error = _is_lower_or_equal(
        row["candidate_mean_abs_logz_error"], row["baseline_mean_abs_logz_error"]
    )
    better_calibration = better_z_outliers and no_worse_mean_error
    ncall_ratio = row["ncall_ratio"]

    if better_calibration and ncall_ratio is not None and ncall_ratio <= 2.0:
        return "better calibration, acceptable cost"
    if better_calibration and ncall_ratio is not None and ncall_ratio > 2.0:
        return "better calibration, expensive"
    if _is_greater(
        row["candidate_frac_abs_z_gt_2"], row["baseline_frac_abs_z_gt_2"]
    ) or _is_greater(
        row["candidate_mean_abs_logz_error"], row["baseline_mean_abs_logz_error"]
    ):
        return "worse calibration"
    if ncall_ratio is not None and ncall_ratio > 2.0:
        return "similar calibration, more expensive"
    return "similar"


def compare_summaries(
    named_summaries: dict[str, list[dict]], baseline_label: str
) -> list[dict]:
    """Compare validation summaries by ``(target, sampler)``."""

    if baseline_label not in named_summaries:
        raise ValueError(f"baseline label not found: {baseline_label}")

    indexed = {
        label: _index_summary(summary)
        for label, summary in named_summaries.items()
    }
    keys = sorted({key for summary in indexed.values() for key in summary})
    baseline_index = indexed[baseline_label]
    rows = []

    for target, sampler in keys:
        baseline = baseline_index.get((target, sampler), {})
        for candidate_label, candidate_index in indexed.items():
            if candidate_label == baseline_label:
                continue
            candidate = candidate_index.get((target, sampler), {})
            baseline_ncall = baseline.get("mean_ncall")
            candidate_ncall = candidate.get("mean_ncall")
            row = {
                "target": target,
                "sampler": sampler,
                "baseline": baseline_label,
                "candidate": candidate_label,
                "baseline_success": baseline.get("success_fraction"),
                "candidate_success": candidate.get("success_fraction"),
                "delta_success": _delta(
                    candidate.get("success_fraction"), baseline.get("success_fraction")
                ),
                "baseline_coverage": baseline.get("coverage_fraction"),
                "candidate_coverage": candidate.get("coverage_fraction"),
                "delta_coverage": _delta(
                    candidate.get("coverage_fraction"),
                    baseline.get("coverage_fraction"),
                ),
                "baseline_mean_abs_logz_error": baseline.get("mean_abs_logz_error"),
                "candidate_mean_abs_logz_error": candidate.get("mean_abs_logz_error"),
                "delta_mean_abs_logz_error": _delta(
                    candidate.get("mean_abs_logz_error"),
                    baseline.get("mean_abs_logz_error"),
                ),
                "baseline_max_abs_z_score": baseline.get("max_abs_z_score"),
                "candidate_max_abs_z_score": candidate.get("max_abs_z_score"),
                "delta_max_abs_z_score": _delta(
                    candidate.get("max_abs_z_score"), baseline.get("max_abs_z_score")
                ),
                "baseline_frac_abs_z_gt_2": baseline.get("frac_abs_z_gt_2"),
                "candidate_frac_abs_z_gt_2": candidate.get("frac_abs_z_gt_2"),
                "delta_frac_abs_z_gt_2": _delta(
                    candidate.get("frac_abs_z_gt_2"), baseline.get("frac_abs_z_gt_2")
                ),
                "baseline_mean_abs_insertion_rank_mean_error": baseline.get(
                    "mean_abs_insertion_rank_mean_error"
                ),
                "candidate_mean_abs_insertion_rank_mean_error": candidate.get(
                    "mean_abs_insertion_rank_mean_error"
                ),
                "delta_mean_abs_insertion_rank_mean_error": _delta(
                    candidate.get("mean_abs_insertion_rank_mean_error"),
                    baseline.get("mean_abs_insertion_rank_mean_error"),
                ),
                "baseline_mean_abs_insertion_rank_std_error": baseline.get(
                    "mean_abs_insertion_rank_std_error"
                ),
                "candidate_mean_abs_insertion_rank_std_error": candidate.get(
                    "mean_abs_insertion_rank_std_error"
                ),
                "delta_mean_abs_insertion_rank_std_error": _delta(
                    candidate.get("mean_abs_insertion_rank_std_error"),
                    baseline.get("mean_abs_insertion_rank_std_error"),
                ),
                "baseline_mean_ncall": baseline_ncall,
                "candidate_mean_ncall": candidate_ncall,
                "ncall_ratio": _ratio(candidate_ncall, baseline_ncall),
                "baseline_recommendation": baseline.get("recommendation"),
                "candidate_recommendation": candidate.get("recommendation"),
            }
            row["verdict"] = _verdict(row)
            rows.append(row)
    return rows


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def print_table(rows: list[dict]) -> None:
    columns = [
        "target",
        "sampler",
        "candidate",
        "delta_coverage",
        "delta_mean_abs_logz_error",
        "delta_frac_abs_z_gt_2",
        "delta_max_abs_z_score",
        "ncall_ratio",
        "verdict",
    ]
    labels = {
        "delta_coverage": "Δcovg",
        "delta_mean_abs_logz_error": "Δmean_abs_err",
        "delta_frac_abs_z_gt_2": "Δfrac|z|>2",
        "delta_max_abs_z_score": "Δmax|z|",
    }
    widths = {
        col: max(len(labels.get(col, col)), *(len(_fmt(row[col])) for row in rows))
        for col in columns
    }
    print("  ".join(labels.get(col, col).ljust(widths[col]) for col in columns))
    print("  ".join("-" * widths[col] for col in columns))
    for row in rows:
        print("  ".join(_fmt(row[col]).ljust(widths[col]) for col in columns))


def _labels_for_paths(paths: list[str], labels: list[str] | None) -> list[str]:
    if labels is None:
        return [Path(path).stem for path in paths]
    if len(labels) != len(paths):
        raise ValueError("--labels must provide one label per JSON file")
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_paths", nargs="+")
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    try:
        labels = _labels_for_paths(args.json_paths, args.labels)
        baseline_label = args.baseline or labels[0]
        named_summaries = {}
        for label, path in zip(labels, args.json_paths, strict=True):
            payload = json.loads(Path(path).read_text())
            named_summaries[label] = summarize_results(payload["results"])
        rows = compare_summaries(named_summaries, baseline_label)
    except ValueError as exc:
        parser.error(str(exc))

    print_table(rows)
    if args.output is not None:
        output = {
            "labels": labels,
            "baseline": baseline_label,
            "comparisons": rows,
        }
        Path(args.output).write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
