from __future__ import annotations

import json
import subprocess
import sys

from validation.compare_validation import compare_summaries


def _summary(**overrides):
    row = {
        "target": "gaussian2d",
        "sampler": "slice",
        "success_fraction": 1.0,
        "coverage_fraction": 0.5,
        "mean_abs_logz_error": 0.2,
        "max_abs_z_score": 2.5,
        "frac_abs_z_gt_2": 0.25,
        "mean_ncall": 100.0,
        "recommendation": "ok",
    }
    row.update(overrides)
    return row


def test_compare_summaries_computes_delta_coverage() -> None:
    rows = compare_summaries(
        {"base": [_summary()], "candidate": [_summary(coverage_fraction=0.75)]},
        "base",
    )

    assert rows[0]["delta_coverage"] == 0.25


def test_compare_summaries_computes_ncall_ratio() -> None:
    rows = compare_summaries(
        {
            "base": [_summary(mean_ncall=100.0)],
            "candidate": [_summary(mean_ncall=150.0)],
        },
        "base",
    )

    assert rows[0]["ncall_ratio"] == 1.5


def test_compare_summaries_missing_metric_fields_do_not_crash() -> None:
    rows = compare_summaries(
        {
            "base": [_summary()],
            "candidate": [
                {
                    "target": "gaussian2d",
                    "sampler": "slice",
                }
            ],
        },
        "base",
    )

    assert rows[0]["candidate_coverage"] is None
    assert rows[0]["delta_coverage"] is None


def test_verdict_better_calibration_acceptable_cost() -> None:
    rows = compare_summaries(
        {
            "base": [_summary(frac_abs_z_gt_2=0.4, mean_abs_logz_error=0.2)],
            "candidate": [
                _summary(frac_abs_z_gt_2=0.1, mean_abs_logz_error=0.2, mean_ncall=150.0)
            ],
        },
        "base",
    )

    assert rows[0]["verdict"] == "better calibration, acceptable cost"


def test_verdict_better_calibration_expensive() -> None:
    rows = compare_summaries(
        {
            "base": [_summary(frac_abs_z_gt_2=0.4, mean_abs_logz_error=0.2)],
            "candidate": [
                _summary(frac_abs_z_gt_2=0.1, mean_abs_logz_error=0.2, mean_ncall=250.0)
            ],
        },
        "base",
    )

    assert rows[0]["verdict"] == "better calibration, expensive"


def test_verdict_worse_calibration_for_larger_z_outlier_fraction() -> None:
    rows = compare_summaries(
        {
            "base": [_summary(frac_abs_z_gt_2=0.1)],
            "candidate": [_summary(frac_abs_z_gt_2=0.3)],
        },
        "base",
    )

    assert rows[0]["verdict"] == "worse calibration"


def test_cli_label_count_mismatch_exits_with_error(tmp_path) -> None:
    path = tmp_path / "validation.json"
    path.write_text(json.dumps({"results": []}))

    result = subprocess.run(
        [
            sys.executable,
            "validation/compare_validation.py",
            str(path),
            "--labels",
            "one",
            "two",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "--labels must provide one label per JSON file" in result.stderr


def test_cli_can_write_output_json(tmp_path) -> None:
    base = tmp_path / "base.json"
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "comparison.json"
    payload = {
        "results": [
            {
                "target": "gaussian2d",
                "sampler": "slice",
                "success": True,
                "logz_error": 0.1,
                "logzerr": 0.2,
                "z_score": 0.5,
                "ncall": 10,
                "warnings": [],
                "replacement_failures": 0,
            }
        ]
    }
    base.write_text(json.dumps(payload))
    candidate.write_text(json.dumps(payload))

    result = subprocess.run(
        [
            sys.executable,
            "validation/compare_validation.py",
            str(base),
            str(candidate),
            "--labels",
            "base",
            "candidate",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    written = json.loads(output.read_text())
    assert written["labels"] == ["base", "candidate"]
    assert written["baseline"] == "base"
    assert written["comparisons"][0]["candidate"] == "candidate"
