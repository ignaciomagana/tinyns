from __future__ import annotations

from validation.summarize_validation import summarize_results


def test_summarize_results_groups_and_computes_metrics() -> None:
    results = [
        {
            "target": "gaussian2d",
            "sampler": "prior",
            "success": True,
            "logz_error": 0.1,
            "logzerr": 0.2,
            "z_score": 0.5,
            "ncall": 10,
            "posterior_ess": 5.0,
            "replacement_mean_ncall": 1.0,
            "warnings": ["a"],
            "replacement_failures": 2,
        },
        {
            "target": "gaussian2d",
            "sampler": "prior",
            "success": False,
            "logz_error": 0.3,
            "logzerr": 0.2,
            "z_score": 1.5,
            "ncall": 30,
            "posterior_ess": 15.0,
            "replacement_mean_ncall": 3.0,
            "warnings": ["b", "c"],
            "replacement_failures": 1,
        },
        {
            "target": "banana2d",
            "sampler": "slice",
            "success": True,
            "logz_error": None,
            "logzerr": 0.4,
            "z_score": None,
            "ncall": 50,
            "posterior_ess": 25.0,
            "replacement_mean_ncall": 5.0,
            "warnings": [],
            "replacement_failures": 0,
        },
    ]

    summaries = summarize_results(results)
    by_group = {(row["target"], row["sampler"]): row for row in summaries}

    gaussian = by_group[("gaussian2d", "prior")]
    assert gaussian["nruns"] == 2
    assert gaussian["success_fraction"] == 0.5
    assert gaussian["mean_logz_error"] == 0.2
    assert gaussian["mean_abs_logz_error"] == 0.2
    assert gaussian["frac_abs_z_gt_2"] == 0.0
    assert gaussian["max_abs_z_score"] == 1.5
    assert gaussian["coverage_fraction"] == 0.5
    assert gaussian["mean_ncall"] == 20.0
    assert gaussian["mean_posterior_ess"] == 10.0
    assert gaussian["mean_replacement_ncall"] == 2.0
    assert gaussian["total_warnings"] == 3
    assert gaussian["total_replacement_failures"] == 3

    banana = by_group[("banana2d", "slice")]
    assert banana["mean_logz_error"] is None
    assert banana["coverage_fraction"] is None
    assert banana["total_warnings"] == 0


def test_summarize_results_recommends_poor_calibration_for_large_z() -> None:
    summaries = summarize_results(
        [
            {
                "target": "gaussian2d",
                "sampler": "slice",
                "success": True,
                "logz_error": 0.4,
                "logzerr": 0.1,
                "z_score": 4.0,
                "ncall": 10,
                "posterior_ess": 100.0,
                "warnings": [],
                "replacement_failures": 0,
            }
        ]
    )

    assert (
        summaries[0]["recommendation"]
        == "poor calibration: at least one >3 sigma evidence error"
    )


def test_summarize_results_recommends_ok_for_clean_group() -> None:
    summaries = summarize_results(
        [
            {
                "target": "gaussian2d",
                "sampler": "rwalk",
                "success": True,
                "logz_error": 0.02,
                "logzerr": 0.1,
                "z_score": 0.2,
                "ncall": 10,
                "posterior_ess": 100.0,
                "live_weight_fraction": 0.01,
                "max_weight_fraction": 0.01,
                "posterior_weight_entropy_fraction": 0.99,
                "warnings": [],
                "replacement_failures": 0,
            }
        ]
    )

    assert summaries[0]["recommendation"] == "ok"


def test_summarize_results_missing_diagnostics_do_not_crash() -> None:
    summaries = summarize_results(
        [
            {
                "target": "gaussian2d",
                "sampler": "prior",
                "success": True,
                "logz_error": 0.1,
                "logzerr": 0.2,
                "z_score": 0.5,
                "ncall": 10,
                "posterior_ess": 5.0,
                "warnings": [],
                "replacement_failures": 0,
            }
        ]
    )

    assert summaries[0]["mean_live_weight_fraction"] is None
    assert summaries[0]["mean_max_weight_fraction"] is None
    assert summaries[0]["mean_entropy_fraction"] is None
