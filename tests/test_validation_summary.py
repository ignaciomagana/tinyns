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
