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


def test_summarize_results_includes_insertion_rank_fields() -> None:
    summaries = summarize_results(
        [
            {
                "target": "gaussian2d",
                "sampler": "slice",
                "success": True,
                "logz_error": None,
                "logzerr": 0.1,
                "z_score": None,
                "ncall": 10,
                "posterior_ess": 100.0,
                "insertion_rank_count": 20,
                "insertion_rank_mean": 0.55,
                "insertion_rank_std": 0.3,
                "insertion_rank_mean_error": 0.05,
                "insertion_rank_std_error": 0.01,
                "insertion_rank_mean_z": 2.0,
                "insertion_rank_std_ratio": 1.1,
                "warnings": [],
                "replacement_failures": 0,
            }
        ]
    )

    assert summaries[0]["mean_insertion_rank_count"] == 20.0
    assert summaries[0]["mean_insertion_rank_mean"] == 0.55
    assert summaries[0]["mean_abs_insertion_rank_mean_error"] == 0.05
    assert summaries[0]["mean_insertion_rank_std"] == 0.3
    assert summaries[0]["mean_abs_insertion_rank_std_error"] == 0.01
    assert summaries[0]["mean_abs_insertion_rank_mean_z"] == 2.0
    assert summaries[0]["max_abs_insertion_rank_mean_z"] == 2.0
    assert summaries[0]["mean_insertion_rank_std_ratio"] == 1.1


def test_recommendation_flags_large_insertion_rank_mean_error() -> None:
    summaries = summarize_results(
        [
            {
                "target": "ring2d",
                "sampler": "rwalk",
                "success": True,
                "logz_error": None,
                "logzerr": 0.1,
                "z_score": None,
                "ncall": 10,
                "posterior_ess": 100.0,
                "live_weight_fraction": 0.01,
                "max_weight_fraction": 0.01,
                "insertion_rank_count": 30,
                "insertion_rank_mean": 0.7,
                "insertion_rank_std": 0.29,
                "insertion_rank_mean_error": 0.2,
                "insertion_rank_std_error": 0.0,
                "warnings": [],
                "replacement_failures": 0,
            }
        ]
    )

    assert (
        summaries[0]["recommendation"]
        == "suspicious insertion ranks: possible constrained-sampler bias"
    )


def test_recommendation_flags_large_insertion_rank_std_error() -> None:
    summaries = summarize_results(
        [
            {
                "target": "ring2d",
                "sampler": "rwalk",
                "success": True,
                "logz_error": None,
                "logzerr": 0.1,
                "z_score": None,
                "ncall": 10,
                "posterior_ess": 100.0,
                "live_weight_fraction": 0.01,
                "max_weight_fraction": 0.01,
                "insertion_rank_count": 30,
                "insertion_rank_mean": 0.5,
                "insertion_rank_std": 0.5,
                "insertion_rank_mean_error": 0.0,
                "insertion_rank_std_error": 0.2,
                "warnings": [],
                "replacement_failures": 0,
            }
        ]
    )

    assert (
        summaries[0]["recommendation"]
        == "suspicious insertion-rank spread: possible constrained-sampler bias"
    )


def test_qualitative_posterior_summaries_work_without_expected_logz() -> None:
    summaries = summarize_results(
        [
            {
                "target": "ring2d",
                "sampler": "slice",
                "success": True,
                "logz_error": None,
                "logzerr": 0.1,
                "z_score": None,
                "ncall": 10,
                "posterior_ess": 100.0,
                "posterior_mean": [0.0, 0.1],
                "posterior_cov": [[1.0, 0.0], [0.0, 4.0]],
                "posterior_std": [1.0, 2.0],
                "warnings": [],
                "replacement_failures": 0,
            }
        ]
    )

    assert summaries[0]["mean_logz_error"] is None
    assert summaries[0]["coverage_fraction"] is None
    assert summaries[0]["mean_posterior_std"] == 1.5


def test_recommendation_flags_large_insertion_rank_z_bias() -> None:
    summaries = summarize_results(
        [
            {
                "target": "ring2d",
                "sampler": "rwalk",
                "success": True,
                "logz_error": None,
                "logzerr": 0.1,
                "z_score": None,
                "ncall": 10,
                "posterior_ess": 100.0,
                "live_weight_fraction": 0.01,
                "max_weight_fraction": 0.01,
                "insertion_rank_count": 3000,
                "insertion_rank_mean": 0.52,
                "insertion_rank_std": 0.29,
                "insertion_rank_mean_error": 0.02,
                "insertion_rank_std_error": 0.0,
                "insertion_rank_mean_z": 4.5,
                "insertion_rank_std_ratio": 1.0,
                "warnings": [],
                "replacement_failures": 0,
            }
        ]
    )

    assert (
        summaries[0]["recommendation"]
        == "statistically significant insertion-rank bias"
    )
