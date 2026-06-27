from __future__ import annotations

import json

from benchmarks.bench_static import (
    build_payload,
    compute_rates,
    main,
    summarize_results,
)


def test_compute_rates() -> None:
    iter_per_s, ncall_per_s = compute_rates(niter=10, ncall=100, seconds=2.0)

    assert iter_per_s == 5.0
    assert ncall_per_s == 50.0


def test_summarize_results_groups_means_and_success_fraction() -> None:
    rows = [
        {
            "target": "gaussian2d",
            "sampler": "rwalk",
            "seconds": 2.0,
            "iterations_per_second": 5.0,
            "likelihood_calls_per_second": 50.0,
            "ncall": 100,
            "mean_replacement_ncall": 3.0,
            "repl_batches": 1.0,
            "max_repl_batches": 1.5,
            "success": True,
        },
        {
            "target": "gaussian2d",
            "sampler": "rwalk",
            "seconds": 4.0,
            "iterations_per_second": 10.0,
            "likelihood_calls_per_second": 100.0,
            "ncall": 200,
            "mean_replacement_ncall": 5.0,
            "repl_batches": 2.0,
            "max_repl_batches": 2.5,
            "success": False,
        },
    ]

    summaries = summarize_results(rows)

    assert len(summaries) == 1
    assert summaries[0]["mean_seconds"] == 3.0
    assert summaries[0]["mean_iter_per_s"] == 7.5
    assert summaries[0]["mean_ncall_per_s"] == 75.0
    assert summaries[0]["mean_ncall"] == 150.0
    assert summaries[0]["mean_repl_ncall"] == 4.0
    assert summaries[0]["mean_repl_batches"] == 1.5
    assert summaries[0]["mean_max_repl_batches"] == 2.0
    assert summaries[0]["success_fraction"] == 0.5


def test_build_payload() -> None:
    results = [{"target": "gaussian2d"}]
    summaries = [{"target": "gaussian2d", "nruns": 1}]

    assert build_payload(results, summaries) == {
        "results": results,
        "summaries": summaries,
    }


def test_cli_smoke_tiny_settings_writes_json(tmp_path) -> None:
    output = tmp_path / "bench.json"

    main(
        [
            "--targets",
            "gaussian2d",
            "--samplers",
            "prior",
            "--seeds",
            "0",
            "--nlive",
            "20",
            "--maxiter",
            "5",
            "--output",
            str(output),
        ]
    )

    assert output.exists()
    payload = json.loads(output.read_text())
    assert "results" in payload
    assert len(payload["results"]) == 1


def test_cli_smoke_jax_rwalk_success_writes_json(tmp_path) -> None:
    output = tmp_path / "bench_jax.json"

    main(
        [
            "--targets",
            "gaussian2d",
            "--samplers",
            "rwalk",
            "--kernel",
            "jax",
            "--seeds",
            "0",
            "--nlive",
            "20",
            "--dlogz",
            "10",
            "--walks",
            "5",
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text())
    assert len(payload["results"]) == 1
    row = payload["results"][0]
    assert row["kernel"] == "jax"
    assert row["success"] is True
    assert "replacement_chains" in row
    assert "replacement_batch_ncall" in row
    assert "repl_batches" in row
    assert "max_repl_batches" in row


def test_benchmark_parser_accepts_replacement_chains() -> None:
    from benchmarks.bench_static import parse_args

    args = parse_args(["--replacement-chains", "4"])

    assert args.replacement_chains == 4
