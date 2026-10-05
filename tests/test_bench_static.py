from __future__ import annotations

import json

from benchmarks.bench_static import (
    build_payload,
    compute_rates,
    main,
    parse_args,
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
            "seconds": 2.0,
            "iterations_per_second": 5.0,
            "likelihood_calls_per_second": 50.0,
            "ncall": 100,
            "mean_replacement_ncall": 3.0,
            "repl_batches": 1.0,
            "max_repl_batches": 1.5,
            "warmup": True,
            "success": True,
        },
        {
            "target": "gaussian2d",
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
    assert summaries[0]["max_repl_batches"] == 2.5
    assert summaries[0]["success_fraction"] == 0.5


def test_build_payload() -> None:
    results = [{"target": "gaussian2d"}]
    summaries = [{"target": "gaussian2d", "nruns": 1}]

    assert build_payload(results, summaries) == {
        "results": results,
        "summaries": summaries,
    }


def test_cli_smoke_writes_json(tmp_path) -> None:
    output = tmp_path / "bench.json"

    main(
        [
            "--targets",
            "gaussian2d",
            "--seeds",
            "0",
            "--nlive",
            "20",
            "--dlogz",
            "10",
            "--walks",
            "5",
            "--block-size",
            "8",
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text())
    assert len(payload["results"]) == 1
    row = payload["results"][0]
    assert row["warmup"] is False
    assert row["success"] is True
    assert row["block_size"] == 8
    assert row["walks"] == 5
    assert row["replacement_chains"] == 1
    assert row["replacement_batch_ncall"] == 5
    for key in (
        "repl_batches",
        "max_repl_batches",
        "mean_replacement_batches",
        "max_replacement_batches",
        "compile_s",
        "mean_ms_per_call",
    ):
        assert key in row
    assert payload["summaries"][0]["block_size"] == 8


def test_benchmark_parser_accepts_replacement_chains() -> None:
    args = parse_args(["--replacement-chains", "4"])

    assert args.replacement_chains == 4


def test_benchmark_parser_accepts_warmup_options() -> None:
    args = parse_args(["--warmup-runs", "1", "--discard-warmup"])

    assert args.warmup_runs == 1
    assert args.discard_warmup is True


def test_benchmark_parser_accepts_replacement_chains_grid() -> None:
    args = parse_args(["--replacement-chains-grid", "1", "4", "16"])

    assert args.replacement_chains_grid == [1, 4, 16]


def test_summarize_results_groups_replacement_chains_separately() -> None:
    rows = [
        {
            "target": "gaussian2d",
            "replacement_chains": 1,
            "seconds": 4.0,
            "iterations_per_second": 5.0,
            "likelihood_calls_per_second": 50.0,
            "ncall": 100,
            "mean_replacement_ncall": 10.0,
            "mean_replacement_batches": 1.0,
            "max_replacement_batches": 2.0,
            "success": True,
            "warmup": False,
        },
        {
            "target": "gaussian2d",
            "replacement_chains": 4,
            "seconds": 2.0,
            "iterations_per_second": 20.0,
            "likelihood_calls_per_second": 100.0,
            "ncall": 200,
            "mean_replacement_ncall": 20.0,
            "mean_replacement_batches": 1.5,
            "max_replacement_batches": 3.0,
            "success": True,
            "warmup": False,
        },
    ]

    summaries = summarize_results(rows)

    by_chains = {row["replacement_chains"]: row for row in summaries}
    assert sorted(by_chains) == [1, 4]
    assert by_chains[1]["relative_speedup_vs_chains1"] == 1.0
    assert by_chains[4]["relative_speedup_vs_chains1"] == 2.0
    assert by_chains[4]["relative_iter_s_vs_chains1"] == 4.0


def test_overnight_jax_validation_parser_defaults_are_safe() -> None:
    from benchmarks.overnight_jax_validation import build_configs, parse_args

    args = parse_args([])

    assert args.nlive <= 25
    assert args.maxiter <= 10
    assert args.block_sizes == [32]
    assert [config.name for config in build_configs(args)] == ["live_cov_B32"]


def test_overnight_jax_validation_quick_writes_expected_keys(tmp_path) -> None:
    from benchmarks.overnight_jax_validation import EXPECTED_KEYS, main

    output = tmp_path / "overnight.json"

    main(
        [
            "--quick",
            "--targets",
            "gaussian2d",
            "--seeds",
            "0",
            "--nlive",
            "20",
            "--maxiter",
            "5",
            "--block-sizes",
            "1",
            "4",
            "--output",
            str(output),
        ]
    )

    assert output.exists()
    rows = json.loads(output.read_text())
    assert [row["config_name"] for row in rows] == ["live_cov_B1", "live_cov_B4"]
    for key in EXPECTED_KEYS:
        assert key in rows[0]


def test_summarize_overnight_jax_validation_prints_tables_and_csv(
    tmp_path, capsys
) -> None:
    from benchmarks.summarize_overnight_jax_validation import main

    no_block = tmp_path / "overnight_jax_validation_B1.json"
    block = tmp_path / "overnight_jax_validation_B16.json"
    csv_path = tmp_path / "summary.csv"
    no_block.write_text(
        json.dumps(
            [
                {
                    "target": "gaussian2d",
                    "config_name": "live_cov_B1",
                    "seconds": 4.0,
                    "ncall": 40,
                    "niter": 10,
                    "logz": -5.0,
                    "logzerr": 0.5,
                    "expected_logz": -6.0,
                    "final_delta_logz": 0.2,
                    "dlogz": 10.0,
                    "replacement_failures": 0,
                    "success": True,
                },
                {
                    "target": "gaussian2d",
                    "config_name": "live_cov_B1",
                    "seconds": 2.0,
                    "ncall": 20,
                    "niter": 8,
                    "logz": -6.5,
                    "logzerr": 0.5,
                    "expected_logz": -6.0,
                    "final_delta_logz": 0.2,
                    "dlogz": 10.0,
                    "replacement_failures": 0,
                    "success": False,
                },
                {
                    "target": "ring2d",
                    "config_name": "live_cov_B1",
                    "seconds": 1.0,
                    "ncall": 10,
                    "niter": 4,
                    "logz": -1.0,
                    "logzerr": 0.0,
                    "expected_logz": None,
                    "replacement_failures": 0,
                    "success": True,
                },
            ]
        )
    )
    block.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "target": "gaussian2d",
                        "config_name": "live_cov_B16",
                        "seconds": 1.0,
                        "ncall": 10,
                        "niter": 6,
                        "logz": -6.0,
                        "logzerr": 0.25,
                        "expected_logz": -6.0,
                        "final_delta_logz": 2.0,
                        "dlogz": 10.0,
                        "replacement_failures": 1,
                        "success": True,
                    }
                ]
            }
        )
    )

    main([str(no_block), str(block), "--csv", str(csv_path)])

    captured = capsys.readouterr()
    assert "Overall by file/target/config" in captured.out
    assert "Accuracy on analytic targets" in captured.out
    assert "Per-target fastest passing config" in captured.out
    assert "B1" in captured.out
    assert "B16" in captured.out
    assert "success_rate" in captured.out
    assert "0.5" in captured.out
    assert "replacement_failures_total=1" in captured.out
    assert "rms_pull" in captured.out
    assert "ring2d" in captured.out
    csv_text = csv_path.read_text()
    assert "run_label,target,config_name" in csv_text
    assert "B1,gaussian2d,live_cov_B1" in csv_text
