from __future__ import annotations

import json

import pytest
from benchmarks.bench_static import (
    build_payload,
    compute_rates,
    main,
    parse_args,
    summarize_results,
    validate_benchmark_args,
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
            "warmup": True,
            "success": True,
        },
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
    assert summaries[0]["max_repl_batches"] == 2.5
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
    assert payload["results"][0]["warmup"] is False


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
    assert "mean_replacement_batches" in row
    assert "max_replacement_batches" in row
    assert "bound_build_time_total" in row
    assert "bound_build_time_mean" in row
    assert "bound_build_time_max" in row
    assert "bound_build_count" in row
    assert "bound_log_volume_final" in row
    assert "bound_log_volume_mean" in row
    assert "bound_log_volume_min" in row
    assert "bound_log_volume_max" in row
    assert "bound_nellipsoids_mean" in row
    assert "bound_nellipsoids_max" in row
    assert "bound_nellipsoids_final" in row


def test_benchmark_parser_accepts_allow_unused_bound() -> None:
    args = parse_args(["--allow-unused-bound"])

    assert args.allow_unused_bound is True


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


def test_validate_benchmark_args_accepts_safe_replacement_chains_grid() -> None:
    args = parse_args(
        [
            "--samplers",
            "rwalk",
            "--kernel",
            "jax",
            "--replacement-chains-grid",
            "1",
            "4",
            "16",
            "--walks",
            "25",
            "--max-attempts",
            "10000",
        ]
    )

    validate_benchmark_args(args)

    assert args.max_attempts == 10000


def test_validate_benchmark_args_rejects_unsafe_replacement_chains_grid() -> None:
    args = parse_args(
        [
            "--samplers",
            "rwalk",
            "--kernel",
            "jax",
            "--replacement-chains-grid",
            "1024",
            "--walks",
            "25",
            "--max-attempts",
            "10000",
        ]
    )

    with pytest.raises(ValueError, match="25600") as exc_info:
        validate_benchmark_args(args)

    message = str(exc_info.value)
    assert "--max-attempts must be at least walks * max(replacement_chains)" in message
    assert "required=25600" in message
    assert "Try --max-attempts 25600 or larger" in message


def test_validate_benchmark_args_auto_max_attempts_uses_four_batches() -> None:
    args = parse_args(
        [
            "--samplers",
            "rwalk",
            "--kernel",
            "jax",
            "--replacement-chains-grid",
            "1024",
            "--walks",
            "25",
            "--max-attempts",
            "10000",
            "--auto-max-attempts",
        ]
    )

    validate_benchmark_args(args)

    assert args.max_attempts == 102400


def test_summarize_results_groups_replacement_chains_separately() -> None:
    rows = [
        {
            "target": "gaussian2d",
            "sampler": "rwalk",
            "kernel": "jax",
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
            "sampler": "rwalk",
            "kernel": "jax",
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
    from benchmarks.overnight_jax_validation import parse_args

    args = parse_args([])

    assert args.nlive <= 25
    assert args.maxiter <= 10
    assert args.include_bounds is False
    assert args.include_block is False


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
            "--output",
            str(output),
        ]
    )

    assert output.exists()
    rows = json.loads(output.read_text())
    assert rows
    for key in EXPECTED_KEYS:
        assert key in rows[0]


def test_summarize_overnight_jax_validation_prints_table(tmp_path, capsys) -> None:
    from benchmarks.summarize_overnight_jax_validation import main

    output = tmp_path / "overnight.json"
    output.write_text(
        json.dumps(
            [
                {
                    "target": "gaussian2d",
                    "config_name": "unbounded_isotropic_rwalk",
                    "seconds": 1.0,
                    "logz": -6.0,
                    "expected_logz": -5.99,
                    "replacement_failures": 0,
                    "success": True,
                }
            ]
        )
    )

    main([str(output)])

    captured = capsys.readouterr()
    assert (
        "target config mean_seconds mean_logz std_logz "
        "mean_abs_logz_error failures"
    ) in captured.out
    assert "gaussian2d unbounded_isotropic_rwalk" in captured.out
