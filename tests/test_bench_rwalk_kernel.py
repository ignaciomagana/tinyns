from __future__ import annotations

import json

import pytest
from benchmarks.bench_rwalk_kernel import main, parse_args, validate_args


def test_parser_accepts_replacement_chains_grid() -> None:
    args = parse_args(["--replacement-chains-grid", "1", "4", "16"])

    assert args.replacement_chains_grid == [1, 4, 16]


def test_parser_accepts_n_replacements() -> None:
    args = parse_args(["--n-replacements", "7"])

    assert args.n_replacements == 7


def test_parser_accepts_warmup_replacements() -> None:
    args = parse_args(["--warmup-replacements", "2"])

    assert args.warmup_replacements == 2


def test_validate_args_rejects_too_small_max_attempts() -> None:
    args = parse_args(
        [
            "--replacement-chains-grid",
            "1",
            "4",
            "--walks",
            "25",
            "--max-attempts",
            "99",
        ]
    )

    with pytest.raises(ValueError, match="required=100"):
        validate_args(args)


def test_cli_smoke_tiny_settings_writes_json(tmp_path) -> None:
    output = tmp_path / "kernel_smoke.json"

    main(
        [
            "--targets",
            "gaussian2d",
            "--replacement-chains-grid",
            "1",
            "4",
            "--walks",
            "5",
            "--nlive",
            "20",
            "--n-replacements",
            "3",
            "--warmup-replacements",
            "1",
            "--max-attempts",
            "100",
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text())
    assert len(payload["results"]) == 2
    assert payload["results"][0]["target"] == "gaussian2d"
    assert payload["results"][0]["kernel"] == "jax"
    assert payload["results"][0]["n_replacements"] == 3
    assert payload["results"][1]["replacement_chains"] == 4
