from __future__ import annotations

import jax.numpy as jnp
from validation.run_validation import insertion_rank_stats, main


class _FakeResult:
    def __init__(self, insertion_indices, metadata=None):
        self._insertion_indices = insertion_indices
        self.metadata = {} if metadata is None else metadata

    def insertion_indices(self):
        return jnp.asarray(self._insertion_indices, dtype=int)


def test_insertion_rank_stats_missing_indices_returns_none_stats() -> None:
    stats = insertion_rank_stats(_FakeResult([], {"insertion_index_nslots": 10}))

    assert stats == {
        "insertion_rank_count": 0,
        "insertion_rank_mean": None,
        "insertion_rank_std": None,
        "insertion_rank_mean_error": None,
        "insertion_rank_std_error": None,
        "insertion_rank_mean_z": None,
        "insertion_rank_std_ratio": None,
    }


def test_insertion_rank_stats_uniform_like_ranks_mean_near_half() -> None:
    stats = insertion_rank_stats(
        _FakeResult(list(range(10)) * 5, {"insertion_index_nslots": 10})
    )

    assert stats["insertion_rank_count"] == 50
    assert stats["insertion_rank_mean"] == 0.5
    assert abs(stats["insertion_rank_mean_error"]) < 1e-12
    assert stats["insertion_rank_std"] is not None
    assert stats["insertion_rank_std_error"] is not None
    assert abs(stats["insertion_rank_mean_z"]) < 1e-12
    assert stats["insertion_rank_std_ratio"] is not None


def test_insertion_rank_stats_biased_ranks_have_large_mean_z() -> None:
    stats = insertion_rank_stats(
        _FakeResult([8, 9] * 50, {"insertion_index_nslots": 10})
    )

    assert stats["insertion_rank_mean_z"] is not None
    assert stats["insertion_rank_mean_z"] > 10.0


def test_validation_parser_accepts_replacement_chains() -> None:
    from validation.run_validation import parse_args

    args = parse_args(["--replacement-chains", "4"])

    assert args.replacement_chains == 4


def test_validation_cli_smoke_writes_replacement_batch_fields(tmp_path) -> None:
    output = tmp_path / "validation.json"

    main([
        "--targets",
        "gaussian2d",
        "--samplers",
        "rwalk",
        "--kernel",
        "jax",
        "--replacement-chains",
        "2",
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
    ])

    payload = __import__("json").loads(output.read_text())
    rows = payload["results"]
    assert len(rows) == 1
    assert "replacement_batch_ncall" in rows[0]
    assert "replacement_mean_batches" in rows[0]
