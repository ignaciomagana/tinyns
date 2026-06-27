from __future__ import annotations

import jax.numpy as jnp
from validation.run_validation import insertion_rank_stats


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
