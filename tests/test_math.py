from __future__ import annotations

import math

import numpy as np
import pytest

from tinyns.math import logdiffexp, logsumexp


def test_logsumexp_matches_numpy_for_simple_values() -> None:
    values = np.array([-3.0, -2.0, -1.0])

    assert logsumexp(values) == pytest.approx(np.log(np.sum(np.exp(values))))


def test_logsumexp_handles_empty_input() -> None:
    assert logsumexp([]) == -math.inf


def test_logsumexp_handles_all_negative_infinity() -> None:
    assert logsumexp([-math.inf, -math.inf]) == -math.inf


def test_logdiffexp_matches_direct_computation() -> None:
    left = math.log(5.0)
    right = math.log(2.0)

    assert logdiffexp(left, right) == pytest.approx(math.log(3.0))


def test_logdiffexp_rejects_negative_result() -> None:
    with pytest.raises(ValueError, match="right <= left"):
        logdiffexp(0.0, 1.0)
