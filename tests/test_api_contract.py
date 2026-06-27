from __future__ import annotations

import numpy as np
import pytest

from tinyns import NestedSampler, NestedSamplingResult


def loglike(theta: np.ndarray) -> float:
    return float(-0.5 * np.dot(theta, theta))


def prior_transform(unit: np.ndarray) -> np.ndarray:
    return 2.0 * unit - 1.0


def test_public_exports() -> None:
    import tinyns

    assert tinyns.__all__ == ["NestedSampler", "NestedSamplingResult"]
    assert tinyns.NestedSampler is NestedSampler
    assert tinyns.NestedSamplingResult is NestedSamplingResult


def test_nested_sampler_stores_configuration() -> None:
    sampler = NestedSampler(
        loglike,
        prior_transform,
        ndim=3,
        nlive=500,
        vectorized=True,
        sample="prior",
        max_attempts=123,
        bootstrap=10,
    )

    assert sampler.loglike is loglike
    assert sampler.prior_transform is prior_transform
    assert sampler.ndim == 3
    assert sampler.nlive == 500
    assert sampler.vectorized is True
    assert sampler.sample == "prior"
    assert sampler.max_attempts == 123
    assert sampler.kwargs == {"bootstrap": 10}


def test_nested_sampler_validates_configuration() -> None:
    with pytest.raises(ValueError, match="ndim"):
        NestedSampler(loglike, prior_transform, ndim=0)

    with pytest.raises(ValueError, match="nlive"):
        NestedSampler(loglike, prior_transform, ndim=3, nlive=0)

    with pytest.raises(ValueError, match="sample"):
        NestedSampler(loglike, prior_transform, ndim=3, sample="slice")

    with pytest.raises(TypeError, match="loglike"):
        NestedSampler(None, prior_transform, ndim=3)  # type: ignore[arg-type]


def test_run_is_declared_but_not_implemented_yet() -> None:
    sampler = NestedSampler(loglike, prior_transform, ndim=3, nlive=500)

    with pytest.raises(NotImplementedError, match="not implemented"):
        sampler.run(key=np.array([0, 0]), dlogz=0.1, maxiter=10, progress=False)
