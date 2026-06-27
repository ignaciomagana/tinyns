from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import random

from tinyns.result import NestedSamplingResult


def make_result() -> NestedSamplingResult:
    return NestedSamplingResult(
        samples_u=jnp.zeros((4, 2)),
        samples=jnp.arange(8, dtype=float).reshape(4, 2),
        logl=jnp.array([-3.0, -2.0, -1.0, -0.5]),
        logwt=jnp.array([-4.0, -2.0, -1.0, -0.25]),
        logz=-0.1,
        logzerr=0.01,
        ncall=10,
        nlive=4,
        ndim=2,
        message="ok",
        metadata={"status": "complete"},
    )


def test_weights_sum_to_one() -> None:
    result = make_result()

    assert np.isclose(float(jnp.sum(result.weights())), 1.0)


def test_resample_equal_returns_requested_shape() -> None:
    result = make_result()

    samples = result.resample_equal(random.PRNGKey(0), n=3)

    assert samples.shape == (3, result.ndim)


def test_posterior_ess_is_positive() -> None:
    result = make_result()

    assert result.posterior_ess() > 0.0


def test_summary_contains_logz() -> None:
    result = make_result()

    summary = result.summary()

    assert isinstance(summary, str)
    assert "logz" in summary


def test_to_dict_contains_expected_keys() -> None:
    result = make_result()

    data = result.to_dict()

    assert set(data) == {
        "samples_u",
        "samples",
        "logl",
        "logwt",
        "logz",
        "logzerr",
        "ncall",
        "nlive",
        "ndim",
        "success",
        "message",
        "metadata",
    }
    assert data["metadata"] == {"status": "complete"}
    assert data["metadata"] is not result.metadata


def test_summary_contains_replacement_information_when_present() -> None:
    result = make_result()
    result.metadata = {
        "mean_replacement_ncall": 2.5,
        "replacement_failures": 1,
    }

    summary = result.summary()

    assert "replacement mean ncall: 2.5" in summary
    assert "replacement failures: 1" in summary
