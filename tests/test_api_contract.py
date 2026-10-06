from __future__ import annotations

import dataclasses
import math

import jax.numpy as jnp
import numpy as np
import pytest
from tests.helpers import run_ns

from tinyns import NestedSampler, NestedSamplingResult, core, loop


def loglike(theta: np.ndarray) -> float:
    return float(-0.5 * np.dot(theta, theta))


def prior_transform(unit: np.ndarray) -> np.ndarray:
    return 2.0 * unit - 1.0


def test_public_exports() -> None:
    import tinyns

    assert tinyns.__all__ == [
        "NestedSampler",
        "NestedSamplingResult",
        "LogZBootstrap",
    ]
    assert tinyns.NestedSampler is NestedSampler
    assert tinyns.NestedSamplingResult is NestedSamplingResult
    assert tinyns.LogZBootstrap is tinyns.result.LogZBootstrap


def test_nested_sampler_stores_configuration() -> None:
    sampler = NestedSampler(
        loglike,
        prior_transform,
        ndim=3,
        nlive=500,
        walks=12,
        replacement_chains=2,
        block_size=8,
    )

    assert sampler.loglike is loglike
    assert sampler.prior_transform is prior_transform
    assert sampler.ndim == 3
    assert sampler.nlive == 500
    assert dataclasses.asdict(sampler._config) == {
        "ndim": 3,
        "nlive": 500,
        "walks": 12,
        "replacement_chains": 2,
        "block_size": 8,
        "cluster_swap": False,
    }


def test_nested_sampler_signature() -> None:
    import inspect

    params = inspect.signature(NestedSampler.__init__).parameters
    assert list(params) == [
        "self",
        "loglike",
        "prior_transform",
        "ndim",
        "nlive",
        "walks",
        "replacement_chains",
        "block_size",
        "cluster_swap",
        "kwargs",
    ]
    assert params["nlive"].default == 500
    assert params["walks"].default is None
    assert params["replacement_chains"].default == 1
    assert params["block_size"].default == 32
    assert params["cluster_swap"].default is None


def test_nested_sampler_rejects_unknown_kwargs() -> None:
    with pytest.raises(TypeError, match=r"'walk'; did you mean 'walks'\?"):
        NestedSampler(loglike, prior_transform, ndim=2, walk=10)

    with pytest.raises(TypeError, match="unexpected keyword argument 'bootstrap'$"):
        NestedSampler(loglike, prior_transform, ndim=2, bootstrap=10)

    # Removed options are not accepted either.
    with pytest.raises(TypeError, match="unexpected keyword argument 'bound'"):
        NestedSampler(loglike, prior_transform, ndim=2, bound="single")
    with pytest.raises(TypeError, match=r"did you mean 'block_size'\?"):
        NestedSampler(loglike, prior_transform, ndim=2, jax_block_size=8)


def test_nested_sampler_validates_configuration() -> None:
    with pytest.raises(ValueError, match="ndim"):
        NestedSampler(loglike, prior_transform, ndim=0)

    with pytest.raises(ValueError, match="nlive"):
        NestedSampler(loglike, prior_transform, ndim=3, nlive=0)

    with pytest.raises(TypeError, match="loglike"):
        NestedSampler(None, prior_transform, ndim=3)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "name, value",
    [
        ("walks", 0),
        ("replacement_chains", 0),
        ("replacement_chains", True),
        ("block_size", 0),
        ("block_size", 2.0),
    ],
)
def test_nested_sampler_rejects_non_positive_integer_options(name, value) -> None:
    with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
        NestedSampler(loglike, prior_transform, ndim=2, **{name: value})


def test_nested_sampler_run_constant_likelihood_gives_finite_logz() -> None:
    sampler = NestedSampler(lambda theta: 0.0, lambda u: u, ndim=2, nlive=30)

    result = sampler.run(
        key=np.array([1, 2], dtype=np.uint32),
        dlogz=0.05,
        maxiter=300,
    )

    assert isinstance(result, NestedSamplingResult)
    assert result.ndim == 2
    assert result.nlive == 30
    assert math.isfinite(result.logz)
    assert jnp.isfinite(result.logz)


def _jax_gaussian_loglike(theta):
    return -0.5 * jnp.sum(theta**2)


def _jax_box_prior(unit):
    return 10.0 * unit - 5.0


def _run_metadata(result) -> dict:
    keys = ("walks", "replacement_chains", "block_size", "scale_initial")
    return {key: result.metadata[key] for key in keys}


def test_default_sampler_runs_live_cov_block_path() -> None:
    sampler = NestedSampler(_jax_gaussian_loglike, _jax_box_prior, ndim=2, nlive=40)
    result = sampler.run(np.array([0, 7], dtype=np.uint32), maxiter=64)

    assert _run_metadata(result) == {
        "walks": 25,
        "replacement_chains": 1,
        "block_size": 32,
        "scale_initial": 0.5,
    }
    assert result.metadata["cluster_swap"] is True
    assert result.niter == 64
    assert math.isfinite(result.logz)


def test_loop_run_defaults_match_nested_sampler() -> None:
    key = np.array([0, 8], dtype=np.uint32)
    direct = loop.run(
        core.Config(2, 30), _jax_gaussian_loglike, _jax_box_prior, key, maxiter=32
    )
    facade = NestedSampler(_jax_gaussian_loglike, _jax_box_prior, ndim=2, nlive=30)
    via_sampler = facade.run(key, maxiter=32)

    assert _run_metadata(direct) == _run_metadata(via_sampler)
    assert direct.metadata["block_size"] == 32
    np.testing.assert_allclose(direct.logl, via_sampler.logl)


@pytest.mark.parametrize("ndim, walks", [(1, 12), (2, 25), (4, 25), (5, 30), (10, 60)])
def test_default_walks_is_max_25_or_six_ndim(ndim: int, walks: int) -> None:
    sampler = NestedSampler(_jax_gaussian_loglike, _jax_box_prior, ndim=ndim)

    assert dataclasses.asdict(sampler._config)["walks"] == walks


def test_default_walks_reaches_run_metadata() -> None:
    result = run_ns(
        np.array([0, 9], dtype=np.uint32),
        _jax_gaussian_loglike,
        _jax_box_prior,
        ndim=5,
        nlive=20,
        maxiter=2,
        dlogz=0.0,
    )

    assert result.metadata["walks"] == 30


def test_huge_walks_still_runs_one_batch_per_replacement() -> None:
    # walks * replacement_chains above the 10_000-call replacement budget
    # still runs one batch.
    result = run_ns(
        np.array([0, 10], dtype=np.uint32),
        _jax_gaussian_loglike,
        _jax_box_prior,
        ndim=2,
        nlive=20,
        walks=6000,
        replacement_chains=2,
        maxiter=2,
        dlogz=0.0,
    )

    assert result.success is False  # maxiter
    assert result.ncall == 20 + 2 * 6000 * 2  # one batch per replacement
