from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from tinyns import NestedSampler, NestedSamplingResult, run_static_nested


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
        "run_static_nested",
    ]
    assert tinyns.NestedSampler is NestedSampler
    assert tinyns.NestedSamplingResult is NestedSamplingResult
    assert tinyns.run_static_nested is run_static_nested
    assert tinyns.LogZBootstrap is tinyns.result.LogZBootstrap


def test_nested_sampler_stores_configuration() -> None:
    with pytest.warns(UserWarning, match="bootstrap"):
        sampler = NestedSampler(
            loglike,
            prior_transform,
            ndim=3,
            nlive=500,
            vectorized=True,
            sample="rwalk",
            max_attempts=123,
            walks=12,
            step_scale=0.2,
            bootstrap=10,
        )

    assert sampler.loglike is loglike
    assert sampler.prior_transform is prior_transform
    assert sampler.ndim == 3
    assert sampler.nlive == 500
    assert sampler.vectorized is True
    assert sampler.sample == "rwalk"
    assert sampler.max_attempts == 123
    # Unknown kwargs still stored for dynesty drop-in compatibility.
    assert sampler.kwargs == {"walks": 12, "step_scale": 0.2, "bootstrap": 10}


def test_nested_sampler_no_warning_for_known_kwargs(recwarn) -> None:
    sampler = NestedSampler(
        loglike,
        prior_transform,
        ndim=2,
        nlive=100,
        sample="rwalk",
        kernel="jax",
        walks=10,
        step_scale=0.1,
        min_accepts=1,
        rwalk_adaptive_step_scale=True,
        rwalk_target_accept=0.3,
    )

    unknown = [
        w
        for w in recwarn.list
        if issubclass(w.category, UserWarning)
        and "unknown keyword" in str(w.message)
    ]
    assert unknown == []
    assert sampler.kwargs["walks"] == 10


def test_nested_sampler_validates_configuration() -> None:
    with pytest.raises(ValueError, match="ndim"):
        NestedSampler(loglike, prior_transform, ndim=0)

    with pytest.raises(ValueError, match="nlive"):
        NestedSampler(loglike, prior_transform, ndim=3, nlive=0)

    with pytest.raises(ValueError, match="sample"):
        NestedSampler(loglike, prior_transform, ndim=3, sample="invalid")

    with pytest.raises(TypeError, match="loglike"):
        NestedSampler(None, prior_transform, ndim=3)  # type: ignore[arg-type]


@pytest.mark.parametrize("sample", ["slice", "rslice", "bound"])
def test_nested_sampler_rejects_removed_samplers(sample: str) -> None:
    with pytest.raises(ValueError, match=r"sample must be one of"):
        NestedSampler(loglike, prior_transform, ndim=3, sample=sample)


def test_run_static_nested_rejects_bound_sampler_mode() -> None:
    with pytest.raises(ValueError, match=r"sample must be one of"):
        run_static_nested(
            0,
            loglike,
            prior_transform,
            ndim=2,
            nlive=10,
            sample="bound",
            maxiter=1,
        )


def test_nested_sampler_vectorized_rwalk_raises_on_run() -> None:
    sampler = NestedSampler(
        lambda theta_batch: -0.5 * jnp.sum(theta_batch**2, axis=1),
        lambda u_batch: 2.0 * u_batch - 1.0,
        ndim=2,
        nlive=10,
        vectorized=True,
        sample="rwalk",
    )

    assert sampler.vectorized is True
    assert sampler.sample == "rwalk"
    with pytest.raises(
        NotImplementedError, match="vectorized rwalk is not implemented yet"
    ):
        sampler.run(key=np.array([4, 5], dtype=np.uint32), maxiter=1)


def test_nested_sampler_run_returns_result() -> None:
    sampler = NestedSampler(loglike, prior_transform, ndim=3, nlive=20, sample="prior")

    result = sampler.run(
        key=np.array([0, 0], dtype=np.uint32),
        dlogz=0.1,
        maxiter=50,
        progress=False,
    )

    assert isinstance(result, NestedSamplingResult)
    assert result.ndim == 3
    assert result.nlive == 20


def test_nested_sampler_run_constant_likelihood_gives_finite_logz() -> None:
    sampler = NestedSampler(lambda theta: 0.0, lambda u: u, ndim=2, nlive=30)

    result = sampler.run(
        key=np.array([1, 2], dtype=np.uint32),
        dlogz=0.05,
        maxiter=300,
    )

    assert math.isfinite(result.logz)
    assert jnp.isfinite(result.logz)


def test_nested_sampler_rwalk_gaussian_returns_finite_logz() -> None:
    sampler = NestedSampler(
        lambda theta: float(-0.5 * theta[0] ** 2 - 0.5 * math.log(2.0 * math.pi)),
        lambda u: 20.0 * u - 10.0,
        ndim=1,
        nlive=40,
        sample="rwalk",
        kernel="python",
        walks=5,
        step_scale=0.2,
        min_accepts=2,
    )

    result = sampler.run(
        key=np.array([2, 3], dtype=np.uint32),
        dlogz=0.1,
        maxiter=300,
    )

    assert math.isfinite(result.logz)
    assert jnp.isfinite(result.logz)
    assert result.metadata["sample"] == "rwalk"
    assert result.metadata["walks"] == 5
    assert result.metadata["step_scale"] == 0.2
    assert result.metadata["min_accepts"] == 2


def test_nested_sampler_rejects_invalid_min_accepts_on_run() -> None:
    sampler = NestedSampler(loglike, prior_transform, ndim=2, nlive=20, min_accepts=0)

    with pytest.raises(ValueError, match="min_accepts"):
        sampler.run(key=np.array([3, 4], dtype=np.uint32), maxiter=1)


def test_rwalk_adaptive_step_scale_validates_public_args() -> None:
    with pytest.raises(ValueError, match="rwalk_target_accept"):
        NestedSampler(
            loglike,
            prior_transform,
            ndim=2,
            sample="rwalk",
            kernel="jax",
            rwalk_adaptive_step_scale=True,
            rwalk_target_accept=1.0,
        )

    with pytest.raises(ValueError, match="rwalk_adaptive_step_scale"):
        NestedSampler(
            loglike,
            prior_transform,
            ndim=2,
            sample="rwalk",
            kernel="python",
            rwalk_adaptive_step_scale=True,
        )


def _jax_gaussian_loglike(theta):
    return -0.5 * jnp.sum(theta**2)


def _jax_box_prior(unit):
    return 10.0 * unit - 5.0


def _run_metadata(result) -> dict:
    keys = ("sample", "kernel", "rwalk_proposal", "walks", "step_scale")
    return {key: result.metadata[key] for key in keys}


def test_default_sampler_runs_live_cov_fast_path() -> None:
    sampler = NestedSampler(_jax_gaussian_loglike, _jax_box_prior, ndim=2, nlive=40)
    result = sampler.run(np.array([0, 7], dtype=np.uint32), maxiter=64)

    assert sampler.sample == "rwalk"
    assert sampler.kernel == "jax"
    assert _run_metadata(result) == {
        "sample": "rwalk",
        "kernel": "jax",
        "rwalk_proposal": "live-cov",
        "walks": 25,
        "step_scale": 0.5,
    }
    assert result.metadata["jax_block_size"] == 32
    assert result.metadata["rwalk_adaptive_step_scale"] is True
    assert math.isfinite(result.logz)


def test_run_static_nested_defaults_match_nested_sampler() -> None:
    key = np.array([0, 8], dtype=np.uint32)
    direct = run_static_nested(
        key, _jax_gaussian_loglike, _jax_box_prior, ndim=2, nlive=30, maxiter=32
    )
    facade = NestedSampler(_jax_gaussian_loglike, _jax_box_prior, ndim=2, nlive=30)
    via_sampler = facade.run(key, maxiter=32)

    assert _run_metadata(direct) == _run_metadata(via_sampler)
    assert direct.metadata["jax_block_size"] == 32
    np.testing.assert_allclose(direct.logl, via_sampler.logl)


@pytest.mark.parametrize("ndim, walks", [(1, 12), (2, 25), (4, 25), (5, 30), (10, 60)])
def test_default_walks_is_max_25_or_six_ndim(ndim: int, walks: int) -> None:
    sampler = NestedSampler(_jax_gaussian_loglike, _jax_box_prior, ndim=ndim)

    assert sampler._checkpoint_config()["walks"] == walks


def test_default_walks_reaches_run_metadata() -> None:
    result = run_static_nested(
        np.array([0, 9], dtype=np.uint32),
        _jax_gaussian_loglike,
        _jax_box_prior,
        ndim=5,
        nlive=20,
        maxiter=2,
        dlogz=0.0,
    )

    assert result.metadata["walks"] == 30


@pytest.mark.parametrize(
    "extra",
    [
        {"kernel": "python"},
        {"bound": "single", "rwalk_seed": "bound"},
        {"replacement_chain_schedule": (1, 2)},
    ],
)
def test_default_proposal_falls_back_to_isotropic(extra) -> None:
    sampler = NestedSampler(
        _jax_gaussian_loglike, _jax_box_prior, ndim=2, nlive=20, walks=5, **extra
    )
    config = sampler._checkpoint_config()
    assert config["rwalk_proposal"] == "isotropic"
    assert config["step_scale"] == 0.1
    assert config["jax_block_size"] == 1
    assert config["rwalk_adaptive_step_scale"] is False

    result = sampler.run(np.array([1, 9], dtype=np.uint32), maxiter=3, dlogz=0.0)
    assert result.metadata["rwalk_proposal"] == "isotropic"
    assert result.metadata["step_scale"] == 0.1


def test_run_static_nested_python_kernel_falls_back_to_isotropic() -> None:
    result = run_static_nested(
        np.array([2, 9], dtype=np.uint32),
        loglike,
        prior_transform,
        ndim=2,
        nlive=10,
        kernel="python",
        walks=3,
        maxiter=3,
        dlogz=0.0,
    )

    assert result.metadata["rwalk_proposal"] == "isotropic"
    assert result.metadata["jax_block_size"] == 1


def test_prior_sampler_defaults_to_python_kernel() -> None:
    sampler = NestedSampler(loglike, prior_transform, ndim=2, sample="prior")

    assert sampler.kernel == "python"


@pytest.mark.parametrize(
    "extra",
    [
        {"kernel": "python"},
        {"bound": "single", "rwalk_seed": "bound"},
        {"replacement_chain_schedule": (1, 2)},
    ],
)
def test_explicit_live_cov_in_unsupported_combination_raises(extra) -> None:
    with pytest.raises(NotImplementedError, match="live-cov"):
        NestedSampler(
            loglike,
            prior_transform,
            ndim=2,
            rwalk_proposal="live-cov",
            **extra,
        )


def test_run_static_nested_explicit_live_cov_python_kernel_raises() -> None:
    with pytest.raises(NotImplementedError, match="live-cov"):
        run_static_nested(
            np.array([3, 9], dtype=np.uint32),
            loglike,
            prior_transform,
            ndim=2,
            nlive=10,
            kernel="python",
            rwalk_proposal="live-cov",
            maxiter=1,
        )


def test_max_attempts_none_resolves_from_walks() -> None:
    import jax.numpy as jnp
    from jax import random

    from tinyns import NestedSampler, run_static_nested

    def loglike(theta):
        return -0.5 * jnp.sum(((theta - 0.5) / 0.1) ** 2)

    def prior_transform(u):
        return u

    sampler = NestedSampler(
        loglike, prior_transform, 3, nlive=20, max_attempts=None, walks=4000,
        replacement_chains=4,
    )
    assert sampler.max_attempts == 16_000
    result = run_static_nested(
        random.PRNGKey(0), loglike, prior_transform, 2, 20, max_attempts=None,
        maxiter=10,
    )
    assert len(result.logl) >= 10
