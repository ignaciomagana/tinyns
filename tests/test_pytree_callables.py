"""Pytree callables: array leaves are jit arguments on the fast path."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

import tinyns.run as run_mod
from tinyns import NestedSampler
from tinyns.run import run_static_nested
from tinyns.samplers import (
    _callable_leaves,
    _combine_callable,
    _make_rwalk_jax_kernel,
    _partition_callable,
    draw_constrained_rwalk_jax,
)

NDIM = 2


def loglike_fn(data, theta):
    # min/max are exact under any reduction order and there is no division
    # (XLA rewrites x / const as x * (1 / const)), so a closure (constants) and
    # a Partial (arguments) evaluate to bit-identical values.
    center = jnp.min(data)
    precision = jnp.max(data)
    return -0.5 * jnp.sum(((theta - center) * precision) ** 2)


def prior_fn(bounds, u):
    return bounds[0] + (bounds[1] - bounds[0]) * u


def make_data(n):
    """Data with min 0.2 (the posterior mean) and max 10 (its precision)."""
    data = jnp.linspace(5.0, 0.3, n)
    return data.at[n // 3].set(0.2).at[n // 2].set(10.0)


BOUNDS = jnp.asarray(((-1.0, -1.0), (1.0, 1.0)))


def plain_loglike(theta):
    return -0.5 * jnp.sum(((theta - 0.2) / 0.1) ** 2)


def plain_prior(u):
    return 2.0 * u - 1.0


def make_closure_loglike(center, width):
    def loglike(theta):
        return -0.5 * jnp.sum(((theta - center) / width) ** 2)

    return loglike


def assert_same_result(a, b):
    assert a.logz == b.logz
    np.testing.assert_array_equal(a.samples, b.samples)
    np.testing.assert_array_equal(a.samples_u, b.samples_u)
    np.testing.assert_array_equal(a.logl, b.logl)
    assert a.ncall == b.ncall


def test_partition_plain_function_has_no_dynamic_leaves() -> None:
    closure = make_closure_loglike(jnp.asarray(0.2), 0.1)
    for fn in (plain_loglike, closure, lambda theta: theta[0]):
        dynamic, static = _partition_callable(fn)
        assert dynamic == ()
        assert _combine_callable(dynamic, static) is fn


def test_partition_round_trips_partial_leaves() -> None:
    data = make_data(10)
    np_data = np.linspace(0.0, 1.0, 5)
    fn = jax.tree_util.Partial(loglike_fn, data, scale=2.5)
    dynamic, static = _partition_callable(fn)
    assert len(dynamic) == 1 and dynamic[0] is data
    rebuilt = _combine_callable(dynamic, static)
    assert rebuilt.func is loglike_fn
    assert rebuilt.keywords == {"scale": 2.5}

    prior = jax.tree_util.Partial(prior_fn, np_data)
    assert len(_callable_leaves(fn, prior)) == 2
    assert _callable_leaves(plain_loglike, plain_prior) == ()
    with pytest.raises(ValueError, match="dynamic callable leaves"):
        _combine_callable((), static)


def test_plain_function_and_equivalent_closure_are_bit_identical() -> None:
    # A plain function and a closure both flatten to [fn]: neither has array
    # leaves, so the kernels take no extra arguments and trace exactly as on
    # the parent branch (whose fixed-seed output this matches bit for bit).
    closure = make_closure_loglike(0.2, 0.1)
    runs = [
        run_static_nested(random.PRNGKey(7), fn, plain_prior, NDIM, 50, maxiter=160)
        for fn in (plain_loglike, closure)
    ]
    assert_same_result(*runs)


@pytest.mark.parametrize("jax_block_size", [None, 1])
def test_partial_loglike_with_large_array_matches_closure(jax_block_size) -> None:
    data = make_data(200_000)
    partial_loglike = jax.tree_util.Partial(loglike_fn, data)
    partial_prior = jax.tree_util.Partial(prior_fn, BOUNDS)

    def closure_loglike(theta):
        return loglike_fn(data, theta)

    def closure_prior(u):
        return prior_fn(BOUNDS, u)

    kwargs = {"maxiter": 96, "dlogz": 0.0}
    if jax_block_size is not None:
        kwargs["jax_block_size"] = jax_block_size
    partial_result = run_static_nested(
        random.PRNGKey(11), partial_loglike, partial_prior, NDIM, 40, **kwargs
    )
    closure_result = run_static_nested(
        random.PRNGKey(11), closure_loglike, closure_prior, NDIM, 40, **kwargs
    )
    assert_same_result(partial_result, closure_result)


def test_draw_constrained_rwalk_jax_accepts_partial_callables() -> None:
    data = make_data(1000)
    loglike = jax.tree_util.Partial(loglike_fn, data)
    prior = jax.tree_util.Partial(prior_fn, BOUNDS)
    live_u = random.uniform(random.PRNGKey(3), (16, NDIM))
    live_logl = jax.vmap(lambda u: loglike(prior(u)))(live_u)
    results = [
        draw_constrained_rwalk_jax(
            random.PRNGKey(5),
            fn,
            pt,
            float(jnp.median(live_logl)),
            live_u,
            live_logl,
            NDIM,
            walks=10,
            proposal="live-cov",
        )
        for fn, pt in (
            (loglike, prior),
            (lambda t: loglike_fn(data, t), lambda u: prior_fn(BOUNDS, u)),
        )
    ]
    np.testing.assert_array_equal(results[0][1], results[1][1])
    assert results[0][3:] == results[1][3:]
    assert results[0][5] is True


def _lower_block_kernel_text(loglike, prior):
    kernel = run_mod._make_static_jax_rwalk_block_kernel(
        loglike, prior, NDIM, 5, 4, 8, "live-cov"
    )
    live_u = jnp.full((20, NDIM), 0.5)
    lowered = kernel.lower(
        random.PRNGKey(0),
        live_u,
        live_u,
        jnp.zeros((20,)),
        jnp.asarray(-jnp.inf),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(20, dtype=jnp.int32),
        jnp.asarray(0.5),
        jnp.asarray(1),
        jnp.asarray(10, dtype=jnp.int32),
        *_callable_leaves(loglike, prior),
    )
    return lowered.as_text()


def test_block_kernel_hlo_does_not_embed_partial_arrays() -> None:
    run_mod._make_static_jax_rwalk_block_kernel.cache_clear()
    _make_rwalk_jax_kernel.cache_clear()
    try:
        prior = jax.tree_util.Partial(prior_fn, BOUNDS)
        texts = {
            n: _lower_block_kernel_text(
                jax.tree_util.Partial(loglike_fn, make_data(n)), prior
            )
            for n in (1_000, 200_000)
        }
        # Only the shape annotations differ; constants would add ~200k values.
        assert abs(len(texts[200_000]) - len(texts[1_000])) < 2_000

        data = make_data(200_000)
        closure_text = _lower_block_kernel_text(
            lambda theta: loglike_fn(data, theta), prior
        )
        assert len(closure_text) > len(texts[200_000]) + 100_000
    finally:
        run_mod._make_static_jax_rwalk_block_kernel.cache_clear()
        _make_rwalk_jax_kernel.cache_clear()


def test_block_kernel_is_cached_per_partial_instance() -> None:
    run_mod._make_static_jax_rwalk_block_kernel.cache_clear()
    try:
        loglike = jax.tree_util.Partial(loglike_fn, make_data(100))
        first = run_mod._make_static_jax_rwalk_block_kernel(
            loglike, plain_prior, NDIM, 5, 4, 8, "live-cov"
        )
        second = run_mod._make_static_jax_rwalk_block_kernel(
            loglike, plain_prior, NDIM, 5, 4, 8, "live-cov"
        )
        assert first is second
    finally:
        run_mod._make_static_jax_rwalk_block_kernel.cache_clear()


def test_partial_loglike_checkpoint_resume_matches_uninterrupted(tmp_path) -> None:
    path = tmp_path / "partial.checkpoint.npz"
    loglike = jax.tree_util.Partial(loglike_fn, make_data(5_000))
    prior = jax.tree_util.Partial(prior_fn, BOUNDS)
    sampler = NestedSampler(loglike, prior, NDIM, nlive=30)

    full = sampler.run(21, maxiter=64, dlogz=0.0)
    sampler.run(21, maxiter=32, dlogz=0.0, checkpoint_path=path)
    resumed = sampler.resume(path, maxiter=64, dlogz=0.0)

    assert resumed.metadata["resumed_from_checkpoint"] is True
    assert_same_result(resumed, full)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kernel": "python"},
        {"sample": "prior"},
        {"bound": "single", "rwalk_seed": "bound"},
        {"replacement_chain_schedule": (1, 2)},
    ],
)
def test_partial_callables_run_on_closure_semantics_paths(kwargs) -> None:
    loglike = jax.tree_util.Partial(loglike_fn, make_data(100))
    prior = jax.tree_util.Partial(prior_fn, BOUNDS)
    result = run_static_nested(
        random.PRNGKey(2), loglike, prior, NDIM, 30, maxiter=20, **kwargs
    )
    assert np.isfinite(result.logz)
