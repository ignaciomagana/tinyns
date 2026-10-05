from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest
from jax import random

from tinyns.samplers import (
    _callable_leaves,
    _callable_specs,
    _evaluate_jax_batch,
    _make_rwalk_jax_kernel_cached,
)


def gaussian_loglike(theta):
    return -0.5 * jnp.sum(theta**2)


def identity_prior_transform(u):
    return u


def draw(
    key,
    loglike,
    prior_transform,
    logl_min,
    live_u,
    live_logl,
    ndim,
    *,
    walks,
    replacement_chains=1,
    scale=0.5,
    max_batches=1,
):
    """Run the cached rwalk kernel once and return its outputs as Python values."""
    kernel = _make_rwalk_jax_kernel_cached(
        *_callable_specs(loglike, prior_transform, ndim),
        ndim,
        walks,
        replacement_chains,
    )
    key, new_u, new_theta, new_logl, ncall, accepted, moves, proposals = kernel(
        key,
        jnp.asarray(logl_min),
        jnp.asarray(live_u),
        jnp.asarray(live_logl),
        jnp.asarray(scale),
        jnp.asarray(max_batches, dtype=jnp.int32),
        *_callable_leaves(loglike, prior_transform, ndim),
    )
    return {
        "u": new_u,
        "theta": new_theta,
        "logl": float(new_logl),
        "ncall": int(ncall),
        "accepted": bool(accepted),
        "moves": int(moves),
        "proposals": int(proposals),
    }


def test_rwalk_kernel_counts_walks_on_easy_target() -> None:
    walks = 5
    out = draw(
        random.PRNGKey(123),
        gaussian_loglike,
        identity_prior_transform,
        -math.inf,
        jnp.full((4, 2), 0.5),
        jnp.zeros(4),
        2,
        walks=walks,
        scale=0.01,
    )

    assert out["ncall"] == walks
    assert out["proposals"] == walks
    assert out["accepted"] is True
    assert math.isfinite(out["logl"])


def test_rwalk_kernel_batched_chains_report_move_acceptance() -> None:
    walks, replacement_chains = 5, 4
    out = draw(
        random.PRNGKey(124),
        gaussian_loglike,
        identity_prior_transform,
        -math.inf,
        jnp.full((8, 2), 0.5),
        jnp.zeros(8),
        2,
        walks=walks,
        replacement_chains=replacement_chains,
        scale=0.01,
        max_batches=5,
    )

    assert out["accepted"] is True
    # With several chains every proposal is evaluated, in or out of the cube.
    assert out["ncall"] == out["proposals"] == walks * replacement_chains
    assert 0 <= out["moves"] <= out["proposals"]


def test_rwalk_kernel_normalizes_scalar_one_dimensional_prior() -> None:
    def scalar_prior(u):
        return 2.0 * u[0] - 1.0

    def vector_loglike(theta):
        return -(theta[0] ** 2)

    live_u = jnp.asarray(((0.2,), (0.4,), (0.6,), (0.8,)))
    live_logl = -((2.0 * live_u[:, 0] - 1.0) ** 2)
    out = draw(
        random.PRNGKey(126),
        vector_loglike,
        scalar_prior,
        -math.inf,
        live_u,
        live_logl,
        1,
        walks=2,
        replacement_chains=2,
    )

    assert out["accepted"] is True
    assert out["ncall"] == 4
    assert out["u"].shape == (1,)
    assert out["theta"].shape == (1,)
    assert math.isfinite(out["logl"])


def test_rwalk_kernel_caches_unhashable_callable_instances() -> None:
    class UnhashablePrior:
        __hash__ = None

        def __call__(self, u):
            return 2.0 * u - 1.0

    class UnhashableLogLike:
        __hash__ = None

        def __call__(self, theta):
            return -jnp.sum(theta**2)

    prior = UnhashablePrior()
    loglike = UnhashableLogLike()
    _make_rwalk_jax_kernel_cached.cache_clear()
    first = _make_rwalk_jax_kernel_cached(*_callable_specs(loglike, prior, 2), 2, 2, 1)
    second = _make_rwalk_jax_kernel_cached(*_callable_specs(loglike, prior, 2), 2, 2, 1)

    assert first is second
    live_u = jnp.asarray(((0.2, 0.3), (0.4, 0.7), (0.8, 0.6), (0.5, 0.5)))
    live_logl = jax.vmap(lambda u: loglike(prior(u)))(live_u)
    out = draw(
        random.PRNGKey(127), loglike, prior, -math.inf, live_u, live_logl, 2, walks=2
    )

    assert out["accepted"] is True
    assert out["proposals"] == 2
    assert out["u"].shape == (2,)
    assert out["theta"].shape == (2,)
    assert math.isfinite(out["logl"])


def test_rwalk_kernel_seeds_above_threshold_succeed_in_first_batch() -> None:
    walks = 3
    live_u = jnp.asarray([[0.1], [0.95]])
    out = draw(
        random.PRNGKey(5),
        lambda theta: theta[0],
        identity_prior_transform,
        0.9,
        live_u,
        live_u[:, 0],
        1,
        walks=walks,
        scale=1e-6,
        max_batches=10,
    )

    assert out["accepted"] is True
    assert out["proposals"] == walks
    assert out["logl"] >= 0.9


def test_rwalk_kernel_exhausts_its_batches_on_an_impossible_threshold() -> None:
    walks, replacement_chains, max_batches = 5, 4, 3
    out = draw(
        random.PRNGKey(43),
        gaussian_loglike,
        identity_prior_transform,
        math.inf,
        jnp.full((8, 2), 0.5),
        jnp.zeros(8),
        2,
        walks=walks,
        replacement_chains=replacement_chains,
        scale=0.01,
        max_batches=max_batches,
    )

    assert out["accepted"] is False
    assert out["proposals"] == max_batches * walks * replacement_chains
    assert out["u"].shape == (2,)
    assert math.isfinite(out["logl"])  # the best point seen


def test_rwalk_kernel_keeps_unmoved_chain_as_seed_copy() -> None:
    """A chain that cannot move is kept as a copy of its seed."""
    live_u = jnp.asarray([[0.2, 0.2], [0.5, 0.5], [0.8, 0.8]])
    live_logl = jnp.asarray([0.0, 1.0, 2.0])

    def loglike(theta):
        # only the exact live points are inside the constraint
        hit = jnp.any(jnp.all(jnp.isclose(theta, live_u), axis=1))
        return jnp.where(hit, jnp.sum(theta) * 2.0 - 0.8 * 2.0 + 2.0, -10.0)

    out = draw(
        random.PRNGKey(0),
        loglike,
        identity_prior_transform,
        0.5,
        live_u,
        live_logl,
        2,
        walks=6,
        scale=0.3,
        max_batches=10,
    )

    assert out["accepted"] is True
    assert out["moves"] == 0
    assert bool(jnp.any(jnp.all(out["u"] == live_u[1:], axis=1)))  # a seed above
    assert out["logl"] > 0.5


def test_rwalk_kernel_rejects_cluster_swap_with_several_chains() -> None:
    specs = _callable_specs(gaussian_loglike, identity_prior_transform, 2)
    with pytest.raises(ValueError, match="cluster_swap"):
        _make_rwalk_jax_kernel_cached(*specs, 2, 5, 2, True)


def test_evaluate_jax_batch_scalar_functions_use_vmap() -> None:
    u_batch = jnp.asarray([[0.1, 0.2], [0.3, 0.4]])
    theta, logl = _evaluate_jax_batch(
        gaussian_loglike, identity_prior_transform, u_batch, 2
    )

    assert theta.shape == (2, 2)
    assert logl.shape == (2,)
    assert jnp.allclose(theta, u_batch)


def test_evaluate_jax_batch_shape_errors() -> None:
    u_batch = jnp.ones((3, 2))
    with pytest.raises(ValueError, match="prior_transform must return shape"):
        _evaluate_jax_batch(gaussian_loglike, lambda u: u[0], u_batch, 2)

    with pytest.raises(ValueError, match="loglike must return a scalar"):
        _evaluate_jax_batch(lambda theta: theta, lambda u: u, u_batch, 2)

