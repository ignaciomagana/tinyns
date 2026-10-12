"""Batched runs: a batch of keys, optionally with one dataset per lane, runs in
one compiled program and matches the separate runs."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.tree_util import Partial

from tinyns import NestedSampler, NestedSamplingResult, core
from tinyns.callables import _split_callables

X64 = bool(jax.config.jax_enable_x64)


def gauss(x):
    return -0.5 * jnp.sum(((x - 0.5) / 0.1) ** 2)


def identity(u):
    return u


def shifted_gauss(mu, sigma, x):
    return -0.5 * jnp.sum(((x - mu) / sigma) ** 2)


def scaled_prior(width, u):
    return width * u


def assert_matches(batched, single) -> None:
    """A lane equals the run of its key alone, up to float roundoff.

    vmap changes XLA's fusion and reduction order, so the lanes are not bit
    identical to the separate runs. In float64 the two agree to roundoff over
    the whole run; in float32 a roundoff difference can flip a Metropolis
    decision late in the run, after which the two are independent draws, so
    the start of the run must agree and the evidence must agree within error.
    """
    assert isinstance(batched, NestedSamplingResult)
    if X64:
        assert (batched.niter, batched.ncall) == (single.niter, single.ncall)
        np.testing.assert_allclose(batched.logz, single.logz, rtol=1e-9)
        np.testing.assert_allclose(batched.samples_u, single.samples_u, atol=1e-9)
        np.testing.assert_allclose(batched.samples, single.samples, atol=1e-9)
    else:
        start = 20 * batched.num_delete
        np.testing.assert_allclose(
            batched.logl[:start], single.logl[:start], rtol=1e-4, atol=1e-4
        )
        assert abs(batched.logz - single.logz) < 3 * single.logzerr


def test_batched_keys_match_separate_runs() -> None:
    sampler = NestedSampler(gauss, identity, 3, 60, num_delete=6)
    keys = jax.random.split(jax.random.key(7), 4)
    results = sampler.run(keys)
    assert isinstance(results, list) and len(results) == 4
    for key, result in zip(keys, results, strict=True):
        assert_matches(result, sampler.run(key))
    assert len({r.logz for r in results}) == 4  # independent lanes
    # Raw uint32 keys batch the same way, and k = 1 runs too.
    raw = jax.random.split(jax.random.PRNGKey(2), 2)
    k1 = NestedSampler(gauss, identity, 2, 40, num_delete=1)
    for key, result in zip(raw, k1.run(raw), strict=True):
        assert_matches(result, k1.run(key))


def test_batched_data_matches_separate_runs() -> None:
    """One compiled program serves many datasets (an SBC or mock campaign)."""
    mu = jnp.asarray([[0.3, 0.3], [0.5, 0.6], [0.7, 0.4], [0.5, 0.5]])
    sigma = jnp.asarray([0.05, 0.1, 0.08, 0.2])  # lanes stop at different steps
    width = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    loglike = Partial(shifted_gauss, mu, sigma[:, None])
    prior = Partial(scaled_prior, width[:, None])
    sampler = NestedSampler(loglike, prior, 2, 60, num_delete=6)
    keys = jax.random.split(jax.random.key(11), 4)
    results = sampler.run(keys, batched_data=True)
    assert len({r.niter for r in results}) > 1
    for i, result in enumerate(results):
        alone = NestedSampler(
            core._lane_callable(loglike, i), core._lane_callable(prior, i), 2, 60,
            num_delete=6,
        ).run(keys[i])
        assert_matches(result, alone)
        np.testing.assert_allclose(
            result.samples, width[i] * np.asarray(result.samples_u), rtol=1e-6
        )

    # New datasets of the same shapes reuse the compiled chunk kernel.
    (_, ll_spec), (_, pt_spec) = _split_callables(loglike, prior, 2)
    cfg = sampler.config
    capacity = max(1, int(np.ceil(core._CHUNK_EFOLDS / cfg.log_shrink)))
    kernel = core._chunk_kernel(ll_spec, pt_spec, cfg, capacity, (0, 0, 0))
    assert kernel._cache_size() == 1
    NestedSampler(Partial(shifted_gauss, mu[::-1], sigma[::-1, None]),
                  Partial(scaled_prior, width[:, None]), 2, 60, num_delete=6).run(
        keys, batched_data=True
    )
    assert kernel._cache_size() == 1


def test_shared_data_with_batched_keys() -> None:
    """Without batched_data the pytree leaves are shared by every lane."""
    loglike = Partial(shifted_gauss, jnp.asarray([0.4, 0.6]), 0.1)
    sampler = NestedSampler(loglike, identity, 2, 40, num_delete=4)
    keys = jax.random.split(jax.random.key(1), 2)
    for key, result in zip(keys, sampler.run(keys), strict=True):
        assert_matches(result, sampler.run(key))


def test_batched_maxiter_maxcall_and_resume(tmp_path) -> None:
    sampler = NestedSampler(gauss, identity, 2, 40, num_delete=4)
    keys = jax.random.split(jax.random.key(5), 3)
    capped = sampler.run(keys, maxiter=40)
    assert [r.niter for r in capped] == [40, 40, 40]
    assert all(r.metadata["status"] == "maxiter" for r in capped)
    calls = sampler.run(keys, maxcall=2000)
    assert all(2000 <= r.ncall < 2000 + 4 * 25 for r in calls)

    full = sampler.run(keys)
    path = tmp_path / "batch.npz"
    sampler.run(keys, maxiter=40, checkpoint=path)
    resumed = sampler.run(keys, checkpoint=path)
    for a, b in zip(resumed, full, strict=True):
        np.testing.assert_array_equal(a.samples_u, b.samples_u)
        assert (a.logz, a.ncall, a.niter) == (b.logz, b.ncall, b.niter)
    with pytest.raises(ValueError, match="batched_data"):
        sampler.run(keys, checkpoint=path, batched_data=True)


def test_batch_arguments_are_validated() -> None:
    keys = jax.random.split(jax.random.key(0), 3)
    sampler = NestedSampler(gauss, identity, 2, 20)
    with pytest.raises(ValueError, match="needs a batch of keys"):
        sampler.run(0, batched_data=True)
    with pytest.raises(ValueError, match="pytree callables"):
        sampler.run(keys, batched_data=True)
    bad = NestedSampler(Partial(shifted_gauss, jnp.zeros((2, 2)), 0.1), identity,
                        2, 20)
    with pytest.raises(ValueError, match="leading axis of 3"):
        bad.run(keys, batched_data=True)
    with pytest.raises(ValueError, match="1-D batch"):
        sampler.run(jax.random.split(jax.random.key(0), (2, 2)))
    with pytest.raises(TypeError, match="PRNG key"):
        sampler.run(jnp.arange(3))
