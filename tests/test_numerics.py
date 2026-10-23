"""Kernel numerics: the walk covariance when few points lie above ``L*``, the
cost of seeding, the open unit cube, float32 cluster frames, the precision of
the internal products, and JAX's strict promotion rules."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from tinyns import Config, NestedSampler, core, init, step

FLOAT = jnp.result_type(float)


def identity(u):
    return u


def gauss(x):
    return -0.5 * jnp.sum(((x - 0.5) / 0.1) ** 2)


# --- the walk covariance when few points lie above L* ------------------------

D10, NLIVE = 10, 500
HALF = 10.0 * (4 / NLIVE) ** (1 / D10)  # a box that holds 4 of 500 initial points


def box_gauss(x):
    """N(0, I) on the box [-HALF, HALF]^10, -inf outside."""
    return jnp.where(jnp.all(jnp.abs(x) < HALF), -0.5 * jnp.sum(x * x), -jnp.inf)


def wide(u):
    return -10.0 + 20.0 * u


def _few_finite(cfg, count, n_min=3, n_max=6):
    """The first ``count`` initial states with ``n_min`` to ``n_max`` finite
    points (the draws depend on the float dtype)."""
    found = []
    for seed in range(200):
        state = init(seed, box_gauss, wide, cfg)
        if n_min <= int(np.isfinite(np.asarray(state.logl)).sum()) <= n_max:
            found.append(state)
            if len(found) == count:
                return found
    raise AssertionError("no such seed")


@pytest.mark.parametrize("which", [0, 1])
def test_few_finite_points_spread_over_the_support(which) -> None:
    """A 10-D Gaussian with compact support: a box that holds about 4 of the
    500 initial points (prior U[-10, 10]^10). While fewer than ndim + 1
    points of the other folds are finite the walk covariance takes all the
    points of the other folds, and once the last -inf point has died the live
    points are uniform in the box. Falling back to the few finite points of
    every fold kept the chains near them: E|x|^2 came out 10-25% low, the
    narrowest principal width 2 to 50 times too small, and logZ 0.7 to 2
    nats (several logzerr) high."""
    cfg = Config(D10, NLIVE)
    state = _few_finite(cfg, 2)[which]
    for _ in range(30):
        state, _ = step(state, box_gauss, wide, cfg)
        if np.all(np.isfinite(np.asarray(state.logl))):
            break
    assert np.all(np.isfinite(np.asarray(state.logl)))
    x = wide(np.asarray(state.u, np.float64))
    # Uniform in the box: E|x|^2 = d HALF^2 / 3, principal widths HALF / sqrt 3.
    r2 = np.mean(np.sum(x * x, axis=1)) / (D10 * HALF**2 / 3)
    width = np.linalg.svd(x - x.mean(axis=0), compute_uv=False)
    width = width / math.sqrt(NLIVE - 1) / (HALF / math.sqrt(3))
    assert abs(r2 - 1.0) < 0.05
    assert width.min() > 0.7 and width.max() < 1.3


def test_two_live_points_do_not_stop_on_a_false_plateau() -> None:
    """With one point above L* the walk covariance was that point's alone,
    zero: the chain returned an exact copy of its seed (counted as moves),
    and a run with nlive 2 stopped on a "plateau" after one step."""
    result = NestedSampler(gauss, identity, 1, 2).run(0)
    assert result.metadata["status"] == "converged" and result.niter > 2


# --- seeding and insertion ranks -------------------------------------------


def _equations(jaxpr):
    """Every equation of ``jaxpr`` and of its sub-jaxprs (scan, cond, ...)."""
    for eqn in jaxpr.eqns:
        yield eqn
        for value in eqn.params.values():
            for sub in value if isinstance(value, (tuple, list)) else (value,):
                sub = getattr(sub, "jaxpr", sub)
                if hasattr(sub, "eqns"):
                    yield from _equations(sub)


def test_seeding_and_ranks_make_no_k_by_m_arrays() -> None:
    """Seeding and the insertion ranks sort the live points once a step: no
    intermediate holds ``num_delete * nlive`` numbers (at nlive 1e4 and
    num_delete 1e3 those took 28 ms of a 44 ms step on a CPU, and 7.6 GiB per
    run at nlive 1e5)."""
    m, k = 1009, 97
    cfg = Config(2, m, num_delete=k, walks=12)
    state = init(0, gauss, identity, cfg)
    closed = jax.make_jaxpr(lambda s: core._step(s, gauss, identity, cfg))(state)
    sizes = [
        math.prod(getattr(v.aval, "shape", ()))
        for eqn in _equations(closed.jaxpr)
        for v in eqn.outvars
    ]
    assert max(sizes) < 8 * m


def reference_seeds(gumbel, uniform, above, worst, folds):
    """The seeding rule of :func:`tinyns.core._seeds`, one dead slot at a
    time: the fold's points above L* by decreasing score (the lower slot
    first on a tie), then with replacement among them (by slot), then among
    all points above L*."""
    m = len(above)
    fold = np.arange(m) % folds
    out, used = [], {j: 0 for j in range(folds)}
    for i, slot in enumerate(worst):
        j = fold[slot]
        mine = np.flatnonzero(above & (fold == j))
        ranked = mine[np.argsort(-gumbel[mine], kind="stable")]
        if used[j] < len(mine):
            out.append(ranked[used[j]])
        else:
            pool = mine if len(mine) else np.flatnonzero(above)
            pick = int(FLOAT.type(uniform[i]) * FLOAT.type(len(pool)))
            out.append(pool[min(pick, len(pool) - 1)])
        used[j] += 1
    return np.array(out)


@pytest.mark.parametrize("m, k, folds, frac", [
    (30, 10, 3, 0.0), (101, 50, 3, 0.8), (64, 32, 2, 0.95), (7, 3, 4, 0.5),
])
def test_seeds_follow_the_rule(m, k, folds, frac) -> None:
    rng = np.random.default_rng(m)
    for trial in range(5):
        logl = rng.normal(size=m)
        worst = np.argsort(logl)[:k]
        above = (logl > logl[worst[-1]]) & (rng.random(m) >= frac * (trial % 2))
        if not above.any():
            continue
        k_seed, k_fill = random.split(random.PRNGKey(trial))
        seeds = core._seeds(
            k_seed, k_fill, jnp.asarray(above), jnp.asarray(worst, jnp.int32),
            folds, FLOAT,
        )
        gumbel = np.asarray(random.gumbel(k_seed, (m,), FLOAT))
        uniform = np.asarray(random.uniform(k_fill, (k,), FLOAT))
        np.testing.assert_array_equal(
            np.asarray(seeds), reference_seeds(gumbel, uniform, above, worst, folds)
        )


