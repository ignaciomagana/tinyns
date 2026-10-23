"""Kernel numerics: the walk covariance when few points lie above ``L*``, the
cost of seeding, the open unit cube, float32 cluster frames, the precision of
the internal products, and JAX's strict promotion rules."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from tinyns import Config, NestedSampler, init, step

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


