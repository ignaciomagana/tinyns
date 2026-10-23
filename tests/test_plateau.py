"""Likelihood plateaus (exact ties): tie-aware live counts and insertion ranks.

The target is a 5-D Gaussian (sd 0.05 at u0 = 0.25) on the unit cube, with
logZ = 0. Setting L = 0 (-inf) or NaN on the half u0 > 0.5 leaves logZ = 0:
that half is prior volume with zero likelihood. About half of the initial
live points then tie at -inf. Before the tie-aware counts, every step was
credited the compression of ``nlive`` live points, and logZ came out
0.17-0.19 high whatever ``nlive`` and ``num_delete`` (Fowlie, Handley and Su
2021 give the rule the sampler now follows).
"""

from __future__ import annotations

import math
from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tinyns import Config, NestedSampler, core, finalise, init, step

D, SIG, PLATEAU = 5, 0.05, -5.0
MU = np.array([0.25, 0.5, 0.5, 0.5, 0.5])
NLIVE = 500


def gauss(u):
    norm = D * math.log(SIG * math.sqrt(2 * math.pi))
    return -0.5 * jnp.sum(((u - MU) / SIG) ** 2) - norm


def half(value):
    """The Gaussian, with ``value`` on the half u0 > 0.5."""

    def loglike(u):
        return jnp.where(u[0] > 0.5, value, gauss(u))

    return loglike


LOGLIKES = {
    "gauss": gauss,
    "inf": half(-jnp.inf),
    "nan": half(jnp.nan),
    "flat": half(PLATEAU),
}
TRUTH = {"flat": math.log1p(0.5 * math.exp(PLATEAU))}


def identity(u):
    return u


@cache
def runs(case: str, k: int, seeds: int, nlive: int = NLIVE):
    """A batch of ``seeds`` runs (cached: tests share them)."""
    sampler = NestedSampler(LOGLIKES[case], identity, D, nlive, num_delete=k)
    return tuple(sampler.run(jax.random.split(jax.random.key(123), seeds)))


def bias(case, k, seeds, nlive=NLIVE):
    """Mean logZ error over the batch and its standard error."""
    z = np.array([r.logz for r in runs(case, k, seeds, nlive)]) - TRUTH.get(case, 0.0)
    return z.mean(), z.std(ddof=1) / math.sqrt(len(z))


def tie_nlive(logl, k, m):
    """The live counts of the dead rows, recounted on the host: a death equal
    to the previous step's contour counts at ``m - j - k * run``, ``run`` the
    consecutive earlier steps at that contour."""
    rows = np.asarray(logl, float).reshape(-1, k)
    lstar = rows[:, -1]
    n = np.tile(m - np.arange(k), (len(rows), 1))
    run = 0
    for s in range(1, len(rows)):
        run = run + 1 if s > 1 and lstar[s - 1] == lstar[s - 2] else 1
        n[s, rows[s] == lstar[s - 1]] -= k * run
    return n.reshape(-1)


# Seeds per batch: k = 1 runs one step per death, so it gets fewer.
CASES = [(1, 12), (25, 16), (NLIVE // 10, 16)]


@pytest.mark.parametrize("k, seeds", CASES)
def test_zero_likelihood_region_is_prior_volume(k, seeds) -> None:
    """L = -inf on half the cube: logZ is unbiased (it was 0.17-0.19 high)."""
    mean, se = bias("inf", k, seeds)
    assert abs(mean) < 3 * se, (mean, se)
    for r in runs("inf", k, seeds):
        n = r.niter
        logl = np.asarray(r.logl)
        q = int(np.sum(logl[:n] == -np.inf))
        assert 150 < q < 350  # about half the initial points tie at -inf
        # The q tied deaths count at m, m-1, ..., m-q+1.
        np.testing.assert_array_equal(np.asarray(r.nlive_i)[:q], NLIVE - np.arange(q))
        np.testing.assert_array_equal(r.nlive_i[:n], tie_nlive(logl[:n], k, NLIVE))


def test_nan_region_is_the_same_as_a_zero_likelihood_region() -> None:
    nan, inf = runs("nan", 25, 16), runs("inf", 25, 16)
    for a, b in zip(nan, inf, strict=True):
        assert a.logz == b.logz
        np.testing.assert_array_equal(a.samples_u, b.samples_u)


def test_insertion_ranks_are_uniform_on_a_plateau() -> None:
    """The in-sampler ranks against the survivors strictly above L*, each
    tested against its own slot count, pass on the -inf toy (the replay of
    the births against nlive - k + 1 slots gave p ~ 1e-27 in the first
    window)."""
    batch = runs("inf", 25, 16)
    first = [r.insertion_test()["windows"][0]["pvalue"] for r in batch]
    pooled = [r.insertion_test()["pvalue"] for r in batch]
    assert np.median(first) > 0.05 and min(first) > 1e-4, first
    assert np.median(pooled) > 0.05 and min(pooled) > 1e-4, pooled
    for r in batch:
        slots = np.asarray(r.insertion_slots)
        # Fewer slots while the tied points remain, nlive - k + 1 after.
        assert slots[0] < NLIVE - 25 + 1 and slots[-1] == NLIVE - 25 + 1
        assert np.all(np.asarray(r.insertion_ranks) < slots)


def test_finite_plateau() -> None:
    """L = -5 on half the cube, above most of the Gaussian half: the plateau
    holds nearly all the live points, and only some 5-10 lie above it when it
    dies. The tie-aware estimate of that fraction, ln(m / (m - q)), is then
    noisy and leaves a residual (+0.09 +- 0.05 over 48 runs, a scatter of
    0.35); without the tie-aware counts logZ was 3.1-3.2 too high. The bound
    allows the residual. New points that land on the plateau tie with the
    survivors there and take a random place among them, so the insertion
    test passes (it gave p ~ 1e-268 in the first window)."""
    batch = runs("flat", 25, 16)
    mean, se = bias("flat", 25, 16)
    assert abs(mean) < 0.3 + 3 * se, (mean, se)
    # A run's chains start from the few points above the plateau, so a
    # single run's ranks can be off (p down to 5e-5 in 16 runs on CI); the
    # median is the test.
    first = [r.insertion_test()["windows"][0]["pvalue"] for r in batch]
    assert np.median(first) > 0.05 and min(first) > 1e-8, first
    # logzerr misses the uncertainty of the plateau's volume; the bootstrap,
    # which draws every shrinkage, has it (0.36 against a scatter of 0.35).
    ratio = [r.logz_bootstrap(n_realizations=200).logzerr / r.logzerr for r in batch]
    assert np.median(ratio) > 2, ratio


def test_no_plateau_control() -> None:
    """Without a plateau the counts are those of plain nested sampling,
    m - j at death j of a step, but at the rare exact ties of an unmoved
    chain's copy with its seed, or of float32 rounding, across a step
    boundary (a survivor tied at L* also takes a slot off the insertion
    rank's range)."""
    mean, se = bias("gauss", 25, 16)
    assert abs(mean) < 3 * se, (mean, se)
    plain = NLIVE - np.arange(25)
    for r in runs("gauss", 25, 16):
        n = r.niter
        logl = np.asarray(r.logl)[:n]
        nlive = np.asarray(r.nlive_i)[:n]
        np.testing.assert_array_equal(nlive, tie_nlive(logl, 25, NLIVE))
        assert np.mean(nlive == np.tile(plain, n // 25)) > 0.999
        assert np.mean(np.asarray(r.insertion_slots) == NLIVE - 24) > 0.95


def test_functional_core_carries_the_tie_bookkeeping() -> None:
    """``init``/``step`` keep the running evidence and prior volume
    tie-aware, so ``delta_logz`` and the running ``logz`` agree with
    ``finalise``; the counts follow the host recount."""
    m, k = 60, 6
    cfg = Config(D, m, num_delete=k, walks=20)
    loglike = LOGLIKES["inf"]
    state = init(3, loglike, identity, cfg)
    assert float(state.lstar_prev) == -np.inf and int(state.tie_run) == 0
    rows = []
    while float(core.delta_logz(state, cfg)) > 0.1:
        state, dead = step(state, loglike, identity, cfg)
        rows.append(dead)
    dead = jax.tree_util.tree_map(lambda *xs: np.stack(xs), *rows)
    result = finalise(state, dead, cfg, dlogz=0.1)
    n = result.niter
    nlive = np.asarray(result.nlive_i)[:n]
    np.testing.assert_array_equal(nlive, tie_nlive(result.logl[:n], k, m))
    assert np.any(nlive < m - k)  # ties were counted
    tol = 1e-9 if jax.config.jax_enable_x64 else 1e-4
    log_x = float(core._log_x(state, cfg))
    assert log_x == pytest.approx(-np.sum(1.0 / nlive), rel=tol)
    assert float(state.log_x_ties) < 0.0
    dead_logz = float(np.logaddexp.reduce(np.asarray(result.logwt)[:n]))
    assert float(state.logz) == pytest.approx(dead_logz, abs=10 * tol)
    assert result.success


def test_ties_under_vmap() -> None:
    """Each lane of a batched run keeps its own tie bookkeeping (the lanes
    leave the plateau at different steps)."""
    m, k = 60, 6
    sampler = NestedSampler(LOGLIKES["inf"], identity, D, m, num_delete=k)
    batch = sampler.run(jax.random.split(jax.random.key(9), 3))
    tied = []
    for lane in batch:
        n = lane.niter
        nlive = np.asarray(lane.nlive_i)[:n]
        np.testing.assert_array_equal(nlive, tie_nlive(lane.logl[:n], k, m))
        tied.append(int(np.sum(nlive < m - k)))
        assert lane.success
    assert min(tied) > 0 and len(set(tied)) > 1, tied
