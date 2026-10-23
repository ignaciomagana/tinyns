"""Kernel numerics: the walk covariance when few points lie above ``L*``, the
cost of seeding, the open unit cube, float32 cluster frames, the precision of
the internal products, and JAX's strict promotion rules."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax, random
from jax.scipy.special import ndtri

from tinyns import Config, NestedSampler, core, init, modes, step

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


def test_insertion_ranks_search_without_a_loop() -> None:
    """The binary searches of the insertion ranks are unrolled: no loop of
    ``ceil(log2(nlive + 1))`` sequential steps is left for a GPU to run."""
    m = 1009
    cfg = Config(2, m, num_delete=97, walks=12)
    state = init(0, gauss, identity, cfg)
    closed = jax.make_jaxpr(lambda s: core._step(s, gauss, identity, cfg))(state)
    levels = math.ceil(math.log2(m + 1))

    def rolled(eqn):
        # JAX 0.12 keeps the search as one primitive, earlier JAX a scan.
        if getattr(eqn.params.get("_prim"), "method", None) == "scan":
            return True
        return (
            eqn.primitive.name == "scan"
            and eqn.params["length"] == levels
            and eqn.params["unroll"] < levels
        )

    assert not any(rolled(eqn) for eqn in _equations(closed.jaxpr))


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


# --- the open unit cube ------------------------------------------------------


def test_initial_draws_of_zero_move_into_the_open_cube(monkeypatch) -> None:
    """A uniform draw of exactly 0 (one in 2^23 per coordinate in float32)
    would map to -inf under ndtri: it becomes half the draws' spacing."""
    uniform = random.uniform

    def with_zeros(key, shape=(), dtype=float, *args, **kwargs):
        u = uniform(key, shape, dtype, *args, **kwargs)
        return u.at[::3, 0].set(0.0) if len(shape) == 2 else u

    monkeypatch.setattr(random, "uniform", with_zeros)
    state = core._init(random.PRNGKey(0), gauss, ndtri, Config(3, 30))
    u = np.asarray(state.u)
    assert np.all((u > 0) & (u < 1))
    np.testing.assert_array_equal(u[::3, 0], jnp.finfo(FLOAT).eps / 2)
    assert np.all(np.isfinite(np.asarray(state.logl)))


@pytest.mark.skipif(FLOAT != jnp.float32, reason="a float32 draw")
def test_float32_seed_with_a_zero_draw_starts_in_the_open_cube() -> None:
    """Seed 4860 draws an exact 0 among its 500 x 4 initial uniforms (with
    the PRNG of JAX 0.5 and later; the test above covers the mechanism on
    every version)."""
    _, sub = random.split(random.PRNGKey(4860))
    if np.all(np.asarray(random.uniform(sub, (500, 4), FLOAT)) > 0):
        pytest.skip("this JAX's PRNG draws no zero for seed 4860")

    def loglike(x):
        return -0.5 * jnp.sum(((x - 0.3) / 0.5) ** 2)

    state = init(4860, loglike, ndtri, Config(4, 500))
    assert np.all(np.asarray(state.u) > 0)


@pytest.mark.parametrize("k", [1, 5])
def test_chains_never_reach_a_face(k) -> None:
    """A likelihood piled against the face u = 1, about 4 float spacings
    wide: proposals that round to 1.0 are rejected, no sample lies on the
    face, and no out-of-cube proposal is evaluated on one."""
    lam = 0.25 / float(jnp.finfo(FLOAT).epsneg)
    faces = []

    def on_face(u):
        faces.append(int(np.sum(np.any((u <= 0) | (u >= 1), axis=-1))))

    def loglike(u):
        jax.debug.callback(on_face, u)
        return lam * jnp.sum(u - 1.0)

    result = NestedSampler(loglike, identity, 1, 20, num_delete=k).run(0)
    u = np.asarray(result.samples_u)
    assert result.success and len(faces) > 0
    assert np.all((u > 0) & (u < 1))
    assert sum(faces) == 0


# --- float32 cluster frames --------------------------------------------------

NARROW = 1e-5


def _narrow_cluster(n=600, seed=0):
    """A cluster ``NARROW`` wide along u2 and 0.1 and 0.29 along u0 and u1."""
    rng = np.random.default_rng(seed)
    x = np.empty((n, 3))
    x[:, 0] = 0.5 + 0.1 * rng.standard_normal(n)
    x[:, 1] = rng.uniform(size=n)
    x[:, 2] = 0.5 + NARROW * rng.standard_normal(n)
    return x


def test_float32_frames_resolve_a_narrow_axis() -> None:
    """The ridge of a frame is relative to each axis: one relative to the
    mean variance made this float32 frame 19 times too wide along its axis
    ``1e-5`` wide."""
    x = _narrow_cluster()
    u = jnp.asarray(x, jnp.float32)
    st = modes.stats(u, jnp.zeros(len(x), jnp.int32), modes.C_MAX)
    fr = modes.frames(st, len(x))
    chol = np.asarray(fr.chol[0], np.float64)
    sd = np.sqrt(np.diag(chol @ chol.T))
    np.testing.assert_allclose(sd / x.std(axis=0, ddof=1), 1.0, rtol=0.05)


def test_float32_separation_along_a_narrow_axis() -> None:
    """Two clusters 10 sd apart along the narrow axis: the merge test's and
    the split search's Fisher separation in float32 match float64."""
    x = _narrow_cluster(seed=1)
    side = np.arange(len(x)) % 2
    x[:, 2] += np.where(side, 5.0, -5.0) * NARROW
    u = jnp.asarray(x, jnp.float32)
    a, b = x[side == 0], x[side == 1]
    dev = np.concatenate([a - a.mean(0), b - b.mean(0)])
    dm = b.mean(0) - a.mean(0)
    j = dm @ np.linalg.solve(dev.T @ dev / (len(x) - 2), dm)
    j = j * (len(x) - 3 - 3) / (len(x) - 2)  # modes._deflate
    st = modes.stats(u, jnp.asarray(side, jnp.int32), modes.C_MAX)
    pair = float(modes._pair_separation(st)[0, 1])
    split, part = modes._split(u, jnp.ones(len(x), bool))
    assert abs(pair / j - 1.0) < 0.02
    assert abs(float(split) / j - 1.0) < 0.05
    assert np.mean(np.asarray(part) == side) in (0.0, 1.0)


# --- the precision of the internal products ----------------------------------


@pytest.mark.parametrize("k", [1, 4])
def test_internal_products_run_at_full_precision(k) -> None:
    """Every product of a step and of a recluster asks for HIGHEST precision,
    so a GPU's TF32 (JAX's default for float32 on an A100 or H100, a 10-bit
    mantissa) does not touch the means of clusters in the unit cube or the
    Hastings ratios. The likelihood here makes no product of its own."""
    cfg = Config(3, 40, num_delete=k)
    state = init(0, gauss, identity, cfg)
    programs = [
        jax.make_jaxpr(lambda s: core._step(s, gauss, identity, cfg))(state),
        jax.make_jaxpr(core._recluster)(state),
    ]
    precisions = [
        eqn.params["precision"]
        for closed in programs
        for eqn in _equations(closed.jaxpr)
        if eqn.primitive.name == "dot_general"
    ]
    assert len(precisions) > 10
    for precision in precisions:
        assert precision is not None
        pair = precision if isinstance(precision, tuple) else (precision,)
        assert all(p == lax.Precision.HIGHEST for p in pair), precision


# --- strict promotion rules --------------------------------------------------


def two_modes(x):
    a = -0.5 * jnp.sum(((x - 0.3) / 0.05) ** 2)
    b = -0.5 * jnp.sum(((x - 0.7) / 0.05) ** 2)
    return jnp.logaddexp(a, b)


def test_runs_under_strict_promotion_rules() -> None:
    """tinyns runs with ``jax_numpy_rank_promotion="raise"`` and
    ``jax_numpy_dtype_promotion="strict"`` (both are part of the compile
    cache key, so the kernels are traced again under them): a batched
    two-mode run, a run with ``num_delete=1``, and the result methods."""
    with jax.numpy_rank_promotion("raise"), jax.numpy_dtype_promotion("strict"):
        batch = NestedSampler(two_modes, identity, 2, 120).run(
            random.split(random.PRNGKey(0), 2)
        )
        lone = NestedSampler(two_modes, identity, 2, 60, num_delete=1).run(1)
        for result in (*batch, lone):
            assert result.success and np.isfinite(result.logz)
            result.summary()
            result.diagnostics()
            result.modes()
        assert max(len(r.modes()) for r in batch) == 2


def test_hop_proposal_indexes_frames_with_int32() -> None:
    """The hop picks its frame with an int32 index, like the chains' fold
    index it is paired with: under x64 an int64 one made JAX 0.4 raise on the
    mixed integer index in strict mode (the test above, on the CI's oldest
    JAX)."""
    seen = []

    def frame(c):
        seen.append(c.dtype)
        return jnp.zeros(2, FLOAT), jnp.eye(2, dtype=FLOAT)

    eligible = jnp.arange(modes.C_MAX) < 3
    logdet = jnp.zeros(modes.C_MAX, FLOAT)
    modes.hop_proposal(random.PRNGKey(0), eligible, logdet, frame)
    assert seen == [jnp.int32]
