"""Pytree callables and closures: large arrays are jit arguments of the kernels.

Pytree callables pass their array leaves; closures have their large jaxpr
constants hoisted.
"""

import gc
import hashlib
import logging
import weakref

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

import tinyns.callables as callables_mod
from tinyns import NestedSampler, core
from tinyns.callables import (
    _callable_leaves,
    _callable_specs,
    _clear_caches,
    _combine_callable,
    _partition_callable,
)
from tinyns.core import _chain_kernel
from tinyns.run import run_static_nested

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
    assert len(_callable_leaves(fn, prior, NDIM)) == 2
    assert _callable_leaves(plain_loglike, plain_prior, NDIM) == ()
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


@pytest.mark.parametrize("block_size", [None, 1])
def test_partial_loglike_with_large_array_matches_closure(block_size) -> None:
    # The closure's 200k-element constant is hoisted to a jit argument, so the
    # two forms compile the same program and agree bit for bit.
    data = make_data(200_000)
    partial_loglike = jax.tree_util.Partial(loglike_fn, data)
    partial_prior = jax.tree_util.Partial(prior_fn, BOUNDS)

    def closure_loglike(theta):
        return loglike_fn(data, theta)

    def closure_prior(u):
        return prior_fn(BOUNDS, u)

    kwargs = {"maxiter": 96, "dlogz": 0.0}
    if block_size is not None:
        kwargs["block_size"] = block_size
    partial_result = run_static_nested(
        random.PRNGKey(11), partial_loglike, partial_prior, NDIM, 40, **kwargs
    )
    closure_result = run_static_nested(
        random.PRNGKey(11), closure_loglike, closure_prior, NDIM, 40, **kwargs
    )
    (hoisted,) = _callable_leaves(closure_loglike, closure_prior, NDIM)
    assert hoisted is data
    assert_same_result(partial_result, closure_result)


def test_rwalk_kernel_accepts_partial_callables() -> None:
    data = make_data(1000)
    loglike = jax.tree_util.Partial(loglike_fn, data)
    prior = jax.tree_util.Partial(prior_fn, BOUNDS)
    live_u = random.uniform(random.PRNGKey(3), (16, NDIM))
    live_logl = jax.vmap(lambda u: loglike(prior(u)))(live_u)
    results = []
    for fn, pt in (
        (loglike, prior),
        (lambda t: loglike_fn(data, t), lambda u: prior_fn(BOUNDS, u)),
    ):
        specs = _callable_specs(fn, pt, NDIM)
        kernel = _chain_kernel(*specs, NDIM, 10, 1)
        results.append(
            kernel(
                random.PRNGKey(5),
                jnp.median(live_logl),
                live_u,
                live_logl,
                jnp.asarray(0.5),
                jnp.asarray(10, dtype=jnp.int32),
                *_callable_leaves(fn, pt, NDIM),
            )
        )
    for a, b in zip(*results, strict=True):
        np.testing.assert_array_equal(a, b)
    assert bool(results[0][5])


def _lower_block_kernel_text(loglike, prior):
    kernel = core._step_kernel(*_callable_specs(loglike, prior, NDIM), NDIM, 5, 4, 8)
    live_u = jnp.full((20, NDIM), 0.5)
    state = core.State(
        key=random.PRNGKey(0),
        u=live_u,
        theta=live_u,
        logl=jnp.zeros((20,)),
        birth=jnp.full((20,), -jnp.inf),
        logz=jnp.asarray(-jnp.inf),
        logx=jnp.asarray(0.0, dtype=jnp.float32),
        it=jnp.asarray(0, dtype=jnp.int32),
        scale=jnp.asarray(0.5),
        ncall=jnp.asarray(0, dtype=jnp.int32),
        failed=jnp.asarray(False),
    )
    lowered = kernel.lower(
        state,
        jnp.asarray(20, dtype=jnp.int32),
        jnp.asarray(10, dtype=jnp.int32),
        None,
        *_callable_leaves(loglike, prior, NDIM),
    )
    return lowered.as_text()


def test_block_kernel_hlo_does_not_embed_partial_arrays() -> None:
    _clear_caches()
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
    finally:
        _clear_caches()


def test_block_kernel_is_cached_per_partial_instance() -> None:
    _clear_caches()
    try:
        loglike = jax.tree_util.Partial(loglike_fn, make_data(100))
        first = core._step_kernel(
            *_callable_specs(loglike, plain_prior, NDIM), NDIM, 5, 4, 8
        )
        second = core._step_kernel(
            *_callable_specs(loglike, plain_prior, NDIM), NDIM, 5, 4, 8
        )
        assert first is second
    finally:
        _clear_caches()


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


# --- closures: large constants are hoisted ---


@pytest.fixture
def fresh_caches():
    _clear_caches()
    yield
    _clear_caches()


# Below _HOIST_MIN_SIZE a closure array stays an embedded constant.
_SMALLEST_HOISTED = callables_mod._HOIST_MIN_SIZE


def closure_prior(u):
    return prior_fn(BOUNDS, u)


def make_closure(data):
    def loglike(theta):
        return loglike_fn(data, theta)

    return loglike


def run(loglike, prior=closure_prior, ndim=NDIM, **kwargs):
    kwargs.setdefault("maxiter", 96)
    kwargs.setdefault("dlogz", 0.0)
    return run_static_nested(random.PRNGKey(11), loglike, prior, ndim, 40, **kwargs)


@pytest.mark.usefixtures("fresh_caches")
def test_block_kernel_hlo_does_not_embed_closure_arrays(monkeypatch) -> None:
    texts = {
        n: _lower_block_kernel_text(make_closure(make_data(n)), closure_prior)
        for n in (_SMALLEST_HOISTED, 200_000)
    }
    assert abs(len(texts[200_000]) - len(texts[_SMALLEST_HOISTED])) < 2_000

    # With hoisting disabled the closure's array is a ~200k-value constant.
    _clear_caches()
    monkeypatch.setattr(callables_mod, "_HOIST_MIN_SIZE", 10**9)
    embedded = _lower_block_kernel_text(make_closure(make_data(200_000)), closure_prior)
    assert len(embedded) > len(texts[200_000]) + 100_000


@pytest.mark.usefixtures("fresh_caches")
def test_small_constants_and_plain_functions_are_not_hoisted(monkeypatch) -> None:
    small = jnp.linspace(0.1, 0.3, callables_mod._HOIST_MIN_SIZE - 1)
    small_np = np.linspace(0.1, 0.3, 50)

    def small_closure(theta):
        center = jnp.mean(small) + 0.0 * jnp.sum(small_np)
        return -0.5 * jnp.sum(((theta - center) / 0.1) ** 2)

    cases = [(plain_loglike, plain_prior), (small_closure, closure_prior)]
    for loglike, prior in cases:
        assert _callable_leaves(loglike, prior, NDIM) == ()
    hoisting = [run(loglike, prior, maxiter=160) for loglike, prior in cases]

    # No hoisting at all is the parent-branch behaviour: bit-identical output.
    _clear_caches()
    monkeypatch.setattr(callables_mod, "_eval_jaxpr", None)
    for (loglike, prior), result in zip(cases, hoisting, strict=True):
        assert_same_result(result, run(loglike, prior, maxiter=160))


@pytest.mark.usefixtures("fresh_caches")
def test_numpy_closure_constant_is_hoisted() -> None:
    data = np.asarray(make_data(200_000))
    loglike = make_closure(data)
    (hoisted,) = _callable_leaves(loglike, closure_prior, NDIM)
    assert isinstance(hoisted, jax.Array) and hoisted.shape == data.shape
    # Cached: the same object on every call, so jit arguments are stable.
    assert _callable_leaves(loglike, closure_prior, NDIM)[0] is hoisted
    texts = [
        _lower_block_kernel_text(make_closure(np.asarray(make_data(n))), closure_prior)
        for n in (_SMALLEST_HOISTED, 200_000)
    ]
    assert abs(len(texts[1]) - len(texts[0])) < 2_000
    partial = jax.tree_util.Partial(loglike_fn, jnp.asarray(data))
    assert_same_result(run(loglike), run(partial))


@pytest.mark.usefixtures("fresh_caches")
def test_nested_jit_operand_is_hoisted() -> None:
    inner = jax.jit(loglike_fn)

    def make_nested(data):
        def loglike(theta):
            return inner(data, theta)

        return loglike

    data = make_data(200_000)
    loglike = make_nested(data)
    (hoisted,) = _callable_leaves(loglike, closure_prior, NDIM)
    assert hoisted is data
    texts = [
        _lower_block_kernel_text(make_nested(make_data(n)), closure_prior)
        for n in (_SMALLEST_HOISTED, 200_000)
    ]
    assert abs(len(texts[1]) - len(texts[0])) < 2_000
    partial = jax.tree_util.Partial(loglike_fn, data)
    assert_same_result(run(loglike), run(partial))


@pytest.mark.usefixtures("fresh_caches")
def test_hoisted_one_dimensional_closure_and_prior() -> None:
    # ndim=1 with a scalar-returning prior transform that also captures a
    # large array: both callables are hoisted and still vmap over chains.
    data = make_data(10_000)
    bounds = jnp.concatenate([jnp.full(5_000, -1.0), jnp.full(5_000, 1.0)])

    def prior(u):
        return jnp.min(bounds) + (jnp.max(bounds) - jnp.min(bounds)) * u[0]

    loglike = make_closure(data)
    assert len(_callable_leaves(loglike, prior, 1)) == 2
    result = run(loglike, prior, ndim=1)
    partial_prior = jax.tree_util.Partial(
        lambda b, u: jnp.min(b) + (jnp.max(b) - jnp.min(b)) * u[0], bounds
    )
    partial = jax.tree_util.Partial(loglike_fn, data)
    assert_same_result(result, run(partial, partial_prior, ndim=1))


@pytest.mark.usefixtures("fresh_caches")
def test_make_jaxpr_failure_falls_back_to_closure_semantics(
    monkeypatch, caplog
) -> None:
    data = make_data(200_000)
    loglike = make_closure(data)
    expected = run(loglike, maxiter=40)

    _clear_caches()

    def broken(*args, **kwargs):
        raise RuntimeError("no jaxpr today")

    monkeypatch.setattr(callables_mod.jax, "make_jaxpr", broken)
    with caplog.at_level(logging.DEBUG, logger="tinyns.callables"):
        assert _callable_leaves(loglike, closure_prior, NDIM) == ()
        result = run(loglike, maxiter=40)
    assert_same_result(result, expected)
    messages = [r for r in caplog.records if "not hoisting" in r.getMessage()]
    # One message per callable (loglike and prior), not one per iteration.
    assert len(messages) == 2


@pytest.mark.usefixtures("fresh_caches")
def test_hoisted_closure_checkpoint_resume_matches_uninterrupted(tmp_path) -> None:
    path = tmp_path / "closure.checkpoint.npz"
    loglike = make_closure(make_data(5_000))
    assert len(_callable_leaves(loglike, closure_prior, NDIM)) == 1
    sampler = NestedSampler(loglike, closure_prior, NDIM, nlive=30)

    full = sampler.run(21, maxiter=64, dlogz=0.0)
    sampler.run(21, maxiter=32, dlogz=0.0, checkpoint_path=path)
    resumed = sampler.resume(path, maxiter=64, dlogz=0.0)

    assert resumed.metadata["resumed_from_checkpoint"] is True
    assert_same_result(resumed, full)


# --- campaigns: one compile per structure, no retained datasets (v0.2.3) ---


@pytest.fixture
def backend_compiles():
    """Count XLA backend compiles while the test runs."""
    events = []

    def listener(event, duration, **kwargs):
        if event == "/jax/core/compile/backend_compile_duration":
            events.append(duration)

    jax.monitoring.register_event_duration_secs_listener(listener)
    yield events
    unregister = getattr(
        jax._src.monitoring, "_unregister_event_duration_listener_by_callback", None
    )
    if unregister is not None:
        unregister(listener)


def shifted_data(i, n=10_000):
    return make_data(n) + 0.05 * i


@pytest.mark.usefixtures("fresh_caches")
def test_same_shape_partials_share_one_compiled_block_kernel(backend_compiles) -> None:
    results, compiles = [], []
    for i in range(4):
        before = len(backend_compiles)
        loglike = jax.tree_util.Partial(loglike_fn, shifted_data(i))
        results.append(run(loglike, plain_prior))
        compiles.append(len(backend_compiles) - before)
    # Dataset 0 compiles everything; datasets 1-3 reuse the block kernel and
    # the live-point pass (v0.2.2 compiled both again for every dataset).
    assert compiles[0] > 0 and compiles[1:] == [0, 0, 0]
    info = core._step_kernel.cache_info()
    assert info.misses == 1 and info.currsize == 1
    assert len({result.logz for result in results}) == 4

    for i, result in enumerate(results):
        _clear_caches()  # an independent fresh run compiles its own kernels
        loglike = jax.tree_util.Partial(loglike_fn, shifted_data(i))
        assert_same_result(result, run(loglike, plain_prior))


def make_closure_prior(bounds):
    def prior(u):
        return prior_fn(bounds[:, :NDIM], u)

    return prior


@pytest.mark.parametrize("form", ["partial", "closure"])
@pytest.mark.usefixtures("fresh_caches")
def test_finished_runs_do_not_keep_datasets_alive(form) -> None:
    refs = []
    for i in range(4):
        data = shifted_data(i)
        bounds = BOUNDS + 0.0  # a fresh array per dataset
        refs += [weakref.ref(data), weakref.ref(bounds)]
        if form == "partial":
            loglike = jax.tree_util.Partial(loglike_fn, data)
            prior = jax.tree_util.Partial(prior_fn, bounds)
        else:  # both closures have hoisted constants
            loglike = make_closure(data)
            prior = make_closure_prior(jnp.tile(bounds, (1, _SMALLEST_HOISTED)))
            refs.append(weakref.ref(prior))
        result = run(loglike, prior, maxiter=64)
        assert np.isfinite(result.logz)
        del data, bounds, loglike, prior, result
    gc.collect()
    assert [ref() for ref in refs] == [None] * len(refs)
    assert not callables_mod._IDENTITY_SPLITS


# Fixed-seed fingerprints of tinyns v0.2.2 (ae330f2) on CPU with jax 0.4.34:
# (logz.hex(), ncall, sha256[:16] of samples, samples_u and logl).
V022_FINGERPRINTS = {
    "block": (
        "-0x1.122a400000000p+2",
        7405,
        "ea9e454c31e2785c",
        "b3ec3331262f69b9",
        "b6ff6462d6dc00fa",
    ),
    "small_closure": (
        "-0x1.122a400000000p+2",
        7405,
        "ea9e454c31e2785c",
        "b3ec3331262f69b9",
        "47b56e8a8af6eb32",
    ),
}
SMALL = jnp.linspace(0.1, 0.3, 50)


def small_closure_loglike(theta):
    return -0.5 * jnp.sum(((theta - jnp.mean(SMALL)) / 0.1) ** 2)


def fingerprint(result):
    def digest(x):
        return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:16]

    return (
        float(result.logz).hex(),
        result.ncall,
        digest(result.samples),
        digest(result.samples_u),
        digest(result.logl),
    )


@pytest.mark.skipif(
    jax.__version__ != "0.4.34" or jax.default_backend() != "cpu",
    reason="fingerprints were recorded with jax 0.4.34 on CPU",
)
@pytest.mark.parametrize("case", sorted(V022_FINGERPRINTS))
def test_plain_functions_match_v022_bit_for_bit(case) -> None:
    loglike = small_closure_loglike if case == "small_closure" else plain_loglike
    result = run_static_nested(
        random.PRNGKey(7), loglike, plain_prior, NDIM, 50, maxiter=300
    )
    assert fingerprint(result) == V022_FINGERPRINTS[case]


@pytest.mark.parametrize("block_size", [1, 32])
def test_wall_time_telemetry(block_size, capsys) -> None:
    result = run_static_nested(
        random.PRNGKey(3),
        plain_loglike,
        plain_prior,
        NDIM,
        30,
        maxiter=96,
        dlogz=0.0,
        block_size=block_size,
        progress=True,
    )
    metadata = result.metadata
    assert metadata["wall_time_s"] > metadata["compile_s"] > 0.0
    assert metadata["mean_ms_per_call"] > 0.0
    assert "calls/s=" in capsys.readouterr().out
