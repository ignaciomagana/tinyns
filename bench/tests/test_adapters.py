"""Adapter timing and stopping: what counts as compilation, where BlackJAX
stops, the padded batches of the CPU likelihood, and the headline's cells."""

from __future__ import annotations

import contextlib
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
from bench import adapters
from bench.targets import get_target

# The XLA compile of one program (or its load from the persistent cache).
BACKEND_COMPILE = "/jax/core/compile/backend_compile_duration"


@contextlib.contextmanager
def backend_compiles():
    """Collect the duration of every XLA compile inside the block."""
    import jax

    seen, on = [], [True]

    def listen(event, duration, **_):
        if on[0] and event == BACKEND_COMPILE:
            seen.append(duration)

    jax.monitoring.register_event_duration_secs_listener(listen)
    try:
        yield seen
    finally:
        on[0] = False  # JAX 0.4 has no public way to unregister it


def _warm_keys():
    """Compile the key ops the adapters run before their timers start."""
    import jax

    jax.random.split(jax.random.PRNGKey(0))


def test_blackjax_compile_s_covers_every_compile_of_the_run():
    """Every XLA compile of a BlackJAX run is inside ``compile_s``, so ``wall -
    compile`` is a compiled run. Its eager post-processing used to compile in
    ``run s`` (about 9 s of a BlackJAX run on the H100). ``nlive`` 120 keeps
    the programs of the other tests out of the way."""
    reason = adapters.unavailable_reason("blackjax_nss")
    if reason:
        pytest.skip(reason)
    _warm_keys()
    cfg = dict(nlive=120, dlogz=0.1, variant="default", opts={})
    with backend_compiles() as seen:
        out = adapters.load("blackjax_nss").run(get_target("gauss_d2"), 3, cfg)
    assert seen, "nothing compiled: the test cannot see the split"
    assert sum(seen) <= out["compile_s"] + 0.05, (sum(seen), out["compile_s"])
    assert out["compile_s"] < out["wall_s"]


def test_tinyns_run_s_holds_no_compile(monkeypatch):
    """The tinyns run that ``run s`` times compiles (almost) nothing: its init
    and step kernels, its prior map and the driver's small ops compile in a
    one-step warm-up run, which is ``compile_s``. They used to compile in the
    timed run, of which ``metadata["compile_s"]`` covered only the step
    kernel."""
    import tinyns

    if adapters.unavailable_reason("tinyns_v1"):
        pytest.skip("tinyns v1 is not installed")
    calls = []
    run = tinyns.NestedSampler.run

    def timed_run(self, *args, **kw):
        with backend_compiles() as seen:
            res = run(self, *args, **kw)
        calls.append(sum(seen))
        return res

    monkeypatch.setattr(tinyns.NestedSampler, "run", timed_run)
    _warm_keys()
    cfg = dict(nlive=136, dlogz=0.1, variant="default", opts={})
    out = adapters.load("tinyns_v1").run(get_target("gauss_d2"), 3, cfg)
    assert sum(calls) > 0, "nothing compiled: the test cannot see the split"
    # the result's run compiled under a tenth of what compile_s holds
    assert calls[-1] <= 0.1 * out["compile_s"], (calls, out["compile_s"])
    assert out["compile_s"] < out["wall_s"]


def _blackjax_reference(target, seed, nlive, dlogz):
    """The adapter's run, one step at a time: (steps, calls) at the first step
    whose state meets the stopping rule. The keys are the adapter's (each
    chunk of ``2 * ceil(nlive / k)`` steps splits one key)."""
    import blackjax
    import jax
    import jax.numpy as jnp

    d, k = target.ndim, max(1, nlive // 10)
    nsteps = 2 * math.ceil(nlive / k)

    def loglike_u(u):
        inside = jnp.all((u >= 0.0) & (u <= 1.0))
        ll = target.loglike(target.prior_transform(jnp.clip(u, 0.0, 1.0)))
        return jnp.where(inside, ll, -jnp.inf)

    def logprior_u(u):
        return jnp.where(jnp.all((u >= 0.0) & (u <= 1.0)), 0.0, -jnp.inf)

    algo = blackjax.nss(
        logprior_fn=logprior_u, loglikelihood_fn=loglike_u, num_delete=k,
        num_inner_steps=max(5, 2 * d),
    )

    @jax.jit
    def step(state, key):
        state, info = algo.step(key, state)
        ui = info.update_info
        lz, lzl = state.integrator.logZ, state.integrator.logZ_live
        calls = jnp.sum(2 + ui.num_expansions + ui.num_shrink)
        return state, calls, jnp.logaddexp(lz, lzl) - lz < dlogz

    key, k_init = jax.random.split(jax.random.PRNGKey(seed))
    state = jax.jit(algo.init)(jax.random.uniform(k_init, (nlive, d)))
    steps, calls = 0, nlive
    while True:
        key, sub = jax.random.split(key)
        for step_key in jax.random.split(sub, nsteps):
            state, c, done = step(state, step_key)
            steps, calls = steps + 1, calls + int(c)
            if done:
                return steps, calls


def test_blackjax_stops_at_the_first_step_that_meets_dlogz():
    """The rule is tested after every step, as in a step-by-step loop; the
    adapter used to test it once per 20-step chunk and ran on (+18-20% calls
    on the 2-D targets at nlive 500)."""
    reason = adapters.unavailable_reason("blackjax_nss")
    if reason:
        pytest.skip(reason)
    t = get_target("gauss_d2")
    cfg = dict(nlive=100, dlogz=0.1, variant="default", opts={})
    for seed in (0, 1):
        out = adapters.load("blackjax_nss").run(t, seed, cfg)
        steps, calls = _blackjax_reference(t, seed, 100, 0.1)
        assert (out["config"]["niter_steps"], out["ncall"]) == (steps, calls)
        assert abs(out["logz"] - t.logz) < 5 * max(out["logzerr"], 0.2)
    # maxiter stops at that step, not at the end of a chunk
    out = adapters.load("blackjax_nss").run(t, 0, dict(cfg, opts={"maxiter": "7"}))
    assert out["config"]["niter_steps"] == 7
    # and one past the int32 range is no limit (the loop compares in int32)
    big = dict(cfg, opts={"maxiter": str(2**31)})
    out = adapters.load("blackjax_nss").run(t, 1, big)
    assert (out["config"]["niter_steps"], out["ncall"]) == (steps, calls)


def test_vectorized_likelihood_compiles_once_per_power_of_two():
    """A batch is padded to the next power of two, so 1 to 33 points take 7
    compiles (UltraNest asks for 56 to 59 batch sizes in one run), and the
    padding never reaches the result."""
    import jax.numpy as jnp
    from bench.adapters._common import numpy_callables

    traces = []

    def loglike(x):
        traces.append(x.shape)  # runs once per compile
        return -0.5 * jnp.sum(x**2)

    target = SimpleNamespace(loglike=loglike, prior_transform_np=lambda u: u)
    f, _ = numpy_callables(target, vectorized=True)
    rng = np.random.default_rng(0)
    for n in range(1, 34):
        x = rng.normal(size=(n, 3))
        assert np.allclose(f(x), -0.5 * np.sum(x**2, axis=1))
    assert len(traces) == 7
    assert f(np.ones(3)).shape == (1,)  # one point, as UltraNest sends it


def _record(sampler, target, seed, nlive):
    return dict(
        schema="tinyns-bench-1", sampler=sampler, target=target, seed=seed,
        ndim=2, status="ok", truth=dict(logz=0.0, mode_mass=None),
        config=dict(cli=dict(nlive=nlive, dlogz=0.1, opts={})),
        hw=dict(gpu=None, cpu_model="cpu", cpu_threads=4),
        logz=0.01 * (seed % 5 - 2), logzerr=0.1, ncall=1000 * nlive, wall_s=1.0,
        compile_s=None, mode_mass=None,
    )


def test_headline_counts_reference_cells(tmp_path):
    """With the reference at two nlive, one target is two reference cells,
    and the headline says it counts cells, not targets."""
    from bench import summarize

    recs = [
        _record(s, "gauss_d2", seed, n)
        for seed in range(10)
        for s, n in (("ref", 500), ("ref", 1000), ("other", 500))
    ]
    path = tmp_path / "r.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    rows = summarize.summarize(summarize.load([path]))
    ratios, _ = summarize.headline(rows, "ref")
    assert (ratios[0]["common"], ratios[0]["ran"]) == (2, 2)
    md = summarize.headline_markdown(rows, "ref")
    assert "| common / ran cells |" in md
    assert "| gauss | other | CPU 4c | 2/2 |" in md
    assert "over the common accurate targets" not in md
