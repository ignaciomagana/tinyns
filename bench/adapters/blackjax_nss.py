"""BlackJAX nested slice sampling, ``blackjax.nss`` (``blackjax/ns/``, PR #947).

``num_inner_steps = max(5, 2 * ndim)`` (hit-and-run slice moves shaped by the
live covariance) follows the ``as_top_level_api`` docstring. ``num_delete =
nlive // 10`` is the bench's choice, not BlackJAX's (its default is 1): it is
tinyns's default batch, so both run that many chains per step. The sampler
runs in the unit cube: the log-prior is 0 inside and -inf outside, and the
log-likelihood is -inf outside, so the slice never accepts an out-of-cube
point.

Driver: a compiled ``lax.while_loop`` of up to ``2 * ceil(nlive / num_delete)``
steps (about two e-folds) per host round trip, which writes each step's dead
points into a preallocated buffer and stops after the first step at which
``logaddexp(logZ, logZ_live) - logZ < dlogz`` (``logZ_live`` is the
integrator's mean-live-likelihood estimate), as a step-by-step loop would.
``logz``/``logzerr`` are the mean/sd over 100 simulated volume sequences of
``blackjax.ns.utils.log_weights`` on the ``finalise``-d run (dead plus final
live points), the BlackJAX convention.

``ncall`` counts per-chain slice evaluations (2 bracket ends + expansions +
shrinks per inner step, plus nlive at init); the vmapped lanes that idle while
the slowest chain finishes are not counted.

``compile_s`` is the ahead-of-time compile of ``init`` and of the chunk, plus
the first-call compiles of the post-processing (``finalise``, ``log_weights``,
the prior map). Those run eagerly, op by op, on arrays whose length depends on
the run, so each compiles on its first call in a process, and each is too small
for the persistent compilation cache: the post-processing is run once more
after the timed run, and the difference of the two times is counted as
compilation. ``config`` keeps both (``post_first_s``, ``post_s``).

Options: ``num_delete``, ``num_inner_steps``, ``maxiter`` (NS steps).
"""

from __future__ import annotations

import math
import time

import numpy as np

from bench.adapters._common import Timer, version_of

VARIANTS = ("default",)


def unavailable_reason():
    import blackjax

    if not hasattr(blackjax, "nss"):
        return f"blackjax {blackjax.__version__} has no nss"
    return None


def run(target, seed, cfg):
    import blackjax
    import jax
    import jax.numpy as jnp
    from blackjax.ns import utils as nsu
    from blackjax.ns.base import NSInfo
    from jax.scipy.special import logsumexp

    opts = dict(cfg.get("opts", {}))
    d, nlive = target.ndim, int(cfg["nlive"])
    num_delete = int(opts.get("num_delete", max(1, nlive // 10)))
    num_inner = int(opts.get("num_inner_steps", max(5, 2 * d)))
    maxiter = int(opts.get("maxiter", 10_000_000))
    nsteps = 2 * math.ceil(nlive / num_delete)
    dlogz = float(cfg["dlogz"])

    def inside(u):
        return jnp.all((u >= 0.0) & (u <= 1.0))

    def loglike_u(u):
        ll = target.loglike(target.prior_transform(jnp.clip(u, 0.0, 1.0)))
        return jnp.where(inside(u), ll, -jnp.inf)

    def logprior_u(u):
        return jnp.where(inside(u), 0.0, -jnp.inf)

    algo = blackjax.nss(
        logprior_fn=logprior_u,
        loglikelihood_fn=loglike_u,
        num_delete=num_delete,
        num_inner_steps=num_inner,
    )

    def init(key):
        return algo.init(jax.random.uniform(key, (nlive, d)))

    def converged(state):
        lz, lzl = state.integrator.logZ, state.integrator.logZ_live
        return jnp.logaddexp(lz, lzl) - lz < dlogz

    def chunk(state, key, it):
        """Up to ``nsteps`` steps; stops after the first that converges or
        reaches ``maxiter``. Returns the state, the next key, the dead points
        (``nsteps * num_delete`` rows, the first ``n * num_delete`` valid),
        the ``n`` steps taken, their calls and whether the run converged."""
        key, sub = jax.random.split(key)
        keys = jax.random.split(sub, nsteps)
        dead = jax.eval_shape(lambda s, k: algo.step(k, s)[1].particles, state, sub)
        buf = jax.tree.map(lambda x: jnp.zeros((nsteps,) + x.shape, x.dtype), dead)

        def more(carry):
            i, s, _, _ = carry
            return (i < nsteps) & (it + i < maxiter) & ~converged(s)

        def body(carry):
            i, s, buf, calls = carry
            s, info = algo.step(keys[i], s)
            ui = info.update_info
            step_calls = jnp.sum(2 + ui.num_expansions + ui.num_shrink)
            calls = calls + step_calls.astype(jnp.int32)
            buf = jax.tree.map(lambda b, x: b.at[i].set(x), buf, info.particles)
            return i + 1, s, buf, calls

        zero = jnp.zeros((), jnp.int32)
        n, state, buf, calls = jax.lax.while_loop(more, body, (zero, state, buf, zero))
        flat = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), buf)
        return state, key, flat, n, calls, converged(state)

    def post(state, dead):
        """logZ draws, mean log weights and samples of the finished run."""
        infos = [
            NSInfo(jax.tree.map(lambda x, n=n: x[: n * num_delete], p), None)
            for p, n in dead
        ]
        final = nsu.finalise(state, infos, update_info=False)
        logw = nsu.log_weights(jax.random.PRNGKey(seed + 1), final, shape=100)
        logzs = np.asarray(logsumexp(logw, axis=0))
        samples = np.asarray(jax.vmap(target.prior_transform)(final.particles.position))
        return logzs, np.asarray(jnp.mean(logw, axis=-1)), samples

    key = jax.random.PRNGKey(seed)
    key, k_init = jax.random.split(key)
    it0 = np.int32(0)
    with Timer() as wall:
        with Timer() as c1:
            init_c = jax.jit(init).lower(k_init).compile()
        state = init_c(k_init)
        with Timer() as c2:
            chunk_c = jax.jit(chunk).lower(state, key, it0).compile()
        dead, ncall, it = [], nlive, 0
        while True:
            state, key, parts, n, calls, done = chunk_c(state, key, np.int32(it))
            n, calls, done = (int(x) for x in jax.device_get((n, calls, done)))
            if n:
                dead.append((parts, n))
            ncall += calls
            it += n
            if done or it >= maxiter:
                break
        with Timer() as p1:
            logzs, logwt, samples = post(state, dead)
    tick = time.perf_counter()
    post(state, dead)  # warm: the same work, compiled
    post_s = time.perf_counter() - tick
    return dict(
        samples=samples,
        logwt=logwt,
        logz=float(np.mean(logzs)),
        logzerr=float(np.std(logzs)),
        ncall=int(ncall),
        ncall_valid=None,
        wall_s=wall.s,
        compile_s=c1.s + c2.s + max(p1.s - post_s, 0.0),
        sampler_version=f"blackjax {version_of('blackjax')}",
        config=dict(
            nlive=nlive,
            dlogz=dlogz,
            num_delete=num_delete,
            num_inner_steps=num_inner,
            steps_per_chunk=nsteps,
            niter_steps=it,
            post_first_s=p1.s,
            post_s=post_s,
        ),
    )
