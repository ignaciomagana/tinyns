"""BlackJAX nested slice sampling, ``blackjax.nss`` (``blackjax/ns/``, PR #947).

Settings follow the ``as_top_level_api`` docstring: ``num_delete = nlive // 10``
and ``num_inner_steps = max(5, 2 * ndim)`` (hit-and-run slice moves shaped by
the live covariance). The sampler runs in the unit cube: the log-prior is 0
inside and -inf outside, and the log-likelihood is -inf outside, so the slice
never accepts an out-of-cube point.

Driver: a jitted ``lax.scan`` of ``2 * ceil(nlive / num_delete)`` steps (about
two e-folds) per host round trip, stopping when
``logaddexp(logZ, logZ_live) - logZ < dlogz`` (``logZ_live`` is the integrator's
mean-live-likelihood estimate). ``logz``/``logzerr`` are the mean/sd over 100
simulated volume sequences of ``blackjax.ns.utils.log_weights`` on the
``finalise``-d run (dead plus final live points), the BlackJAX convention.

``ncall`` counts per-chain slice evaluations (2 bracket ends + expansions +
shrinks per inner step, plus nlive at init); the vmapped lanes that idle while
the slowest chain finishes are not counted. ``compile_s`` is the ahead-of-time
compile of ``init`` and of the scan chunk.

Options: ``num_delete``, ``num_inner_steps``, ``maxiter`` (NS steps).
"""

from __future__ import annotations

import math

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

    def chunk(state, key):
        def body(s, k):
            s, info = algo.step(k, s)
            ui = info.update_info
            calls = jnp.sum(2 + ui.num_expansions + ui.num_shrink)
            return s, (info.particles, calls)

        return jax.lax.scan(body, state, jax.random.split(key, nsteps))

    key = jax.random.PRNGKey(seed)
    key, k_init = jax.random.split(key)
    with Timer() as wall:
        u0 = jax.random.uniform(k_init, (nlive, d))
        with Timer() as c1:
            init_c = jax.jit(algo.init).lower(u0).compile()
        state = init_c(u0)
        with Timer() as c2:
            chunk_c = jax.jit(chunk).lower(state, key).compile()
        dead, ncall, it = [], nlive, 0
        while True:
            key, sub = jax.random.split(key)
            state, (parts, calls) = chunk_c(state, sub)
            dead.append(jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), parts))
            ncall += int(jnp.sum(calls))
            it += nsteps
            lz, lzl = float(state.integrator.logZ), float(state.integrator.logZ_live)
            if np.logaddexp(lz, lzl) - lz < dlogz or it >= maxiter:
                break
        final = nsu.finalise(state, [NSInfo(p, None) for p in dead], update_info=False)
        logw = nsu.log_weights(jax.random.PRNGKey(seed + 1), final, shape=100)
        logzs = np.asarray(logsumexp(logw, axis=0))
        samples = np.asarray(jax.vmap(target.prior_transform)(final.particles.position))
    return dict(
        samples=samples,
        logwt=np.asarray(jnp.mean(logw, axis=-1)),
        logz=float(np.mean(logzs)),
        logzerr=float(np.std(logzs)),
        ncall=int(ncall),
        ncall_valid=None,
        wall_s=wall.s,
        compile_s=c1.s + c2.s,
        sampler_version=f"blackjax {version_of('blackjax')}",
        config=dict(
            nlive=nlive,
            dlogz=dlogz,
            num_delete=num_delete,
            num_inner_steps=num_inner,
            steps_per_chunk=nsteps,
            niter_steps=it,
        ),
    )
