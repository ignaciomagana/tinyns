"""JAXNS 3 (race-tree nested sampling; ``jaxns.core.NestedSampler``).

Runs in its own env (``bench_jaxns``): jaxns pins ``tfp-nightly`` and
``jaxctx``. The model is one vector prior ``Uniform(lo, hi)`` and the target
likelihood. Variants: ``default`` (all JAXNS defaults: ``30 * ndim`` root
chains, ``5 * ndim`` slices per chain, stop at ``dlogZ = log1p(1e-3)``) and
``nlive`` (``root_allocation_degree = nlive``). ``logz``/``logzerr`` are the
classic ``log_Z_mean``/``log_Z_uncert``; weights are ``log_dp``.
"""

from __future__ import annotations

import numpy as np

from bench.adapters._common import Timer, version_of

VARIANTS = ("default", "nlive")


def unavailable_reason():
    try:
        from jaxns.core import NestedSampler  # noqa: F401
    except ImportError as err:  # jaxns 2 has no jaxns.core.NestedSampler
        return f"jaxns 3 API not found: {err}"
    return None


def run(target, seed, cfg):
    import jax
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp
    from jaxns.core import NestedSampler
    from jaxns.model import Model
    from jaxns.priors import Prior

    tfpd = tfp.distributions
    lo, hi = jnp.asarray(target.lo), jnp.asarray(target.hi)

    def prior_model():
        x = Prior(tfpd.Uniform(low=lo, high=hi), name="x").realise()
        return target.loglike(x)

    kw = {}
    if cfg["variant"] == "nlive":
        kw["root_allocation_degree"] = int(cfg["nlive"])
    with Timer() as t:
        sampler = NestedSampler(model=Model(prior_model=prior_model), **kw)
        state = sampler.run(key=jax.random.PRNGKey(seed))
        res = state.to_result().trim()
        x = res.X_samples["x"]  # a jaxctx ScopedDict keyed by prior name
        samples = np.asarray(x).reshape(-1, target.ndim)
    return dict(
        samples=samples,
        logwt=np.asarray(res.log_dp),
        logz=float(res.log_Z_mean),
        logzerr=float(res.log_Z_uncert),
        ncall=int(res.total_num_likelihood_evaluations),
        ncall_valid=None,
        wall_s=t.s,
        compile_s=None,
        sampler_version=f"jaxns {version_of('jaxns')}",
        config=dict(**kw, termination_reason=int(res.termination_reason)),
    )
