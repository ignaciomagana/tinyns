"""Nautilus (``nautilus-sampler`` 1.0.6).

Variants: ``default`` (``Sampler`` and ``run`` defaults: ``n_live=2000``,
``f_live=0.01``, ``n_eff=10000``) and ``discard`` (``discard_exploration=True``,
which drops the exploration-phase points for an unbiased posterior/evidence).
Nautilus has its own live-set size and ignores ``--nlive`` unless
``--opt n_live=...`` is given. Vectorised likelihood (jit+vmap on CPU),
``seed=seed``. ``--opt pool=N`` passes ``pool=(None, N)``: N worker processes
for the sampler's own calculations (it trains its ``n_networks = 4`` networks
in parallel), none for the likelihood, which is already a vectorised call.
Without it Nautilus trains the networks one after the other.

Nautilus reports no evidence uncertainty; ``logzerr`` is None. (Its paper uses
``1 / sqrt(n_eff)`` as a rough scale; ``ess`` is in the record.)
"""

from __future__ import annotations

import numpy as np

from bench.adapters._common import Timer, numpy_callables, version_of

VARIANTS = ("default", "discard")


def unavailable_reason():
    return None


def run(target, seed, cfg):
    from nautilus import Sampler

    opts = dict(cfg.get("opts", {}))
    loglike, ptform = numpy_callables(target, vectorized=True)
    kw = {}
    if "n_live" in opts:
        kw["n_live"] = int(opts["n_live"])
    if "pool" in opts:
        kw["pool"] = (None, int(opts["pool"]))
    discard = cfg["variant"] == "discard"
    with Timer() as t:
        sampler = Sampler(
            ptform,
            loglike,
            n_dim=target.ndim,
            vectorized=True,
            pass_dict=False,
            seed=seed,
            **kw,
        )
        sampler.run(discard_exploration=discard, verbose=False)
        points, log_w, _ = sampler.posterior()
    return dict(
        samples=np.asarray(points),
        logwt=np.asarray(log_w),
        logz=float(sampler.log_z),
        logzerr=None,
        ncall=int(sampler.n_like),
        ncall_valid=None,
        wall_s=t.s,
        compile_s=None,
        sampler_version=f"nautilus-sampler {version_of('nautilus-sampler')}",
        config=dict(
            n_live=int(sampler.n_live),
            pool=opts.get("pool"),
            discard_exploration=discard,
            n_eff=float(sampler.n_eff),
        ),
    )
