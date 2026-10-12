"""dynesty static ``NestedSampler``.

Variants: ``default`` (``bound='multi', sample='auto'``: uniform sampling in
multi-ellipsoids below 10 D, random walk at 10-20 D, slice sampling above) and
``rwalk100`` (``sample='rwalk', walks=100``). The likelihood is the target's
JAX function, jitted on CPU and called one point at a time; ``rstate`` is
``np.random.default_rng(seed)``. Options: ``maxcall``, ``walks``, ``bound``.
"""

from __future__ import annotations

import numpy as np

from bench.adapters._common import Timer, numpy_callables, version_of

VARIANTS = ("default", "rwalk100")


def unavailable_reason():
    return None


def run(target, seed, cfg):
    import dynesty

    opts = dict(cfg.get("opts", {}))
    loglike, ptform = numpy_callables(target, vectorized=False)
    kw = dict(bound=opts.get("bound", "multi"), sample="auto")
    if cfg["variant"] == "rwalk100":
        kw.update(sample="rwalk", walks=100)
    if "walks" in opts:
        kw["walks"] = int(opts["walks"])
    with Timer() as t:
        sampler = dynesty.NestedSampler(
            loglike,
            ptform,
            target.ndim,
            nlive=cfg["nlive"],
            rstate=np.random.default_rng(seed),
            **kw,
        )
        sampler.run_nested(
            dlogz=cfg["dlogz"],
            print_progress=False,
            maxcall=int(opts["maxcall"]) if "maxcall" in opts else None,
        )
        res = sampler.results
    return dict(
        samples=np.asarray(res.samples),
        logwt=np.asarray(res.logwt),
        logz=float(res.logz[-1]),
        logzerr=float(res.logzerr[-1]),
        ncall=int(np.sum(res.ncall)),
        ncall_valid=None,
        wall_s=t.s,
        compile_s=None,
        sampler_version=f"dynesty {version_of('dynesty')}",
        config=dict(
            nlive=cfg["nlive"],
            dlogz=cfg["dlogz"],
            **kw,
            sample_resolved=getattr(sampler, "method", None),
        ),
    )
