"""UltraNest ``ReactiveNestedSampler``.

Variants: ``default`` (MLFriends region rejection sampling, the UltraNest
default) and ``slice`` (``SliceSampler(nsteps=2 * ndim,
generate_direction=generate_mixture_random_direction)``, the step sampler
UltraNest recommends above ~10 D). ``min_num_live_points = nlive``; every
other run setting stays at its default (``dlogz=0.5``, ``frac_remain=0.01``,
``min_ess=400``). The CLI ``--dlogz`` is NOT passed: UltraNest's ``dlogz`` is a
target evidence *uncertainty*, and 0.1 makes it add live points (613-1027 at
nlive 100 on gauss_d2); use ``--opt dlogz=...`` to set it. Vectorised
likelihood (jit+vmap on CPU), no output files; seeded through
``np.random.seed`` (UltraNest draws from the global state).
Options: ``nsteps`` (slice), ``max_ncalls``.
"""

from __future__ import annotations

import numpy as np

from bench.adapters._common import Timer, numpy_callables, version_of

VARIANTS = ("default", "slice")


def unavailable_reason():
    return None


def run(target, seed, cfg):
    import logging

    import ultranest
    import ultranest.stepsampler as ss

    opts = dict(cfg.get("opts", {}))
    d = target.ndim
    loglike, ptform = numpy_callables(target, vectorized=True)
    np.random.seed(seed)
    dlogz = float(opts.get("dlogz", 0.5))
    conf = dict(nlive=cfg["nlive"], dlogz=dlogz, region="MLFriends")
    with Timer() as t:
        sampler = ultranest.ReactiveNestedSampler(
            [f"x{i}" for i in range(d)],
            loglike,
            ptform,
            vectorized=True,
            log_dir=None,
        )
        logging.getLogger("ultranest").setLevel(logging.WARNING)
        if cfg["variant"] == "slice":
            nsteps = int(opts.get("nsteps", 2 * d))
            sampler.stepsampler = ss.SliceSampler(
                nsteps=nsteps,
                generate_direction=ss.generate_mixture_random_direction,
            )
            conf.update(stepsampler="SliceSampler(mixture)", nsteps=nsteps)
        res = sampler.run(
            min_num_live_points=cfg["nlive"],
            dlogz=dlogz,
            show_status=False,
            viz_callback=False,
            max_ncalls=int(opts["max_ncalls"]) if "max_ncalls" in opts else None,
        )
    ws = res["weighted_samples"]
    w = np.asarray(ws["weights"], float)
    with np.errstate(divide="ignore"):
        logwt = np.log(w)
    return dict(
        samples=np.asarray(ws["points"]),
        logwt=logwt,
        logz=float(res["logz"]),
        logzerr=float(res["logzerr"]),
        ncall=int(res["ncall"]),
        ncall_valid=None,
        wall_s=t.s,
        compile_s=None,
        sampler_version=f"ultranest {version_of('ultranest')}",
        config=conf,
    )
