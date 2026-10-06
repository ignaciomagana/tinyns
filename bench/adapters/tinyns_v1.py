"""tinyns v1 (the ``v1`` branch; ``__version__`` 1.x):

    tinyns.NestedSampler(loglike, prior_transform, ndim, nlive, *,
                         num_delete=None, walks=None).run(key, dlogz=...)

Defaults resolve in ``Config``: ``num_delete = max(1, nlive // 10)`` chains run
vmapped per step, ``walks = max(25, 6 * ndim)`` steps each. ``ncall`` counts
every evaluation, including out-of-cube lanes that the vmapped chains accept
and waste; ``metadata["ncall_valid"]`` counts only the in-cube ones.
``compile_s`` is ``metadata["compile_s"]``: v1 compiles its kernel in a
zero-step chunk and times it apart (it is inside ``wall_s``).

This adapter ignores ``TINYNS_V02_SRC``; only ``tinyns_v02`` reads it. Within a
single Python process the first adapter to import ``tinyns`` decides which
tree is loaded. ``run.py`` uses one process per seed, so this does not arise
there.

Variants: ``default`` and ``k1`` (``num_delete = 1``: one unbatched chain).
Options (``--opt``): ``num_delete``, ``walks``, ``maxiter``, ``maxcall``.
"""

from __future__ import annotations

import inspect

import numpy as np

from bench.adapters._common import Timer

VARIANTS = ("default", "k1")


def unavailable_reason():
    import tinyns

    params = inspect.signature(tinyns.NestedSampler).parameters
    if "num_delete" not in params:
        return f"tinyns {tinyns.__version__} predates the v1 API (no num_delete)"
    return None


def run(target, seed, cfg):
    import jax

    import tinyns

    opts = dict(cfg.get("opts", {}))
    kw = {}
    if cfg["variant"] == "k1":
        kw["num_delete"] = 1
    for k in ("num_delete", "walks"):
        if k in opts:
            kw[k] = int(opts[k])
    run_kw = {k: int(opts[k]) for k in ("maxiter", "maxcall") if k in opts}
    sampler = tinyns.NestedSampler(
        target.loglike, target.prior_transform, target.ndim, cfg["nlive"], **kw
    )
    with Timer() as t:
        res = sampler.run(jax.random.PRNGKey(seed), dlogz=cfg["dlogz"], **run_kw)
    meta = res.metadata or {}
    conf = getattr(sampler, "config", None)
    ncv = meta.get("ncall_valid")
    return dict(
        samples=np.asarray(res.samples),
        logwt=np.asarray(res.logwt),
        logz=float(res.logz),
        logzerr=float(res.logzerr),
        ncall=int(res.ncall),
        ncall_valid=None if ncv is None else int(ncv),
        wall_s=t.s,
        compile_s=meta.get("compile_s"),
        sampler_version=f"tinyns {tinyns.__version__} ({tinyns.__file__})",
        config=dict(
            nlive=cfg["nlive"],
            dlogz=cfg["dlogz"],
            **kw,
            **run_kw,
            num_delete_resolved=int(res.num_delete),
            walks_resolved=getattr(conf, "walks", None),
            niter=int(res.niter),
            success=bool(res.success),
            message=res.message,
        ),
    )
