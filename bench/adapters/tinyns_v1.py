"""tinyns v1 (the ``v1`` branch), against the planned API:

    tinyns.NestedSampler(loglike, prior_transform, ndim, nlive,
                         num_delete=..., walks=...).run(key)

This adapter becomes usable when v1 PR 1 lands (``NestedSampler`` grows a
``num_delete`` argument); until then ``unavailable_reason`` says so. Result
fields read with fallbacks, since PR 1 may still rename them.

Options (``--opt``): ``num_delete`` (default: the sampler's, nlive // 10),
``walks``. Variants: ``default`` and ``k1`` (num_delete = 1).
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
    sampler = tinyns.NestedSampler(
        target.loglike, target.prior_transform, target.ndim, cfg["nlive"], **kw
    )
    with Timer() as t:
        res = sampler.run(jax.random.PRNGKey(seed), dlogz=cfg["dlogz"])
    meta = getattr(res, "metadata", None) or {}

    def field(name, default=None):
        return getattr(res, name, meta.get(name, default))

    ncv = field("ncall_valid")
    return dict(
        samples=np.asarray(res.samples),
        logwt=np.asarray(res.logwt),
        logz=float(res.logz),
        logzerr=float(res.logzerr),
        ncall=int(res.ncall),
        ncall_valid=None if ncv is None else int(ncv),
        wall_s=t.s,
        compile_s=field("compile_s"),
        sampler_version=f"tinyns {tinyns.__version__} ({tinyns.__file__})",
        config=dict(
            nlive=cfg["nlive"],
            dlogz=cfg["dlogz"],
            **kw,
            num_delete_resolved=field("num_delete"),
            walks_resolved=field("walks"),
        ),
    )
