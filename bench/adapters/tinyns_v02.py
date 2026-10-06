"""tinyns 0.x: the ``release/0.x`` branch at 2f67111.

That commit is the v0.3 series in progress (PRs 1-6 merged on main), which is
v0.2.5 behaviour with the code pruned; its ``__version__`` still reads 0.2.5.
``NestedSampler(loglike, prior_transform, ndim, nlive, *, walks=None,
replacement_chains=1, block_size=32, cluster_swap=None)``.

To run it in an env whose installed tinyns is v1, point ``TINYNS_V02_SRC`` at a
checkout's ``src`` directory (``git worktree add ../tinyns-0x release/0.x``);
the adapter puts it first on ``sys.path`` before importing ``tinyns``.

Variants: ``default`` (cluster swap on, the 0.2.5 default) and ``noswap``.
"""

from __future__ import annotations

import os
import sys

import numpy as np

from bench.adapters import AdapterUnavailable
from bench.adapters._common import Timer

VARIANTS = ("default", "noswap")


def _tinyns():
    src = os.environ.get("TINYNS_V02_SRC")
    if src and "tinyns" not in sys.modules:
        sys.path.insert(0, src)
    import tinyns

    if not str(tinyns.__version__).startswith("0."):
        raise AdapterUnavailable(
            f"tinyns {tinyns.__version__} is not 0.x; set TINYNS_V02_SRC"
        )
    return tinyns


def unavailable_reason():
    try:
        _tinyns()
    except AdapterUnavailable as err:
        return str(err)
    return None


def run(target, seed, cfg):
    import jax

    tinyns = _tinyns()
    opts = dict(cfg.get("opts", {}))
    kw = dict(cluster_swap=cfg["variant"] != "noswap")
    for k in ("walks", "replacement_chains", "block_size"):
        if k in opts:
            kw[k] = int(opts[k])
    sampler = tinyns.NestedSampler(
        target.loglike, target.prior_transform, target.ndim, cfg["nlive"], **kw
    )
    with Timer() as t:
        res = sampler.run(jax.random.PRNGKey(seed), dlogz=cfg["dlogz"])
    meta = res.metadata or {}
    return dict(
        samples=np.asarray(res.samples),
        logwt=np.asarray(res.logwt),
        logz=float(res.logz),
        logzerr=float(res.logzerr),
        ncall=int(res.ncall),
        ncall_valid=None,
        wall_s=t.s,
        compile_s=meta.get("compile_s"),
        sampler_version=f"tinyns {tinyns.__version__} ({tinyns.__file__})",
        config=dict(
            nlive=cfg["nlive"],
            dlogz=cfg["dlogz"],
            **kw,
            walks_resolved=meta.get("walks"),
            success=bool(res.success),
        ),
    )
