"""Run one bake-off cell: (target, arm, nlive, k) for a list of seeds.

    python bench/bakeoff/run.py --target sepW_d18 --arm B_ell --nlive 500 --k 50 \\
        --seeds 0-39 --out results.jsonl

All seeds run as one batched tinyns run (``core.run`` with a batch of keys:
one compiled program, the step vmapped over the seeds), so a cell is one GPU
job. One ``tinyns-bakeoff-1`` JSON line per seed is appended to ``--out``
under ``flock``. ``wall_s`` and ``compile_s`` are the batch's (shared by all
its seeds); ``wall_per_seed_s`` divides the wall time by the seed count.

Per seed the record holds the evidence (``logz``, ``logzerr``), the calls
(``ncall``, ``ncall_valid``), the oracle mode masses (the target's
responsibility function on the weighted samples, as in ``bench/run.py``) and
their Kish effective sample sizes, the mode-move telemetry (``hops``,
``hop_tries``, ``hop_acceptance``, ``mode_history``) and, for mixtures, the
detection diagnostics:

- ``isolation_niter[k]``: the first death above the main mode's density at
  the centre of minor mode ``k`` (where the main contour stops containing
  it; chains stop crossing around there);
- ``detection_niter``: the first death at which the clustering held at least
  two clusters (``eligible_niter``: two frames eligible for the moves);
- ``minor_live[k]``: the oracle live count of minor mode ``k`` (live points
  whose responsibility for it exceeds 1/2) every ``live_stride`` deaths, so
  the summary can read it at any detection time, also for the arm ``N``;
- ``modes``: :meth:`tinyns.NestedSamplingResult.modes` (mass, urn sd,
  min_live, unresolved), the per-mode report read from the run's labels.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.run import append_jsonl, cpu_model, git_sha, parse_seeds  # noqa: E402

SCHEMA = "tinyns-bakeoff-1"
# B_ell is tinyns's always-on hop; N turns it off (the private Config._hop),
# keeping the rest (the folds, the clustering and the local walk).
ARMS = ("N", "B_ell")


def isolation_levels(target) -> list[float] | None:
    """Log likelihood at which each minor mode becomes isolated: the main
    component's (weighted) log density at the minor mode's centre."""
    from bench.targets import mixture_spec

    spec = mixture_spec(target.name)
    if spec is None:
        return None
    import numpy as np

    mu0, s0 = spec["means"][0], spec["covs"][0]
    d = len(mu0)
    const = math.log(spec["weights"][0]) - 0.5 * (
        d * math.log(2 * math.pi) + np.linalg.slogdet(s0)[1]
    )
    out = []
    for mu in spec["means"][1:]:
        y = mu - mu0
        out.append(float(const - 0.5 * y @ np.linalg.solve(s0, y)))
    return out


def birth_niter(dead_logl, logl_birth):
    """Number of deaths before each sample was born (0 for the first set)."""
    import numpy as np

    return np.searchsorted(dead_logl, logl_birth, side="right")


def live_trajectory(result, resp, stride):
    """Oracle live counts of the minor modes every ``stride`` deaths.

    A sample is live from its birth (the deaths before its birth contour) to
    its own death (its index among the dead points; the final live points
    never die): it is live at ``g`` deaths if ``born <= g <= died``. Returns
    a list per minor mode.
    """
    import numpy as np

    niter = result.niter
    logl = np.asarray(result.logl)
    born = birth_niter(logl[:niter], np.asarray(result.logl_birth))
    died = np.r_[np.arange(niter), np.full(len(logl) - niter, niter + 1)]
    grid = np.arange(0, niter + 1, stride)
    out = []
    for k in range(1, resp.shape[1]):
        mine = resp[:, k] > 0.5
        alive = np.zeros(len(grid) + 1, int)  # the last slot takes the overflow
        np.add.at(alive, np.searchsorted(grid, born[mine], side="left"), 1)
        np.add.at(alive, np.searchsorted(grid, died[mine], side="right"), -1)
        out.append(np.cumsum(alive)[:-1].tolist())
    return out


def lane_record(target, result, resp, args, seed, cfg, extra):
    import numpy as np

    lw = np.asarray(result.logwt, float)
    w = np.exp(lw - lw.max())
    w /= w.sum()
    rec = dict(
        seed=seed,
        logz=float(result.logz),
        logzerr=float(result.logzerr),
        ncall=int(result.ncall),
        ncall_valid=result.metadata.get("ncall_valid"),
        niter=int(result.niter),
        success=bool(result.success),
        message=result.message,
        ess=float(1.0 / np.sum(w**2)),
        acceptance=result.metadata.get("acceptance"),
        hops=result.metadata.get("hops"),
        hop_tries=result.metadata.get("hop_tries"),
        hop_acceptance=result.metadata.get("hop_acceptance"),
        mode_history=result.metadata.get("mode_history"),
        modes=result.modes(),
    )
    hist = rec["mode_history"] or []
    rec["detection_niter"] = next((n for n, c, _ in hist if c >= 2), None)
    rec["eligible_niter"] = next((n for n, _, e in hist if e >= 2), None)
    rec["max_clusters"] = max((c for _, c, _ in hist), default=None)
    if resp is not None:
        wr = w[:, None] * resp
        rec["mode_mass"] = wr.sum(0).tolist()
        rec["mode_ess"] = (wr.sum(0) ** 2 / np.maximum((wr**2).sum(0), 1e-300)).tolist()
        levels = isolation_levels(target)
        if levels is not None:
            dead = np.asarray(result.logl[: result.niter])
            rec["isolation_niter"] = [
                int(np.searchsorted(dead, level, side="right")) for level in levels
            ]
        stride = max(1, round(cfg.nlive / 4))
        rec["live_stride"] = stride
        rec["minor_live"] = live_trajectory(result, resp, stride)
    rec.update(extra)
    return rec


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--target", required=True)
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--nlive", type=int, default=500)
    p.add_argument("--k", type=int, default=None, help="num_delete (default nlive/10)")
    p.add_argument("--walks", type=int, default=None)
    p.add_argument("--seeds", default="0-39")
    p.add_argument("--dlogz", type=float, default=0.1)
    p.add_argument("--out", required=True)
    p.add_argument("--x64", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tag", default=None)
    args = p.parse_args(argv)
    if "jax" not in sys.modules:  # in a fresh process, before JAX starts
        os.environ["JAX_ENABLE_X64"] = "1" if args.x64 else "0"

    import jax
    import numpy as np

    from bench.targets import get_target
    from tinyns import __version__, core

    target = get_target(args.target)
    seeds = parse_seeds(args.seeds)
    k = args.k if args.k is not None else max(1, args.nlive // 10)
    cfg = core.Config(
        target.ndim, args.nlive, k, args.walks, _hop=args.arm == "B_ell"
    )
    keys = jax.numpy.stack([jax.random.PRNGKey(s) for s in seeds])
    t0 = time.perf_counter()
    results = core.run(
        keys, target.loglike, target.prior_transform, cfg, dlogz=args.dlogz
    )
    wall = time.perf_counter() - t0
    resp_fn = None
    if target.responsibility is not None:
        resp_fn = jax.jit(jax.vmap(target.responsibility))
    dev = jax.devices()[0]
    common = dict(
        schema=SCHEMA,
        target=target.name,
        ndim=target.ndim,
        arm=args.arm,
        nlive=args.nlive,
        num_delete=k,
        walks=cfg.walks,
        recluster_every=cfg.recluster_every,
        dlogz=args.dlogz,
        nseeds=len(seeds),
        truth=dict(
            logz=target.logz,
            mode_mass=list(target.mode_mass) if target.mode_mass else None,
        ),
        wall_s=wall,
        wall_per_seed_s=wall / len(seeds),
        compile_s=results[0].metadata.get("compile_s"),
        tinyns=__version__,
        git_sha=git_sha(),
        jax=jax.__version__,
        x64=bool(jax.config.jax_enable_x64),
        hw=dict(
            host=socket.gethostname(),
            platform=dev.platform,
            device=dev.device_kind,
            cpu_model=cpu_model(),
            slurm_job=os.environ.get("SLURM_JOB_ID"),
        ),
        tag=args.tag,
        ts=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    )
    for seed, result in zip(seeds, results, strict=True):
        resp = None
        if resp_fn is not None:
            x = np.asarray(result.samples)
            resp = np.concatenate(
                [
                    np.asarray(resp_fn(x[s : s + 65536]), float)
                    for s in range(0, len(x), 65536)
                ]
            )
            resp = np.nan_to_num(resp)
        rec = lane_record(target, result, resp, args, seed, cfg, common)
        append_jsonl(args.out, rec)
        masses = rec.get("mode_mass")
        print(
            f"{args.target} {args.arm} m={args.nlive} k={k} seed={seed} "
            f"dlogz={rec['logz'] - (target.logz or 0):+.3f} err={rec['logzerr']:.3f} "
            f"ncall={rec['ncall']} hop_acc={rec['hop_acceptance']} "
            f"modes={masses and [round(v, 4) for v in masses]}",
            flush=True,
        )
    print(f"cell wall {wall:.1f}s (compile {common['compile_s']:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
