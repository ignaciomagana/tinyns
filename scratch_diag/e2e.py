"""End-to-end: tinyns runs on a bench mixture; oracle minor masses per seed,
per-mode insertion ranks and duplicate rates.
usage: e2e.py target nlive seeds_lo seeds_hi batch out.jsonl [cfg_json]"""
import bisect, json, math, sys, time
import numpy as np
import os
import jax, jax.numpy as jnp
from tinyns import core, modes
if os.environ.get("TINYNS_WALK_MIN"):  # experiments: the walk-frame size
    modes.WALK_MIN_POINTS = int(os.environ["TINYNS_WALK_MIN"])
from bench.targets import get_target, mixture_spec
name, m, lo, hi, B, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
extra = json.loads(sys.argv[7]) if len(sys.argv) > 7 else {}
tg = get_target(name)
cfg = core.Config(tg.ndim, m, max(1, m // 10), None, **extra)
resp = jax.jit(jax.vmap(tg.responsibility)) if tg.responsibility else None


def per_mode_insertion(r, mode, nmodes):
    """Normalized ranks of each new point among the survivors of its own mode."""
    logl = np.asarray(r.logl, float)
    order = np.argsort(np.asarray(r.logl_birth, float), kind="stable")
    k, niter = int(r.num_delete), int(r.niter)
    n0 = len(logl) - niter
    lives = [[] for _ in range(nmodes)]
    for i in order[:n0]:
        bisect.insort(lives[mode[i]], logl[i])
    births = order[n0:]
    ranks = [[] for _ in range(nmodes)]
    steps = [[] for _ in range(nmodes)]
    traj = []
    stride = max(1, int(r.nlive) // (4 * k))
    for step in range(niter // k):
        if step % stride == 0:
            traj.append([step * k, float(logl[max(step * k - 1, 0)]), [len(x) for x in lives]])
        for i in range(step * k, (step + 1) * k):  # deaths of this step (dead rows in order)
            lst = lives[mode[i]]
            j = bisect.bisect_left(lst, logl[i])
            if j < len(lst) and lst[j] == logl[i]:
                del lst[j]
        group = births[step * k:(step + 1) * k]
        for i in group:
            lst = lives[mode[i]]
            n = len(lst)
            if n > 0:
                ranks[mode[i]].append((bisect.bisect_right(lst, logl[i]) + 0.5) / (n + 1))
                steps[mode[i]].append(step)
        for i in group:
            bisect.insort(lives[mode[i]], logl[i])
    return [np.array(x) for x in ranks], [np.array(x) for x in steps], traj


for s0 in range(lo, hi, B):
    seeds = list(range(s0, min(hi, s0 + B)))
    keys = jnp.stack([jax.random.PRNGKey(s) for s in seeds])
    t0 = time.time()
    res = core.run(keys, tg.loglike, tg.prior_transform, cfg)
    wall = time.time() - t0
    with open(out, "a") as f:
        for s, r in zip(seeds, res):
            md = r.metadata
            rec = dict(target=name, nlive=m, seed=s, cfg=extra, walk_min=os.environ.get("TINYNS_WALK_MIN"), logz=float(r.logz), logzerr=float(r.logzerr),
                       truth=tg.logz, ncall=int(r.ncall), niter=int(r.niter), hop_acc=md.get("hop_acceptance"),
                       acc=md.get("acceptance"), unmoved=md.get("unmoved_fraction"), hist=md.get("mode_history"),
                       wall=wall / len(seeds))
            if resp is not None:
                x = np.asarray(r.samples); lw = np.asarray(r.logwt, float)
                w = np.exp(lw - lw.max()); w /= w.sum()
                rr = np.nan_to_num(np.concatenate([np.asarray(resp(x[i:i + 65536])) for i in range(0, len(x), 65536)]))
                rec["mass"] = (w[:, None] * rr).sum(0).tolist(); rec["truth_mass"] = list(tg.mode_mass)
                mode = rr.argmax(1)
                ranks, steps, traj = per_mode_insertion(r, mode, rr.shape[1])
                rec["traj"] = traj
                # after the minor mode's isolation, roughly: second half of the run
                half = r.niter // r.num_delete // 2
                rec["ins"] = [dict(n=int(len(a)), mean=float(a.mean()) if len(a) else None,
                                   n2=int((b >= half).sum()), mean2=float(a[b >= half].mean()) if (b >= half).any() else None)
                              for a, b in zip(ranks, steps)]
                u = np.asarray(r.samples_u)
                born = np.arange(len(u)) >= 0
                dup = []
                for j in range(rr.shape[1]):
                    sel = u[mode == j]
                    nu = len(np.unique(sel, axis=0)) if len(sel) else 0
                    dup.append(1 - nu / max(len(sel), 1))
                rec["dup"] = dup
            f.write(json.dumps(rec) + "\n")
    print(f"seeds {seeds[0]}-{seeds[-1]} wall {wall:.1f}s", flush=True)
