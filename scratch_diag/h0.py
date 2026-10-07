"""H0: per-chain invariance with the partition refit per chain.

Fixed 'others' O (m-1 iid uniform points in a two-mode constrained region),
a fresh uniform seed s per chain. Arms, paired on the same seed and chain key:
  cur   : recluster(O + s) warm-started, rank-one LOO of s (production)
  orc   : frames of recluster(O) (exact by construction)
  rw    : random walk only
  nolo  : recluster(O + s), no LOO (power control)
Region: union of the two component ellipsoids {comp_k > L*} of a bench mixture
at the level L* where the minor volume fraction is FRAC.
usage: h0.py target m nchains batch seed [frac] [scale] [walks]
"""
import math, sys, time, json
import numpy as np
import jax, jax.numpy as jnp
from jax import random
from tinyns import core, modes
from bench.targets import mixture_spec
from scipy.special import gammaln, logsumexp
from scipy.optimize import brentq

name, m, nchains, B, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
FRAC = float(sys.argv[6]) if len(sys.argv) > 6 else 0.06
t = mixture_spec(name); d = t["d"]
w = np.array(t["weights"]); mus = np.stack(t["means"]); covs = np.stack(t["covs"])
chols = np.linalg.cholesky(covs); ichols = np.linalg.inv(chols)
A = np.log(w) - 0.5 * (d * math.log(2 * math.pi) + np.linalg.slogdet(covs)[1])
logdetL = np.log(np.abs(np.diagonal(chols, axis1=1, axis2=2))).sum(1)

def frac(L):
    lv = logdetL + 0.5 * d * np.log(2 * (A - L))
    return math.exp(lv[1] - logsumexp(lv))
lo, hi = min(A) - 2000.0, min(A) - 1e-9
LSTAR = brentq(lambda L: frac(L) - FRAC, lo, hi)
R2 = 2 * (A - LSTAR)
lv = logdetL + 0.5 * d * np.log(R2); P = np.exp(lv - logsumexp(lv))

def sample(rng, n):
    out = []
    while sum(len(o) for o in out) < n:
        nb = 2 * n
        k = rng.choice(2, size=nb, p=P)
        z = rng.standard_normal((nb, d)); z /= np.linalg.norm(z, axis=1)[:, None]
        z *= rng.uniform(size=(nb, 1)) ** (1 / d)
        x = mus[k] + np.einsum("nij,nj->ni", chols[k], z * np.sqrt(R2[k])[:, None])
        y = np.einsum("kij,nkj->nki", ichols, x[:, None] - mus[None])
        cnt = np.sum(np.sum(y * y, -1) <= R2[None], 1)
        ok = np.all((x >= 0) & (x <= 1), 1) & (rng.uniform(size=nb) * cnt < 1)
        out.append(x[ok])
    return np.concatenate(out)[:n]

dtype = jnp.result_type(float)
jmus, jich, jA = jnp.asarray(mus, dtype), jnp.asarray(ichols, dtype), jnp.asarray(A, dtype)
def comps(x):
    y = jnp.einsum("kij,kj->ki", jich, x[None] - jmus)
    return jA - 0.5 * jnp.sum(y * y, -1)
def loglike(x):
    return jnp.max(comps(x))
def in_minor(x):
    return comps(x)[1] > LSTAR

rng = np.random.default_rng(seed)
_ref = sample(np.random.default_rng(10**6 + seed), 2_000_000)
_y = np.einsum("ij,nj->ni", ichols[1], _ref - mus[1])
PREF = float(np.mean(np.sum(_y * _y, 1) <= R2[1])); del _ref, _y
others = jnp.asarray(sample(rng, m - 1), dtype)
lab_o, st_o = modes.recluster(others, jnp.zeros(m - 1, jnp.int32))
fr_o = modes.frames(st_o, m)
print(f"# {name} d={d} m={m} L*={LSTAR:.3f} vol frac {P[1]:.5f} clusters {int(jnp.sum(st_o.count>0))} "
      f"counts {np.asarray(st_o.count[st_o.count>0]).astype(int)} eligible {np.asarray(fr_o.eligible).sum()}", flush=True)
chol = core._live_chol(others, jnp.ones(m - 1, bool))
walks = int(sys.argv[8]) if len(sys.argv) > 8 else core.default_walks(d)
lstar = jnp.asarray(LSTAR, dtype)

def run_chain(key, s, fr, hop=True, scale=1.0, local=False):
    out = core._chain(key, s, loglike(s), lstar, chol, jnp.asarray(scale, dtype), loglike,
                      lambda v: v, walks, False, fr, hop, local)
    return out[0], out[2], out[5], out[6]

# scale: pilot to ~0.25 RW acceptance on the region
if len(sys.argv) > 7 and float(sys.argv[7]) > 0:
    SCALE = float(sys.argv[7])
else:
    SCALE = 0.5
    sp = jnp.asarray(sample(rng, 512), dtype)
    for _ in range(12):
        f = jax.jit(jax.vmap(lambda k, s: run_chain(k, s, fr_o, False, SCALE)))
        _, mv, _, _ = f(random.split(random.PRNGKey(99), 512), sp)
        acc = float(jnp.mean(mv)) / walks
        SCALE *= math.exp(1.5 * (acc - 0.25))
    print(f"# scale {SCALE:.4f} (rw acceptance {acc:.3f}) walks {walks}", flush=True)

import os
FAST = os.environ.get("H0_FAST") == "1"

def one(key, s):
    lab_s = modes.nearest(fr_o, s[None])[0]
    U = jnp.concatenate([others, s[None]])
    warm = jnp.concatenate([lab_o, lab_s[None]])
    if FAST:  # no per-chain recluster: the oracle partition plus the seed
        lab, st = warm, modes.stats(U, warm, modes.C_MAX)
    else:
        lab, st = modes.recluster(U, warm)
    changed = jnp.sum(lab[:-1] != lab_o)
    ncl = jnp.sum(st.count > 0)
    fr_cur = modes.chain_frames(st, s, lab[-1], True, m)
    fr_nolo = modes.frames(st, m)
    res = {}
    for nm, fr, hop, loc in ARMS:
        fr = {"cur": fr_cur, "orc": fr_o, "nolo": fr_nolo}[fr]
        u, mv, h, tr = run_chain(key, s, fr, hop, SCALE, loc)
        res[nm] = (in_minor(u), h, tr, mv - h)
    return res, changed, ncl, jnp.sum(fr_cur.eligible), in_minor(s)

ARMS = [("cur", "cur", True, False), ("orc", "orc", True, False), ("rw", "orc", False, False),
        ("nolo", "nolo", True, False), ("loc", "orc", True, True), ("rwloc", "orc", False, True)]
if os.environ.get("H0_ARMS"):
    ARMS = [a for a in ARMS if a[0] in os.environ["H0_ARMS"].split(",")]
f = jax.jit(jax.vmap(one))
key = random.PRNGKey(seed + 12345)
tot = {a[0]: np.zeros(3) for a in ARMS}
rwacc = {a[0]: np.zeros(4) for a in ARMS}  # walk moves: seeds main, minor; counts
pair = np.zeros(4)  # sum diff cur-orc, sum diff^2, nolo-orc, ^2
nchg = 0; ncl_hist = {}; nel_hist = {}; n = 0; t0 = time.time(); seedminor = 0
while n < nchains:
    key, sub = random.split(key)
    s = jnp.asarray(sample(rng, B), dtype)
    res, chg, ncl, nel, sm = f(random.split(sub, B), s)
    for nm in tot:
        a, h, tr, wm = res[nm]
        tot[nm] += [float(jnp.sum(a)), float(jnp.sum(h)), float(jnp.sum(tr))]
        smn = np.asarray(sm); wm = np.asarray(wm, float) / (walks - np.asarray(tr))
        rwacc[nm] += [wm[~smn].sum(), wm[smn].sum(), (~smn).sum(), smn.sum()]
    z0 = np.zeros(B)
    dc = np.asarray(res["cur"][0], float) - np.asarray(res["orc"][0], float) if "cur" in res and "orc" in res else z0
    dn = np.asarray(res["nolo"][0], float) - np.asarray(res["orc"][0], float) if "nolo" in res and "orc" in res else z0
    pair += [dc.sum(), (dc**2).sum(), dn.sum(), (dn**2).sum()]
    nchg += int(jnp.sum(chg > 0)); seedminor += int(jnp.sum(sm))
    for v in np.asarray(ncl): ncl_hist[int(v)] = ncl_hist.get(int(v), 0) + 1
    for v in np.asarray(nel): nel_hist[int(v)] = nel_hist.get(int(v), 0) + 1
    n += B
    if n % (B * 10) == 0 or n >= nchains:
        line = {nm: f"{tot[nm][0] / n:.5f}({(tot[nm][0] / n - PREF) / math.sqrt(PREF * (1 - PREF) / n):+.1f}z)" for nm in tot}
        mc, mn = pair[0] / n, pair[2] / n
        sc = math.sqrt(max(pair[1] / n - mc**2, 0) / n); sn = math.sqrt(max(pair[3] / n - mn**2, 0) / n)
        print(f"n={n} frac {line} vol {PREF:.5f} | cur-orc {mc:+.6f}+-{sc:.6f} nolo-orc {mn:+.6f}+-{sn:.6f} "
              f"| partition changed {nchg/n:.4f} ncl {ncl_hist} nelig {nel_hist} hopacc {tot[ARMS[0][0]][1]/max(tot[ARMS[0][0]][2],1):.3f} "
              f"seedminor {seedminor/n:.4f} t={time.time()-t0:.0f}s", flush=True)
        print("   walk acceptance main/minor: " + " ".join(f"{nm} {v[0]/max(v[2],1):.3f}/{v[1]/max(v[3],1):.3f}" for nm, v in rwacc.items())
              + " | hop acc " + " ".join(f"{nm} {tot[nm][1]/max(tot[nm][2],1):.3f}" for nm in tot), flush=True)
