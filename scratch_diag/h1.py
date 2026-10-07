"""H1: steady-state population on a FIXED two-mode region (no NS shrinkage).
Random deletions, seeds from survivors, the real chains/frames/clustering.
If every chain were exact for iid live points, the minor fraction would
average its volume fraction; a population-level drain shows up as a deficit.
usage: h1.py target m k steps B seed frac variant(L|X) local(0|1)"""
import math, sys, os, time
import numpy as np
import jax, jax.numpy as jnp
from jax import random, lax
from tinyns import core, modes
from bench.targets import mixture_spec
from scipy.special import logsumexp
from scipy.optimize import brentq
name, m, k, steps, B, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
FRAC = float(sys.argv[7]); variant = sys.argv[8]; local = sys.argv[9] == "1"
if os.environ.get("TINYNS_WALK_MIN"): modes.WALK_MIN_POINTS = int(os.environ["TINYNS_WALK_MIN"])
if os.environ.get("TINYNS_SHRINK"): modes.SHRINK = float(os.environ["TINYNS_SHRINK"])
t = mixture_spec(name); d = t["d"]
w = np.array(t["weights"]); mus = np.stack(t["means"]); covs = np.stack(t["covs"])
chols = np.linalg.cholesky(covs); ichols = np.linalg.inv(chols)
A = np.log(w) - 0.5 * (d * math.log(2 * math.pi) + np.linalg.slogdet(covs)[1])
logdetL = np.log(np.abs(np.diagonal(chols, axis1=1, axis2=2))).sum(1)
def frac(L):
    lv = logdetL + 0.5 * d * np.log(2 * (A - L)); return math.exp(lv[1] - logsumexp(lv))
LSTAR = brentq(lambda L: frac(L) - FRAC, min(A) - 2000.0, min(A) - 1e-9)
R2 = 2 * (A - LSTAR); lv = logdetL + 0.5 * d * np.log(R2); P = np.exp(lv - logsumexp(lv))
dtype = jnp.result_type(float)
jmus, jich, jA, jch = (jnp.asarray(x, dtype) for x in (mus, ichols, A, chols))
jR2, jP = jnp.asarray(R2, dtype), jnp.asarray(P, dtype)
def comps(x):
    y = jnp.einsum("kij,kj->ki", jich, x[None] - jmus); return jA - 0.5 * jnp.sum(y * y, -1)
def loglike(x): return jnp.max(comps(x))
lstar = jnp.asarray(LSTAR, dtype)
def sample(key, n):  # exact uniform draws (cube ignored: check)
    k1, k2, k3 = random.split(key, 3)
    c = random.categorical(k1, jnp.log(jP), shape=(n,))
    z = random.normal(k2, (n, d), dtype); z = z / jnp.linalg.norm(z, axis=1, keepdims=True)
    z = z * random.uniform(k3, (n, 1), dtype) ** (1 / d)
    return jmus[c] + jnp.einsum("nij,nj->ni", jch[c], z * jnp.sqrt(jR2[c])[:, None])
walks = core.default_walks(d); every = max(1, round(m / (4 * k)))
half = jnp.arange(m) % 2

def recl(u, lab):
    if variant == "L":
        lab, st = modes.recluster(u, lab); return lab, st
    hm = m // 2
    l0, s0 = modes.recluster(u[0::2], lab[0::2]); l1, s1 = modes.recluster(u[1::2], lab[1::2])
    lab = jnp.zeros(m, jnp.int32).at[0::2].set(l0).at[1::2].set(l1)
    return lab, modes.Stats(*(jnp.stack([a, b]) for a, b in zip(s0, s1)))

def run(key):
    key, sub = random.split(key)
    u = sample(sub, m); lab = jnp.zeros(m, jnp.int32); fitted = jnp.ones(m, bool)
    lab, st = recl(u, lab)
    def body(carry, i):
        key, u, lab, st, fitted, log_scale = carry
        def do_recl(_):
            l, s = recl(u, lab); return l, s, jnp.ones(m, bool)
        lab, st, fitted = lax.cond(i % every == 0, do_recl, lambda _: (lab, st, fitted), None)
        key, kd, ks, kc = random.split(key, 4)
        dead = random.choice(kd, m, (k,), replace=False)
        alive = jnp.ones(m, bool).at[dead].set(False)
        if variant == "L":
            g = jnp.where(alive, random.gumbel(ks, (m,)), -jnp.inf)
            seeds = lax.top_k(g, k)[1]
        else:  # same-half seeds
            g = random.gumbel(ks, (m,))
            dh = half[dead]
            rank = jnp.take_along_axis(jnp.cumsum(jax.nn.one_hot(dh, 2, dtype=jnp.int32), 0) - 1, dh[:, None], 1)[:, 0]
            tops = jnp.stack([lax.top_k(jnp.where(alive & (half == h), g, -jnp.inf), k)[1] for h in (0, 1)])
            seeds = tops[dh, rank]
        scale = jnp.exp(log_scale)
        def chain(key, i):
            if variant == "L":
                mask = alive.at[seeds].set(False)
                ch = core._live_chol(u, mask)
                fr = modes.chain_frames(st, u[i], lab[i], fitted[i], m)
            else:
                h = half[i]
                ch = core._live_chol(u, alive & (half != h))
                fr = modes.frames(modes.Stats(*(x[1 - h] for x in st)), m)
            out = core._chain(key, u[i], loglike(u[i]), lstar, ch, scale, loglike, lambda v: v, walks, False, fr, True, local)
            return out[0], out[2] - out[5], out[6]
        new_u, wm, tr = jax.vmap(chain)(random.split(kc, k), seeds)
        acc = jnp.sum(wm) / jnp.maximum(k * walks - jnp.sum(tr), 1)
        log_scale = jnp.clip(log_scale + 0.5 * min(1.0, k / 32) * jnp.clip(acc - 0.25, -0.5, 0.5), math.log(1e-3), math.log(10.0))
        u = u.at[dead].set(new_u)
        if variant == "L":
            fr0 = modes.frames(st, m)
            lab = lab.at[dead].set(modes.nearest(fr0, new_u)); fitted = fitted.at[dead].set(False)
        else:
            fr0 = modes.frames(modes.Stats(*(x[0] for x in st)), m); fr1 = modes.frames(modes.Stats(*(x[1] for x in st)), m)
            lab = lab.at[dead].set(jnp.where(half[dead] == 0, modes.nearest(fr0, new_u), modes.nearest(fr1, new_u)))
        nminor = jnp.sum(jax.vmap(lambda x: comps(x)[1] > lstar)(u))
        return (key, u, lab, st, fitted, log_scale), nminor
    _, nm = lax.scan(body, (key, u, lab, st, fitted, jnp.asarray(math.log(0.5), dtype)), jnp.arange(steps))
    return nm
t0 = time.time()
nm = jax.jit(jax.vmap(run))(random.split(random.PRNGKey(seed), B))
nm = np.asarray(nm, float) / m  # (B, steps)
burn = steps // 5
per = nm[:, burn:].mean(1)
print(f"{name} m={m} k={k} variant={variant} local={local} walkmin={modes.WALK_MIN_POINTS} shrink={modes.SHRINK} vol frac {P[1]:.4f} "
      f"mean minor frac {per.mean():.4f} +- {per.std(ddof=1)/math.sqrt(B):.4f} rel {per.mean()/P[1]-1:+.4f} t={time.time()-t0:.0f}s", flush=True)
np.save(f"h1_{name}_m{m}_{variant}{int(local)}_s{seed}.npy", nm)
