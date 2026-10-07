"""Perfect nested sampling (exact constrained draws) on a bench mixture, to
calibrate the NS mode-mass estimator: is there an inherent bias?"""
import math, sys
import numpy as np
from scipy.special import logsumexp, gammaln
sys.path.insert(0, sys.argv[5] if len(sys.argv) > 5 else ".")
from bench.targets import mixture_spec

name = sys.argv[1]; m = int(sys.argv[2]); nrep = int(sys.argv[3]); seed0 = int(sys.argv[4])
t = mixture_spec(name); d = t["d"]
w = np.array(t["weights"]); mus = np.stack(t["means"]); covs = np.stack(t["covs"])
K = len(w)
chols = np.linalg.cholesky(covs); ichols = np.linalg.inv(chols)
A = np.log(w) - 0.5 * (d * math.log(2 * math.pi) + np.linalg.slogdet(covs)[1])
logdetL = np.log(np.abs(np.diagonal(chols, axis1=1, axis2=2))).sum(1)
logball = 0.5 * d * math.log(math.pi) - gammaln(0.5 * d + 1)

def comps(x):  # (n,K)
    y = np.einsum("kij,nkj->nki", ichols, x[:, None, :] - mus[None])
    return A[None] - 0.5 * np.sum(y * y, -1)

def loglike(x):
    return logsumexp(comps(x), axis=1)

def draw(rng, L, n):
    """n exact uniform draws from {loglike > L} in the cube."""
    r2 = 2 * (A - L + math.log(K))  # E_k = {comp_k > L - log K}
    ok = r2 > 0
    logv = np.where(ok, logdetL + 0.5 * d * np.log(np.maximum(r2, 1e-300)), -np.inf)
    p = np.exp(logv - logsumexp(logv))
    out = []; need = n
    while need > 0:
        nb = max(2 * need, 64)
        k = rng.choice(K, size=nb, p=p)
        z = rng.standard_normal((nb, d)); z /= np.linalg.norm(z, axis=1)[:, None]
        z *= rng.uniform(size=(nb, 1)) ** (1 / d)
        x = mus[k] + np.einsum("nij,nj->ni", chols[k], z * np.sqrt(r2[k])[:, None])
        c = comps(x)
        cnt = np.sum(c > (L - math.log(K)), axis=1)
        inside = np.all((x >= 0) & (x <= 1), axis=1) & (logsumexp(c, axis=1) > L)
        acc = inside & (rng.uniform(size=nb) * np.maximum(cnt, 1) < 1)
        out.append(x[acc]); need -= acc.sum()
    return np.concatenate(out)[:n]

def resp(x):
    c = comps(x); return np.exp(c - logsumexp(c, axis=1)[:, None])

R0 = float(sys.argv[6]) if len(sys.argv) > 6 else 12.0  # start level: main radius R0
L0 = A[0] - 0.5 * R0 ** 2
k = max(1, m // 10)
res = []
for rep in range(nrep):
    rng = np.random.default_rng(seed0 + rep)
    u = draw(rng, L0, m); lu = loglike(u)
    dl, dr = [], []
    logX = 0.0; logw = []
    while True:
        order = np.argsort(lu); worst = order[:k]
        lstar = lu[worst[-1]]
        for j, i in enumerate(worst):
            n = m - j
            logw.append(lu[i] + logX + math.log(-math.expm1(-1.0 / n)))
            logX -= 1.0 / n
            dr.append(resp(u[i:i+1])[0])
        new = draw(rng, lstar, k); u[worst] = new; lu[worst] = loglike(new)
        # stop at dlogz 0.1
        lz = logsumexp(logw)
        if np.logaddexp(lz, logX + lu.max()) - lz < 0.1:
            break
    # final live points: counts m..1
    order = np.argsort(lu)
    for j, i in enumerate(order):
        n = m - j
        logw.append(lu[i] + logX + math.log(-math.expm1(-1.0 / n)))
        logX -= 1.0 / n
        dr.append(resp(u[i:i+1])[0])
    logw = np.array(logw); dr = np.array(dr)
    ww = np.exp(logw - logw.max()); mass = (ww[:, None] * dr).sum(0) / ww.sum()
    res.append(mass)
    print(rep, np.round(mass, 5), len(logw), flush=True)
res = np.array(res)
lg = np.log(res[:, 1] / (1 - res[:, 1])) - math.log(w[1] / (1 - w[1]))
print(f"{name} m={m} reps={nrep} minor mass {res[:,1].mean():.5f}+-{res[:,1].std(ddof=1)/math.sqrt(nrep):.5f}"
      f" truth {w[1]} logit bias {lg.mean():+.4f}+-{lg.std(ddof=1)/math.sqrt(nrep):.4f} sd {lg.std(ddof=1):.4f}")
np.save(f"perfect_{name}_m{m}_s{seed0}.npy", res)
