import json, math, sys
import numpy as np
from scipy.special import logsumexp
sys.path.insert(0, "/hildafs/projects/phy230014p/magana/scratch_v1fix/src")
from bench.targets import mixture_spec
for p in sys.argv[1:]:
    rs = [json.loads(l) for l in open(p)]
    t = mixture_spec(rs[0]["target"]); d = t["d"]
    w = np.array(t["weights"]); covs = np.stack(t["covs"])
    A = np.log(w) - 0.5 * (d * math.log(2 * math.pi) + np.linalg.slogdet(covs)[1])
    ld = 0.5 * np.linalg.slogdet(covs)[1]
    m = rs[0]["nlive"]
    def f(L):
        r2 = 2 * (A - L)
        lv = np.where(r2 > 0, ld + 0.5 * d * np.log(np.maximum(r2, 1e-300)), -np.inf)
        return math.exp(lv[1] - logsumexp(lv))
    bins = {}
    for r in rs:
        for i, L, n in r["traj"]:
            e = int(i / m)  # e-folds
            tot = sum(n); fr = n[1] / tot; ft = f(L)
            bins.setdefault(e // 4, []).append((fr, ft, fr - ft))
    print(p.split("/")[-1], "n seeds", len(rs))
    print("  efolds  live-frac   truth    diff(se)      rel")
    for b in sorted(bins):
        a = np.array(bins[b])
        nseed = len(rs)
        print(f"  {4*b:3d}-{4*b+3:3d} {a[:,0].mean():.4f}  {a[:,1].mean():.4f}  {a[:,2].mean():+.4f}({a[:,2].std()/math.sqrt(nseed):.4f})  {a[:,0].mean()/a[:,1].mean()-1:+.3f}")
