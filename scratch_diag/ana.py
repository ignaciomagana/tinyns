import json, math, sys, glob
import numpy as np
def load(p):
    out = []
    for l in open(p):
        try: out.append(json.loads(l))
        except Exception: pass
    return out
for p in sys.argv[1:]:
    rs = load(p)
    if not rs: continue
    seen = {}
    for r in rs: seen[r["seed"]] = r
    rs = list(seen.values()); n = len(rs)
    def ms(x):
        x = np.asarray(x, float); return x.mean(), x.std(ddof=1) / math.sqrt(len(x)), x.std(ddof=1)
    line = f"{p.split('/')[-1]:40s} n={n:4d}"
    if "mass" in rs[0]:
        tm = rs[0]["truth_mass"]
        for j in range(1, len(tm)):
            p_ = np.array([r["mass"][j] for r in rs]); kept = p_ > 1e-3 * tm[j]
            lg = np.log(p_[kept] / (1 - p_[kept])) - math.log(tm[j] / (1 - tm[j]))
            b, se, sd = ms(lg)
            line += f" | mode{j} logit {b:+.4f}+-{se:.4f} sd {sd:.3f} lost {1-kept.mean():.2f}"
            ins = [r["ins"][j]["mean2"] for r in rs if r["ins"][j]["mean2"] is not None]
            i0 = [r["ins"][0]["mean2"] for r in rs if r["ins"][0]["mean2"] is not None]
            if ins: line += f" ins2 minor {np.mean(ins):.4f}+-{np.std(ins)/math.sqrt(len(ins)):.4f} main {np.mean(i0):.4f}+-{np.std(i0)/math.sqrt(len(i0)):.4f}"
            line += f" dup minor {np.mean([r['dup'][j] for r in rs]):.4f} main {np.mean([r['dup'][0] for r in rs]):.5f}"
    dz = np.array([r["logz"] - r["truth"] for r in rs]); b, se, sd = ms(dz)
    err = np.mean([r["logzerr"] for r in rs])
    line += f" | logZ {b:+.3f}+-{se:.3f} scat/err {sd/err:.2f} hop {np.mean([r['hop_acc'] or 0 for r in rs]):.3f} acc {np.mean([r['acc'] for r in rs]):.3f} ncall {np.mean([r['ncall'] for r in rs]):.4g}"
    print(line)
