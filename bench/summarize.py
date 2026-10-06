"""Summarise ``tinyns-bench-1`` JSONL records as a Markdown table.

    python bench/summarize.py results.jsonl [more.jsonl ...] [--out table.md]

One row per (sampler, target, nlive). A repeated seed keeps its last record.
Columns:

``ok``          successful runs / all runs (timeouts and errors counted apart)
``dlogZ``       mean of logz - truth, +- its standard error
``scat/err``    sd(logz - truth) / mean(logzerr): 1 for honest error bars
``z_rms``       rms of (logz - truth) / logzerr, which also counts the bias
``logit sd``    sd over seeds of logit(mode mass) for the minor modes (the
                worst one when there are several) with a 95% bootstrap CI over
                seeds; seeds that lost the mode are left out of the sd
``logit bias``  mean logit(mass) - logit(true mass) for that mode
``lost``        fraction of seeds whose mass for some minor mode is below
                10% of its true mass
``ncall``, ``wall s``, ``compile s``  medians
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict

import numpy as np

LOST_FRACTION = 0.1


def load(paths):
    recs = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("schema") != "tinyns-bench-1":
                    continue
                nlive = (r.get("config") or {}).get("cli", {}).get("nlive")
                recs[(r["sampler"], r["target"], nlive, r["seed"])] = r
    groups = defaultdict(list)
    for (sampler, target, nlive, _), r in recs.items():
        groups[(sampler, target, nlive)].append(r)
    return groups


def _logit(p):
    return math.log(p / (1.0 - p))


def bootstrap_sd_ci(x, n_boot=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 5:  # too few seeds for a meaningful interval
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    sds = x[idx].std(axis=1, ddof=1)
    return float(np.percentile(sds, 2.5)), float(np.percentile(sds, 97.5))


def mode_stats(ok):
    """Worst minor mode: (sd, ci_lo, ci_hi, bias, lost fraction, mode index)."""
    truth = ok[0]["truth"].get("mode_mass")
    if not truth:
        return None
    truth = np.asarray(truth, float)
    masses = np.array([r["mode_mass"] for r in ok if r.get("mode_mass")], float)
    if masses.size == 0:
        return None
    minor = [k for k in range(len(truth)) if k != int(np.argmax(truth))]
    lost = np.any(masses[:, minor] < LOST_FRACTION * truth[minor], axis=1)
    worst = None
    for k in minor:
        m = masses[:, k]
        keep = (m >= LOST_FRACTION * truth[k]) & (m < 1.0)
        lg = np.array([_logit(v) for v in m[keep]])
        sd = float(lg.std(ddof=1)) if len(lg) > 1 else float("nan")
        bias = float(lg.mean() - _logit(truth[k])) if len(lg) else float("nan")
        lo, hi = bootstrap_sd_ci(lg)
        if worst is None or (
            not math.isnan(sd) and (math.isnan(worst[0]) or sd > worst[0])
        ):
            worst = (sd, lo, hi, bias, k)
    sd, lo, hi, bias, k = worst
    return dict(sd=sd, lo=lo, hi=hi, bias=bias, lost=float(lost.mean()), mode=k)


def summarize(groups):
    rows = []
    for (sampler, target, nlive), recs in sorted(
        groups.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2] or 0)
    ):
        ok = [r for r in recs if r["status"] == "ok" and r.get("logz") is not None]
        row = dict(
            sampler=sampler,
            target=target,
            nlive=nlive,
            n=len(recs),
            n_ok=len(ok),
            n_timeout=sum(r["status"] == "timeout" for r in recs),
            n_error=sum(r["status"] == "error" for r in recs),
        )
        if ok:
            truth = ok[0]["truth"]["logz"]
            dz = (
                np.array([r["logz"] - truth for r in ok]) if truth is not None else None
            )
            errs = np.array(
                [r["logzerr"] for r in ok if r.get("logzerr") is not None], float
            )
            if dz is not None:
                row["dz_mean"] = float(dz.mean())
                row["dz_se"] = (
                    float(dz.std(ddof=1) / math.sqrt(len(dz)))
                    if len(dz) > 1
                    else float("nan")
                )
                row["scatter"] = float(dz.std(ddof=1)) if len(dz) > 1 else float("nan")
                if len(errs) == len(dz) and len(errs):
                    row["err_mean"] = float(errs.mean())
                    row["scat_over_err"] = row["scatter"] / row["err_mean"]
                    row["z_rms"] = float(np.sqrt(np.mean((dz / errs) ** 2)))
            row["modes"] = mode_stats(ok)
            row["ncall"] = float(np.median([r["ncall"] for r in ok]))
            row["wall_s"] = float(np.median([r["wall_s"] for r in ok]))
            comp = [r["compile_s"] for r in ok if r.get("compile_s") is not None]
            row["compile_s"] = float(np.median(comp)) if comp else None
        rows.append(row)
    return rows


def _f(x, fmt="{:.3f}"):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return fmt.format(x)


def to_markdown(rows):
    head = (
        "| target | sampler | nlive | ok | dlogZ | scat/err | z_rms "
        "| logit sd [95% CI] | logit bias | lost | ncall | wall s | compile s |"
    )
    out = [head, "|" + "---|" * (head.count("|") - 1)]
    for r in rows:
        ok = f"{r['n_ok']}/{r['n']}"
        if r["n_timeout"] or r["n_error"]:
            ok += f" ({r['n_timeout']}T {r['n_error']}E)"
        dz = f"{r['dz_mean']:+.3f} ± {_f(r.get('dz_se'))}" if "dz_mean" in r else "-"
        m = r.get("modes")
        if m:
            f2 = "{:.2f}"
            sd = f"{_f(m['sd'], f2)} [{_f(m['lo'], f2)}, {_f(m['hi'], f2)}]"
            bias, lost = _f(m["bias"], "{:+.2f}"), _f(m["lost"], "{:.2f}")
        else:
            sd = bias = lost = "-"
        out.append(
            f"| {r['target']} | {r['sampler']} | {r['nlive']} | {ok} | {dz} "
            f"| {_f(r.get('scat_over_err'), '{:.2f}')} "
            f"| {_f(r.get('z_rms'), '{:.2f}')} "
            f"| {sd} | {bias} | {lost} | {_f(r.get('ncall'), '{:.3g}')} "
            f"| {_f(r.get('wall_s'), '{:.1f}')} "
            f"| {_f(r.get('compile_s'), '{:.1f}')} |"
        )
    return "\n".join(out) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("files", nargs="+")
    p.add_argument(
        "--out", default=None, help="write the Markdown here (default stdout)"
    )
    args = p.parse_args(argv)
    md = to_markdown(summarize(load(args.files)))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(md)
    else:
        print(md, end="")


if __name__ == "__main__":
    main()
