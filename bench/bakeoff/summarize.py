"""Bake-off tables and the decision rule of the v1 plan, from runner JSONL.

    python bench/bakeoff/summarize.py results.jsonl [more.jsonl] [--out report.md]

A cell is (target, nlive, num_delete, arm). For each mixture cell, per minor
mode: the lost fraction (mass below 1e-3 of the truth), the seed sd of
``logit(w)`` over the seeds that kept the mode with a bootstrap 95% CI, the
exact-draw floor, the logit bias, ``Z_k / truth``, the logZ bias and
``scatter / logzerr``, the hop acceptance, the detection lag, and the call and
wall-time ratios to ``N`` in the same cell. For each control cell: the logZ
bias, ``scatter / logzerr`` and the cost ratios.

**Floor.** ``sqrt(1 / ESS_minor + 1 / ESS_main)`` from the Kish effective
sizes of the dead points' weights within each mode (median over seeds): the
scatter of the mode mass if the posterior samples were independent exact
draws. ``--floor floors.json`` (``{"target|nlive": sd}``) overrides it, e.g.
with the exact-sampling toy of the multimodality prototype.

**Resolvable.** A cell is resolvable if the oracle live count of the minor
mode at detection is at least ``3 d`` (median over seeds). Detection is the
median first niter with two clusters over the seeds of the tracking arms of
the same (target, nlive, num_delete), so the ``N`` cells use the same time.

**Decision rule** (plan, "Bake-off"):

1. Eliminate an arm with a logit or ``Z2`` bias above 3 se in a resolvable
   cell; a control logZ bias above 3 se or ``scatter / logzerr`` above 1.3;
   more lost modes than ``N``; or calls above 1.15x or wall time above 1.3x
   ``N`` (geometric means over the cells).
2. User gate: logit sd at most 0.4 in every resolvable cell.
3. Score: geometric mean over the resolvable cells of sd / floor. Arms within
   20% of the best score are ranked by cost (calls, then wall time). ``BC``
   is adopted only if it beats both ``B_t`` and ``C`` by more than 20%.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict

import numpy as np

ARMS = ("N", "B_ell", "B_t", "C", "BC")
LOST = 1e-3  # a mode with mass below this fraction of its truth is lost
BIAS_SE = 3.0
CONTROL_SCATTER = 1.3
CALLS_MAX, WALL_MAX = 1.15, 1.3
USER_GATE = 0.4
TIE = 0.2
RESOLVE_PER_DIM = 3.0


def load(paths):
    recs = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            recs.extend(json.loads(line) for line in f if line.strip())
    return recs


def logit(p):
    return math.log(p / (1.0 - p))


def bootstrap_sd(x, rng, n=2000):
    x = np.asarray(x, float)
    if len(x) < 3:
        return math.nan, math.nan
    idx = rng.integers(0, len(x), size=(n, len(x)))
    sds = np.std(x[idx], axis=1, ddof=1)
    return float(np.percentile(sds, 2.5)), float(np.percentile(sds, 97.5))


def mean_se(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return math.nan, math.nan
    se = float(np.std(x, ddof=1) / math.sqrt(len(x))) if len(x) > 1 else math.nan
    return float(np.mean(x)), se


def geomean(x):
    x = [v for v in x if v and math.isfinite(v) and v > 0]
    return float(np.exp(np.mean(np.log(x)))) if x else math.nan


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    return f"{x:.{nd}f}"


def cells(recs):
    out = defaultdict(list)
    for r in recs:
        out[(r["target"], r["nlive"], r["num_delete"], r["arm"])].append(r)
    return out


def detection_niter(by_cell, target, nlive, k):
    """Median first niter with two clusters over the tracking arms' seeds."""
    vals = [
        r["detection_niter"]
        for arm in ARMS[1:]
        for r in by_cell.get((target, nlive, k, arm), [])
        if r.get("detection_niter") is not None
    ]
    return float(np.median(vals)) if vals else None


def minor_live_at(recs, niter, mode):
    vals = []
    for r in recs:
        traj = (r.get("minor_live") or [None] * (mode + 1))[mode]
        if traj is None or niter is None:
            continue
        i = min(int(niter // r["live_stride"]), len(traj) - 1)
        vals.append(traj[i])
    return float(np.median(vals)) if vals else None


def summarize_cell(key, recs, by_cell, floors, rng):
    target, nlive, k, arm = key
    truth = recs[0]["truth"]
    d = recs[0]["ndim"]
    logz_true = truth["logz"]
    out = dict(
        target=target,
        nlive=nlive,
        k=k,
        arm=arm,
        d=d,
        n=len(recs),
        mixture=bool(truth.get("mode_mass")),
    )
    logz = np.array([r["logz"] for r in recs])
    err = np.array([r["logzerr"] for r in recs])
    out["logz_bias"], out["logz_se"] = mean_se(logz - logz_true)
    out["scatter_err"] = (
        float(np.std(logz, ddof=1) / np.mean(err)) if len(recs) > 1 else math.nan
    )
    out["ncall"] = float(np.mean([r["ncall"] for r in recs]))
    out["wall"] = float(np.mean([r["wall_s"] for r in recs]))
    acc = [r["hop_acceptance"] for r in recs if r.get("hop_acceptance") is not None]
    out["hop_acc"] = float(np.median(acc)) if acc else None
    if not out["mixture"]:
        return out
    masses = np.array([r["mode_mass"] for r in recs])
    ess = np.array([r["mode_ess"] for r in recs])
    det = detection_niter(by_cell, target, nlive, k)
    out["detection_niter"] = det
    modes = []
    for j in range(1, len(truth["mode_mass"])):
        w_true = truth["mode_mass"][j]
        w = masses[:, j]
        kept = w > LOST * w_true
        lg = np.array([logit(v) for v in w[kept]])
        sd = float(np.std(lg, ddof=1)) if kept.sum() > 1 else math.nan
        lo, hi = bootstrap_sd(lg, rng)
        bias, bias_se = mean_se(lg - logit(w_true))
        ratio = np.exp(logz - logz_true) * w / w_true
        z2, z2_se = mean_se(ratio)
        floor = floors.get(f"{target}|{nlive}")
        if floor is None:
            floor = float(np.median(np.sqrt(1 / ess[:, j] + 1 / ess[:, 0])))
        iso = [r["isolation_niter"][j - 1] for r in recs if r.get("isolation_niter")]
        lag = None
        if det is not None and iso:
            lag = (det - float(np.median(iso))) / nlive
        live = minor_live_at(recs, det, j - 1)
        modes.append(
            dict(
                mode=j,
                truth=w_true,
                lost=float(np.mean(~kept)),
                sd=sd,
                sd_lo=lo,
                sd_hi=hi,
                floor=floor,
                logit_bias=bias,
                logit_se=bias_se,
                z_ratio=z2,
                z_se=z2_se,
                lag_efolds=lag,
                live_at_detection=live,
                resolvable=live is not None and live >= RESOLVE_PER_DIM * d,
            )
        )
    out["modes"] = modes
    return out


def add_cost_ratios(rows):
    base = {(r["target"], r["nlive"], r["k"]): r for r in rows if r["arm"] == "N"}
    for r in rows:
        b = base.get((r["target"], r["nlive"], r["k"]))
        r["calls_ratio"] = r["ncall"] / b["ncall"] if b else math.nan
        r["wall_ratio"] = r["wall"] / b["wall"] if b else math.nan


def decide(rows):
    """The decision rule; returns (per-arm verdict dicts, chosen arm, notes)."""
    arms = sorted({r["arm"] for r in rows}, key=ARMS.index)
    verdict = {a: dict(arm=a, reasons=[], gate_fail=[], ratios=[]) for a in arms}
    lost = defaultdict(float)
    for r in rows:
        v = verdict[r["arm"]]
        cell = f"{r['target']} m={r['nlive']} k={r['k']}"
        if r["mixture"]:
            for m in r["modes"]:
                lost[r["arm"]] += m["lost"] * r["n"]
                if not m["resolvable"]:
                    continue
                if abs(m["logit_bias"]) > BIAS_SE * m["logit_se"]:
                    v["reasons"].append(f"logit bias {m['logit_bias']:+.3f} ({cell})")
                if abs(m["z_ratio"] - 1) > BIAS_SE * m["z_se"]:
                    v["reasons"].append(
                        f"Z{m['mode'] + 1}/truth {m['z_ratio']:.3f} ({cell})"
                    )
                if not m["sd"] <= USER_GATE:
                    v["gate_fail"].append(f"{cell} mode {m['mode']}: sd {fmt(m['sd'])}")
                v["ratios"].append(m["sd"] / m["floor"])
        else:
            if abs(r["logz_bias"]) > BIAS_SE * r["logz_se"]:
                v["reasons"].append(f"control logZ bias {r['logz_bias']:+.3f} ({cell})")
            if r["scatter_err"] > CONTROL_SCATTER:
                v["reasons"].append(
                    f"control scatter/err {r['scatter_err']:.2f} ({cell})"
                )
    for a in arms:
        v = verdict[a]
        own = [r for r in rows if r["arm"] == a]
        v["calls"] = geomean([r["calls_ratio"] for r in own])
        v["wall"] = geomean([r["wall_ratio"] for r in own])
        v["lost"] = lost[a]
        v["score"] = geomean(v["ratios"])
        if a == "N":
            continue
        if lost[a] > lost.get("N", math.inf):
            v["reasons"].append(f"lost modes {lost[a]:.0f} > N {lost['N']:.0f}")
        if v["calls"] > CALLS_MAX:
            v["reasons"].append(f"calls {v['calls']:.2f}x N")
        if v["wall"] > WALL_MAX:
            v["reasons"].append(f"wall {v['wall']:.2f}x N")
    alive = [
        a
        for a in arms
        if a != "N"
        and not verdict[a]["reasons"]
        and not verdict[a]["gate_fail"]
        and math.isfinite(verdict[a]["score"])
    ]
    notes = []
    if not alive:
        return verdict, None, ["no arm survives the elimination and the user gate"]
    best = min(verdict[a]["score"] for a in alive)
    tied = [a for a in alive if verdict[a]["score"] <= (1 + TIE) * best]
    if "BC" in tied:
        singles = [verdict[a]["score"] for a in ("B_t", "C") if a in verdict]
        if not all(verdict["BC"]["score"] < s / (1 + TIE) for s in singles):
            tied.remove("BC")
            notes.append("BC does not beat both B_t and C by more than 20%")
    if not tied:
        tied = [a for a in alive if a != "BC"]
    tied.sort(key=lambda a: (verdict[a]["calls"], verdict[a]["wall"]))
    if len(tied) > 1:
        notes.append(f"within 20% of the best score: {', '.join(tied)}; cheapest wins")
    return verdict, tied[0], notes


def table_row(*cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def cell_order(r):
    return (r["target"], r["nlive"], r["k"], ARMS.index(r["arm"]))


MIX_HEAD = (
    "target", "m", "k", "arm", "n", "mode", "lost", "logit sd [95% CI]", "floor",
    "sd/floor", "logit bias ± se", "Z/truth ± se", "logZ bias ± se", "scat/err",
    "hop acc", "lag (e-folds)", "live at det.", "resolvable", "calls/N", "wall/N",
)
CTL_HEAD = (
    "target", "m", "k", "arm", "n", "logZ bias ± se", "scat/err", "hop acc",
    "calls/N", "wall/N",
)
DECISION_HEAD = (
    "arm", "eliminated by", "user gate (sd <= 0.4)", "score (gm sd/floor)",
    "calls/N", "wall/N", "lost seeds",
)


def report(rows, verdict, chosen, notes) -> str:
    lines = ["# v1 bake-off", "", "## Mixtures (per minor mode)", ""]
    lines += [table_row(*MIX_HEAD), "|" + "---|" * len(MIX_HEAD)]
    for r in sorted((r for r in rows if r["mixture"]), key=cell_order):
        for m in r["modes"]:
            lines.append(table_row(
                r["target"], r["nlive"], r["k"], r["arm"], r["n"], m["mode"],
                fmt(m["lost"], 2),
                f"{fmt(m['sd'])} [{fmt(m['sd_lo'])}, {fmt(m['sd_hi'])}]",
                fmt(m["floor"]), fmt(m["sd"] / m["floor"], 2),
                f"{m['logit_bias']:+.3f} ± {fmt(m['logit_se'])}",
                f"{fmt(m['z_ratio'])} ± {fmt(m['z_se'])}",
                f"{r['logz_bias']:+.3f} ± {fmt(r['logz_se'])}",
                fmt(r["scatter_err"], 2), fmt(r["hop_acc"]),
                fmt(m["lag_efolds"], 1), fmt(m["live_at_detection"], 0),
                "yes" if m["resolvable"] else "no",
                fmt(r["calls_ratio"], 2), fmt(r["wall_ratio"], 2),
            ))
    lines += ["", "## Controls", "", table_row(*CTL_HEAD), "|" + "---|" * len(CTL_HEAD)]
    for r in sorted((r for r in rows if not r["mixture"]), key=cell_order):
        lines.append(table_row(
            r["target"], r["nlive"], r["k"], r["arm"], r["n"],
            f"{r['logz_bias']:+.3f} ± {fmt(r['logz_se'])}",
            fmt(r["scatter_err"], 2), fmt(r["hop_acc"]),
            fmt(r["calls_ratio"], 2), fmt(r["wall_ratio"], 2),
        ))
    lines += ["", "## Decision", ""]
    lines += [table_row(*DECISION_HEAD), "|" + "---|" * len(DECISION_HEAD)]
    for a, v in verdict.items():
        more = " ..." if len(v["reasons"]) > 4 else ""
        reasons = "; ".join(v["reasons"][:4]) + more
        gate = "pass" if not v["gate_fail"] else f"{len(v['gate_fail'])} cells fail"
        lines.append(table_row(
            a, reasons or "-", gate, fmt(v["score"], 2), fmt(v["calls"], 2),
            fmt(v["wall"], 2), f"{v['lost']:.0f}",
        ))
    lines += ["", f"**Chosen arm: {chosen or 'none'}**", ""]
    lines += [f"- {n}" for n in notes]
    for a, v in verdict.items():
        if v["gate_fail"]:
            lines.append(f"- {a} fails the user gate in: " + "; ".join(v["gate_fail"]))
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("jsonl", nargs="+")
    p.add_argument("--floor", default=None, help="JSON {'target|nlive': sd}")
    p.add_argument("--out", default=None)
    p.add_argument("--json", default=None, help="also write the cell rows as JSON")
    args = p.parse_args(argv)
    floors = {}
    if args.floor:
        with open(args.floor, encoding="utf-8") as f:
            floors = json.load(f)
    by_cell = cells(load(args.jsonl))
    rng = np.random.default_rng(0)
    rows = [summarize_cell(k, v, by_cell, floors, rng) for k, v in by_cell.items()]
    add_cost_ratios(rows)
    verdict, chosen, notes = decide(rows)
    text = report(rows, verdict, chosen, notes)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(dict(rows=rows, verdict=verdict, chosen=chosen), f, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
