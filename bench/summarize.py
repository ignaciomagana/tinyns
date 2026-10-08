"""Merge ``tinyns-bench-1`` JSONL records into the head-to-head report.

    python bench/summarize.py results_gpu.jsonl results_cpu.jsonl \
        --out report.md --csv cells.csv --json cells.json

A cell is one (sampler, target, nlive); a sampler run with ``--opt`` settings is
its own sampler, labelled ``name [key=value]``. A repeated seed keeps its last
record. The report has three parts.

**Cells**, one row each:

``hw``          where the cell ran: ``GPU <model>`` or ``CPU <n>c`` (cores)
``nlive``       the live points the sampler used (Nautilus and JAXNS set
                their own; see ``live_points``)
``ok``          successful runs / all runs, then timeouts (T) and errors (E)
``dlogZ``       mean of logz - truth, +- its standard error
``rms``         rms of logz - truth, in nats (the accuracy of one run)
``scat/err``    sd(logz - truth) / mean(logzerr): 1 for honest error bars
``in 3s``       fraction of runs with |logz - truth| < 3 logzerr
``logit bias``  mean logit(mass) - logit(true mass) of a minor mode, +- its
                standard error; the mode is the one with the largest sd
``logit sd``    sd over seeds of that logit(mass), with a 95% bootstrap CI;
                seeds that lost the mode are left out of both
``lost``        fraction of seeds where some minor mode has under 10% of its
                true mass
``ncall``       median likelihood calls of the successful runs
``wall s``      median wall time over the successful and the timed-out runs
                (``>`` when that median is a timeout); ``compile s`` likewise
``acc``         ``yes`` when the cell is accurate (``accurate``), else why not

**Accurate cells**: per sampler and target family, how many cells are accurate.

**Headline** (only when ``--reference``, default ``tinyns_v1``, has records):
per family, each competitor's calls, wall time, rms and logit sd relative to the
reference (geometric means over the targets where both are accurate), the same
costs at equal rms, and the reference's rank by those (``headline``).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict

import numpy as np

LOST_FRACTION = 0.1
# A cell is accurate when at least this fraction of its runs finished, ...
MIN_OK = 0.9
# ... its logZ bias is within 3 standard errors or this many nats (the dlogz
# tolerance the runs were stopped at), and, for a mixture, at most
# MAX_LOST of its seeds lost a mode and the logit bias of the mode weight is
# within 3 standard errors or MODE_TOL.
LOGZ_TOL = 0.1
MODE_TOL = 0.1
MAX_LOST = 0.1
FAMILIES = (
    "gauss", "rosen", "funnel", "loggamma", "eggbox", "sepW", "connW", "sepM", "mix3"
)


def label(rec) -> str:
    """Sampler spec, plus the ``--opt`` settings of the run when it had any."""
    opts = (rec.get("config") or {}).get("cli", {}).get("opts") or {}
    if not opts:
        return rec["sampler"]
    return f"{rec['sampler']} [{','.join(f'{k}={opts[k]}' for k in sorted(opts))}]"


def load(paths):
    """{(sampler label, target, cli nlive): [records]} from JSONL files."""
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
                recs[(label(r), r["target"], nlive, r["seed"])] = r
    groups = defaultdict(list)
    for (sampler, target, nlive, _), r in recs.items():
        groups[(sampler, target, nlive)].append(r)
    return groups


def family(target: str) -> str:
    return target.rsplit("_d", 1)[0]


def _target_order(target: str):
    fam = family(target)
    rank = FAMILIES.index(fam) if fam in FAMILIES else len(FAMILIES)
    m = re.search(r"_d(\d+)$", target)
    return rank, fam, int(m.group(1)) if m else 0


def hw_label(rec) -> str:
    hw = rec.get("hw") or {}
    if hw.get("gpu"):
        return "GPU " + hw["gpu"].replace("NVIDIA ", "").split()[0]
    return f"CPU {hw.get('cpu_threads', '?')}c"


def live_points(sampler: str, recs) -> str:
    """The live points a cell ran with, as the table shows them.

    Nautilus ignores ``--nlive`` (its record holds ``n_live``), JAXNS's
    default variant runs ``30 * ndim`` chains, and UltraNest treats its value
    as a minimum.
    """
    r = recs[0]
    conf = r.get("config") or {}
    name = sampler.split()[0]
    if name.startswith("nautilus"):
        own = [x["config"].get("n_live") for x in recs if x["status"] == "ok"]
        return str(own[0]) if own else "own"
    if name == "jaxns":
        return str(30 * r["ndim"])
    nlive = conf.get("cli", {}).get("nlive")
    return f">={nlive}" if name.startswith("ultranest") else str(nlive)


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


def logz_stats(dz, err=None) -> dict:
    """Bias, scatter and calibration of ``dz = logz - truth`` over seeds.

    ``err`` is each run's ``logzerr`` (or None): with it, ``scat_over_err`` is
    ``sd(dz) / mean(err)`` and ``in3`` the fraction with ``|dz| < 3 err``.
    """
    dz = np.asarray(dz, float)
    nan = float("nan")
    n = len(dz)
    sd = float(dz.std(ddof=1)) if n > 1 else nan
    out = dict(
        dz_mean=float(dz.mean()),
        dz_se=sd / math.sqrt(n) if n > 1 else nan,
        scatter=sd,
        rms=float(np.sqrt(np.mean(dz**2))),
    )
    if err is not None and len(err) == n:
        err = np.asarray(err, float)
        out["err_mean"] = float(err.mean())
        out["scat_over_err"] = sd / out["err_mean"]
        out["in3"] = float(np.mean(np.abs(dz) < 3.0 * err))
    return out


def mode_stats(masses, truth) -> dict | None:
    """Mode-weight statistics over seeds, from the oracle masses of each run.

    ``masses`` is (seeds, modes), ``truth`` the exact masses. A minor mode is
    any mode but the heaviest. For each, the logit of its mass is taken over
    the seeds that kept it (at least ``LOST_FRACTION`` of its true mass).
    Returns the sd (with a bootstrap CI), bias and standard error of the mode
    with the largest sd, ``z_max`` (the largest ``|bias| / se`` of any minor
    mode) and ``lost`` (the fraction of seeds that lost some minor mode).
    """
    truth = np.asarray(truth, float)
    masses = np.asarray(masses, float).reshape(-1, len(truth))
    if not len(masses):
        return None
    nan = float("nan")
    minor = [k for k in range(len(truth)) if k != int(np.argmax(truth))]
    lost = np.any(masses[:, minor] < LOST_FRACTION * truth[minor], axis=1)
    worst, z_max = None, nan
    for k in minor:
        m = masses[:, k]
        keep = (m >= LOST_FRACTION * truth[k]) & (m < 1.0)
        lg = np.array([_logit(v) for v in m[keep]])
        sd = float(lg.std(ddof=1)) if len(lg) > 1 else nan
        bias = float(lg.mean() - _logit(truth[k])) if len(lg) else nan
        se = sd / math.sqrt(len(lg)) if len(lg) > 1 else nan
        if se > 0 and (math.isnan(z_max) or abs(bias) / se > z_max):
            z_max = abs(bias) / se
        if worst is None or (
            not math.isnan(sd) and (math.isnan(worst["sd"]) or sd > worst["sd"])
        ):
            lo, hi = bootstrap_sd_ci(lg)
            worst = dict(sd=sd, lo=lo, hi=hi, bias=bias, bias_se=se, mode=k)
    return dict(worst, z_max=z_max, lost=float(lost.mean()))


def _within(value, se, tol) -> bool:
    """|value| below 3 standard errors or below ``tol``; NaN is not."""
    if value is None or math.isnan(value):
        return False
    return abs(value) < tol or (se is not None and abs(value) < 3.0 * se)


def accurate(row) -> str:
    """'' when the cell is accurate, else the reasons: T (too many runs
    timed out or failed), Z (logZ bias), M (a mode lost, or its weight biased)."""
    why = []
    if row["n_ok"] < MIN_OK * row["n"]:
        why.append("T")
    if row["n_ok"] and not _within(row.get("dz_mean"), row.get("dz_se"), LOGZ_TOL):
        why.append("Z")
    if row.get("lost") is not None and (
        row["lost"] > MAX_LOST
        or not _within(row.get("logit_bias"), row.get("logit_bias_se"), MODE_TOL)
    ):
        why.append("M")
    return ",".join(why)


def _median_wall(recs):
    """Median wall time over ok and timed-out runs, and whether it is a timeout."""
    pairs = sorted(
        (r["wall_s"], r["status"] == "timeout")
        for r in recs
        if r["status"] in ("ok", "timeout") and r.get("wall_s") is not None
    )
    if not pairs:
        return None, False
    wall = float(np.median([w for w, _ in pairs]))
    return wall, pairs[(len(pairs) - 1) // 2][1] or pairs[len(pairs) // 2][1]


def summarize(groups):
    """One row (a flat dict) per cell, sorted by target then sampler."""
    rows = []
    for (sampler, target, nlive), recs in sorted(
        groups.items(),
        key=lambda kv: (_target_order(kv[0][1]), kv[0][0], kv[0][2] or 0),
    ):
        ok = [r for r in recs if r["status"] == "ok" and r.get("logz") is not None]
        wall, censored = _median_wall(recs)
        row = dict(
            target=target,
            family=family(target),
            ndim=recs[0].get("ndim"),
            sampler=sampler,
            hw="; ".join(sorted({hw_label(r) for r in recs})),
            nlive=nlive,
            live_points=live_points(sampler, recs),
            n=len(recs),
            n_ok=len(ok),
            n_timeout=sum(r["status"] == "timeout" for r in recs),
            n_error=sum(r["status"] == "error" for r in recs),
            wall_s=wall,
            wall_is_timeout=censored,
        )
        if ok:
            truth = ok[0]["truth"]
            errs = [r["logzerr"] for r in ok if r.get("logzerr") is not None]
            if truth["logz"] is not None:
                dz = [r["logz"] - truth["logz"] for r in ok]
                row.update(logz_stats(dz, errs if len(errs) == len(ok) else None))
            masses = [r["mode_mass"] for r in ok if r.get("mode_mass")]
            m = mode_stats(masses, truth["mode_mass"]) if masses else None
            if m:
                row.update(
                    logit_bias=m["bias"], logit_bias_se=m["bias_se"],
                    logit_sd=m["sd"], logit_sd_lo=m["lo"], logit_sd_hi=m["hi"],
                    lost=m["lost"], mode=m["mode"],
                )
            row["ncall"] = float(np.median([r["ncall"] for r in ok]))
            comp = [r["compile_s"] for r in ok if r.get("compile_s") is not None]
            row["compile_s"] = float(np.median(comp)) if comp else None
        row["why_not"] = accurate(row)
        row["accurate"] = not row["why_not"]
        rows.append(row)
    return rows


CSV_COLUMNS = (
    "target", "family", "ndim", "sampler", "hw", "nlive", "live_points", "n", "n_ok",
    "n_timeout", "n_error", "dz_mean", "dz_se", "rms", "scatter", "err_mean",
    "scat_over_err", "in3", "logit_bias", "logit_bias_se", "logit_sd", "logit_sd_lo",
    "logit_sd_hi", "lost", "ncall", "wall_s", "wall_is_timeout", "compile_s",
    "accurate", "why_not",
)


def _f(x, fmt="{:.3f}"):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return fmt.format(x)


def _pm(mean, se, fmt):
    return "-" if mean is None or math.isnan(mean) else f"{_f(mean, fmt)} ± {_f(se)}"


def cells_markdown(rows) -> str:
    head = (
        "| target | sampler | hw | nlive | ok | dlogZ | rms | scat/err | in 3s "
        "| logit bias | logit sd [95% CI] | lost | ncall | wall s | compile s | acc |"
    )
    out = [head, "|" + "---|" * (head.count("|") - 1)]
    f2 = "{:.2f}"
    for r in rows:
        ok = f"{r['n_ok']}/{r['n']}"
        if r["n_timeout"] or r["n_error"]:
            ok += f" ({r['n_timeout']}T {r['n_error']}E)"
        if r.get("lost") is None:
            bias = sd = lost = "-"
        else:
            bias = _pm(r["logit_bias"], r["logit_bias_se"], "{:+.2f}")
            sd = f"{_f(r['logit_sd'], f2)} [{_f(r['logit_sd_lo'], f2)}, "
            sd += f"{_f(r['logit_sd_hi'], f2)}]"
            lost = _f(r["lost"], f2)
        wall = (">" if r["wall_is_timeout"] else "") + _f(r["wall_s"], "{:.0f}")
        out.append(
            f"| {r['target']} | {r['sampler']} | {r['hw']} | {r['live_points']} | {ok} "
            f"| {_pm(r.get('dz_mean'), r.get('dz_se'), '{:+.3f}')} "
            f"| {_f(r.get('rms'))} | {_f(r.get('scat_over_err'), f2)} "
            f"| {_f(r.get('in3'), f2)} | {bias} | {sd} | {lost} "
            f"| {_f(r.get('ncall'), '{:.3g}')} | {wall} "
            f"| {_f(r.get('compile_s'), '{:.1f}')} "
            f"| {'yes' if r['accurate'] else 'no: ' + r['why_not']} |"
        )
    return "\n".join(out) + "\n"


def _families(rows):
    present = {r["family"] for r in rows}
    return [f for f in FAMILIES if f in present] + sorted(present - set(FAMILIES))


def accurate_markdown(rows) -> str:
    """Accurate cells / cells per sampler (rows) and target family (columns)."""
    fams = _families(rows)
    count = defaultdict(lambda: [0, 0])
    for r in rows:
        c = count[(r["sampler"], r["family"])]
        c[0] += r["accurate"]
        c[1] += 1
    out = ["| sampler | " + " | ".join(fams) + " |", "|---|" + "---|" * len(fams)]
    for s in sorted({r["sampler"] for r in rows}):
        cells = [
            "{}/{}".format(*count[(s, f)]) if (s, f) in count else "-" for f in fams
        ]
        out.append(f"| {s} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def _geomean(ratios):
    return math.exp(sum(math.log(x) for x in ratios) / len(ratios)) if ratios else None


def _pair(a, cells):
    """The cell of ``cells`` (one sampler, one target) closest in nlive to ``a``."""

    def distance(b):
        return abs(math.log((b["nlive"] or 1) / (a["nlive"] or 1)))

    return min(cells, key=distance)


def headline(rows, reference: str):
    """Per (family, competitor): costs relative to ``reference``.

    A reference cell pairs with the competitor's cell of the same target (of
    the nearest ``nlive`` when there are several). ``common`` counts the
    targets where both are accurate, out of those both ran. The ratios are
    competitor / reference, as geometric means over the common targets:

    ``ncall``, ``wall_s``   as run. Above 1: the reference is cheaper.
    ``rms``, ``logit_sd``   above 1: the reference is more precise.
    ``ncall_eq``, ``wall_eq``  the cost ratio times the squared rms ratio: the
        cost at equal logZ rms, if each sampler's cost goes as 1 / rms^2 (as
        when nlive is raised). This is the comparison at comparable accuracy
        when the two ran at different precision.

    Returns the ratio rows and, per family, the reference's rank by
    ``ncall_eq`` and by ``wall_eq`` among the samplers with a common target
    (1 is the cheapest).
    """
    by = defaultdict(list)
    for r in rows:
        by[(r["sampler"], r["target"])].append(r)
    ratios = []
    for fam in _families(rows):
        ref = [r for r in rows if r["sampler"] == reference and r["family"] == fam]
        for s in sorted({r["sampler"] for r in rows} - {reference}):
            pairs = [(a, by[(s, a["target"])]) for a in ref if (s, a["target"]) in by]
            pairs = [(a, _pair(a, cells)) for a, cells in pairs]
            if not pairs:
                continue
            both = [(a, b) for a, b in pairs if a["accurate"] and b["accurate"]]

            def ratio(k, k2=None, both=both):
                vals = [
                    b[k] / a[k] * ((b[k2] / a[k2]) ** 2 if k2 else 1.0)
                    for a, b in both
                    if all(x.get(key) for x in (a, b) for key in (k, k2) if key)
                ]
                return _geomean(vals)

            ratios.append(
                dict(
                    family=fam, sampler=s, common=len(both), ran=len(pairs),
                    hw=pairs[0][1]["hw"],
                    ncall=ratio("ncall"), wall_s=ratio("wall_s"), rms=ratio("rms"),
                    ncall_eq=ratio("ncall", "rms"), wall_eq=ratio("wall_s", "rms"),
                    logit_sd=ratio("logit_sd"),
                )
            )
    ranks = []
    for fam in _families(rows):
        ref = [r for r in rows if r["sampler"] == reference and r["family"] == fam]
        if not ref:
            continue
        rs = [x for x in ratios if x["family"] == fam and x["common"]]

        def rank(key, rs=rs):
            return 1 + sum(x[key] is not None and x[key] < 1 for x in rs)

        ranks.append(
            dict(
                family=fam,
                accurate=sum(r["accurate"] for r in ref),
                cells=len(ref),
                of=len(rs) + 1,
                rank_ncall=rank("ncall_eq"),
                rank_wall=rank("wall_eq"),
            )
        )
    return ratios, ranks


def headline_markdown(rows, reference: str) -> str:
    if not any(r["sampler"] == reference for r in rows):
        return (
            f"No `{reference}` records in the input, so there is no headline "
            "table yet (`--reference` ranks another sampler).\n"
        )
    ratios, ranks = headline(rows, reference)
    out = [
        f"`{reference}` per target family: its accurate cells, and its rank by "
        "cost at equal logZ rms among the samplers that are accurate on a "
        "common target.\n",
        "| family | accurate | rank by calls | rank by wall time |",
        "|---|---|---|---|",
    ]
    for r in ranks:
        out.append(
            f"| {r['family']} | {r['accurate']}/{r['cells']} "
            f"| {r['rank_ncall']} of {r['of']} | {r['rank_wall']} of {r['of']} |"
        )
    out += [
        "",
        f"Competitor / `{reference}`, geometric mean over the common accurate "
        f"targets. Above 1: `{reference}` needs fewer calls or less time, or has "
        "the smaller rms or logit sd. `at equal rms` is the cost ratio times "
        "the squared rms ratio (cost taken to go as 1 / rms^2). Wall times "
        "compare across hardware labels only as measured.\n",
        "| family | competitor | hw | common / ran | calls | wall time | rms "
        "| calls at equal rms | wall time at equal rms | logit sd |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    g = "{:.3g}"
    for x in ratios:
        out.append(
            f"| {x['family']} | {x['sampler']} | {x['hw']} | {x['common']}/{x['ran']} "
            f"| {_f(x['ncall'], g)} | {_f(x['wall_s'], g)} | {_f(x['rms'], g)} "
            f"| {_f(x['ncall_eq'], g)} | {_f(x['wall_eq'], g)} "
            f"| {_f(x['logit_sd'], g)} |"
        )
    return "\n".join(out) + "\n"


def hardware(groups) -> list[str]:
    """One line per hardware label: the devices and hosts behind it."""
    seen = defaultdict(set)
    for recs in groups.values():
        for r in recs:
            hw = r.get("hw") or {}
            part = hw.get("slurm_partition")
            seen[hw_label(r)].add(
                (hw.get("gpu") or "no GPU", hw.get("cpu_model"), hw.get("cpu_threads"),
                 f"Slurm {part}" if part else "no Slurm")
            )
    lines = []
    for k, v in sorted(seen.items()):
        devs = [f"{g}, {c}, {n} threads visible, {p}" for g, c, n, p in sorted(v)]
        lines.append(f"- `{k}`: " + "; ".join(devs))
    return lines


def report(groups, reference: str = "tinyns_v1") -> tuple[str, list[dict]]:
    rows = summarize(groups)
    nrec = sum(len(v) for v in groups.values())
    md = [
        "# Nested-sampler head-to-head",
        "",
        f"{nrec} runs in {len(rows)} cells (sampler x target x nlive). "
        "The columns are described in `bench/summarize.py`; the samplers' "
        "settings and the caveats in `bench/README.md`.",
        "",
        "Hardware (wall times compare only within one label):",
        *hardware(groups),
        "",
        "A cell is accurate (`acc`) when at least 90% of its runs finished "
        f"(else `T`), its logZ bias is within 3 se or {LOGZ_TOL} nats (else `Z`) "
        "and, for a mixture, at most 10% of its seeds lost a mode and the logit "
        f"bias of the mode weight is within 3 se or {MODE_TOL} (else `M`).",
        "",
        "## Headline",
        "",
        headline_markdown(rows, reference),
        "## Accurate cells per sampler and target family",
        "",
        accurate_markdown(rows),
        "## Cells",
        "",
        cells_markdown(rows),
    ]
    return "\n".join(md), rows


def write_csv(path, rows) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _json_safe(x):
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return bool(x) if isinstance(x, np.bool_) else x


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("files", nargs="+")
    p.add_argument("--out", default=None, help="write the Markdown here, not stdout")
    p.add_argument("--csv", default=None, help="write one row per cell as CSV")
    p.add_argument("--json", default=None, help="write the cell rows as JSON")
    p.add_argument(
        "--reference", default="tinyns_v1", help="the sampler the headline ranks"
    )
    args = p.parse_args(argv)
    md, rows = report(load(args.files), args.reference)
    if args.csv:
        write_csv(args.csv, rows)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            safe = [{k: _json_safe(v) for k, v in r.items()} for r in rows]
            json.dump(dict(schema="tinyns-bench-cells-1", cells=safe), f, indent=1)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(md)
    else:
        print(md, end="")


if __name__ == "__main__":
    main()
