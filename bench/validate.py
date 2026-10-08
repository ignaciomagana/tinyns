"""The statistical release gate: tinyns on standard targets, PASS or FAIL.

    python bench/validate.py                  # quick tier: about 6 min on 4 CPU cores
    python bench/validate.py --tier full      # every case; a GPU or a Slurm node
    python bench/validate.py --cases gauss_d2,sepW_d10 --seeds 32 --json gate.json

Each case runs ``--seeds`` independent runs of one target as one batch of keys
(one compiled program) and compares them with the target's exact evidence and
mode masses (``bench/targets.py``). The exit code is 1 if any case fails.

Criteria (``judge``), with se the standard error over the seeds:

``bias``      |mean(logz - truth)| < 3 se
``scatter``   sd(logz - truth) / mean(logzerr) in [0.75, 1.3]. The ratio of N
              seeds has a relative error of 1 / sqrt(2 (N - 1)), so the case
              fails only when the ratio is outside the band by more than 3 of
              those (the ``3 se`` interval is printed).
``modes``     mixtures only: logit bias of every minor mode's weight < 3 se,
              its sd over seeds <= 0.4, and no seed lost a mode (weight below
              10% of the truth). The weights are oracle masses: the target's
              exact responsibility averaged over the posterior samples.
``flags``     no run ended without converging, and no mode of
              ``result.modes()`` is flagged ``unresolved``
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCATTER_BAND = (0.75, 1.3)
MAX_LOGIT_SD = 0.4
NSIGMA = 3.0

# target: (nlive in the quick tier, nlive in the full tier, walks). A tier
# skips the cases whose nlive is None; walks None is the sampler default.
# - quick keeps to d <= 10 and nlive 500; full runs the default nlive (1000).
# - Rosenbrock needs more steps per chain than the default (25 * ndim; see the
#   NestedSampler docstring).
# - The 6% minor mode of sepW needs 5 * ndim live points through its bulk to be
#   resolved (tinyns.result.UNRESOLVED_PER_DIM): 0.06 * nlive is 120 against
#   50 at d = 10, and 240 against 90 at d = 18.
CASES = {
    "gauss_d2": (500, 1000, None),
    "gauss_d10": (500, 1000, None),
    "gauss_d32": (None, 1000, None),
    "rosen_d4": (500, 1000, 100),
    "rosen_d10": (None, 1000, 250),
    "funnel_d10": (500, 1000, None),
    "loggamma_d10": (500, 1000, None),
    "eggbox_d2": (500, 1000, None),
    "sepW_d10": (2000, 2000, None),
    "sepW_d18": (None, 4000, None),
}
TIERS = {"quick": dict(column=0, seeds=16), "full": dict(column=1, seeds=64)}


def run_case(target_name, nlive, walks=None, seeds=16, seed=0) -> dict:
    """Run ``seeds`` batched tinyns runs of one target; return its statistics."""
    import jax
    import numpy as np

    import tinyns
    from bench.run import weighted_summary
    from bench.summarize import logz_stats, mode_stats
    from bench.targets import get_target

    target = get_target(target_name)
    sampler = tinyns.NestedSampler(
        target.loglike, target.prior_transform, target.ndim, nlive, walks=walks
    )
    keys = jax.random.split(jax.random.PRNGKey(seed), seeds)
    t0 = time.perf_counter()
    results = sampler.run(keys)
    wall = time.perf_counter() - t0
    row = dict(
        case=target_name,
        ndim=target.ndim,
        nlive=nlive,
        walks=int(sampler.config.walks),
        seeds=seeds,
        wall_s=wall,
        ncall=float(np.median([r.ncall for r in results])),
        logz=[float(r.logz) for r in results],
        logzerr=[float(r.logzerr) for r in results],
        not_converged=sum(not r.success for r in results),
        unresolved=sum(any(m["unresolved"] for m in r.modes()) for r in results),
        nmodes=[len(r.modes()) for r in results],
    )
    row.update(logz_stats(np.array(row["logz"]) - target.logz, row["logzerr"]))
    if target.mode_mass is not None:
        masses = [weighted_summary(target, r.samples, r.logwt)[0] for r in results]
        row["mode_mass"] = masses
        m = mode_stats(masses, target.mode_mass)
        row.update(
            logit_bias=m["bias"], logit_bias_se=m["bias_se"], logit_z=m["z_max"],
            logit_sd=m["sd"], lost=m["lost"],
        )
    row["failed"] = judge(row)
    return row


def scatter_interval(ratio: float, seeds: int) -> tuple[float, float]:
    """The ``NSIGMA`` interval of a scatter/logzerr ratio measured on ``seeds``
    runs: the log of a sample sd has standard error ``1 / sqrt(2 (N - 1))``."""
    width = math.exp(NSIGMA / math.sqrt(2.0 * max(seeds - 1, 1)))
    return ratio / width, ratio * width


def judge(row: dict) -> list[str]:
    """The criteria a case fails (see the module docstring); [] is a PASS."""
    failed = []
    if not abs(row["dz_mean"]) < NSIGMA * row["dz_se"]:
        failed.append("bias")
    lo, hi = scatter_interval(row["scat_over_err"], row["seeds"])
    if not (lo <= SCATTER_BAND[1] and hi >= SCATTER_BAND[0]):
        failed.append("scatter")
    if "lost" in row and not (
        row["lost"] == 0 and row["logit_z"] < NSIGMA and row["logit_sd"] <= MAX_LOGIT_SD
    ):
        failed.append("modes")
    if row["not_converged"] or row["unresolved"]:
        failed.append("flags")
    return failed


def to_markdown(rows) -> str:
    head = (
        "| case | nlive | walks | seeds | dlogZ ± se | z | scat/err [3 se] "
        "| logit bias ± se | logit sd | lost | unres | ncall | wall s | verdict |"
    )
    out = [head, "|" + "---|" * (head.count("|") - 1)]
    for r in rows:
        lo, hi = scatter_interval(r["scat_over_err"], r["seeds"])
        if "lost" in r:
            modes = (
                f"{r['logit_bias']:+.3f} ± {r['logit_bias_se']:.3f} "
                f"| {r['logit_sd']:.3f} | {r['lost']:.2f}"
            )
        else:
            modes = "- | - | -"
        flags = f"{r['unresolved']}/{r['seeds']}"
        if r["not_converged"]:
            flags += f" ({r['not_converged']} not converged)"
        verdict = "FAIL: " + ", ".join(r["failed"]) if r["failed"] else "PASS"
        out.append(
            f"| {r['case']} | {r['nlive']} | {r['walks']} | {r['seeds']} "
            f"| {r['dz_mean']:+.3f} ± {r['dz_se']:.3f} "
            f"| {r['dz_mean'] / r['dz_se']:+.1f} "
            f"| {r['scat_over_err']:.2f} [{lo:.2f}, {hi:.2f}] | {modes} | {flags} "
            f"| {r['ncall']:.3g} | {r['wall_s']:.0f} | {verdict} |"
        )
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tier", choices=sorted(TIERS), default="quick")
    p.add_argument("--cases", default=None, help="comma-separated subset of the cases")
    p.add_argument("--seeds", type=int, default=None, help="runs per case")
    p.add_argument("--nlive", type=int, default=None, help="override every case")
    p.add_argument("--seed", type=int, default=0, help="the key the runs split from")
    p.add_argument("--x64", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--json", default=None, help="write the rows (per-seed values too)")
    args = p.parse_args(argv)

    if "jax" not in sys.modules:  # must precede the first JAX import
        os.environ["JAX_ENABLE_X64"] = "1" if args.x64 else "0"
    import jax

    jax.config.update("jax_enable_x64", args.x64)
    import tinyns
    from bench.run import _finite

    tier = TIERS[args.tier]
    names = args.cases.split(",") if args.cases else list(CASES)
    unknown = [n for n in names if n not in CASES]
    if unknown:
        raise SystemExit(f"unknown cases {unknown}; known: {list(CASES)}")
    seeds = args.seeds or tier["seeds"]
    rows = []
    for name in names:
        nlive = args.nlive or CASES[name][tier["column"]]
        if nlive is None:
            if args.cases:
                print(f"{name} is not in the {args.tier} tier", file=sys.stderr)
            continue
        row = run_case(name, nlive, CASES[name][2], seeds, args.seed)
        rows.append(row)
        verdict = "FAIL: " + ", ".join(row["failed"]) if row["failed"] else "PASS"
        print(f"{name}: {verdict} ({row['wall_s']:.0f} s)", file=sys.stderr, flush=True)
    dev = jax.devices()[0]
    print(
        f"tinyns {tinyns.__version__}, jax {jax.__version__} on {dev.platform}, "
        f"x64 {bool(jax.config.jax_enable_x64)}, {args.tier} tier, key {args.seed}\n"
    )
    print(to_markdown(rows), end="")
    nfail = sum(bool(r["failed"]) for r in rows)
    print(f"\n{len(rows) - nfail}/{len(rows)} cases pass")
    if args.json:
        meta = dict(
            schema="tinyns-validate-1", tier=args.tier, tinyns=tinyns.__version__,
            jax=jax.__version__, platform=dev.platform, x64=args.x64, seed=args.seed,
        )
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(_finite(dict(meta, cases=rows)), f, indent=1)
    return 1 if nfail or not rows else 0


if __name__ == "__main__":
    sys.exit(main())
