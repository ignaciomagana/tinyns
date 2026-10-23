"""The statistical release gate: tinyns on standard targets, PASS or FAIL.

    python bench/validate.py                  # quick tier: about 7 min on 4 CPU cores
    python bench/validate.py --tier full      # every case; a GPU or a Slurm node
    python bench/validate.py --cases gauss_d2,sepW_d10 --seeds 32 --json gate.json

Each case runs ``--seeds`` independent runs of one target as one batch of keys
(one compiled program) and compares them with the target's exact evidence,
mode masses and, for the Gaussians, posterior (``bench/targets.py``). The
exit code is 1 if any case fails.

Criteria (``judge``), with se the standard error over the seeds:

``bias``      |mean(logz - truth)| < 3 se
``scatter``   sd(logz - truth) / mean(logzerr) in [0.75, 1.3]. The ratio of N
              seeds has a relative error of 1 / sqrt(2 (N - 1)), so the case
              fails only when the ratio is outside the band by more than 3 of
              those (the ``3 se`` interval is printed).
``posterior`` Gaussians only: along each eigenvector of the true covariance,
              the offset of the posterior mean (in units of the true sd
              there) and the log of the posterior variance over the true
              one are zero within their scatter over the seeds. The largest
              |mean| / se of those 2 ndim numbers must stay below the bound
              at which Student's t (N - 1 dof) gives one false alarm in 370
              cases, as a single 3 se test does (Bonferroni over the 2 ndim).
``modes``     mixtures only: logit bias of every minor mode's weight < 3 se,
              its sd over seeds within the case's limit (``Case.logit_sd``;
              0.4 at most), and no seed lost a mode (weight below 10% of the
              truth). The weights are oracle masses: the target's exact
              responsibility averaged over the posterior samples.
``hop``       mixtures only: in at least 90% of the runs the inter-mode hop
              was tried and accepted at least ``HOP_ACCEPT_WARN`` (2%) of
              its proposals.
``count``     the number of modes ``result.modes()`` reports is the true
              one in at least 90% of the runs. Where the truth has more
              modes than the sampler's clustering can separate (the eggbox),
              a run may report fewer, but then it must say so: a mode that
              is not ``tracked``, or a full-slots warning.
``flags``     no run ended without converging, and no mode of
              ``result.modes()`` is flagged ``unresolved`` (but for the
              cases whose smallest mode holds too few live points to be
              resolved: see ``CASES``)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCATTER_BAND = (0.75, 1.3)
MAX_LOGIT_SD = 0.4
NSIGMA = 3.0

MAX_MISCOUNT = 0.1


class Case(NamedTuple):
    """A gate case: ``nlive`` in the quick tier (None: not in it) and in the
    full tier; ``walks`` (None: the sampler default); the smallest and the
    largest number of modes a run may report, and ``small`` if the smallest
    mode is too small for ``unresolved`` to count as a failure; the largest
    logit sd of the minor-mode weights at the full-tier ``nlive`` (it scales
    as ``1 / sqrt(nlive)`` at another ``nlive``, up to ``MAX_LOGIT_SD``)."""

    quick: int | None
    full: int
    walks: int | None = None
    modes: tuple[int, int] = (1, 1)
    small: bool = False
    logit_sd: float = MAX_LOGIT_SD


# - quick keeps to d <= 10 and nlive 500 (more where a mixture needs it);
#   full runs the default nlive (1000).
# - Rosenbrock runs at walks = 25 * ndim. The default walks (60 at d = 10)
#   leave the logZ of rosen_d10 0.05 +- 0.04 low at nlive 1000 (README,
#   Choosing walks; -0.09 +- 0.04 in 32 more float32 runs), close to the 3 se
#   of 64 seeds, so a case at the default would fail now and then for a bias
#   the README states. At 25 * ndim the bias is gone, and the cases test the
#   sampler on a curved valley. The default walks are gated on the other
#   cases, none of them curved.
# - ndtri_d6 puts a Gaussian posterior in a corner of the unit cube, through a
#   nonlinear prior transform (``bench.targets.ndtri_corner``). The default
#   walks (36) do not decorrelate the chains there, and the case fails bias
#   and scatter in both tiers (key 0: logZ 0.60 +- 0.19 low, scatter 3.0
#   times logzerr, quick; 0.24 +- 0.08 and 3.5 times, full; x64). At 72 walks
#   the scatter is 1.7 times logzerr; at 144 (4 times the default) it is 1.05
#   at nlive 1000 and 1.2 at 500, with no bias (128 runs each, f32 and x64).
# - The 6% minor mode of sepW needs 5 * ndim live points through its bulk to be
#   resolved (tinyns.result.UNRESOLVED_PER_DIM): 0.06 * nlive is 120 against
#   50 at d = 10, 240 against 90 at d = 18 and 60 against 20 at d = 4.
#   sepWbox_d4 is sepW_d4 in the box [-3, 5]^4 (``bench.targets.BOX``): the
#   mode masses of its samples are scored in the box's coordinates.
# - logit_sd: with the hop the logit sd of the sepW minor mode is 0.02 to
#   0.07 at the gate's nlive, and without it (Config._hop=False) 0.25 to 0.37
#   (H100 and CPU, 16 and 64 seeds), which the accuracy floor of 0.4 lets
#   through; 0.12 tells them apart.
# - modes: the smallest and the largest number of modes a run may report.
#   - rosen_d10 has a second lobe at x_1 = -1 with 2.52% of the mass (by the
#     quadrature of ``bench.targets``, P(x_1 < 0)), joined to the main mode by
#     a bridge of 0.8% of the mass (|x_1| < 0.3): one mode or two are both
#     right, and at nlive 1000 the lobe holds about 25 live points, under
#     5 * ndim. At d = 4 the two arms are one mode (6.3% and 6.8%).
#   - eggbox has 18 modes of mass 0.08 (8), 0.04 (8) and 0.02 (2), which the
#     clustering does not separate, so the hop does not balance them: a run
#     reports the ones its live points still hold apart, not tracked, and the
#     0.02 modes hold about 10 live points at nlive 500. Measured: 13 to 18
#     at nlive 1000 (192 runs), 4 to 18 at 500 (960 runs), every count below
#     18 flagged. The floor of 3 sits below that and still fails a run that
#     reports one or two modes.
CASES = {
    "gauss_d2": Case(500, 1000),
    "gauss_d10": Case(500, 1000),
    "gauss_d32": Case(None, 1000),
    "rosen_d4": Case(500, 1000, walks=100),
    "rosen_d10": Case(None, 1000, walks=250, modes=(1, 2), small=True),
    "funnel_d10": Case(500, 1000),
    "loggamma_d10": Case(500, 1000, modes=(4, 4)),
    "eggbox_d2": Case(500, 1000, modes=(3, 18), small=True),
    "ndtri_d6": Case(500, 1000, walks=144),
    "sepWbox_d4": Case(1000, 1000, modes=(2, 2), logit_sd=0.12),
    "sepW_d10": Case(2000, 2000, modes=(2, 2), logit_sd=0.12),
    "sepW_d18": Case(None, 4000, modes=(2, 2), logit_sd=0.12),
}
TIERS = {"quick": dict(column=0, seeds=16), "full": dict(column=1, seeds=64)}


def run_case(target_name, nlive, seeds=16, seed=0, case=None) -> dict:
    """Run ``seeds`` batched tinyns runs of one target; return its statistics.

    ``case``: a :class:`Case` for the walks, the mode counts a run may report
    and the logit-sd limit (default: the sampler's walks, no ``count``
    criterion, and ``MAX_LOGIT_SD``)."""
    if seeds < 2:
        raise ValueError("a case needs at least 2 seeds: its criteria are scatters")
    import jax
    import numpy as np

    import tinyns
    from bench.run import weighted_summary
    from bench.summarize import logz_stats, mode_stats
    from bench.targets import get_target

    target = get_target(target_name)
    sampler = tinyns.NestedSampler(
        target.loglike, target.prior_transform, target.ndim, nlive,
        walks=case.walks if case else None,
    )
    keys = jax.random.split(jax.random.PRNGKey(seed), seeds)
    t0 = time.perf_counter()
    results = sampler.run(keys)
    wall = time.perf_counter() - t0
    checks = [r.diagnostics() for r in results]
    found = [d["modes"] for d in checks]
    bound = [  # the run says that its mode count is a lower bound
        any(not m["tracked"] for m in d["modes"])
        or bool(len(d["modes"]) > 1 and d["mode_slots_full"])
        for d in checks
    ]
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
        unresolved=sum(any(m["unresolved"] for m in ms) for ms in found),
        nmodes=[len(ms) for ms in found],
        lower_bound=bound,
    )
    if case is not None:
        row.update(
            modes_min=case.modes[0], modes_max=case.modes[1], small_mode=case.small
        )
    row.update(logz_stats(np.array(row["logz"]) - target.logz, row["logzerr"]))
    if "cov" in target.meta:
        mean, cov = target.meta["mean"], target.meta["cov"]
        offsets = [posterior_offsets(r.samples, r.logwt, mean, cov) for r in results]
        row["post_mean"] = [z.tolist() for z, _ in offsets]
        row["post_logvar"] = [v.tolist() for _, v in offsets]
        row.update(posterior_stats(row["post_mean"], row["post_logvar"]))
    if target.mode_mass is not None:
        masses = [weighted_summary(target, r.samples, r.logwt)[0] for r in results]
        row["mode_mass"] = masses
        m = mode_stats(masses, target.mode_mass)
        row.update(
            logit_bias=m["bias"], logit_bias_se=m["bias_se"], logit_z=m["z_max"],
            logit_sd=m["sd"], lost=m["lost"],
            max_logit_sd=MAX_LOGIT_SD if case is None else min(
                MAX_LOGIT_SD, case.logit_sd * math.sqrt(case.full / nlive)
            ),
        )
        if len(target.mode_mass) > 1:
            row["hop_acceptance"] = [d["hop_acceptance"] for d in checks]
    row["failed"] = judge(row)
    return row


def posterior_offsets(samples, logwt, mean, cov):
    """``(z, v)`` of one run along the eigenvectors ``e_i`` of the true
    covariance ``cov`` (variances ``s_i^2``): ``z_i = e_i . (m - mean) / s_i``
    and ``v_i = log(e_i^T C e_i / s_i^2)``, where ``m`` and ``C`` are the
    run's weighted posterior mean and covariance (in float64)."""
    import numpy as np

    lam, e = np.linalg.eigh(np.asarray(cov, float))
    lw = np.asarray(logwt, float)
    lw = np.where(np.isfinite(lw), lw, -np.inf)
    w = np.exp(lw - lw.max())
    w /= w.sum()
    y = (np.asarray(samples, float) - np.asarray(mean, float)) @ e
    m = w @ y
    var = w @ (y - m) ** 2
    return m / np.sqrt(lam), np.log(var / lam)


def posterior_stats(post_mean, post_logvar) -> dict:
    """The ``posterior`` criterion's numbers: ``post_t``, the largest |mean| /
    se over the seeds of the 2 ndim offsets of :func:`posterior_offsets`, and
    ``post_bound``, its limit (:func:`t_bound`)."""
    import numpy as np

    x = np.concatenate([np.asarray(post_mean), np.asarray(post_logvar)], axis=1)
    n = len(x)
    se = x.std(axis=0, ddof=1) / math.sqrt(n)
    t = np.where(se > 0, np.abs(x.mean(axis=0)) / np.where(se > 0, se, 1.0), np.inf)
    return dict(post_t=float(t.max()), post_bound=t_bound(x.shape[1], n - 1))


def t_tail(t: float, dof: int) -> float:
    """``P(|T| > t)`` for Student's t with ``dof`` degrees of freedom: with
    ``T = sqrt(dof) tan(theta)``, the density of ``theta`` is
    ``cos(theta)^(dof - 1) / B(dof / 2, 1 / 2)`` on (-pi/2, pi/2). It falls
    by e^-200 or more within ``20 / sqrt(dof)`` of the lower limit."""
    import numpy as np

    from bench.targets import _trapezoid

    start = math.atan(t / math.sqrt(dof))
    theta = np.linspace(start, min(math.pi / 2, start + 20 / math.sqrt(dof)), 4001)
    norm = math.exp(math.lgamma((dof + 1) / 2) - math.lgamma(dof / 2)) / math.sqrt(
        math.pi
    )
    return 2.0 * norm * float(_trapezoid(np.cos(theta) ** (dof - 1), theta))


def t_bound(ntests: int, dof: int) -> float:
    """The bound on |t| for ``ntests`` two-sided Student-t tests with ``dof``
    degrees of freedom: under the null, one of them or more exceeds it with
    (at most) the probability that one normal deviate exceeds ``NSIGMA``
    (Bonferroni)."""
    alpha = math.erfc(NSIGMA / math.sqrt(2.0)) / ntests
    lo, hi = 0.0, 1e3
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if t_tail(mid, dof) > alpha else (lo, mid)
    return hi


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
    if "post_t" in row and not row["post_t"] < row["post_bound"]:
        failed.append("posterior")
    if "lost" in row and not (
        row["lost"] == 0
        and row["logit_z"] < NSIGMA
        and row["logit_sd"] <= row.get("max_logit_sd", MAX_LOGIT_SD)
    ):
        failed.append("modes")
    if "hop_acceptance" in row and hop_failures(row) > MAX_MISCOUNT * row["seeds"]:
        failed.append("hop")
    if "modes_min" in row and miscounted(row) > MAX_MISCOUNT * row["seeds"]:
        failed.append("count")
    if row["not_converged"] or (row["unresolved"] and not row.get("small_mode")):
        failed.append("flags")
    return failed


def hop_failures(row: dict) -> int:
    """The runs whose inter-mode hop was never tried (``hop_acceptance`` is
    None) or accepted less than ``HOP_ACCEPT_WARN`` of its proposals."""
    from tinyns.result import HOP_ACCEPT_WARN

    return sum(a is None or a < HOP_ACCEPT_WARN for a in row["hop_acceptance"])


def miscounted(row: dict) -> int:
    """The runs whose mode count is wrong: outside ``modes_min..modes_max``,
    or, where the two differ, below ``modes_max`` without saying that the
    count is a lower bound (``lower_bound``; one mode needs no such flag when
    one mode is right)."""
    lo, hi = row["modes_min"], row["modes_max"]
    return sum(
        not lo <= n <= hi or (lo > 1 and n < hi and not flagged)
        for n, flagged in zip(row["nmodes"], row["lower_bound"], strict=True)
    )


def to_markdown(rows) -> str:
    head = (
        "| case | nlive | walks | seeds | dlogZ ± se | z | scat/err [3 se] "
        "| post t [max] | logit bias ± se | logit sd [max] | lost | hop "
        "| modes | unres | ncall | wall s | verdict |"
    )
    out = [head, "|" + "---|" * (head.count("|") - 1)]
    for r in rows:
        lo, hi = scatter_interval(r["scat_over_err"], r["seeds"])
        se = r["dz_se"]
        z = f"{r['dz_mean'] / se:+.1f}" if math.isfinite(se) and se > 0 else "-"
        post = "-"
        if "post_t" in r:
            post = f"{r['post_t']:.1f} [{r['post_bound']:.1f}]"
        if "lost" in r:
            modes = (
                f"{r['logit_bias']:+.3f} ± {r['logit_bias_se']:.3f} "
                f"| {r['logit_sd']:.3f} [{r['max_logit_sd']:.2g}] | {r['lost']:.2f}"
            )
        else:
            modes = "- | - | -"
        hop = "-"
        if "hop_acceptance" in r:
            tried = [a for a in r["hop_acceptance"] if a is not None]
            hop = f"{min(tried):.2f}" if tried else "off"
            if hop_failures(r):
                hop += f" ({hop_failures(r)}/{r['seeds']} low)"
        low, high = min(r["nmodes"]), max(r["nmodes"])
        count = str(low) if low == high else f"{low}-{high}"
        if "modes_min" in r:
            count += f" ({r['seeds'] - miscounted(r)}/{r['seeds']} ok)"
        flags = f"{r['unresolved']}/{r['seeds']}"
        if r["not_converged"]:
            flags += f" ({r['not_converged']} not converged)"
        verdict = "FAIL: " + ", ".join(r["failed"]) if r["failed"] else "PASS"
        out.append(
            f"| {r['case']} | {r['nlive']} | {r['walks']} | {r['seeds']} "
            f"| {r['dz_mean']:+.3f} ± {se:.3f} | {z} "
            f"| {r['scat_over_err']:.2f} [{lo:.2f}, {hi:.2f}] | {post} | {modes} "
            f"| {hop} | {count} | {flags} "
            f"| {r['ncall']:.3g} | {r['wall_s']:.0f} | {verdict} |"
        )
    return "\n".join(out) + "\n"


def unresolvable(name: str, nlive: int, case: Case) -> str | None:
    """A warning when the smallest mode of a case's target holds fewer than
    ``UNRESOLVED_PER_DIM * ndim`` live points at ``nlive``: the case then
    fails ``flags``."""
    from bench.targets import get_target
    from tinyns.result import UNRESOLVED_PER_DIM

    target = get_target(name)
    if target.mode_mass is None or case.small:
        return None
    held, needed = min(target.mode_mass) * nlive, UNRESOLVED_PER_DIM * target.ndim
    if held >= needed:
        return None
    return (
        f"{name}: at nlive {nlive} its smallest mode holds about {held:.0f} live "
        f"points, fewer than the {needed:.0f} that resolve it: expect a FAIL on "
        "flags"
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tier", choices=sorted(TIERS), default="quick")
    p.add_argument(
        "--cases",
        default=None,
        help="comma-separated cases, run whatever their tier (at the tier's "
        "nlive, or the full tier's for a case outside the tier)",
    )
    p.add_argument("--seeds", type=int, default=None, help="runs per case (>= 2)")
    p.add_argument("--nlive", type=int, default=None, help="override every case")
    p.add_argument("--seed", type=int, default=0, help="the key the runs split from")
    p.add_argument("--x64", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--json", default=None, help="write the rows (per-seed values too)")
    args = p.parse_args(argv)
    if args.seeds is not None and args.seeds < 2:
        p.error("--seeds must be at least 2: the criteria are scatters over seeds")

    if "jax" not in sys.modules:  # must precede the first JAX import
        os.environ["JAX_ENABLE_X64"] = "1" if args.x64 else "0"
    import jax

    jax.config.update("jax_enable_x64", args.x64)
    import tinyns
    from bench.run import _finite, git_sha

    tier = TIERS[args.tier]
    names = args.cases.split(",") if args.cases else list(CASES)
    unknown = [n for n in names if n not in CASES]
    if unknown:
        raise SystemExit(f"unknown cases {unknown}; known: {list(CASES)}")
    seeds = args.seeds or tier["seeds"]
    rows = []
    for name in names:
        case = CASES[name]
        base = (case.quick, case.full)[tier["column"]]
        if base is None and not args.cases:
            continue  # not in this tier
        nlive = args.nlive or base or case.full
        if base is None and not args.nlive:
            note = f"{name}: not in the {args.tier} tier; nlive {nlive}"
            print(note, file=sys.stderr)
        warning = unresolvable(name, nlive, case)
        if warning:
            print(f"warning: {warning}", file=sys.stderr)
        row = run_case(name, nlive, seeds, args.seed, case)
        rows.append(row)
        verdict = "FAIL: " + ", ".join(row["failed"]) if row["failed"] else "PASS"
        print(f"{name}: {verdict} ({row['wall_s']:.0f} s)", file=sys.stderr, flush=True)
    dev = jax.devices()[0]
    sha = git_sha()
    where = Path(tinyns.__file__).resolve().parent
    meta = dict(
        schema="tinyns-validate-1", tier=args.tier, tinyns=tinyns.__version__,
        git_sha=sha, tinyns_path=str(where), jax=jax.__version__,
        platform=dev.platform, x64=args.x64, seed=args.seed, seeds=seeds,
        nlive=args.nlive, cases=rows,
    )
    if args.json:  # before the table: the runs are kept even if printing fails
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(_finite(meta), f, indent=1)
    custom = f", nlive {args.nlive} in every case" if args.nlive else ""
    print(
        f"tinyns {tinyns.__version__} at {sha or 'an unknown commit'}, "
        f"jax {jax.__version__} on {dev.platform}, "
        f"x64 {bool(jax.config.jax_enable_x64)}, {args.tier} tier, key {args.seed}, "
        f"{seeds} seeds{custom}"
    )
    if ROOT / "src" not in where.parents:
        print(f"tinyns imported from {where}, not from this checkout")
    print()
    print(to_markdown(rows), end="")
    nfail = sum(bool(r["failed"]) for r in rows)
    print(f"\n{len(rows) - nfail}/{len(rows)} cases pass")
    return 1 if nfail or not rows else 0


if __name__ == "__main__":
    sys.exit(main())
