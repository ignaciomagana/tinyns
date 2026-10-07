"""Queue lines for the full bake-off grid (one line per cell, 40 seeds batched).

    python bench/bakeoff/emit.py > queue/bakeoff_jobs.txt
    python bench/bakeoff/emit.py --estimate      # the H100 time estimate only

The lines follow ``bench/h100_plan.sh``: paths relative to ``$P`` on js2h100,
each line holds ``gpu.lock`` so no two cells time the GPU at once. Override
the paths with ``BENCH_PY``, ``BAKEOFF_RUN``, ``OUT`` and ``TAG``.

Grid (plan, "Bake-off"):

- mixtures: ``sepW_d{4,10,18,32}``, ``sepM_d{10,18,32}``, ``connW_d{10,18,32}``,
  ``mix3_d10`` and the banana-twisted ``sepWtw_d10``;
- controls: correlated Gaussians ``gauss_d{18,32}``, ``rosen_d4``,
  ``funnel_d10``;
- every arm at nlive 500 and 2000 with ``k = nlive / 10``, plus ``k = 1`` at
  nlive 500 for ``N`` and ``B_ell`` on ``sepW_d18``; seeds 0-39 (the v1
  bake-off ran five arms, 152 cells; ``B_ell`` won and the others are gone).

**Estimate.** A cell runs ``10 T`` steps (``T`` e-folds to convergence, ``k =
m / 10``) of ``walks`` sequential chain steps, whatever ``m``. The model:
55 us per chain step (the PR 1 H100 speed run: gauss_d32, m 500, k 50,
walks 192, 952 steps in 10 s, one run, float32), times 1.5 for 40 batched
lanes at m 500 and 3 at m 2000, plus one recluster every ``recluster_every``
steps at 20 ms (m 500) or 60 ms (m 2000) (both arms cluster), plus 60 s of
compilation per cell. ``k = 1`` cells run ``m T`` steps. ``T`` per target is
the information plus a few e-folds (from the prototype geometry and the
Gaussian truths). Treat the result as good to a factor of 2-3; the CPU
smoke run gives the relative cost of the arms.
"""

from __future__ import annotations

import argparse
import os
import sys

ARMS = ("N", "B_ell")
MIXTURES = (
    [f"sepW_d{d}" for d in (4, 10, 18, 32)]
    + [f"sepM_d{d}" for d in (10, 18, 32)]
    + [f"connW_d{d}" for d in (10, 18, 32)]
    + ["mix3_d10", "sepWtw_d10"]
)
CONTROLS = ["gauss_d18", "gauss_d32", "rosen_d4", "funnel_d10"]
NLIVE = (500, 2000)
SEEDS = "0-39"
K1_CELLS = [("sepW_d18", "N"), ("sepW_d18", "B_ell")]
# e-folds to convergence (information plus a few e-folds)
EFOLDS = {
    "sepW_d4": 20,
    "sepW_d10": 45,
    "sepW_d18": 75,
    "sepW_d32": 130,
    "sepM_d10": 45,
    "sepM_d18": 75,
    "sepM_d32": 130,
    "connW_d10": 45,
    "connW_d18": 75,
    "connW_d32": 130,
    "mix3_d10": 45,
    "sepWtw_d10": 45,
    "gauss_d18": 55,
    "gauss_d32": 95,
    "rosen_d4": 20,
    "funnel_d10": 45,
}
STEP_US, LANE_FACTOR = 55.0, {500: 1.5, 2000: 3.0}
RECLUSTER_S, COMPILE_S = {500: 0.02, 2000: 0.06}, 60.0


def walks(d):
    return max(25, 6 * d, d * d // 6)


def cells():
    for target in MIXTURES + CONTROLS:
        for nlive in NLIVE:
            for arm in ARMS:
                yield target, arm, nlive, nlive // 10
    for target, arm in K1_CELLS:
        yield target, arm, 500, 1


def estimate_s(target, arm, nlive, k):
    d = int(target.rsplit("_d", 1)[1])
    steps = nlive * EFOLDS[target] / k
    chain = steps * walks(d) * STEP_US * 1e-6 * (LANE_FACTOR[nlive] if k > 1 else 1.0)
    every = max(1, int(nlive / (4 * k) + 0.5))
    clustering = steps / every * RECLUSTER_S[nlive]
    return COMPILE_S + chain + clustering


def line(target, arm, nlive, k):
    py = os.environ.get("BENCH_PY", "env/bench/bin/python")
    run = os.environ.get("BAKEOFF_RUN", "src/tinyns-bench/bench/bakeoff/run.py")
    out = os.environ.get("OUT", "runs/v1_bakeoff")
    tag = os.environ.get("TAG", "v1-bakeoff-h100")
    return (
        f"flock gpu.lock env XLA_PYTHON_CLIENT_PREALLOCATE=false {py} {run} "
        f"--target {target} --arm {arm} --nlive {nlive} --k {k} --seeds {SEEDS} "
        f"--out {out}/results.jsonl --tag {tag}"
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--estimate", action="store_true", help="print the estimate only")
    args = p.parse_args(argv)
    grid = list(cells())
    total = sum(estimate_s(*c) for c in grid)
    if not args.estimate:
        for c in grid:
            print(line(*c))
    by_arm = {a: sum(estimate_s(*c) for c in grid if c[1] == a) for a in ARMS}
    summary = ", ".join(f"{a} {s / 3600:.1f} h" for a, s in by_arm.items())
    print(
        f"# {len(grid)} cells, estimated H100 time {total / 3600:.1f} h ({summary})",
        file=sys.stdout if args.estimate else sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
