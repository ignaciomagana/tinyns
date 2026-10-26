# Competitor benchmark harness

Standard targets with a known evidence and known mode masses, and three tools
built on them. It is not part of the package.

- `validate.py` is the **release gate**: tinyns alone, PASS or FAIL.
- `run.py` runs tinyns or a competitor and records one JSONL line per run.
- `summarize.py` merges those records into the **head-to-head report**.
- `RESULTS.md` is that report for v1, with the settings and how to read it.

```
bench/
  targets.py           targets: JAX loglike, box prior, true logZ, oracle mode masses
  adapters/            one module per sampler: run(target, seed, cfg) -> dict
  validate.py          the release gate: tinyns on the standard targets -> PASS/FAIL
  run.py               CLI: one sampler x one target x seeds -> results.jsonl
  summarize.py         JSONL files -> head-to-head report (Markdown, CSV, JSON)
  RESULTS.md           the measured head-to-head and the full gate, for v1
  h100_plan.sh         env build + sweep job lines for js2h100 (review before use)
  slurm_cpu.sh         the CPU sampler lines as a Slurm array (Hilda)
  requirements-*.txt   pins for the two envs
  tests/               pytest (collected by the repo's CI)
```

## Validation gate

Run it before a release, and after any change to the sampler:

```bash
python bench/validate.py                 # quick tier: about 7 min on 4 CPU cores
python bench/validate.py --tier full     # every case: a GPU, or a Slurm node
python bench/validate.py --no-x64        # the same in float32
```

It prints a PASS/FAIL table and exits with code 1 if a case fails. Each case
is independent runs of one target, one per seed, run as one batch of keys.

| case | quick nlive | full nlive | walks |
|---|---|---|---|
| `gauss_d2`, `gauss_d10` | 500 | 1000 | default |
| `gauss_d32` | - | 1000 | default |
| `rosen_d4` | 500 | 1000 | 100 |
| `rosen_d10` | - | 1000 | 250 |
| `funnel_d10`, `loggamma_d10`, `eggbox_d2` | 500 | 1000 | default |
| `ndtri_d6` | 500 | 1000 | 144 |
| `sepWbox_d4` | 1000 | 1000 | default |
| `sepW_d10` | 2000 | 2000 | default |
| `sepW_d18` | - | 4000 | default |

- The quick tier runs 16 seeds, and 64 for `gauss_d2`, its cheapest case
  (below); the full tier runs 64 at the default `nlive`.
- Rosenbrock runs at `walks = 25 * ndim`, where its logZ is unbiased, so the
  two cases test the sampler on a curved valley. The default walks (60 at
  d = 10) leave the 10-D Rosenbrock 0.05 +- 0.04 low at `nlive=1000` (the
  README, Choosing `walks`), close to the 3 se of 64 seeds: a case at the
  default would fail now and then for a bias the README already states. The
  default walks are gated on the other cases, none of them curved.
- `ndtri_d6` has a standard normal prior (`x = ndtri(u)`) and the data 3 prior
  sd off centre in all 6 coordinates, so the posterior sits in a corner of
  the unit cube, where the prior transform is far from affine. It runs at 4
  times the default walks. At the default (36) the chains do not decorrelate
  there, and the case fails `scatter` in both tiers and `bias` in the full
  tier: logZ 0.53 +- 0.18 low (z -2.9) with a scatter of 3.0 times `logzerr`
  in the quick tier (key 0, x64, on the release code), and 0.24 +- 0.08 low,
  3.6 times, in the full tier (x64). At 72 walks the scatter was still 1.7
  times `logzerr` (32 runs at `nlive=1000`); at 144 it is 1.05 at
  `nlive=1000` and 1.2 at 500.
- The mixtures run at an `nlive` that gives the 6% minor mode more than
  `5 * ndim` live points, which is what it takes to resolve it.
  `sepWbox_d4` is `sepW_d4` moved from the unit cube to the box [-3, 5]^4, so
  its mode masses are scored on samples in coordinates that are not the
  sampler's own.

A case passes when all of these hold (se is the standard error over seeds):

| criterion | rule |
|---|---|
| `bias` | \|mean logZ - truth\| < 3 se |
| `scatter` | sd(logZ) / mean(logzerr) in [0.75, 1.3], within its noise (below) |
| `posterior` | Gaussians: along each eigenvector of the true covariance, the mean offset of the posterior (in units of the true sd) and the log of its variance over the true one are zero within their scatter over the seeds (Student's t, Bonferroni over the `2 * ndim` tests) |
| `modes` | mixtures: logit bias of each minor mode weight < 3 se, its sd at most the case's limit, no seed lost a mode |
| `hop` | mixtures: the inter-mode hop was tried and accepted at least 2% of its proposals in at least 90% of the runs |
| `count` | `result.modes()` reports the true number of modes in at least 90% of the runs |
| `flags` | every run converged, and `result.modes()` flags no mode `unresolved` |

- **What the scatter criterion catches.** Measured on N seeds the ratio has a
  relative error of `1 / sqrt(2 (N - 1))`: 18% at 16 seeds, 9% at 64. A case
  fails only when the ratio is outside the band by more than 3 of those, so
  it passes measured ratios from 0.43 to 2.25 at 16 seeds and from 0.57 to
  1.70 at 64. As factors by which `logzerr` is off (too small, or too large):

  | seeds | caught in half the cases | caught in 95% |
  |---|---|---|
  | 16 (quick) | 2.3 too small, 2.25 too large | 3.2, 3.0 |
  | 64 (full) | 1.7, 1.73 | 2.0, 2.0 |

  A `logzerr` 40% too small (ratio 1.67) passes one full-tier case 61% of
  the time; if every case had it, the full tier would fail 99% of the time.
- **The posterior check** reads the Gaussian cases' samples, which no other
  criterion does. A run's offsets scatter by about 0.02 (mean, in units of
  the sd) and 0.03 (log variance) per direction at `nlive=500`, and 0.013 and
  0.022 at 1000. With bounds of 4.3 to 5.1 at 16 seeds and about 3.6 to 4.4
  at 64, a case fails an offset of about 0.02 sd in one direction's mean or
  3% in its variance in the quick tier, and 0.006 sd or 1% in the full tier.
  `gauss_d2` runs 64 seeds in both tiers. Measured by moving every run's
  offsets on 512 batches of keys 0 to 255 (float32 and x64), its check
  fails a mean 0.02 sd off along one eigenvector in every batch of 64 and in
  a quarter of the batches of 16, and a variance 3% too large in every
  batch of 64 and 43% of those of 16. Real runs agree (x64, keys 0 to 31):
  with the likelihood's mean moved by 0.02 sd the case failed 32 of 32
  batches of 64 and 7 of 32 of 16, and with its covariance 3% wider 32 and
  17 of 32.
  Samples in the wrong coordinates (the unit cube instead of the box) fail
  by a t of hundreds.
- **The limit on the mode-weight scatter** is 0.12 in logit for the `sepW`
  cases and 0.4 (the accuracy floor) for LogGamma. With the hop the sepW
  minor modes scatter by 0.02 to 0.07; with it off (`Config._hop=False`) by
  0.25 to 0.37, which the floor alone let through. The limit holds at the
  full tier's `nlive` and scales as `1 / sqrt(nlive)` when `--nlive`
  overrides it, up to 0.4. A hop that never runs fails `hop` too.
- **The hop limit** of 2% sits far below what the hop accepts: at least 0.20
  in LogGamma and 0.72 in the sepW cases, `sepW_d18` at `nlive=4000`
  included (416 runs on CPU, float32 and x64).
- **Mode weights are oracle masses**: the target's exact responsibility
  averaged over the posterior samples, not `result.modes()`.
- **The mode count** is that of `result.modes()`. The truth is 1 for the
  Gaussians, `rosen_d4`, the funnel and `ndtri_d6`, 4 for LogGamma and 2 for
  the mixtures. Two cases have a range, and may flag `unresolved`:
  - `rosen_d10` has a second lobe at `x_1 = -1` with 2.52% of the mass
    (`P(x_1 < 0)` by the quadrature that gives its evidence), joined to the
    main mode by a bridge of 0.8% (`|x_1| < 0.3`). One mode or two are both
    right. At nlive 1000 the lobe holds about 25 live points, fewer than
    `5 * ndim`, so a run that reports it flags it.
  - `eggbox_d2` has 18 modes that the sampler's clustering does not separate
    (a lattice has no two-way split). `modes()` counts the ones the live
    points still hold apart: 13 to 18 at nlive 1000 (192 runs), 4 to 18 at
    500 (960 runs). The case passes with 3 to 18 modes, and a run that
    reports fewer than 18 must say that its count is a lower bound (a mode
    that is not `tracked`, or the full-slots warning). Its 2% modes hold
    about 10 live points at nlive 500.
- **Expect a false FAIL now and then.** The `bias` and logit-bias tests are
  at 3 se with se estimated from the seeds, so a correct sampler fails each
  in 0.9% of the cases at 16 seeds and 0.4% at 64 (Student's t), and the
  `posterior` test in 0.27% (measured on `gauss_d2` and `gauss_d10`, float32
  and x64: `posterior` failed 1 of 144 batches of 16 seeds, `bias` 3). Over
  a tier that is a false FAIL in about 1 run in 8 of the quick tier and 1 in
  14 of the full one. Some cases fail more often: the calibration runs put
  single cases at up to 4% at 16 seeds. A false FAIL also stays with its
  key while the runs do not change: at 16 seeds the default key 0 failed
  the posterior check of `gauss_d2` in x64 on 1.0.0 and on 1.1.0 (t 5.9
  against 4.3), with the same runs. That check is calibrated: over keys 0
  to 255 at 16 seeds (float32 and x64) the per-seed offsets are normal and
  independent from seed to seed, and the case failed 3 of 512 batches on
  `posterior` and 8 on `bias`, against 1.4 and 4.6 expected; key 0 is a
  draw of about 1 in 9000. `gauss_d2` runs 64 seeds instead, which cost
  about 4 s: the 64 keys begin with the 16, and the other 48 runs bring key
  0 to t 2.0. At 64 seeds it failed 3 of the 512 batches. Rerun a failed
  case with another key and more seeds (`--cases NAME --seed 1 --seeds 128`)
  before believing it.
- **Pair A/B comparisons.** In float32 a seed's trajectory differs between
  CPU node types (on Hilda, HENON and SIRIUS nodes) as well as between CPU
  and GPU. Compare two versions or settings on the same seeds, run on one
  node type, and compare them seed by seed. An unpaired comparison once
  showed a fake trend: in the hop-eligibility study, 17 against 26 lost
  modes in 120 seeds, which vanished when the runs were paired.
- `--json FILE` keeps every seed's logZ, logzerr, mode masses, hop
  acceptance and posterior offsets. It also records the commit the gate ran
  at (`git_sha`, with `-dirty` when a tracked file under `src`, `bench` or
  `pyproject.toml` differs from it), which the header line prints too, and
  where `tinyns` was imported from.
- `--cases` runs the named cases whatever the tier, at the tier's `nlive` or,
  for a case outside the tier, the full tier's. `--nlive` overrides the
  `nlive` of the cases the tier (or `--cases`) runs, and warns when a
  mixture's smallest mode then holds fewer than `5 * ndim` live points (the
  case fails `flags`). `--seeds` sets the seeds of every case (the tier's,
  and the 64 of `gauss_d2`), at least 2.

## Targets

`python -m bench.targets` lists them; `python -m bench.targets --check`
recomputes the cached quadrature values.

| name | d | logZ truth | modes |
|---|---|---|---|
| `gauss_d{2,8,16,32,64}` | 2-64 | analytic, `-d log 20` | - |
| `ndtri_d{d}` | any | analytic | - (a normal prior, the data 3 prior sd off centre) |
| `rosen_d{2,10}` | 2, 10 | 1-D transfer-operator quadrature | - (d = 10: a 2.5% lobe at `x_1 = -1`) |
| `funnel_d10` | 10 | 1-D quadrature over v | - |
| `loggamma_d{2,10,30}` | 2-30 | analytic (CDFs), ~ -2.3e-5 | 4 x 0.25 |
| `eggbox_d2` | 2 | 2-D quadrature, 235.856 | not scored (18) |
| `sepW_d{4,10,18,32}` | 4-32 | analytic, 0 | 0.94 / 0.06 |
| `connW_d{4,10,18,32}` | 4-32 | analytic, 0 | 0.94 / 0.06 |
| `sepWtw_d{4,10,18,32}` | 4-32 | analytic, 0 | 0.94 / 0.06 (banana-twisted main mode) |
| `sepWbox_d{4,10,18,32}` | 4-32 | analytic, 0 | 0.94 / 0.06 (in the box [-3, 5]^d) |
| `sepM_d{10,18,32}` | 10-32 | analytic, 0 | 0.94 / 0.06 |
| `mix3_d10` | 10 | analytic, 0 | 0.7 / 0.2 / 0.1 |
| `needle_d{4,10,18,32}` | 4-32 | analytic, 0 | 0.8 / 0.2 (each mode a core with two thin arms) |
| `cauchy_d{d}` | >= 7 | analytic (CDFs, box included) | 0.797 / 0.203 (products of Cauchy and normal factors) |

- **Gaussians**: covariance `Q diag(s^2) Q^T`, with `s` log-spaced over
  [0.1, 1] and `Q` a random rotation. The mean is in [-1, 1]^d and the prior
  box is [-10, 10]^d, so the truncated mass is below 1e-18 and the posterior
  is the Gaussian (its mean and covariance are in `meta`).
- **`ndtri`**: a standard normal prior with `lnL = sum_i log N(x_i; 3, 0.3^2)`,
  so the posterior, `N(2.75, 0.29^2)` in each coordinate, lies in a corner of
  the unit cube (`u` from 0.985 to 0.9996). The target is written in the unit
  cube: its box is [0, 1]^d and its likelihood applies `ndtri`, so every
  sampler sees a box prior, and its samples are `u`.
- **Mixtures** are the multimodality-prototype targets (`mm_targets.build`),
  ported verbatim; at d = 4, 10 and 18 the likelihood and responsibility match
  the original exactly. They live in
  the unit cube, with the minor mode at weight 0.06:
  - `sep`: the minor mode is offset by 10 marginal sigmas, which is 5.8 sigma
    per offset dim at d >= 10.
  - `conn`: an offset of 4 sigmas, so the two modes touch.
  - `W`/`M`: the minor-mode volume ratio, 0.21 or 0.03.
  - `box`: the target in the box [-3, 5]^d (`BOX`), with the likelihood of
    the unit-cube target at `u = (x + 3) / 8`, so logZ and the mode masses
    do not change.
  - The minor-mode shapes are set for d = 4, 10, 18 and 32 only; another d
    raises a ValueError.
  - d = 32 reuses the d = 18 shapes: the minor mode is narrowed by the same
    factor in 15 of the 32 dims, so its volume ratio (0.21 or 0.03) is the
    d = 18 one, and it is oriented unlike the main mode (in the main mode's
    whitened frame its variances span 0.02 to 33). The targets are well
    posed (the cube truncates a mass below 3e-15), but the minor mode holds
    6% of the live points through its posterior bulk: 30 (about `d`) at
    nlive 500 and 120 (4 `d`) at 2000, fewer than a covariance-adapted walk
    needs (about 5 `d`), so the d = 32 cells measure a sampler's
    small-population limit.
- **Modes that are not convex** (`needle`, `cauchy`; not in the standard
  sweep): the contours of each mode have thin arms, so the clustering cuts
  the modes into lumps and the hop balances them only in part. They test
  what `result.modes()` and `diagnostics()` report then
  (`tests/test_modes.py`).
- **Mode masses**: for every sampler, the mass of mode k is the
  weighted mean of the oracle responsibility `w_k N_k / sum_j w_j N_j` over the
  sampler's samples. Its exact expectation is `w_k`.

## Samplers

| spec | device | notes |
|---|---|---|
| `tinyns_v02[:noswap]` | gpu | `release/0.x` = 2f67111; see below |
| `tinyns_v1[:k1]` | gpu | the installed v1 package; `k1` is `num_delete=1` |
| `blackjax_nss` | gpu | `blackjax.nss`, `num_delete = nlive/10` (the bench's choice; BlackJAX's default is 1), `num_inner_steps = max(5, 2d)` |
| `dynesty[:rwalk100]` | cpu | `bound='multi', sample='auto'`, or `sample='rwalk', walks=100` |
| `ultranest[:slice]` | cpu | MLFriends, or `SliceSampler(nsteps=2d, mixture directions)` |
| `nautilus[:discard]` | cpu | defaults (`n_live=2000`), or `discard_exploration=True`; `--opt pool=N` |
| `jaxns[:nlive]` | gpu | JAXNS 3 defaults, or `root_allocation_degree = nlive` (own env) |

Each adapter's docstring gives its exact settings. These differences matter
when you read the table:

- **tinyns 0.x** is the frozen `release/0.x` branch at 2f67111. That is the
  v0.3 series in progress (PRs 1-6 merged on main). It behaves like v0.2.5 with
  the code pruned, and its `__version__` still says 0.2.5.
- **Termination.** `--dlogz` (default 0.1) goes to tinyns, BlackJAX and
  dynesty; the BlackJAX adapter tests it after every step, inside its
  compiled loop. The others keep their own stopping rules:
  - Nautilus stops at `f_live=0.01` and `n_eff=10000`.
  - JAXNS stops at `dlogZ = log1p(1e-3)`, which is a remaining-evidence
    fraction of 1e-3 at the contour likelihood, not a log-evidence
    remainder. In low dimensions that is later than tinyns's rule: about
    1.5 times the calls at d = 2 (1.0 to 2.4 per seed), 1.2 to 1.3 at d = 4
    to 8 and under 10% more from d = 16, with the same logZ and logzerr.
  - UltraNest keeps `dlogz=0.5`, because its `dlogz` is a target uncertainty:
    a value of 0.1 makes it add live points. It also keeps `frac_remain=0.01`
    (1% of the evidence left in the live points, against about 9.5% for
    `dlogz = 0.1`), which costs about 1.25 times the calls on `gauss_d2`.
- **Evidence error.** Nautilus reports no `logzerr`, so its `scat/err` is
  blank.
- **`ncall`** is the number of likelihood evaluations the sampler counts. The
  vmapped samplers (BlackJAX, JAXNS) count per-chain evaluations; lanes that
  sit idle inside a vmap are not counted.
- **Time.** `wall_s` covers the whole run, compilation included.
  `compile_s` is the compilation part, every compile of the run, where it
  can be separated (tinyns and BlackJAX); `run s` = wall - compile is then a
  compiled run.
  - tinyns v1 runs a one-step warm-up of the same sampler first and counts
    it as `compile_s`: it compiles (or loads) the init and step kernels, the
    prior map and the small ops of the driver. `metadata["compile_s"]` alone
    covers only the step kernel.
  - BlackJAX compiles `init` and its loop ahead of time; its post-processing
    (`finalise`, `log_weights`, the prior map) runs op by op, and each op
    compiles on its first call for a run of a new length, which is in effect
    every run: most are too small for the persistent cache, and the cached
    ones serve only a run of the same length. The adapter runs the
    post-processing twice and counts the difference as compilation (4.1 to
    6.9 s a run on an H100 NVL, `RESULTS.md`).
  - tinyns 0.x's `compile_s` is what 0.x times itself: the time until its
    first block of iterations ends.
- **Devices.** The CPU samplers call the jitted JAX likelihood on CPU
  (`JAX_PLATFORMS=cpu`). dynesty calls it one point at a time; UltraNest and
  Nautilus call it in batches through `vmap`, padded to the next power of two
  so that the batch sizes (56 to 59 in an UltraNest MLFriends run) compile
  only a few programs. Their compiles are inside `wall_s`.
- **Precision.** x64 is on by default (`--no-x64` turns it off).

## Environments

There are two venvs, because jaxns 3 pins `tfp-nightly` and `jaxctx`.

```bash
python3 -m venv env/bench
env/bench/bin/pip install -r bench/requirements-bench.txt
env/bench/bin/pip install -e .              # this checkout: tinyns v1
git worktree add ../tinyns-0x origin/release/0.x   # tinyns 0.x source tree (2f67111)
export TINYNS_V02_SRC=$PWD/../tinyns-0x/src

python3 -m venv env/bench_jaxns
env/bench_jaxns/bin/pip install -r bench/requirements-bench_jaxns.txt
```

**Two tinyns versions, one env.** The repo and the installed package are v1,
so `import tinyns` gives v1. `tinyns_v02` needs the `release/0.x` source tree.
`TINYNS_V02_SRC` must point at that tree's `src` directory (the one that
contains `tinyns/`). The adapter puts it first on `sys.path` before it imports
`tinyns`. This works because every seed runs in its own process.

- Without `TINYNS_V02_SRC`, `tinyns_v02` reports itself unavailable: it
  writes no record and exits with code 2. It never runs v1 under the 0.x
  label.
- `tinyns_v1` ignores the variable and always imports the installed v1.
  Within a single Python process (pytest, for example), whichever adapter
  imports `tinyns` first decides which tree is loaded.
- `sampler_version` records the path tinyns was imported from, so a record
  shows which tree it used.

A `git archive origin/release/0.x src | tar -x -C <dir>` copy works as well as
a worktree.

For a CPU-only machine, use `jax` in place of `jax[cuda12]`. Run every script
from the repo root, or point at it by path: `run.py` puts the repo root on
`sys.path`.

## Running

```bash
env/bench/bin/python bench/run.py --sampler dynesty:rwalk100 --target sepW_d10 \
    --seeds 0-39 --nlive 500 --out results.jsonl --timeout 3600
env/bench_jaxns/bin/python bench/run.py --sampler jaxns --target gauss_d8 \
    --seeds 0-19 --out results.jsonl
env/bench/bin/python bench/summarize.py results.jsonl --out table.md
```

How `run.py` works:

- Each seed runs in a fresh spawned subprocess. The backend and x64 are set
  before JAX is imported, and the subprocess is killed at `--timeout`, which
  writes a `status: timeout` record.
- Several jobs can append to the same file, because each write takes `flock`.
- The flags:
  - `--opt key=value` passes a sampler option, as listed in the adapter
    docstrings.
  - `--device cpu|gpu` overrides the sampler's default device.
  - `--jax-cache DIR` turns on the persistent compilation cache. `compile_s`
    then measures cache loads after the first seed.
  - `--exclusive` marks the run as having had its cores to itself.
- An unavailable sampler (not installed, or the wrong tinyns version) exits
  with code 2 and writes nothing.

**Record schema** `tinyns-bench-1` has these keys:

- `schema, sampler, sampler_version, git_sha, target, ndim, seed`
  (`git_sha` ends in `-dirty` when a tracked file under `src`, `bench` or
  `pyproject.toml` differs from the commit; untracked files do not count)
- `truth {logz, logz_source, mode_mass}`
- `config {<resolved sampler settings>, variant, cli {nlive, dlogz, opts}}`
- `hw {host, gpu, platform, cpu_model, cpu_threads, exclusive[, slurm_job, slurm_partition]}, jax, x64`
- `status (ok/timeout/error), error, wall_s, compile_s, ncall, ncall_valid`
- `logz, logzerr, mode_mass, n_samples, ess (Kish), ts`

## Head-to-head report

```bash
python bench/summarize.py results_gpu.jsonl results_cpu.jsonl \
    --out report.md --csv cells.csv --json cells.json
```

It merges any number of JSONL files. A cell is one (sampler, target, nlive);
runs with `--opt` settings form their own sampler, labelled
`name [key=value]`, so a matched-settings rerun sits next to the default one.

- `--sha tinyns_v1=afa58dd` keeps only the `tinyns_v1` records written at
  that commit (`git_sha`) from an unmodified checkout, and says how many it
  left out; `--sha tinyns_v1=afa58dd-dirty` keeps those of a modified one.
  Use it when a results file holds runs of an older or a modified build.
- `--reference-nlive 500,1000` keeps the headline to the reference's cells at
  those `nlive`, when it also ran at values no competitor ran at.

The report has four parts:

1. **Headline**: for each target family, the reference sampler's rank, and
   each competitor's calls, wall time, logZ rms and mode-weight sd relative to
   it. `--reference` picks the sampler (default `tinyns_v1`); without records
   of it the report says so and carries on.
   - The samplers do not run at the same precision, so the table also gives
     the costs **at equal rms**: the cost ratio times the squared rms ratio,
     which takes each sampler's cost to go as `1 / rms^2` (as it does when
     nlive is raised). The rank uses these.
   - A reference cell pairs with the competitor's cell of the same target, of
     the nearest nlive when the competitor has several.
2. **Samplers**: version, commit, hardware, runs, timeouts and errors.
3. **Accurate cells**: per sampler and family, how many cells are accurate,
   with one row per `nlive` for a sampler that ran at several.
4. **Cells**: one row per cell. The module docstring describes the columns.
   `wall s` is the median over seeds; `run s` is wall minus compile, where
   the sampler times its compilation apart; `first s` is the cell's first
   run on a GPU, which is a cold start when the later seeds load their
   programs from `--jax-cache`.

The same rows go to `--csv` (flat) and `--json`.

**Accurate** is the report's meaning of "comparable accuracy". A cell is
accurate when:

- at least 90% of its runs finished;
- its logZ bias is within 3 se or within 0.1 nats (the `dlogz` the runs stop
  at);
- for a mixture, at most 10% of its seeds lost a mode, and the logit bias of
  the mode weight is within 3 se or 0.1.

The headline's ratios are geometric means over the reference cells where both
samplers are accurate, and `common / ran cells` counts reference cells, not
targets: with `--reference-nlive 500,1000` every target is two cells, each
paired with the competitor's one cell. A sampler that is cheap where it is
accurate, and accurate on few targets, still ranks well there: read the rank
next to the accurate-cells table.

**Read the costs with these in mind:**

- **Hardware.** Wall times compare only within one hardware label. The GPU
  samplers ran on one H100; the CPU samplers ran on 4 cores of a Xeon
  E5-2695 v3 each, and call the likelihood from Python. One such call costs
  35 us through the jitted JAX function, against 0.3 us per point in a batch.
- **Precision differs.** The samplers do not stop at the same accuracy, so
  compare `rms` before `ncall`:
  - Nautilus runs 2000 live points to an effective sample size of 10000.
  - JAXNS's default is `30 * ndim` live points: 60 at d = 2, 1920 at d = 64.
  - UltraNest stops at `dlogz = 0.5` and may add live points.
  - JAXNS and UltraNest stop later than `dlogz = 0.1` in low dimensions
    (see Termination above), which adds calls but not accuracy.
- **Timeouts.** A run stops at 1 h on the GPU and at 4 h on a CPU. `ncall` is
  the median of the finished runs only; `wall s` counts the timeouts too.
- **Seeds.** 20 per target (40 for the mixtures); the CPU samplers have 10 at
  d >= 32.
- **Records of the v1 sweep.** `RESULTS.md` was measured before the adapter
  fixes above (BlackJAX's stop and `compile_s`, tinyns's warm-up, the padded
  CPU batches). Its Known gaps say what each changes. The sweep's cells
  have not been measured again; BlackJAX NSS and tinyns v1 were timed again
  with the fixed adapters on 9 targets, on an H100 NVL (`RESULTS.md`,
  *BlackJAX NSS and tinyns re-timed*).

## Multimodality bake-off (`bench/bakeoff/`)

The v1 plan's bake-off of the inter-mode moves (see `tinyns/modes.py`). It
chose `B_ell`, now tinyns's always-on hop (report:
`/hildafs/projects/phy220048p/magana/darksirens-core-data/tinyns_h100_2026-09-30/v1_bakeoff/REPORT.md`);
the losing arms `B_t`, `C` and `BC` are deleted. The runner keeps two arms for
re-tests: `B_ell` (the default) and `N` (the private `Config._hop=False`:
random-walk steps only; the clustering still runs).

```bash
python bench/bakeoff/run.py --target sepW_d18 --arm B_ell --nlive 500 --k 50 \
    --seeds 0-39 --out results.jsonl      # one cell: 40 seeds, one batched run
python bench/bakeoff/emit.py > jobs.txt   # queue lines of the full grid (66 cells)
python bench/bakeoff/emit.py --estimate   # its H100 time estimate
python bench/bakeoff/summarize.py results.jsonl --out report.md
```

- `run.py` runs all seeds of a cell as one batched tinyns run (`core.run`
  with a batch of keys), so a cell is one GPU job, and writes one
  `tinyns-bakeoff-1` line per seed: logZ, logzerr, ncall, ncall_valid, the
  batch's wall and compile time, the oracle mode masses and their Kish
  effective sizes, the hop acceptance, the cluster history, the oracle
  isolation and the detection iteration, the oracle live count of each
  minor mode every quarter e-fold, and `result.modes()`.
- `summarize.py` applies the plan's decision rule: per-cell tables (logit sd
  with a bootstrap CI against the exact-draw floor, lost fraction, logit and
  Z/truth biases, logZ bias and scatter/logzerr, hop acceptance, detection
  lag, calls and wall time relative to `N`), the eliminations, the user gate
  (logit sd at most 0.4 in every resolvable cell) and the score.
- `emit.py` writes one `flock gpu.lock` queue line per cell, in the format of
  `h100_plan.sh` (paths relative to `$P` on js2h100).

## Tests

`pytest bench/tests` covers:

- the target truths: 2-D grid integrals, the cached quadratures, the `ndtri`
  evidence by quadrature, the exact mode masses checked with direct draws
  from each mixture, and the `box` variant against the unit-cube one;
- every installed adapter, on `gauss_d2` with nlive 100;
- the adapters' timing and stopping (`test_adapters.py`): every compile of a
  tinyns or BlackJAX run falls inside `compile_s`, BlackJAX stops at the
  first step that meets `dlogz`, and the CPU likelihood compiles once per
  power-of-two batch;
- `run.py` and `summarize.py` end to end, including a timeout record;
- the report on synthetic records: merging, the accuracy flags, the headline
  ratios and ranks, a missing reference, the commit filter (clean and
  modified builds apart), the cold-start and compiled times, cells of
  several `nlive`;
- the commit stamp of `run.py`: the `-dirty` flag of `git_sha`, in a scratch
  repository;
- the gate: its criteria on synthetic rows (the mode count and its
  lower-bound flag, the eggbox floor, the logit-sd limit and the hop among
  them), the posterior check on exact draws and its Student-t bounds, the
  tier, seeds and override rules, degenerate inputs, and one run of its code
  path on tiny settings, with the hop on and off.

Adapters whose package is missing are skipped. CI has only tinyns installed,
so there it runs the tinyns adapter that matches the installed version.
