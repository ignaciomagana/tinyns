# Competitor benchmark harness

Standard targets with a known evidence and known mode masses, and three tools
built on them. It is not part of the package.

- `validate.py` is the **release gate**: tinyns alone, PASS or FAIL.
- `run.py` runs tinyns or a competitor and records one JSONL line per run.
- `summarize.py` merges those records into the **head-to-head report**.

```
bench/
  targets.py           targets: JAX loglike, box prior, true logZ, oracle mode masses
  adapters/            one module per sampler: run(target, seed, cfg) -> dict
  validate.py          the release gate: tinyns on the standard targets -> PASS/FAIL
  run.py               CLI: one sampler x one target x seeds -> results.jsonl
  summarize.py         JSONL files -> head-to-head report (Markdown, CSV, JSON)
  h100_plan.sh         env build + sweep job lines for js2h100 (review before use)
  slurm_cpu.sh         the CPU sampler lines as a Slurm array (Hilda)
  requirements-*.txt   pins for the two envs
  tests/               pytest (collected by the repo's CI)
```

## Validation gate

Run it before a release, and after any change to the sampler:

```bash
python bench/validate.py                 # quick tier: about 6 min on 4 CPU cores
python bench/validate.py --tier full     # every case: a GPU, or a Slurm node
python bench/validate.py --no-x64        # the same in float32
```

It prints a PASS/FAIL table and exits with code 1 if a case fails. Each case
is `--seeds` independent runs of one target, run as one batch of keys.

| case | quick nlive | full nlive | walks |
|---|---|---|---|
| `gauss_d2`, `gauss_d10` | 500 | 1000 | default |
| `gauss_d32` | - | 1000 | default |
| `rosen_d4` | 500 | 1000 | 100 |
| `rosen_d10` | - | 1000 | 250 |
| `funnel_d10`, `loggamma_d10`, `eggbox_d2` | 500 | 1000 | default |
| `sepW_d10` | 2000 | 2000 | default |
| `sepW_d18` | - | 4000 | default |

- The quick tier runs 16 seeds, the full tier 64 at the default `nlive`.
- Rosenbrock runs at `walks = 25 * ndim`: the default is calibrated on
  Gaussians and is too short for a curved valley.
- The mixtures run at an `nlive` that gives the 6% minor mode more than
  `5 * ndim` live points, which is what it takes to resolve it.

A case passes when all of these hold (se is the standard error over seeds):

| criterion | rule |
|---|---|
| `bias` | \|mean logZ - truth\| < 3 se |
| `scatter` | sd(logZ) / mean(logzerr) in [0.75, 1.3] |
| `modes` | mixtures: logit bias of each minor mode weight < 3 se, its sd <= 0.4, no seed lost a mode |
| `flags` | every run converged, and `result.modes()` flags no mode `unresolved` |

- **The scatter ratio is noisy.** Measured on N seeds it has a relative error
  of `1 / sqrt(2 (N - 1))`: 18% at 16 seeds, 9% at 64. A case fails only when
  the ratio is outside the band by more than 3 of those; the table prints that
  interval. So the quick tier catches an error bar that is off by a factor of
  about 2, and the full tier one that is off by 40%.
- **Mode weights are oracle masses**: the target's exact responsibility
  averaged over the posterior samples, not `result.modes()`.
- **Expect a false FAIL now and then.** Each criterion is a 3-sigma test, and
  the full tier runs about 25 of them. Rerun a failed case with another key
  (`--cases NAME --seed 1`) and more seeds before believing it.
- `--json FILE` keeps every seed's logZ, logzerr and mode masses.
- `--cases`, `--seeds` and `--nlive` override the tier, for a closer look at
  one case.

## Targets

`python -m bench.targets` lists them; `python -m bench.targets --check`
recomputes the cached quadrature values.

| name | d | logZ truth | modes |
|---|---|---|---|
| `gauss_d{2,8,16,32,64}` | 2-64 | analytic, `-d log 20` | - |
| `rosen_d{2,10}` | 2, 10 | 1-D transfer-operator quadrature | - |
| `funnel_d10` | 10 | 1-D quadrature over v | - |
| `loggamma_d{2,10,30}` | 2-30 | analytic (CDFs), ~ -2.3e-5 | 4 x 0.25 |
| `eggbox_d2` | 2 | 2-D quadrature, 235.856 | not scored (18) |
| `sepW_d{4,10,18,32}` | 4-32 | analytic, 0 | 0.94 / 0.06 |
| `connW_d{4,10,18,32}` | 4-32 | analytic, 0 | 0.94 / 0.06 |
| `sepWtw_d{d}` | any | analytic, 0 | 0.94 / 0.06 (banana-twisted main mode) |
| `sepM_d{10,18,32}` | 10-32 | analytic, 0 | 0.94 / 0.06 |
| `mix3_d10` | 10 | analytic, 0 | 0.7 / 0.2 / 0.1 |

- **Gaussians**: covariance `Q diag(s^2) Q^T`, with `s` log-spaced over
  [0.1, 1] and `Q` a random rotation. The mean is in [-1, 1]^d and the prior
  box is [-10, 10]^d, so the truncated mass is below 1e-18.
- **Mixtures** are the multimodality-prototype targets (`mm_targets.build`),
  ported verbatim; at d = 4, 10 and 18 the likelihood and responsibility match
  the original exactly. They live in
  the unit cube, with the minor mode at weight 0.06:
  - `sep`: the minor mode is offset by 10 marginal sigmas, which is 5.8 sigma
    per offset dim at d >= 10.
  - `conn`: an offset of 4 sigmas, so the two modes touch.
  - `W`/`M`: the minor-mode volume ratio, 0.21 or 0.03.
  - d = 32 reuses the d = 18 shapes: the minor mode is narrowed by the same
    factor in 15 of the 32 dims, so its volume ratio (0.21 or 0.03) is the
    d = 18 one, and it is oriented unlike the main mode (in the main mode's
    whitened frame its variances span 0.02 to 33). The targets are well
    posed (the cube truncates a mass below 3e-15), but the minor mode holds
    6% of the live points through its posterior bulk: 30 (about `d`) at
    nlive 500 and 120 (4 `d`) at 2000, fewer than a covariance-adapted walk
    needs (about 5 `d`), so the d = 32 cells measure a sampler's
    small-population limit.
- **Mode masses**: for every sampler, the mass of mode k is the
  weighted mean of the oracle responsibility `w_k N_k / sum_j w_j N_j` over the
  sampler's samples. Its exact expectation is `w_k`.

## Samplers

| spec | device | notes |
|---|---|---|
| `tinyns_v02[:noswap]` | gpu | `release/0.x` = 2f67111; see below |
| `tinyns_v1[:k1]` | gpu | the installed v1 package; `k1` is `num_delete=1` |
| `blackjax_nss` | gpu | `blackjax.nss`, `num_delete = nlive/10`, `num_inner_steps = max(5, 2d)` |
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
  dynesty. The others keep their own stopping rules:
  - Nautilus stops at `f_live=0.01` and `n_eff=10000`.
  - JAXNS stops at `dlogZ = log1p(1e-3)`.
  - UltraNest keeps `dlogz=0.5`, because its `dlogz` is a target uncertainty:
    a value of 0.1 makes it add live points. It also keeps `frac_remain=0.01`.
- **Evidence error.** Nautilus reports no `logzerr`, so its `scat/err` is
  blank.
- **`ncall`** is the number of likelihood evaluations the sampler counts. The
  vmapped samplers (BlackJAX, JAXNS) count per-chain evaluations; lanes that
  sit idle inside a vmap are not counted.
- **Time.** `wall_s` covers the whole run, compilation included.
  `compile_s` is the compilation part when it can be separated (tinyns,
  BlackJAX).
- **Devices.** The CPU samplers call the jitted JAX likelihood on CPU
  (`JAX_PLATFORMS=cpu`). dynesty calls it one point at a time; UltraNest and
  Nautilus call it in batches through `vmap`.
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

The report has three parts:

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
2. **Accurate cells**: per sampler and family, how many cells are accurate.
3. **Cells**: one row per cell. The module docstring describes the columns.

The same rows go to `--csv` (flat) and `--json`.

**Accurate** is the report's meaning of "comparable accuracy". A cell is
accurate when:

- at least 90% of its runs finished;
- its logZ bias is within 3 se or within 0.1 nats (the `dlogz` the runs stop
  at);
- for a mixture, at most 10% of its seeds lost a mode, and the logit bias of
  the mode weight is within 3 se or 0.1.

The headline's ratios are geometric means over the targets where both samplers
are accurate. A sampler that is cheap where it is accurate, and accurate on
few targets, still ranks well there: read the rank next to the accurate-cells
table.

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
- **Timeouts.** A run stops at 1 h on the GPU and at 4 h on a CPU. `ncall` is
  the median of the finished runs only; `wall s` counts the timeouts too.
- **Seeds.** 20 per target (40 for the mixtures); the CPU samplers have 10 at
  d >= 32.

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

- the target truths: 2-D grid integrals, the cached quadratures, and the exact
  mode masses checked with direct draws from each mixture;
- every installed adapter, on `gauss_d2` with nlive 100;
- `run.py` and `summarize.py` end to end, including a timeout record;
- the report on synthetic records: merging, the accuracy flags, the headline
  ratios and ranks, a missing reference;
- the gate: its criteria on synthetic rows, and one run of its code path on
  tiny settings.

Adapters whose package is missing are skipped. CI has only tinyns installed,
so there it runs the tinyns adapter that matches the installed version.
