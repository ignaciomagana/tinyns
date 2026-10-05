# Benchmarks

All scripts here run the default sampler: the live-cov random walk in jitted
blocks of `block_size` iterations (default 32). They expose only the sampler's
own knobs (`--walks`, `--replacement-chains`, `--block-size`).

## Performance benchmarks

A lightweight benchmark harness is available:

```bash
python benchmarks/bench_static.py \
  --targets gaussian2d correlated_gaussian2d \
  --seeds 0 1 2 \
  --nlive 200 \
  --dlogz 0.1 \
  --output bench.json
```

The benchmark reports wall time, compile time (`compile_s`, the time to the
end of the first block), iterations/sec, likelihood calls/sec, replacement
cost, and basic diagnostics. It is intended to guide optimization, not to
replace validation.

### Benchmarking replacement chains

The first run of a configuration includes compilation. Use a warmup run when
comparing chain counts:

```bash
python benchmarks/bench_static.py \
  --targets gaussian2d correlated_gaussian2d \
  --replacement-chains-grid 1 4 16 64 \
  --seeds 0 1 2 \
  --nlive 200 \
  --dlogz 0.1 \
  --warmup-runs 1 \
  --discard-warmup \
  --output bench_chain_sweep.json
```

For batched chains, scalar `ncall` is not a wall-clock cost proxy. Prefer wall
time, iterations/sec, and replacement batch counts. With one chain, proposals
that leave the unit cube do not call the likelihood, so `ncall` is below
`walks` per replacement; with several chains every proposal is evaluated.

### Overnight validation wrapper

For opt-in local or overnight validation runs, use the shell wrapper:

```bash
benchmarks/run_overnight_jax_validation.sh
```

By default, the wrapper writes timestamped JSON results under
`benchmarks/results/` and then prints a summary table. Override `NLIVE`,
`DLOGZ`, `SEEDS`, `TARGETS`, `MAXITER`, or `OUTPUT` in the environment to
customize a run. This script is opt-in and intended for local/overnight runs.
It is not part of CI.

### Block-size validation recipes

These recipes are documentation-only validation workflows. They should be run
manually, not in CI. Makefile shortcuts are available for the common
workflows: `make quick-validation`, `make overnight-b32`, `make overnight-b16`,
`make overnight-comparison` (block size 1), and `make summarize-overnight`.

`overnight_jax_validation.py` runs one configuration per `--block-sizes`
entry, named `live_cov_B<size>`:

```bash
python benchmarks/overnight_jax_validation.py \
  --targets gaussian2d correlated_gaussian2d ring2d banana2d eggbox2d \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --nlive 500 \
  --dlogz 0.1 \
  --maxiter 10000 \
  --block-sizes 1 16 32 64 \
  --output overnight_jax_validation_sweep.json

python benchmarks/summarize_overnight_jax_validation.py \
  overnight_jax_validation_sweep.json
```

The summary reports the success rate, replacement failures, wall time,
`ncall`/`niter` growth from block overshoot, logZ accuracy on analytic
targets, and `final_delta_logz`. Speedups are relative to `live_cov_B1`, when
present. Convergence is checked between blocks, so larger blocks may increase
`ncall`/`niter` slightly while reducing dispatch overhead.

`block_size=32` is the default. Block size changes only where convergence is
checked and how often the step scale adapts (once per block), not the
replacement kernel.

#### How to read the results

Prefer configurations with 100% success and zero replacement failures. For
analytic targets, check `pull = (logz - expected_logz) / logzerr`: RMS pull
around 1 is good, while large `max_abs_pull` or RMS pull greater than 2 means
the configuration needs investigation. Lower wall time is only useful if logZ
behavior remains sane.

### Expensive-likelihood validation guidance

Block mode helps most when dispatch overhead is a meaningful part of runtime.
If the likelihood is extremely expensive, the speedup may shrink because
likelihood evaluation dominates. The `heavy_gaussian2d` validation target adds
artificial likelihood cost:

```bash
python benchmarks/bench_static.py \
  --targets heavy_gaussian2d \
  --replacement-chains-grid 1 4 16 \
  --seeds 0 \
  --nlive 200 \
  --dlogz 0.5 \
  --warmup-runs 1 \
  --discard-warmup \
  --output bench_heavy_chain_sweep.json
```

### External expensive-likelihood benchmarking

Use `benchmarks/templates/external_expensive_likelihood_template.py` as a
minimal starting point for benchmarking `tinyns` on external, user-provided
expensive JAX likelihoods. The template intentionally depends only on JAX and
TinyNS and shows where to plug in a user likelihood and prior transform.

Do not add domain data or external scientific packages to TinyNS itself. Keep
domain-specific likelihoods in user repositories or external benchmark
scripts. Do not add external likelihood packages as TinyNS dependencies, do
not add domain-specific code here, and do not put overnight domain benchmark
runs in CI.

When benchmarking, keep the sampling problem fixed across runs:

- use the same likelihood, data files, masks, injections, likelihood settings, and priors;
- use the same seed list;
- use the same `nlive`;
- use the same `dlogz` stopping threshold;
- set the progress interval high enough that terminal output is not a material part of the timing;
- compare evidence calibration and replacement failures before treating wall-time speedups as meaningful;
- separate compile time (`compile_s`) from sampling time (`mean_ms_per_call`).

Start from the defaults, `NestedSampler(loglike, prior_transform, ndim)`. For
a campaign over many datasets, pass the data through a pytree callable
(`jax.tree_util.Partial(loglike_fn, data)`) so that datasets of the same shape
share one compiled kernel.

### 10D GW-like stress target

`benchmarks/templates/gw_like_10d_tinyns_b32_figures.py` is a self-contained
synthetic 10D GW-like stress target. It includes mass-ratio/chirp-mass
curvature, hard bounded `q` and `chi_p` priors, distance/inclination amplitude
degeneracy, a sky banana plus mirror mode, wrapped phase/polarization
structure, and spin/mass-ratio coupling.

The 10D GW-like template is intentionally harder than the included
low-dimensional validation targets. It should be treated as a
constrained-replacement mixing stress test,
not as a new default configuration.
It is not a production gravitational-wave parameter-estimation likelihood,
and mechanically clean behavior on this target should not be presented as
evidence that TinyNS is production-ready for arbitrary GW parameter
estimation. It is not part of the release gate.

Weak random-walk mixing can reach high posterior ESS and apparent convergence
while still showing badly biased insertion-rank diagnostics. Raising `walks` is the
remedy; compare against a run with several times the default:

```bash
python benchmarks/templates/gw_like_10d_tinyns_b32_figures.py \
  --seeds 0 1 2 \
  --nlive 2000 \
  --dlogz 0.11 \
  --maxiter 150000 \
  --walks 160 \
  --output-dir gw_like_10d_walks160_seeds012 \
  --progress
```

This command is intentionally expensive.

When reading 10D GW-like output, healthy signs include:

- `success=True`;
- `replacement_failures=0`;
- no warnings;
- `insertion_rank_mean_z` close to 0;
- `insertion_rank_std_ratio` close to 1;
- small `live_weight_fraction`;
- large posterior ESS;
- logZ stable across seeds/configurations.

Bad signs include:

- `insertion_rank_mean_z` several sigma from 0;
- large logZ shifts as `walks` changes;
- large `replacement_max_batches`;
- replacement failures;
- low posterior ESS;
- large `live_weight_fraction`.
