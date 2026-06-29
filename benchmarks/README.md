# Benchmarks

## Performance benchmarks

A lightweight benchmark harness is available:

```bash
python benchmarks/bench_static.py \
  --targets gaussian2d correlated_gaussian2d \
  --samplers rwalk slice rslice \
  --seeds 0 1 2 \
  --nlive 200 \
  --dlogz 0.1 \
  --output bench.json
```

For the recommended JAX random-walk fast path, run:

```bash
python benchmarks/bench_static.py \
  --targets gaussian2d correlated_gaussian2d \
  --samplers rwalk \
  --kernel jax \
  --seeds 0 1 2 \
  --nlive 200 \
  --dlogz 0.1 \
  --output bench_rwalk_jax.json
```

The benchmark reports wall time, iterations/sec, likelihood calls/sec, replacement cost, and basic diagnostics. It is intended to guide optimization, not to replace validation.

### Benchmarking JAX rwalk replacement chains

For JAX kernels, first-run timings may include compilation. Use a warmup run when comparing chain counts:

```bash
python benchmarks/bench_static.py \
  --targets gaussian2d correlated_gaussian2d \
  --samplers rwalk \
  --kernel jax \
  --replacement-chains-grid 1 4 16 64 256 1024 \
  --seeds 0 1 2 \
  --nlive 200 \
  --dlogz 0.1 \
  --max-attempts 102400 \
  --warmup-runs 1 \
  --discard-warmup \
  --output bench_rwalk_jax_chain_sweep.json
```

When sweeping large `replacement_chains`, make sure `--max-attempts` is at least `walks * max(replacement_chains)`. For example, with `walks=25` and `replacement_chains=1024`, use at least `--max-attempts 25600`. For benchmarking, `--max-attempts 102400` allows up to four replacement batches at 1024 chains. Alternatively, `--auto-max-attempts` is a benchmark convenience that raises `--max-attempts` only when needed for JAX `rwalk` runs.

For batched chains, scalar `ncall` is not a wall-clock cost proxy. Prefer wall time, iterations/sec, and replacement batch counts.

### Replacement-kernel microbenchmark

The full static benchmark can be dominated by the Python nested-sampling loop on cheap targets. To measure the JAX replacement kernel itself, use:

```bash
python benchmarks/bench_rwalk_kernel.py \
  --targets gaussian2d correlated_gaussian2d \
  --replacement-chains-grid 1 4 16 64 256 1024 \
  --walks 25 \
  --nlive 200 \
  --n-replacements 1000 \
  --warmup-replacements 100 \
  --max-attempts 102400 \
  --output bench_rwalk_kernel.json
```

### Heavy synthetic likelihood benchmark

Cheap Gaussian targets are useful for correctness and overhead checks, but they do not represent heavy catalog/injection likelihoods. To test GPU batching behavior, use:

```bash
python benchmarks/bench_rwalk_kernel.py \
  --targets heavy_gaussian2d \
  --replacement-chains-grid 1 4 16 64 256 1024 2048 \
  --walks 25 \
  --nlive 200 \
  --n-replacements 500 \
  --warmup-replacements 50 \
  --work-size 100000 \
  --max-attempts 204800 \
  --output bench_heavy_rwalk_kernel_chain_sweep.json
```

Increase `--work-size` to mimic more expensive likelihoods.

Use this benchmark to understand GPU throughput scaling. Use the full `bench_static.py` benchmark to understand end-to-end nested-sampling wall time.


### Overnight JAX validation wrapper

For opt-in local or overnight validation runs, use the shell wrapper:

```bash
benchmarks/run_overnight_jax_validation.sh
```

By default, the wrapper writes timestamped JSON results under `benchmarks/results/` and then prints a summary table. Override `NLIVE`, `DLOGZ`, `SEEDS`, `TARGETS`, `MAXITER`, or `OUTPUT` in the environment to customize a run. This script is opt-in and intended for local/overnight runs. It is not part of CI.

To reproduce the recommended fast unbounded JAX rwalk validation with the
current fastest validated block size, run:

```bash
python benchmarks/overnight_jax_validation.py \
  --targets gaussian2d correlated_gaussian2d ring2d banana2d eggbox2d \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --nlive 500 \
  --dlogz 0.1 \
  --maxiter 10000 \
  --include-block \
  --jax-block-size 32 \
  --output overnight_jax_validation_block_B32.json
```

### External expensive-likelihood benchmarks

Some users may want to benchmark `tinyns` on external, user-provided expensive JAX likelihoods, such as catalog or dark-siren likelihoods. Keep those benchmarks outside the core package: do not add the external likelihood package as a `tinyns` dependency, do not add domain-specific code to `tinyns`, and do not put overnight benchmark runs in CI.

When benchmarking the optimized path, hold the sampling problem fixed across runs:

- use a fixed random seed, or a documented fixed seed list;
- use the same `nlive`;
- use the same `dlogz` stopping threshold;
- use exactly the same data files, injections, masks, and likelihood settings;
- set the progress interval high enough that terminal output is not a material part of the timing;
- compare wall time, scalar `ncall`, `logZ`, and replacement metadata such as replacement batches, per-replacement calls, chain usage, and success/failure counts.

Recommended starting configurations for external expensive JAX likelihoods are:

Unbounded 2D baseline:

```bash
--sample rwalk \
--kernel jax \
--walks 1 \
--replacement-chains 16
```

10D baseline:

```bash
--sample rwalk \
--kernel jax \
--walks 5 \
--replacement-chains 16 \
--rwalk-proposal live-cov
```

Bounded 10D candidate:

```bash
--sample rwalk \
--kernel jax \
--bound multi \
--rwalk-seed bound \
--rwalk-proposal live-cov \
--walks 5 \
--replacement-chains 16 \
--bound-update-interval 25
```

Fast JAX candidate once fused or block modes are available:

```bash
--sample rwalk \
--kernel jax \
--bound multi \
--rwalk-seed bound \
--rwalk-proposal live-cov \
--fused-bound-rwalk \
--jax-block-size 10
```

Do not compare wall time between runs that print progress every iteration; progress output can dominate timings for otherwise fast runs. When comparing against dynesty, use the same likelihood, the same seed family, the same `nlive`, and the same stopping threshold. For tiny or cheap likelihoods, Python dispatch and JAX launch overhead can dominate, so fewer scalar likelihood calls do not necessarily imply faster wall time. For expensive JAX likelihoods, prefer batched or vectorized candidate evaluation when it is available, and judge performance primarily with wall time together with replacement metadata rather than scalar `ncall` alone.
