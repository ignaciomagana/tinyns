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

Use this benchmark to understand GPU throughput scaling. Use the full `bench_static.py` benchmark to understand end-to-end nested-sampling wall time.
