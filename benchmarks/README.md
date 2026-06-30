# Benchmarks

## Performance benchmarks

A lightweight benchmark harness is available:

```bash
python benchmarks/bench_static.py \
  --targets gaussian2d correlated_gaussian2d \
  --samplers prior rwalk \
  --seeds 0 1 2 \
  --nlive 200 \
  --dlogz 0.1 \
  --output bench.json
```

For the recommended JAX random-walk fast path on the included benchmark targets, run:

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

### Cached JAX block rwalk validation recipes

These recipes are documentation-only validation workflows for the unbounded JAX `rwalk` path. They do not change sampler defaults and should be run manually, not in CI. The currently validated fast path on the included benchmark targets is:

```text
sample="rwalk"
kernel="jax"
rwalk_proposal="isotropic"
walks=5
replacement_chains=1
jax_block_size=32
```

Recent validation found `jax_block_size=32` fastest overall among the validated unbounded cached JAX block runs, with `jax_block_size=16` slightly more conservative. Ordinary no-block isotropic `rwalk` was much slower in that validation. Live-cov and bounded/fused bounded paths remain experimental and should not be promoted from these results.


#### Post-cleanup overnight validation

After removing slice/rslice and the legacy `sample="bound"` mode, the recommended unbounded cached-block JAX rwalk path was rerun on the included validation targets. The B32 path remained fully successful across 50 runs, with zero replacement failures and evidence diagnostics consistent with the no-block isotropic baseline. Timing means exclude first-run compile/warmup outliers. Success and replacement-failure counts include all runs.

| Config | Success | Replacement failures | Mean sec | Mean ncall | Analytic RMS pull | Max abs pull | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B32 cached block | 50/50 | 0 | ~4.14 | ~35,501 | ~1.23 | ~2.56 | Recommended fast path |
| B16 cached block | 50/50 | 0 | ~4.42 | ~35,023 | ~1.23 | ~2.56 | Conservative fallback |
| No-block isotropic | 50/50 | 0 | ~20.66 | ~34,636 | ~1.23 | ~2.56 | Clean but slower |
| Live-cov | 50/50 | 0 | ~19.82 | ~15,582 | ~3.27 | ~7.97 | Experimental; do not promote |
| Adaptive rwalk | 50/50 | 0 | ~24.28 | ~40,187 | ~1.35 | ~2.24 | Experimental |
| Single bound | 50/50 | 0 | ~72.33 | ~392,211 | ~1.36 | ~2.90 | Experimental; slow |
| Multi bound | 50/50 | 0 | ~83.62 | ~392,554 | ~1.37 | ~2.97 | Experimental; slow |
| Fused bounded | 49/50 | 1 | ~95.72 | ~392,229 | ~1.37 | ~2.97 | Experimental; one known failure |

B32 was about 5x faster than the no-block isotropic JAX rwalk baseline while preserving the same analytic evidence behavior. B32 was also about 6–7% faster than B16, at the cost of only about 1.4% more scalar likelihood calls.

| Target | B32 speedup over no-block isotropic |
| --- | ---: |
| gaussian2d | ~5.78x |
| correlated_gaussian2d | ~5.71x |
| ring2d | ~4.63x |
| banana2d | ~5.05x |
| eggbox2d | ~3.82x |

These results support keeping `jax_block_size=32` as the recommended fast path for unbounded JAX rwalk on the included benchmark targets. `jax_block_size=16` remains a conservative fallback. `jax_block_size=1` disables block mode.

These results do not promote live-cov, bounds, fused bounds, or bounded block mode. Live-cov had concerning analytic pull behavior, and fused bounded still had one eggbox replacement failure. Experimental settings still require target-specific validation.

#### Recommended fast-path validation

To reproduce the recommended fast unbounded JAX `rwalk` validation with the current fastest validated block size, run:

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

`jax_block_size=32` is the currently recommended validated fast setting for unbounded JAX `rwalk` on the included benchmark targets, not a universal optimum. `jax_block_size=16` is a conservative alternative. `jax_block_size=1` disables block mode.

#### Block-size sweep

Use this copy-paste sweep to compare convergence and timing as the cached JAX block size changes:

```bash
for B in 2 4 8 16 32; do
  python benchmarks/overnight_jax_validation.py \
    --targets gaussian2d correlated_gaussian2d ring2d banana2d eggbox2d \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --nlive 500 \
    --dlogz 0.1 \
    --maxiter 10000 \
    --include-block \
    --jax-block-size "$B" \
    --output "overnight_jax_validation_block_B${B}.json"
done
```

This validates the success/failure rate, replacement failures, wall time, `ncall`/`niter` growth from block overshoot, logZ accuracy on analytic targets, and `final_delta_logz` overshoot as block size increases. Larger block sizes can reduce wall time, but they may increase `ncall`/`niter` because convergence is checked between blocks.

#### No-block and bounded comparison / experimental candidates

Use this comparison command when comparing the recommended B32 block mode against no-block and bounded configurations. This is not the primary validation command:

```bash
python benchmarks/overnight_jax_validation.py \
  --targets gaussian2d correlated_gaussian2d ring2d banana2d eggbox2d \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --nlive 500 \
  --dlogz 0.1 \
  --maxiter 10000 \
  --include-bounds \
  --output overnight_jax_validation_no_block.json
```

This gives comparison against ordinary unbounded isotropic `rwalk`, unbounded live-cov `rwalk`, adaptive `rwalk`, and bounded single/multi/fused paths. Treat no-block as a comparison baseline, and treat live-cov and bounded/fused bounded results as experimental unless they are separately validated for the intended workload.

#### How to read the results

Prefer configurations with 100% success and zero replacement failures. For analytic targets, check `pull = (logz - expected_logz) / logzerr`: RMS pull around 1 is good, while large `max_abs_pull` or RMS pull greater than 2 means the configuration needs investigation. Lower wall time is only useful if logZ behavior remains sane.

When comparing block sizes, inspect `seconds`, `ncall`, `niter`, `final_delta_logz`, success counts, replacement failures, RMS pull, and maximum absolute pull together. B32 was fastest in the recent validation with only mild extra `ncall`; B16 is safer if you want less convergence overshoot.

#### Summarizing overnight validation files

The overnight summarizer supports multiple input files. After running the no-block baseline and selected block-size validations, summarize them with:

```bash
python benchmarks/summarize_overnight_jax_validation.py \
  overnight_jax_validation_no_block.json \
  overnight_jax_validation_block_B16.json \
  overnight_jax_validation_block_B32.json
```

### Expensive-likelihood validation guidance

Block mode helps most when Python/JAX dispatch overhead is a meaningful part of runtime. If the likelihood is extremely expensive, the speedup may shrink because likelihood evaluation dominates. The `bench_rwalk_kernel.py` heavy synthetic likelihood benchmark above is the best in-repository tool for testing GPU batching and artificial likelihood cost; increase `--work-size` there to mimic more expensive likelihoods.

For a quick end-to-end block-mode smoke run, use:

```bash
python benchmarks/overnight_jax_validation.py \
  --targets gaussian2d correlated_gaussian2d ring2d banana2d eggbox2d \
  --seeds 0 1 2 \
  --nlive 200 \
  --dlogz 0.5 \
  --maxiter 3000 \
  --include-block \
  --jax-block-size 32 \
  --output expensive_like_block_smoke.json
```

This is only a smoke recipe unless the benchmark targets include artificial likelihood cost; it is not a true expensive-likelihood benchmark by itself.

### External expensive-likelihood benchmarks

Some users may want to benchmark `tinyns` on external, user-provided expensive JAX likelihoods, such as catalog or dark-siren likelihoods. Keep those benchmarks outside the core package: do not add the external likelihood package as a `tinyns` dependency, do not add domain-specific code to `tinyns`, and do not put overnight benchmark runs in CI.

When benchmarking the optimized path, hold the sampling problem fixed across runs:

- use a fixed random seed, or a documented fixed seed list;
- use the same `nlive`;
- use the same `dlogz` stopping threshold;
- use exactly the same data files, injections, masks, and likelihood settings;
- set the progress interval high enough that terminal output is not a material part of the timing;
- compare wall time, scalar `ncall`, `logZ`, and replacement metadata such as replacement batches, per-replacement calls, chain usage, and success/failure counts.

Starting configurations for external expensive JAX likelihoods should distinguish the recommended unbounded isotropic path from experimental candidates:

- Start with the recommended unbounded cached block mode below.
- Test live-cov, bounds, fused bounds, or bounded block mode only as experimental candidates after the primary path is calibrated.
- Compare evidence calibration and replacement failures before treating wall-time speedups as meaningful.

Recommended unbounded isotropic cached block baseline to validate first:

```bash
--sample rwalk \
--kernel jax \
--walks 5 \
--replacement-chains 1 \
--rwalk-proposal isotropic \
--jax-block-size 32
```

Experimental 10D live-cov candidate for separate validation:

```bash
--sample rwalk \
--kernel jax \
--walks 5 \
--replacement-chains 16 \
--rwalk-proposal live-cov
```

Experimental bounded 10D candidate for separate validation:

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

Experimental bounded/fused candidate for separate validation (do not treat this as production-ready from the overnight block results alone):

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
