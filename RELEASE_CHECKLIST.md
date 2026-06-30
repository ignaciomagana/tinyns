# tinyns release checklist

## Pre-release checks

- [ ] `ruff check .`
- [ ] `pytest`
- [ ] Run core validation:

```bash
python validation/run_validation.py \
  --targets gaussian2d correlated_gaussian2d ring2d banana2d eggbox2d \
  --samplers rwalk \
  --kernel jax \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --nlive 200 \
  --dlogz 0.1 \
  --output validation_release_rwalk_jax.json

python validation/summarize_validation.py validation_release_rwalk_jax.json
```

- [ ] Run a CPU/Python baseline validation:

```bash
python validation/run_validation.py \
  --targets gaussian2d correlated_gaussian2d \
  --samplers rwalk \
  --kernel python \
  --seeds 0 1 2 3 4 \
  --nlive 200 \
  --dlogz 0.1 \
  --output validation_release_rwalk_python.json

python validation/summarize_validation.py validation_release_rwalk_python.json
```

- [ ] Run benchmark smoke:

```bash
python benchmarks/bench_static.py \
  --targets gaussian2d correlated_gaussian2d \
  --samplers rwalk \
  --kernel jax \
  --seeds 0 1 2 \
  --nlive 200 \
  --dlogz 0.1 \
  --output bench_release_rwalk_jax.json
```

- [ ] Run examples:

  - [ ] `python examples/gaussian_2d.py`
  - [ ] `python examples/gaussian_2d_rwalk.py`
  - [ ] `python examples/gaussian_2d_rwalk_jax.py`
  - [ ] `python examples/gaussian_2d_rwalk_jax_block.py`
  - [ ] checkpoint/resume example, if present
  - [ ] progress/callback example, if present


## Repeatable release validation

Use the Makefile shortcuts for the routine release path so sampler changes can be checked without remembering the long benchmark commands. The primary release gate is `make overnight-b32`. The B16, B64/B128, and no-block/bounds comparison runs are optional diagnostics. Failures isolated to experimental live-cov/bounded/fused-bounded paths should be tracked, but they do not block the core B32 release path unless they reveal shared infrastructure breakage:

1. [ ] Run `make test`.
2. [ ] Run `make quick-validation`.
3. [ ] For release validation, run `make overnight-b32`.
4. [ ] Optional comparison: run `make overnight-b16` and `make overnight-comparison`; B64/B128 sweeps are optional performance diagnostics for cheap likelihoods or external target-specific benchmarking.
5. [ ] Run `make summarize-overnight`.
6. [ ] Confirm B32 has 100% success, zero replacement failures, and sane analytic pulls.
7. [ ] Confirm experimental failures do not affect the core release path.

The overnight Makefile targets are opt-in local validation commands and must not be added to CI. Generated JSON outputs are local artifacts and should not be committed.

The primary release gate remains B32. B64/B128 sweeps are optional performance diagnostics for cheap likelihoods or external target-specific benchmarking. They should not replace the B32 gate unless future validation shows a clear robustness and efficiency advantage.

## Documentation checks

- [ ] README quickstart works
- [ ] sampler recommendation table is up to date
- [ ] validation README command works
- [ ] benchmark command works
- [ ] limitations are explicit

## Current known limitations

- static nested sampling only
- no dynamic nested sampling
- ellipsoidal bounding is experimental and not part of the recommended fast path
- no full-Python-free compiled nested-sampling loop
- `kernel="jax"` currently supports only `sample="rwalk"`
- not a probabilistic programming framework

- Recommended path release gate:
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
  - [ ] Confirm 50/50 success.
  - [ ] Confirm zero replacement failures.
  - [ ] Confirm analytic RMS pull is sane, roughly near 1.
  - [ ] Confirm B32 remains faster than no-block isotropic.
  - [ ] Confirm any bounded/fused bounded failures are treated as experimental-path failures and do not block the core release unless they indicate shared infrastructure breakage.
