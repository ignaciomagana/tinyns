# Changelog

## v0.1.0-alpha

- Narrowed TinyNS public sampler surface to `sample="rwalk"` and `sample="prior"`.
- Removed legacy `slice`, `rslice`, and `sample="bound"` public sampler paths.
- Documented the recommended unbounded isotropic JAX rwalk fast path with `jax_block_size=32`.
- Added cached JAX block rwalk validation summaries and release checklist commands.
- Added optional B16/B32/B64/B128 block-size smoke documentation; B32 remains the default recommendation.
- Added a self-contained 10D GW-like stress-test template with diagnostic figures.
- Added or documented robust block-mode termination behavior for late replacement failures.
- Kept live-cov, bounds, fused bounds, and larger block sizes experimental.

## Release caveats

TinyNS v0.1.0-alpha is intended as a small static nested sampler with a validated low-dimensional JAX rwalk fast path. High-dimensional, strongly curved, or multimodal targets require target-specific validation. The included 10D GW-like benchmark is a stress test for constrained-replacement mixing, not a production GW parameter-estimation pipeline.
