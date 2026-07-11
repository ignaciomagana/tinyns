# Changelog

## v0.1.0-alpha

- Narrowed TinyNS public sampler surface to `sample="rwalk"` and `sample="prior"`.
- Removed legacy `slice`, `rslice`, and `sample="bound"` public sampler paths.
- Documented the recommended unbounded isotropic JAX rwalk fast path with `jax_block_size=32`.
- Added cached JAX block rwalk validation summaries and release checklist commands.
- Added optional B16/B32/B64/B128 block-size smoke documentation; B32 remains the default recommendation.
- Added a self-contained 10D GW-like stress-test template with diagnostic figures.
- Added or documented robust block-mode termination behavior for late replacement failures.
- Added a `logzerr` diagnostics breakdown and additional rwalk/replacement telemetry (`repl_ncall`, `repl_chains`, rescue/usage stage counts) to results and metadata.
- Added an experimental `rwalk_adaptive_step_scale` JAX-only rwalk option that adapts the isotropic proposal scale from constrained-replacement acceptance telemetry (default off, `step_scale=0.1` unchanged); the acceptance telemetry it adapts from was fixed to reflect true rwalk move acceptance rather than an earlier proxy.
- **Removed** `rwalk_proposal="live-cov"` and `rwalk_cov_jitter`: validation found concerning evidence pulls, and per-axis unit-cube reflection is not symmetry-preserving for correlated proposals. `rwalk_proposal` stays and now accepts only `"isotropic"`; passing anything else raises a clear removal error.
- **Removed** the `replacement_chain_schedule` + `jax_block_size > 1` combination: the in-JIT adaptive kernel evaluated all `max_chains` likelihoods at every stage regardless of schedule position, saving no compute while understating `ncall`/`chains_used` telemetry. Adaptive replacement-chain schedules remain supported at `jax_block_size=1` (the default).
- Fixed block-mode `ncall` undercounting failed-offset and rescue-ladder likelihood calls on an unrescued replacement failure; these are now always counted, matching per-iteration accounting.
- Fixed checkpoint cadence so `checkpoint_interval` is honored in block mode: checkpoints are now written on elapsed nested-iteration deltas rather than an exact-modulo check, so a large `jax_block_size` no longer skips the first several periodic saves (e.g. block 32 / interval 100 previously did not checkpoint before iteration 800).
- Fixed resume labeling: a resumed run that had already converged now reports `success=True, "converged"` instead of an incorrect `"maxiter reached"` message, and a resumed run at or past `maxiter` without convergence is labeled `"maxiter reached"` rather than defaulting to `"converged"`; fixed in both block and per-iteration modes.
- Persisted `effective_step_scale` in checkpoints as an optional `.npz` field (no checkpoint format-version bump; older checkpoints without the field still load) so adaptive step-scale state survives resume; added `rwalk_adaptive_step_scale`/`rwalk_target_accept` config validation on resume.
- Added a warning (not an error) when `NestedSampler` receives unknown keyword arguments; unknown kwargs are still stored, preserving dynesty drop-in compatibility.
- Validated `maxiter >= 1`; a fresh run with `maxiter=0` previously fell through and was mislabeled `"converged"`.
- Deduplicated run-loop state/bound-rebuild helpers, bound-rwalk JAX wrappers, and tuple-based ellipsoid-bound primitives; behavior-preserving, no public API change.
- Kept bounds, fused bounds, and larger block sizes experimental.

## Release caveats

TinyNS v0.1.0-alpha is intended as a small static nested sampler with a validated low-dimensional JAX rwalk fast path. High-dimensional, strongly curved, or multimodal targets require target-specific validation. The included 10D GW-like benchmark is a stress test for constrained-replacement mixing, not a production GW parameter-estimation pipeline.
