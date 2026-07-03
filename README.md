# tinyns

`tinyns` is a tiny, dynesty-style nested sampler for JAX-friendly likelihoods.
The core public API is deliberately small: provide `loglike`,
`prior_transform`, and `ndim`, then call `NestedSampler(...).run(key)`.

TinyNS is not many samplers; it is one excellent tiny static nested sampler,
plus reference baselines. TinyNS deliberately keeps the sampler surface small.
The main optimized path is static nested sampling with JAX random-walk
replacement. Other samplers are kept as reference baselines or experimental
research knobs, not as equally supported production paths.

## Install

From source:

```bash
git clone <repo-url>
cd tinyns
python -m pip install .
```

Editable development install:

```bash
git clone <repo-url>
cd tinyns
python -m pip install -e '.[dev]'
```

## Minimal working example

```python
import jax
import jax.numpy as jnp
from tinyns import NestedSampler


def prior_transform(u):
    return -10.0 + 20.0 * u


def loglike(theta):
    return -0.5 * theta[0] ** 2 - 0.5 * jnp.log(2 * jnp.pi)


key = jax.random.PRNGKey(0)
sampler = NestedSampler(loglike, prior_transform, ndim=1, nlive=200)
result = sampler.run(key, dlogz=0.1)

print(result.summary())
```

## Minimal API

| API | Purpose |
| --- | --- |
| `NestedSampler(loglike, prior_transform, ndim, ...)` | Dynesty-style sampler facade for static nested sampling. |
| `NestedSamplingResult` | Result container with samples, weights, evidence estimates, status, and metadata. |
| `result.summary()` | Human-readable run summary. |
| `result.diagnostics()` | Plain-dict diagnostics, including ESS, call counts, and warnings. |
| `result.resample_equal(key, n=None)` | Equally weighted posterior samples via systematic resampling. |
| `result.to_numpy()` | Plain dictionary with array fields converted to NumPy arrays. |
| `result.to_dynesty_dict()` | Lightweight dynesty-compatible dictionary using matching tinyns fields. |

## Saving and loading results

Final results can be saved without extra dependencies using NumPy `.npz`:

```python
from tinyns import NestedSamplingResult

result.save_npz("run.npz")
loaded = NestedSamplingResult.load_npz("run.npz")
```

The `.npz` file stores weighted samples, evidence estimates, status, and
JSON-serialized metadata. Equal-weight posterior samples are not stored because
they can be regenerated with `resample_equal`.

HDF5 is not part of the core package to keep dependencies minimal. If needed,
HDF5 support can be added later as an optional extra.

## Progress and callbacks

`tinyns` has dependency-free progress reporting:

```python
result = sampler.run(key, progress=True, progress_interval=50)
```

For custom logging or early stopping, pass a callback:

```python
def callback(state):
    print(state["iter"], state["logz"], state["dlogz"])
    if state["iter"] > 1000:
        return False


result = sampler.run(key, callback=callback, callback_interval=25)
```

Returning `False` from the callback stops the run gracefully and returns a
partial `NestedSamplingResult`.

## Sampler recommendations

TinyNS is intentionally organized around one recommended fast path, with
reference baselines and experimental research knobs separated from that primary
route. The currently recommended fast path was validated on the included
benchmark targets; it should still be revalidated for new target geometries.

| Tier | Options | Status |
| --- | --- | --- |
| Recommended fast path | `sample="rwalk"`, `kernel="jax"`, `rwalk_proposal="isotropic"`, `walks=5`, `replacement_chains=1`, `jax_block_size=32` | Best validated path on included benchmarks |
| Reference baseline | `sample="rwalk"`, `kernel="python"` | Simple CPU/Python correctness/debug baseline |
| Reference baseline | `sample="prior"` | Conceptual brute-force constrained-prior baseline |
| Experimental | `rwalk_proposal="live-cov"` | Not promoted due to concerning validation pulls |
| Experimental | bounds / fused bounds / bounded block | Useful research direction; not production-ready |
| Experimental | adaptive replacement-chain schedules | Useful tuning knob; not the main recommended path |

Removed: slice/random-slice samplers were removed to keep TinyNS small. Use dynesty for slice-based external comparisons.

### Recommended fast JAX rwalk path

For unbounded JAX rwalk on the included validation targets, the recommended
fast path is the cached block kernel with isotropic proposals:

```python
from tinyns import NestedSampler

sampler = NestedSampler(
    loglike,
    prior_transform,
    ndim,
    sample="rwalk",
    kernel="jax",
    walks=5,
    replacement_chains=1,
    rwalk_proposal="isotropic",
    jax_block_size=32,
)

result = sampler.run(key, dlogz=0.1)
```

`jax_block_size > 1` batches several nested-sampling replacement iterations
into one cached, jitted JAX block. This reduces Python/JAX dispatch overhead,
which usually gives a large speedup for cheap or moderately expensive JAX
likelihoods. For very expensive likelihoods, the speedup may be smaller because
likelihood cost dominates dispatch overhead. Convergence is checked between
blocks, not after every individual nested iteration, so a run may overshoot the
requested `dlogz` threshold by up to roughly `jax_block_size - 1` iterations.

Use `jax_block_size=32` for the fastest validated unbounded JAX rwalk path. Use
`jax_block_size=16` if you want a more conservative block size with slightly
less convergence overshoot. Leave `jax_block_size=1` for the most conservative
behavior, which disables block mode. This recommendation is based on current
validation on the included benchmark targets; it is not a proof for all
likelihoods.


### Bound update interval

For `bound="multi"`, rebuilding every iteration can be expensive. Use
`bound_update_interval` to reuse a bound for multiple nested-sampling
iterations. Larger intervals reduce Python/bound-building overhead but can make
bounds stale. Validate evidence before relying on results.


### Batched JAX replacement chains

For GPU-native likelihoods, `rwalk+jax` can run several independent replacement chains in parallel:

```python
sampler = NestedSampler(
    loglike,
    prior_transform,
    ndim,
    sample="rwalk",
    kernel="jax",
    walks=25,
    replacement_chains=16,
)
```

Here `walks` is the length of each chain, while `replacement_chains` is the number of independent chains run in parallel per replacement batch. This can improve wall time on GPU by evaluating many proposals in parallel.

The replacement remains valid only if a successful chain is selected without favoring higher-likelihood endpoints. `tinyns` selects randomly among successful chains.

> Warning: Increasing `replacement_chains` increases likelihood evaluations per replacement attempt. It is useful only when the likelihood benefits from batched/device parallelism.

For batched JAX chains, `ncall` counts scalar likelihood evaluations, not wall-clock-equivalent work. A replacement with `walks=25` and `replacement_chains=16` costs 400 scalar likelihood evaluations, but those chains are evaluated in parallel on device. Use wall time and replacement batch counts when judging batched performance.

For JAX rwalk, `repl_ncall` is scalar likelihood calls per replacement. In fixed chain mode, `repl_chains` reports the effective number of parallel chains used per replacement. In adaptive mode, `usage=...` reports how often each stage in the replacement chain schedule was used. Use these diagnostics to choose the smallest chain count or schedule that avoids retry tails.


### Adaptive JAX replacement-chain schedules

Fixed `replacement_chains` runs the same number of independent chains for every replacement. This can waste work when most chains succeed.

For JAX rwalk, `tinyns` can instead start with a small batch and escalate only if needed:

```python
sampler = NestedSampler(
    loglike,
    prior_transform,
    ndim,
    sample="rwalk",
    kernel="jax",
    walks=25,
    replacement_chain_schedule=(1, 4, 16, 64, 256),
)
```

Use this when replacement difficulty varies across the nested-sampling run. The sampler returns as soon as any stage succeeds and randomly selects among successful chains in that stage. This avoids always paying for large batches.

> Warning: Adaptive schedules do not replace validation. Check evidence calibration and insertion-rank diagnostics on representative targets.

For non-JAX likelihoods, or when debugging sampler behavior, use `kernel="python"` with `sample="rwalk"`.

`kernel="jax"` currently supports `sample="rwalk"` only. The top-level nested-sampling loop remains in Python; only the constrained replacement kernel is compiled.

For local constrained samplers, step-count parameters are decorrelation lengths:

- `walks`: number of reflected random-walk proposals per replacement attempt for `rwalk`
- `min_accepts`: minimum number of accepted constrained rwalk moves required for the replacement to be considered valid

The sampler does not return merely after the first accepted local move; it runs the requested local update length.

`sample="prior"` supports vectorized replacement proposals with
`vectorized=True`; the full nested-sampling loop remains a small Python loop.
Vectorized `rwalk` replacement sampling is not implemented yet.

### Bounding

`tinyns` supports `bound="none"` by default. Bounds are experimental modifiers for rwalk, not a separate public sampler mode. Use `sample="rwalk"` with `bound="single"` or `bound="multi"` and `rwalk_seed="bound"`. Bounds are built in unit-cube coordinates from the live points and enlarged by `bound_enlargement`.

The only currently recommended fast path is unbounded JAX rwalk with isotropic proposals and cached block mode. Live-cov proposals and bounded/fused-bounded paths remain experimental and require target-specific validation.

Bounding is experimental. Validate evidence and insertion-rank diagnostics on representative targets before relying on it for scientific results.


### Experimental adaptive rwalk step scale

`rwalk_adaptive_step_scale=True` is an explicitly experimental JAX-only rwalk option that adapts the isotropic rwalk proposal scale from constrained-replacement acceptance telemetry. The default remains off, `step_scale=0.1` remains unchanged, and the recommended B32 path is still `sample="rwalk"`, `kernel="jax"`, `rwalk_proposal="isotropic"`, `walks=5`, `step_scale=0.1`, `min_accepts=1`, `replacement_chains=1`, `replacement_chain_schedule=None`, `bound="none"`, and `jax_block_size=32`.

This is intended for hard-target diagnostics where a fixed rwalk scale is a poor compromise. It is not a dynesty replacement, does not add slice/rslice or `sample="bound"`, and is not a substitute for checking insertion-rank diagnostics, replacement diagnostics, and seed/config stability on the target.

### Bounded rwalk

For experimental dynesty-style bounded rwalk, use both a bound and bound seeding:

```python
sampler = NestedSampler(
    loglike,
    prior_transform,
    ndim,
    sample="rwalk",
    kernel="jax",
    bound="multi",
    rwalk_seed="bound",
    rwalk_proposal="live-cov",
    walks=5,
    replacement_chains=16,
)
```

Setting `bound="multi"` alone does not define a bounded rwalk transition unless `rwalk_seed="bound"` is also enabled. `tinyns` raises a clear error for `bound != "none"` with live-seeded rwalk unless `allow_unused_bound=True`. Use `allow_unused_bound=True` only when you intentionally want to build bounds for diagnostics or overhead measurements while keeping ordinary live-seeded rwalk.

`fused_bound_rwalk=True` currently means the bounded seed draw and rwalk transition are exposed as one replacement path and share accounting. It is not yet a single compiled seed+rwalk kernel. A future implementation may replace this wrapper fusion with a true single-dispatch JAX kernel.

For `bound="none"`, `jax_block_size=32` uses a cached JAX `lax.scan` over several nested-sampling iterations and is the recommended fast path described above. For bounded rwalk, block mode remains experimental: the current mode reuses a fixed bound across a Python-level block and is mainly a stepping stone toward a fully compiled bounded block kernel.

### Multiellipsoid bounding

`bound="multi"` is an experimental dynesty-style union-of-ellipsoids bound. It recursively splits the live points using a dependency-free PCA/median split and samples from the volume-weighted union of ellipsoids with overlap correction.

An experimental bounded/fused candidate configuration for separate validation is:

```python
NestedSampler(
    loglike,
    prior_transform,
    ndim,
    sample="rwalk",
    kernel="jax",
    bound="multi",
    rwalk_seed="bound",
    rwalk_proposal="live-cov",
    walks=5,
    replacement_chains=16,
)
```

This mode is experimental and is not the recommended fast path. Check evidence calibration, insertion-rank diagnostics, and seed stability before using it for science.

### JAX bound representation

`tinyns` keeps Python-friendly bound objects for readability, but also provides an internal padded `JaxEllipsoidBound` representation. The padded representation is used as a bridge toward fast JAX replacement kernels and should not change public sampling behavior.

## Current validation status

The recommended fast path (`sample="rwalk"`, `kernel="jax"`, `rwalk_proposal="isotropic"`, `walks=5`, `replacement_chains=1`, `jax_block_size=32`) has been checked with repeated-seed validation on the included benchmark targets:

- `gaussian2d`
- `correlated_gaussian2d`
- `ring2d`
- `banana2d`
- `eggbox2d`

The analytic Gaussian targets show good evidence calibration in the current validation suite, and qualitative targets show acceptable insertion-rank diagnostics. TinyNS focuses on one optimized static nested-sampling path, JAX rwalk with cached block mode, plus small reference baselines. Bounds and `rwalk_proposal="live-cov"` are tracked separately as experimental. Users should still validate on their own target geometry before relying on evidence values.

### `min_accepts`

For `rwalk`, `min_accepts` requires multiple accepted
constrained moves before returning a replacement. The default is
`min_accepts=1`.

Increasing `min_accepts` can increase likelihood-call cost and is not guaranteed
to improve evidence calibration. In the current validation suite,
`min_accepts=3` did not generally improve calibration and made several runs
worse. Treat it as an experimental diagnostic knob rather than a recommended
default.

## Design philosophy

- **Tiny:** keep dependencies and abstractions minimal.
- **Dynesty-style:** expose familiar `loglike + prior_transform + ndim` entry
  points and lightweight dynesty-compatible exports.
- **JAX-friendly:** support JAX arrays, PRNG keys, and JAX-compatible callbacks
  without requiring users to define model objects.
- **Correctness and diagnostics first:** prefer clear bookkeeping, tests, and
  warnings over premature speedups.

## Limitations

`tinyns` is an early v0.1-oriented implementation for correctness experiments
and low-dimensional examples. It is intentionally not a probabilistic
programming language.

- Static nested sampling only.
- No dynamic nested sampling.
- Multiellipsoid bounding is experimental.
- No full vectorized `rwalk` replacement sampler.
- Not a PPL; users provide functions, not model objects.
- Replacement attempts are capped by `max_attempts`; hitting the cap returns
  `success=False` with a partial result rather than raising during the run.
- Evidence and live-point bookkeeping are included, but error estimates are
  lightweight diagnostics for this toy implementation.

## Additional examples

### Primary

- `examples/gaussian_2d_rwalk_jax_block.py`: recommended validated fast path using JAX `rwalk`, isotropic proposals, one replacement chain, and cached block mode with `jax_block_size=32`.

### Reference

- `examples/gaussian_2d.py`: 2D Gaussian with prior rejection.
- `examples/gaussian_2d_rwalk.py`: reflected Python random-walk constrained sampling.
- `examples/gaussian_2d_rwalk_jax.py`: simple JAX-native random-walk replacement path without the full cached block configuration.

### Utility

- `examples/progress_and_callback.py`: dependency-free progress and callbacks.
- `examples/vectorized_gaussian_2d.py`: vectorized prior-rejection proposals.
- `examples/checkpoint_resume.py`: checkpoint and resume with the rwalk baseline.

### Experimental

- `examples/banana_2d.py` and `examples/eggbox_2d.py`: qualitative target demos that can exercise non-primary sampler options for experimentation.
- `examples/repeated_gaussian_evidence.py` and `examples/save_load_result.py`: workflow/diagnostic demos for coverage or reproducibility checks.

## Validation

For repeated-seed validation on analytic targets:

```bash
python validation/run_validation.py --output validation_results.json
python validation/summarize_validation.py validation_results.json
```

The validation harness is intended to catch calibration and reliability issues,
not to be a formal speed benchmark.
The validation summary includes heuristic calibration warnings such as repeated
large evidence z-scores, high final-live weight fraction, and concentrated
posterior weights.

## Checkpoint and resume

Long static nested-sampling runs can save active checkpoints:

```python
result = sampler.run(
    key,
    checkpoint_path="run.checkpoint.npz",
    checkpoint_interval=100,
)
```

A checkpoint stores the active live points, accumulated dead points, PRNG key,
iteration counters, and sampler metadata. It does not serialize user functions.
To resume, reconstruct the same sampler and call:

```python
result = sampler.resume("run.checkpoint.npz")
```

The sampler configuration must match the checkpoint. Checkpoints are distinct
from final result files saved with `result.save_npz(...)`. Checkpoints use NumPy
`.npz` files to avoid extra dependencies. Checkpoint/resume is intended for
static nested sampling; dynamic nested sampling is not implemented yet.
