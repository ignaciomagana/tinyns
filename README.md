# tinyns

`tinyns` is a tiny, dynesty-style nested sampler for JAX-friendly likelihoods.
The core public API is deliberately small: provide `loglike`,
`prior_transform`, and `ndim`, then call `NestedSampler(...).run(key)`.

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

`tinyns` supports several constrained-replacement samplers:

| sampler | use case | caveats |
|---|---|---|
| `prior` | Conceptual baseline; brute-force rejection from the prior | Can be extremely expensive as likelihood constraints tighten |
| `rwalk` | Recommended robust default for low-dimensional problems | More likelihood calls than slice-like samplers |
| `slice` | Fast coordinate-wise constrained slice updates | Can under-cover evidence uncertainty; validate on correlated targets |
| `rslice` | Fast random-direction constrained slice updates | Experimental; validate before relying on evidence estimates |

For local constrained samplers, step-count parameters are decorrelation lengths:

- `walks`: number of reflected random-walk proposals per replacement attempt for `rwalk`
- `slices`: number of coordinate or random-direction slice updates per replacement attempt for `slice` and `rslice`
- `slice_steps`: shrinkage proposal budget per slice update
- `min_accepts`: minimum number of accepted constrained moves required for the replacement to be considered valid

The sampler does not return merely after the first accepted local move; it runs the requested local update length.

The current recommended starting point for nontrivial low-dimensional problems is:

```python
sampler = NestedSampler(
    loglike,
    prior_transform,
    ndim,
    sample="rwalk",
)
```

For faster exploratory runs, `slice` may be useful, but evidence
calibration should be checked with repeated validation runs. Treat `rslice` as
experimental.

`sample="prior"` supports vectorized replacement proposals with
`vectorized=True`; the full nested-sampling loop remains a small Python loop.
Vectorized `rwalk`, `slice`, and `rslice` are not implemented yet.

### JAX replacement kernel

For JAX-native likelihoods, the recommended reliable fast path is:

```python
sampler = NestedSampler(
    loglike,
    prior_transform,
    ndim,
    sample="rwalk",
    kernel="jax",
    walks=25,
    step_scale=0.1,
)
```

This keeps the top-level nested-sampling loop in Python but runs each constrained random-walk replacement on device. It avoids proposal-by-proposal Python/GPU synchronization while preserving the same replacement semantics as the Python `rwalk` kernel.

The JAX rwalk kernel matches the Python rwalk replacement semantics: it retries fresh live-point seeds until a full `walks`-step chain has at least `min_accepts` accepted moves or `max_attempts` is exhausted.

`kernel="jax"` currently supports `sample="rwalk"` only. Use `kernel="python"` for `prior`, `slice`, and `rslice`.

### `min_accepts`

For `rwalk`, `slice`, and `rslice`, `min_accepts` requires multiple accepted
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
- No full ellipsoidal bounding.
- No full vectorized `rwalk`, `slice`, or `rslice` replacement sampler.
- Not a PPL; users provide functions, not model objects.
- Replacement attempts are capped by `max_attempts`; hitting the cap returns
  `success=False` with a partial result rather than raising during the run.
- Evidence and live-point bookkeeping are included, but error estimates are
  lightweight diagnostics for this toy implementation.

## Additional examples

- `examples/gaussian_2d.py`: 2D Gaussian with prior rejection.
- `examples/gaussian_2d_rwalk.py`: reflected random-walk constrained sampling.
- `examples/gaussian_2d_slice.py`: coordinate-wise constrained slice sampling.
- `examples/progress_and_callback.py`: dependency-free progress and callbacks.
- `examples/vectorized_gaussian_2d.py`: vectorized prior-rejection proposals.

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
