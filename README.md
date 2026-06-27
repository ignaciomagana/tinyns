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

## Replacement samplers

| `sample` | Description |
| --- | --- |
| `"prior"` | Brute-force rejection from the prior; robust and correctness-first, but inefficient as dimension or likelihood concentration grows. |
| `"rwalk"` | Reflected random-walk constrained sampler in the unit cube. |
| `"slice"` | Coordinate-wise constrained slice sampler in the unit cube. |
| `"rslice"` | Random-direction constrained slice sampler in the unit cube; often better for correlated targets than coordinate-wise `"slice"`. |

`sample="prior"` supports vectorized replacement proposals with
`vectorized=True`; the full nested-sampling loop remains a small Python loop.
Vectorized `rwalk`, `slice`, and `rslice` are not implemented yet.

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
large evidence z-scores, high live-point weight fraction, and concentrated
posterior weights.
