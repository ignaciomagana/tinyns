# tinyns

`tinyns` is a tiny, dynesty-style nested sampler for JAX-friendly likelihoods.
It provides a compact interface that transforms unit-cube samples through a user
prior, evaluates a log likelihood, and returns a structured
`NestedSamplingResult` containing posterior samples, weights, and evidence
estimates.

## Current status and limitations

This is an early, static nested-sampling implementation intended for correctness
experiments and low-dimensional examples. The default replacement sampler,
`sample="prior"`, is brute-force rejection from the prior: it repeatedly draws a
fresh unit-cube point and keeps it only if its likelihood exceeds the current
nested-sampling threshold. That approach is correctness-first and useful for toy
problems, but it becomes inefficient very quickly as dimension or likelihood
concentration grows.

In particular:

- this is static nested sampling only; dynamic nested sampling is not
  implemented;
- no slice sampler is implemented;
- vectorized replacement is implemented for `sample="prior"` rejection draws;
- vectorized `sample="rwalk"` is not implemented yet and currently raises
  before the run starts;
- replacement attempts are capped by `max_attempts`, and hitting that cap returns
  `success=False` with the partial result rather than raising during the run;
- evidence and live-point bookkeeping are included, but error estimates are only
  lightweight diagnostics for this toy implementation.

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


## Vectorized prior-rejection example

If `sample="prior"` and `vectorized=True`, `tinyns` batches constrained
prior-rejection replacement proposals. This can reduce callback overhead when
your `prior_transform` and `loglike` naturally accept arrays, but it does **not**
yet JIT or vectorize the whole nested-sampling loop. The loop remains a small
Python loop, and vectorized `sample="rwalk"` is not implemented yet.

```python
def prior_transform(u):
    # u has shape (batch, 2). Return shape (batch, 2).
    return -10.0 + 20.0 * u


def loglike(theta):
    # theta has shape (batch, 2). Return shape (batch,).
    return -0.5 * jnp.sum(theta**2, axis=1) - jnp.log(2.0 * jnp.pi)


sampler = NestedSampler(
    loglike,
    prior_transform,
    ndim=2,
    nlive=200,
    sample="prior",
    vectorized=True,
    batch_size=256,
)
result = sampler.run(key, dlogz=0.5)
print(result.summary())
```

See `examples/vectorized_gaussian_2d.py` for a complete 2D Gaussian example
that prints the expected log evidence and equally resampled posterior
mean/standard deviation.


## Random-walk constrained sampler

For slightly less wasteful constrained draws, `sample="rwalk"` starts from an
existing live point and performs reflected Gaussian random-walk moves in the unit
cube, accepting only proposals whose likelihood remains above the current nested
sampling threshold. The `walks` option controls how many accepted-or-rejected
proposal steps are attempted for each replacement, and `step_scale` controls the
Gaussian proposal scale in unit-cube coordinates.

This sampler is still deliberately simple. It is useful for small examples, but
it is not yet a production-grade multimodal sampler.

```python
sampler = NestedSampler(
    loglike,
    prior_transform,
    ndim=2,
    nlive=200,
    sample="rwalk",
    walks=25,
    step_scale=0.1,
)
result = sampler.run(key, dlogz=0.5)
print(result.summary())
```
