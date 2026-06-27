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
- no fully vectorized replacement sampler is implemented yet;
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
