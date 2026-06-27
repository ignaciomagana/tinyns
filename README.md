# tinyns

`tinyns` is a tiny, dynesty-style nested sampler for JAX-friendly likelihoods.
It provides a compact interface that transforms unit-cube samples through a user
prior, evaluates a log likelihood, and returns a structured
`NestedSamplingResult` containing posterior samples, weights, and evidence
estimates.

## Current status and limitations

This is an early, static nested-sampling implementation intended for correctness
experiments and low-dimensional examples. The only replacement sampler currently
implemented is brute-force rejection from the prior: it repeatedly draws a fresh
unit-cube point and keeps it only if its likelihood exceeds the current nested
sampling threshold. That approach is simple and useful for toy problems, but it
becomes inefficient very quickly as dimension or likelihood concentration grows.

In particular:

- only `sample="prior"` is supported;
- no random-walk, slice, or MCMC constrained sampler is implemented yet;
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
