# tinyns

`tinyns` is a tiny, dynesty-style nested sampler for JAX-friendly likelihoods.
It provides a compact interface that transforms unit-cube samples through a user
prior, evaluates a log likelihood, and returns a structured
`NestedSamplingResult` containing posterior samples, weights, and evidence
estimates.

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
