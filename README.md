# tinyns

`tinyns` aims to be a tiny, dynesty-style nested sampler for JAX-friendly
likelihoods. The project is currently locking down its public API before the
full sampling algorithm is implemented.

## Goal

The goal is to provide a compact nested-sampling interface that can transform
unit-cube samples through a user prior, evaluate a log likelihood, and return a
structured `NestedSamplingResult` containing posterior samples, weights, and
evidence estimates.

## Minimal future usage

```python
import jax.numpy as jnp
from jax import random

from tinyns import NestedSampler


def loglike(theta):
    return -0.5 * jnp.dot(theta, theta)


def prior_transform(unit):
    return 2.0 * unit - 1.0


sampler = NestedSampler(loglike, prior_transform, ndim=2, nlive=500)
result = sampler.run(random.PRNGKey(0), dlogz=0.1, progress=True)
print(result.logz, result.logzerr)
```

`NestedSampler.run` is not implemented yet; the example above documents the
intended shape of the public API.
