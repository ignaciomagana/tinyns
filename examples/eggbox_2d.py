"""Qualitative 2D eggbox stress test for tinyns.

This example uses the unit-cube prior directly and an eggbox-like likelihood
with many separated modes. It is intended as a lightweight pathological stress
test for constrained samplers, not as a recommendation benchmark.
"""

import jax
import jax.numpy as jnp

from tinyns import NestedSampler

NDIM = 2
NLIVE = 80


def prior_transform(u):
    """Use the unit square as the parameter space."""

    return u


def loglike(theta):
    """Multimodal eggbox-like log likelihood on the unit square."""

    x = 10.0 * jnp.pi * theta[0]
    y = 10.0 * jnp.pi * theta[1]
    return 5.0 * jnp.log(2.0 + jnp.cos(x) * jnp.cos(y))


def main():
    sampler = NestedSampler(loglike, prior_transform, ndim=NDIM, nlive=NLIVE)
    result = sampler.run(jax.random.PRNGKey(72), dlogz=0.5, maxiter=600)

    print(result.summary())
    print(f"diagnostics: {result.diagnostics()}")
    print("note: no analytic evidence is assumed for this stress test.")


if __name__ == "__main__":
    main()
