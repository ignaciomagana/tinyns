"""Run tinyns on a two-dimensional constant likelihood.

The likelihood is one everywhere on the unit-square prior, so the evidence is
exactly 1 and the expected log evidence is 0.
"""

import jax
import jax.numpy as jnp

from tinyns import NestedSampler

NDIM = 2
EXPECTED_LOGZ = 0.0


def prior_transform(u):
    """Identity transform from the unit cube to parameter space."""

    return u


def loglike(theta):
    """Constant unit likelihood over the prior volume."""

    return 0.0


def main():
    key = jax.random.PRNGKey(0)
    run_key, resample_key = jax.random.split(key)

    sampler = NestedSampler(loglike, prior_transform, ndim=NDIM, nlive=200)
    result = sampler.run(run_key, dlogz=0.01)

    print(result.summary())
    print(f"expected logZ: {EXPECTED_LOGZ}")

    samples = result.resample_equal(resample_key, n=1_000)
    print(f"sample mean: {jnp.mean(samples, axis=0)}")
    print(f"sample covariance:\n{jnp.cov(samples, rowvar=False)}")


if __name__ == "__main__":
    main()
