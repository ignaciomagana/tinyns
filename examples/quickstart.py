"""Quickstart: evidence and posterior of a 3-D correlated Gaussian.

The prior is uniform on [-10, 10]^3 and the likelihood is a normalized
Gaussian well inside it, so the evidence is the inverse prior volume:
logZ = -3 log 20.
"""

import math

import jax
import jax.numpy as jnp

from tinyns import NestedSampler

NDIM = 3
MEAN = jnp.array([1.0, -2.0, 0.5])
COV = jnp.array([[1.0, 0.8, 0.0], [0.8, 1.0, 0.3], [0.0, 0.3, 0.5]])
PRECISION = jnp.linalg.inv(COV)
LOG_NORM = -0.5 * (NDIM * math.log(2 * math.pi) + jnp.linalg.slogdet(COV)[1])
TRUE_LOGZ = -NDIM * math.log(20.0)


def prior_transform(u):
    """Map the unit cube to the uniform prior on [-10, 10]^3."""
    return -10.0 + 20.0 * u


def loglike(theta):
    """Log likelihood of one point."""
    r = theta - MEAN
    return LOG_NORM - 0.5 * r @ PRECISION @ r


def main():
    sampler = NestedSampler(loglike, prior_transform, ndim=NDIM)
    result = sampler.run(jax.random.key(0))

    print(result.summary())
    print(f"true logZ: {TRUE_LOGZ:.4f}")

    samples = result.resample_equal(jax.random.key(1), n=4000)
    print(f"posterior mean: {jnp.mean(samples, axis=0)}")
    print(f"posterior covariance:\n{jnp.cov(samples, rowvar=False)}")
    return result


if __name__ == "__main__":
    main()
