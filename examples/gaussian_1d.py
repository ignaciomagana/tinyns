"""Run tinyns on a normalized one-dimensional standard normal likelihood.

The prior is uniform on [-10, 10]. Since nearly all likelihood mass is inside
that interval, the expected log evidence is approximately -log(20).
"""

import jax
import jax.numpy as jnp

from tinyns import NestedSampler

NDIM = 1
PRIOR_WIDTH = 20.0
EXPECTED_LOGZ = -jnp.log(PRIOR_WIDTH)


def prior_transform(u):
    """Map the unit interval to a uniform prior on [-10, 10]."""

    return -10.0 + PRIOR_WIDTH * u


def loglike(theta):
    """Normalized standard normal log likelihood."""

    return -0.5 * theta[0] ** 2 - 0.5 * jnp.log(2.0 * jnp.pi)


def main():
    key = jax.random.PRNGKey(1)
    run_key, resample_key = jax.random.split(key)

    sampler = NestedSampler(loglike, prior_transform, ndim=NDIM, nlive=100)
    result = sampler.run(run_key, dlogz=0.5)

    print(result.summary())
    print(f"expected logZ: {EXPECTED_LOGZ}")

    samples = result.resample_equal(resample_key, n=1_000)
    print(f"sample mean: {jnp.mean(samples[:, 0])}")
    print(f"sample std: {jnp.std(samples[:, 0], ddof=1)}")


if __name__ == "__main__":
    main()
