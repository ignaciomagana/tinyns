"""Run tinyns with vectorized likelihood and prior callbacks.

The prior is uniform on [-10, 10]^2. Since nearly all likelihood mass is inside
that square, the expected log evidence is approximately -log(400).

Vectorized mode batches prior-rejection replacement draws. The nested-sampling
loop itself is still a small Python loop.
"""

import jax
import jax.numpy as jnp

from tinyns import NestedSampler

NDIM = 2
PRIOR_WIDTH = 20.0
PRIOR_VOLUME = PRIOR_WIDTH**NDIM
EXPECTED_LOGZ = -jnp.log(PRIOR_VOLUME)


def prior_transform(u):
    """Map a batch of unit-square points to a uniform prior on [-10, 10]^2."""

    return -10.0 + PRIOR_WIDTH * u


def loglike(theta):
    """Return batched normalized two-dimensional standard-normal log likelihoods."""

    return -0.5 * jnp.sum(theta**2, axis=1) - jnp.log(2.0 * jnp.pi)


def main():
    key = jax.random.PRNGKey(4)
    run_key, resample_key = jax.random.split(key)

    sampler = NestedSampler(
        loglike,
        prior_transform,
        ndim=NDIM,
        nlive=200,
        sample="prior",
        vectorized=True,
        batch_size=256,
    )
    result = sampler.run(run_key, dlogz=0.5)

    print(result.summary())
    print(f"expected logZ: {EXPECTED_LOGZ}")

    samples = result.resample_equal(resample_key, n=1_000)
    print(f"posterior mean: {jnp.mean(samples, axis=0)}")
    print(f"posterior std: {jnp.std(samples, axis=0)}")


if __name__ == "__main__":
    main()
