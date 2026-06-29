# Reference-only example.
# slice/rslice are retained for debugging and comparison, but are not part of
# the recommended fast path. Prefer JAX rwalk with cached block mode for
# performance-sensitive use.
"""Run tinyns on a 2D Gaussian using the reference-only random-direction slice sampler.

This sampler is frozen, not optimized, and retained for comparison/debugging;
it is not part of the recommended fast path.

The prior is uniform on [-10, 10]^2. Since nearly all likelihood mass is inside
that square, the expected log evidence is approximately -log(400).
"""

import jax
import jax.numpy as jnp

from tinyns import NestedSampler

NDIM = 2
PRIOR_WIDTH = 20.0
PRIOR_VOLUME = PRIOR_WIDTH**NDIM
EXPECTED_LOGZ = -jnp.log(PRIOR_VOLUME)


def prior_transform(u):
    """Map the unit square to a uniform prior on [-10, 10]^2."""

    return -10.0 + PRIOR_WIDTH * u


def loglike(theta):
    """Normalized two-dimensional standard normal log likelihood."""

    return -0.5 * jnp.sum(theta**2) - jnp.log(2.0 * jnp.pi)


def main():
    key = jax.random.PRNGKey(5)
    run_key, resample_key = jax.random.split(key)

    sampler = NestedSampler(
        loglike,
        prior_transform,
        ndim=NDIM,
        nlive=200,
        sample="rslice",
        slices=5,
        slice_steps=10,
        step_scale=0.1,
        min_accepts=3,
    )
    result = sampler.run(run_key, dlogz=0.5)

    print(result.summary())
    print(f"expected logZ: {EXPECTED_LOGZ}")

    samples = result.resample_equal(resample_key, n=1_000)
    print(f"sample mean: {jnp.mean(samples, axis=0)}")
    print(f"sample covariance:\n{jnp.cov(samples, rowvar=False)}")


if __name__ == "__main__":
    main()
