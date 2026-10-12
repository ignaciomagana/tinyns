"""A two-mode Gaussian mixture in 4-D, with per-mode masses from ``modes()``.

The modes hold 75% and 25% of the posterior mass, are 14 standard deviations
apart and differ in width. A random walk cannot cross between them; the
sampler's inter-mode hop keeps the number of live points in each mode in
proportion to its volume, so the masses come out right.
"""

import math

import jax
import jax.numpy as jnp

from tinyns import NestedSampler

NDIM = 4
MEANS = jnp.array([[3.5] * NDIM, [-3.5] * NDIM])
SIGMAS = jnp.array([0.5, 0.3])
MASSES = jnp.array([0.75, 0.25])
TRUE_LOGZ = -NDIM * math.log(20.0)


def prior_transform(u):
    """Map the unit cube to the uniform prior on [-10, 10]^4."""
    return -10.0 + 20.0 * u


def loglike(theta):
    """Log of a mixture of two normalized isotropic Gaussians."""
    r2 = jnp.sum((theta - MEANS) ** 2, axis=1)
    log_norm = -0.5 * NDIM * jnp.log(2 * jnp.pi * SIGMAS**2)
    return jax.nn.logsumexp(jnp.log(MASSES) + log_norm - 0.5 * r2 / SIGMAS**2)


def main():
    sampler = NestedSampler(loglike, prior_transform, ndim=NDIM)
    result = sampler.run(jax.random.key(0))

    # With more than one mode the summary ends with a per-mode table.
    print(result.summary())
    print(f"true logZ: {TRUE_LOGZ:.4f}, true masses: 0.75, 0.25")

    for i, mode in enumerate(result.modes()):
        print(
            f"mode {i}: mass {mode['mass']:.3f}, smallest live count "
            f"{mode['min_live']}, unresolved: {mode['unresolved']}"
        )
    return result


if __name__ == "__main__":
    main()
