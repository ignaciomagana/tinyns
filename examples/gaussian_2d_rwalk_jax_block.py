"""A 2D Gaussian with the default sampler settings."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tinyns import NestedSampler


def prior_transform(u):
    """Map the unit square to a uniform prior on [-5, 5]^2."""

    return -5.0 + 10.0 * u


def loglike(theta):
    """Normalized two-dimensional standard normal log likelihood."""

    return -0.5 * jnp.sum(theta**2) - jnp.log(2.0 * jnp.pi)


def main():
    key = jax.random.PRNGKey(0)

    # The defaults: num_delete=max(1, nlive // 10) replacements per step, each
    # a live-covariance random walk of walks=tinyns.core.default_walks(ndim)
    # steps (25 here).
    sampler = NestedSampler(loglike, prior_transform, ndim=2, nlive=200)

    result = sampler.run(key, dlogz=0.1)
    print(result.summary())
    print(result.diagnostics())


if __name__ == "__main__":
    main()
