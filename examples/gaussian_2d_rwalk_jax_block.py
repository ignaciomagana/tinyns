"""Recommended fast-path 2D Gaussian example using cached JAX block rwalk."""

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

    sampler = NestedSampler(
        loglike,
        prior_transform,
        ndim=2,
        nlive=200,
        sample="rwalk",
        kernel="jax",
        walks=5,
        replacement_chains=1,
        rwalk_proposal="isotropic",
        jax_block_size=32,
    )

    result = sampler.run(key, dlogz=0.1)
    print(result.summary())
    print(result.diagnostics())


if __name__ == "__main__":
    main()
