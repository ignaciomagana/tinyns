"""Run tinyns on a 2D Gaussian with the JAX random-walk kernel."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tinyns import NestedSampler


def prior_transform(u):
    """Map the unit square to a uniform prior on [-10, 10]^2."""

    return -10.0 + 20.0 * u


def loglike(theta):
    """Normalized two-dimensional standard normal log likelihood."""

    return -0.5 * jnp.sum(theta**2) - jnp.log(2.0 * jnp.pi)


def main():
    sampler = NestedSampler(
        loglike,
        prior_transform,
        ndim=2,
        nlive=200,
        sample="rwalk",
        kernel="jax",
        walks=25,
    )
    result = sampler.run(jax.random.PRNGKey(0), dlogz=0.5)

    print(result.summary())
    print(result.diagnostics())


if __name__ == "__main__":
    main()
