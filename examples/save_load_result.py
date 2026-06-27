"""Save and load a tiny one-dimensional nested-sampling result."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tinyns import NestedSampler, NestedSamplingResult

NDIM = 1
PRIOR_WIDTH = 20.0
RESULT_PATH = "tinyns_example_result.npz"


def prior_transform(u):
    """Map the unit interval to a uniform prior on [-10, 10]."""

    return -10.0 + PRIOR_WIDTH * u


def loglike(theta):
    """Normalized standard normal log likelihood."""

    return -0.5 * theta[0] ** 2 - 0.5 * jnp.log(2.0 * jnp.pi)


def main():
    key = jax.random.PRNGKey(7)
    run_key, resample_key = jax.random.split(key)

    sampler = NestedSampler(loglike, prior_transform, ndim=NDIM, nlive=40)
    result = sampler.run(run_key, dlogz=1.0)
    result.save_npz(RESULT_PATH)
    loaded = NestedSamplingResult.load_npz(RESULT_PATH)

    print("Original summary:")
    print(result.summary())
    print("\nLoaded summary:")
    print(loaded.summary())

    samples = loaded.resample_equal(resample_key, n=5)
    print(f"\nEqual-weight posterior sample shape: {samples.shape}")
    print(f"Saved result to: {RESULT_PATH}")


if __name__ == "__main__":
    main()
