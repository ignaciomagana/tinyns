"""Checkpoint a run, stop it early, and resume it.

``run(key, checkpoint=path)`` writes ``path`` (at most every 10 minutes and
at the end) and resumes from it when it exists. Here the first call stops at
``maxiter``; the second call picks the run up from the file. The resumed
result is bit-identical to an uninterrupted run with the same key.
"""

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp

from tinyns import NestedSampler

NDIM = 5


def prior_transform(u):
    """Map the unit cube to the uniform prior on [-10, 10]^5."""
    return -10.0 + 20.0 * u


def loglike(theta):
    """Normalized standard normal log likelihood."""
    return -0.5 * jnp.sum(theta**2) - 0.5 * NDIM * jnp.log(2 * jnp.pi)


def main():
    sampler = NestedSampler(loglike, prior_transform, ndim=NDIM, nlive=500)
    key = jax.random.key(0)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "run.npz"

        first = sampler.run(key, checkpoint=path, maxiter=3000)
        print(f"first call:  {first.message} (niter {first.niter})")

        resumed = sampler.run(key, checkpoint=path)
        print(f"second call: {resumed.message} (niter {resumed.niter})")
        print(f"resumed: {resumed.metadata['resumed']}")

    straight = sampler.run(key)
    print(f"logZ resumed:       {resumed.logz:.6f} +/- {resumed.logzerr:.6f}")
    print(f"logZ uninterrupted: {straight.logz:.6f} +/- {straight.logzerr:.6f}")
    return first, resumed, straight


if __name__ == "__main__":
    main()
