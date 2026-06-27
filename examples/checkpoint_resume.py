"""Checkpoint and resume a small 2D Gaussian nested-sampling run."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax.numpy as jnp
from jax import random

from tinyns import NestedSampler


def loglike(theta):
    theta = jnp.asarray(theta)
    return -0.5 * jnp.sum(((theta - 0.5) / 0.1) ** 2)


def prior_transform(u):
    return jnp.asarray(u)


def main():
    checkpoint_path = Path("run.checkpoint.npz")
    sampler = NestedSampler(loglike, prior_transform, ndim=2, nlive=50, sample="rslice")

    partial = sampler.run(
        random.PRNGKey(123),
        maxiter=10,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=5,
    )
    print("Initial partial result:")
    print(partial.summary())

    resumed_sampler = NestedSampler(
        loglike, prior_transform, ndim=2, nlive=50, sample="rslice"
    )
    resumed = resumed_sampler.resume(checkpoint_path, maxiter=20)
    print("\nResumed result:")
    print(resumed.summary())
    print(f"\nCheckpoint path: {checkpoint_path}")


if __name__ == "__main__":
    main()
