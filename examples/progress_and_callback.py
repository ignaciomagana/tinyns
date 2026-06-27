"""Demonstrate tinyns progress reporting and callbacks."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from tinyns import NestedSampler


def prior_transform(u):
    """Map the unit square to a broad square prior."""

    return 10.0 * u - 5.0


def loglike(theta):
    """Standard 2D Gaussian log likelihood."""

    return float(-0.5 * jnp.sum(theta**2) - math.log(2.0 * math.pi))


def main() -> None:
    key = jax.random.PRNGKey(0)
    sampler = NestedSampler(
        loglike,
        prior_transform,
        ndim=2,
        nlive=40,
        sample="slice",
        slices=3,
        slice_steps=5,
        step_scale=0.2,
    )

    callback_states = []

    def callback(state):
        callback_states.append(state)

    result = sampler.run(
        key,
        dlogz=0.5,
        maxiter=150,
        progress=True,
        progress_interval=25,
        callback=callback,
        callback_interval=10,
    )

    print(result.summary())
    print(f"callback calls: {len(callback_states)}")


if __name__ == "__main__":
    main()
