"""Qualitative 2D eggbox stress test for tinyns.

This example uses the unit-cube prior directly and an eggbox-like likelihood
with many separated modes. It is intended as a lightweight pathological stress
test for constrained samplers, not as a guaranteed production benchmark.
"""

import jax
import jax.numpy as jnp

from tinyns import NestedSampler

NDIM = 2
NLIVE = 80


def prior_transform(u):
    """Use the unit square as the parameter space."""

    return u


def loglike(theta):
    """Multimodal eggbox-like log likelihood on the unit square."""

    x = 10.0 * jnp.pi * theta[0]
    y = 10.0 * jnp.pi * theta[1]
    return 5.0 * jnp.log(2.0 + jnp.cos(x) * jnp.cos(y))


def run_sampler(sample, key):
    """Run one constrained sampler configuration and print diagnostics."""

    kwargs = {
        "sample": sample,
        "nlive": NLIVE,
        "step_scale": 0.08,
    }
    if sample == "rwalk":
        kwargs["walks"] = 15
    elif sample == "slice":
        kwargs["slices"] = 4
        kwargs["slice_steps"] = 6

    try:
        sampler = NestedSampler(loglike, prior_transform, ndim=NDIM, **kwargs)
    except ValueError as exc:
        print(f"\n=== sample={sample!r} unavailable: {exc} ===")
        return

    result = sampler.run(key, dlogz=0.5, maxiter=600)

    print(f"\n=== sample={sample!r} ===")
    print(result.summary())
    print(f"diagnostics: {result.diagnostics()}")
    print("note: no analytic evidence is assumed for this stress test.")


def main():
    key = jax.random.PRNGKey(72)
    sample_keys = jax.random.split(key, 2)
    for sample, sample_key in zip(("rwalk", "slice"), sample_keys, strict=True):
        run_sampler(sample, sample_key)


if __name__ == "__main__":
    main()
