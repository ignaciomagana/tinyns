"""Qualitative 2D banana-shaped likelihood stress test for tinyns.

The prior maps the unit square to [-5, 5]^2. The likelihood is intentionally
curved and non-Gaussian, so this example is for qualitative sampler diagnostics
rather than exact evidence validation.
"""

import jax
import jax.numpy as jnp

from tinyns import NestedSampler

NDIM = 2
NLIVE = 80
PRIOR_LOW = -5.0
PRIOR_HIGH = 5.0
PRIOR_WIDTH = PRIOR_HIGH - PRIOR_LOW


def prior_transform(u):
    """Map the unit square to a uniform prior on [-5, 5]^2."""

    return PRIOR_LOW + PRIOR_WIDTH * u


def loglike(theta):
    """Curved banana-shaped log likelihood."""

    x = theta[0]
    y = theta[1]
    banana = y - 0.2 * (x**2 - 4.0)
    return -0.5 * (x / 1.8) ** 2 - 0.5 * (banana / 0.35) ** 2


def run_sampler(sample, key):
    """Run one constrained sampler configuration and print diagnostics."""

    kwargs = {
        "sample": sample,
        "nlive": NLIVE,
        "step_scale": 0.12,
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
    key = jax.random.PRNGKey(73)
    for sample, sample_key in zip(("rwalk", "slice"), jax.random.split(key, 2)):
        run_sampler(sample, sample_key)


if __name__ == "__main__":
    main()
