"""Qualitative 2D banana-shaped likelihood stress test for tinyns.

The prior maps the unit square to [-5, 5]^2. The likelihood is intentionally
curved and non-Gaussian, so this example is for qualitative sampler diagnostics
rather than exact evidence validation.
"""

import jax

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


def main():
    sampler = NestedSampler(loglike, prior_transform, ndim=NDIM, nlive=NLIVE)
    result = sampler.run(jax.random.PRNGKey(73), dlogz=0.5, maxiter=600)

    print(result.summary())
    print(f"diagnostics: {result.diagnostics()}")
    print("note: no analytic evidence is assumed for this stress test.")


if __name__ == "__main__":
    main()
