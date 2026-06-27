"""Compare repeated evidence estimates for a simple Gaussian problem.

This example runs tinyns several times on a normalized two-dimensional standard
normal likelihood with a uniform prior on [-10, 10]^2. Since essentially all
likelihood mass lies inside that prior box, the analytic expected log evidence
is -log(400).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from tinyns import NestedSampler

NDIM = 2
PRIOR_WIDTH = 20.0
PRIOR_VOLUME = PRIOR_WIDTH**NDIM
EXPECTED_LOGZ = -math.log(PRIOR_VOLUME)
NLIVE = 200
DLOGZ = 0.1
SEEDS = range(5)


def prior_transform(u):
    """Map the unit square to a uniform prior on [-10, 10]^2."""

    return -10.0 + PRIOR_WIDTH * u


def loglike(theta):
    """Normalized two-dimensional standard-normal log likelihood."""

    return -0.5 * jnp.sum(theta**2) - math.log(2.0 * math.pi)


def available_configs() -> list[tuple[str, int]]:
    """Return supported ``(sampler, min_accepts)`` validation configs."""

    configs = [("prior", 1), ("rwalk", 1), ("rwalk", 3)]
    for sample in ("rslice",):
        try:
            NestedSampler(loglike, prior_transform, ndim=NDIM, sample=sample)
        except ValueError:
            continue
        configs.extend([(sample, 1), (sample, 3)])
    return configs


def format_warnings(warnings: list[str]) -> str:
    """Format diagnostics warnings for a compact table cell."""

    return "; ".join(warnings) if warnings else "-"


def main() -> None:
    """Run each supported sampler for seeds 0 through 4 and print a table."""

    print(f"expected logZ: {EXPECTED_LOGZ:.6f}")
    print(f"nlive: {NLIVE}, dlogz: {DLOGZ}")
    print()
    print(
        f"{'sampler':<8} {'min_acc':>7} {'seed':>4} {'logz':>11} {'delta':>11} "
        f"{'logzerr':>9} {'ncall':>8} {'success':>7} warnings"
    )
    print("-" * 96)

    for sample, min_accepts in available_configs():
        for seed in SEEDS:
            sampler = NestedSampler(
                loglike,
                prior_transform,
                ndim=NDIM,
                nlive=NLIVE,
                sample=sample,
                min_accepts=min_accepts,
            )
            result = sampler.run(jax.random.PRNGKey(seed), dlogz=DLOGZ)
            diagnostics = result.diagnostics()
            warnings = diagnostics.get("warnings", [])

            print(
                f"{sample:<8} {min_accepts:7d} {seed:4d} {result.logz:11.6f} "
                f"{result.logz - EXPECTED_LOGZ:11.6f} {result.logzerr:9.6f} "
                f"{result.ncall:8d} {str(result.success):>7} "
                f"{format_warnings(warnings)}"
            )


if __name__ == "__main__":
    main()
