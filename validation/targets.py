"""Validation targets for repeated-seed tinyns reliability checks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class ValidationTarget:
    """Small container describing a validation likelihood and expectations."""

    name: str
    ndim: int
    prior_transform: Callable
    loglike: Callable
    expected_logz: float | None = None
    expected_mean: np.ndarray | None = None
    expected_cov: np.ndarray | None = None
    description: str = ""


def constant_cube(ndim: int = 2) -> ValidationTarget:
    """Return a constant likelihood on the unit cube."""

    def prior_transform(u):
        return u

    def loglike(theta):
        del theta
        return 0.0

    return ValidationTarget(
        name=f"constant{ndim}d",
        ndim=ndim,
        prior_transform=prior_transform,
        loglike=loglike,
        expected_logz=0.0,
        expected_mean=np.full(ndim, 0.5),
        expected_cov=np.eye(ndim) / 12.0,
        description="Constant likelihood on the unit cube.",
    )


def gaussian_1d(width: float = 20.0) -> ValidationTarget:
    """Return a normalized 1D standard-normal likelihood under a wide prior."""

    def prior_transform(u):
        return -0.5 * width + width * jnp.asarray(u)

    def loglike(theta):
        x = jnp.asarray(theta)[0]
        return -0.5 * x**2 - 0.5 * jnp.log(2.0 * jnp.pi)

    return ValidationTarget(
        name="gaussian1d",
        ndim=1,
        prior_transform=prior_transform,
        loglike=loglike,
        expected_logz=-float(np.log(width)),
        expected_mean=np.array([0.0]),
        expected_cov=np.array([[1.0]]),
        description="Normalized 1D Gaussian likelihood with a wide uniform prior.",
    )


def gaussian_2d(width: float = 20.0) -> ValidationTarget:
    """Return a normalized 2D standard-normal likelihood under a wide prior."""

    def prior_transform(u):
        return -0.5 * width + width * jnp.asarray(u)

    def loglike(theta):
        theta = jnp.asarray(theta)
        return -0.5 * jnp.sum(theta**2) - jnp.log(2.0 * jnp.pi)

    return ValidationTarget(
        name="gaussian2d",
        ndim=2,
        prior_transform=prior_transform,
        loglike=loglike,
        expected_logz=-float(np.log(width**2)),
        expected_mean=np.zeros(2),
        expected_cov=np.eye(2),
        description="Normalized 2D Gaussian likelihood with a wide uniform prior.",
    )


def correlated_gaussian_2d(width: float = 20.0, rho: float = 0.8) -> ValidationTarget:
    """Return a normalized correlated 2D Gaussian likelihood."""

    cov = np.array([[1.0, rho], [rho, 1.0]])
    inv_cov = jnp.asarray(np.linalg.inv(cov))
    log_norm = -0.5 * float(2 * np.log(2.0 * np.pi) + np.log(np.linalg.det(cov)))

    def prior_transform(u):
        return -0.5 * width + width * jnp.asarray(u)

    def loglike(theta):
        theta = jnp.asarray(theta)
        return -0.5 * theta @ inv_cov @ theta + log_norm

    return ValidationTarget(
        name="correlated_gaussian2d",
        ndim=2,
        prior_transform=prior_transform,
        loglike=loglike,
        expected_logz=-float(np.log(width**2)),
        expected_mean=np.zeros(2),
        expected_cov=cov,
        description="Normalized correlated 2D Gaussian likelihood.",
    )


def gaussian_10d(width: float = 20.0) -> ValidationTarget:
    """Return a normalized 10D standard-normal likelihood under a wide prior."""

    ndim = 10

    def prior_transform(u):
        return -0.5 * width + width * jnp.asarray(u)

    def loglike(theta):
        theta = jnp.asarray(theta)
        return -0.5 * jnp.sum(theta**2) - 0.5 * ndim * jnp.log(2.0 * jnp.pi)

    return ValidationTarget(
        name="gaussian10d",
        ndim=ndim,
        prior_transform=prior_transform,
        loglike=loglike,
        expected_logz=-float(np.log(width**ndim)),
        expected_mean=np.zeros(ndim),
        expected_cov=np.eye(ndim),
        description="Normalized 10D Gaussian likelihood with a wide uniform prior.",
    )


def correlated_gaussian_10d(width: float = 20.0, rho: float = 0.5) -> ValidationTarget:
    """Return a normalized equicorrelated 10D Gaussian likelihood."""

    ndim = 10
    cov = np.full((ndim, ndim), rho)
    np.fill_diagonal(cov, 1.0)
    inv_cov = jnp.asarray(np.linalg.inv(cov))
    log_norm = -0.5 * float(ndim * np.log(2.0 * np.pi) + np.log(np.linalg.det(cov)))

    def prior_transform(u):
        return -0.5 * width + width * jnp.asarray(u)

    def loglike(theta):
        theta = jnp.asarray(theta)
        return -0.5 * theta @ inv_cov @ theta + log_norm

    return ValidationTarget(
        name="correlated_gaussian10d",
        ndim=ndim,
        prior_transform=prior_transform,
        loglike=loglike,
        expected_logz=-float(np.log(width**ndim)),
        expected_mean=np.zeros(ndim),
        expected_cov=cov,
        description="Normalized equicorrelated 10D Gaussian likelihood.",
    )


def banana_2d() -> ValidationTarget:
    """Return a qualitative banana-shaped stress target."""

    def prior_transform(u):
        return -5.0 + 10.0 * jnp.asarray(u)

    def loglike(theta):
        x = theta[0]
        y = theta[1]
        banana = y - 0.2 * (x**2 - 4.0)
        return -0.5 * (x / 1.8) ** 2 - 0.5 * (banana / 0.35) ** 2

    return ValidationTarget(
        name="banana2d",
        ndim=2,
        prior_transform=prior_transform,
        loglike=loglike,
        description="Banana-shaped unnormalized likelihood stress target.",
    )


def ring_2d() -> ValidationTarget:
    """Return a qualitative 2D annulus stress target."""

    r0 = 1.0
    sigma_r = 0.08

    def prior_transform(u):
        return -2.0 + 4.0 * jnp.asarray(u)

    def loglike(theta):
        theta = jnp.asarray(theta)
        x, y = theta
        r = jnp.sqrt(x * x + y * y)
        return -0.5 * ((r - r0) / sigma_r) ** 2

    return ValidationTarget(
        name="ring2d",
        ndim=2,
        prior_transform=prior_transform,
        loglike=loglike,
        expected_logz=None,
        expected_mean=None,
        expected_cov=None,
        description=(
            "2D annulus likelihood in a square prior; qualitative geometry stress test"
        ),
    )


def eggbox_2d() -> ValidationTarget:
    """Return a qualitative multimodal eggbox stress target."""

    def prior_transform(u):
        return u

    def loglike(theta):
        x = 10.0 * jnp.pi * theta[0]
        y = 10.0 * jnp.pi * theta[1]
        return 5.0 * jnp.log(2.0 + jnp.cos(x) * jnp.cos(y))

    return ValidationTarget(
        name="eggbox2d",
        ndim=2,
        prior_transform=prior_transform,
        loglike=loglike,
        description="Multimodal eggbox-like unnormalized likelihood stress target.",
    )


def heavy_gaussian_2d(work_size: int = 100_000) -> ValidationTarget:
    """Return a deterministic catalog-like 2D benchmark likelihood.

    The target is intentionally heavier than the cheap Gaussian validation
    targets: each scalar likelihood evaluation reduces over fixed pseudo-data of
    length ``work_size``. It is benchmark-only and has no analytic evidence.
    """

    if work_size <= 0:
        raise ValueError("work_size must be positive")

    key = jax.random.PRNGKey(12345)
    data = jax.random.normal(key, (int(work_size), 2))
    log_work_size = jnp.log(jnp.asarray(work_size, dtype=data.dtype))

    def prior_transform(u):
        return -5.0 + 10.0 * jnp.asarray(u)

    def loglike(theta):
        theta = jnp.asarray(theta)
        diff = data - theta[None, :]
        log_terms = -0.5 * jnp.sum(diff * diff, axis=1)
        max_log_term = jnp.max(log_terms)
        shifted_sum = jnp.sum(jnp.exp(log_terms - max_log_term))
        return max_log_term + jnp.log(shifted_sum) - log_work_size

    return ValidationTarget(
        name="heavy_gaussian2d",
        ndim=2,
        prior_transform=prior_transform,
        loglike=loglike,
        expected_logz=None,
        expected_mean=None,
        expected_cov=None,
        description=(
            "Deterministic heavy catalog-like 2D Gaussian mixture benchmark target."
        ),
    )


def available_targets() -> dict[str, Callable[[], ValidationTarget]]:
    """Return available named validation target constructors."""

    return {
        "constant2d": constant_cube,
        "gaussian1d": gaussian_1d,
        "gaussian2d": gaussian_2d,
        "correlated_gaussian2d": correlated_gaussian_2d,
        "gaussian10d": gaussian_10d,
        "correlated_gaussian10d": correlated_gaussian_10d,
        "banana2d": banana_2d,
        "ring2d": ring_2d,
        "eggbox2d": eggbox_2d,
        "heavy_gaussian2d": heavy_gaussian_2d,
    }


def get_target(name: str) -> ValidationTarget:
    """Return a validation target by name."""

    targets = available_targets()
    if name not in targets:
        raise KeyError(f"unknown validation target {name!r}")
    return targets[name]()
