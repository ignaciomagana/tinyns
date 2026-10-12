"""Helpers shared by the adapters."""

from __future__ import annotations

import time

import numpy as np


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.s = time.perf_counter() - self.t0


def numpy_callables(target, vectorized: bool):
    """Jitted CPU likelihood callables for the numpy-driven samplers.

    Returns ``(loglike, prior_transform)`` on numpy inputs. With ``vectorized``
    both take an (n, d) batch. The prior transform is the target's affine map
    in numpy; the likelihood is the target's JAX function, jitted (and vmapped
    for batches).
    """
    import jax

    if vectorized:
        f = jax.jit(jax.vmap(target.loglike))

        def loglike(x):
            return np.asarray(f(np.atleast_2d(x)), dtype=float)

        def prior_transform(u):
            return target.prior_transform_np(np.atleast_2d(u))

    else:
        f = jax.jit(target.loglike)

        def loglike(x):
            return float(f(x))

        prior_transform = target.prior_transform_np
    return loglike, prior_transform


def version_of(dist: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(dist)
    except PackageNotFoundError:
        return "unknown"
