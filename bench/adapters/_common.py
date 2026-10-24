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

    A jitted function compiles once per input shape, and the samplers ask for
    batches of many sizes (UltraNest's MLFriends 56 to 59 in one run), so a
    batch is padded to the next power of two with copies of its first row
    and the padding is sliced off: at most ``log2(n) + 1`` compiles.
    """
    import jax

    if vectorized:
        f = jax.jit(jax.vmap(target.loglike))

        def loglike(x):
            x = np.atleast_2d(x)
            n = len(x)
            size = 1 << max(n - 1, 0).bit_length()
            if size > n:
                x = np.concatenate([x, np.repeat(x[:1], size - n, axis=0)])
            return np.asarray(f(x), dtype=float)[:n]

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
