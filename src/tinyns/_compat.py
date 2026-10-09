"""Shims for old JAX. tinyns supports JAX from 0.4.31; from 0.4.38 on nothing
here touches JAX's internals."""

from __future__ import annotations

import jax
from jax import lax

if jax.__version_info__ >= (0, 4, 38):
    optimization_barrier = lax.optimization_barrier
else:
    # JAX before 0.4.38 has no batching rule for the barrier, and the chains
    # call it under vmap. The barrier is the identity, so the rule binds the
    # batched operands and keeps their batch dimensions (the rule that JAX
    # registers itself from 0.4.38 on).
    from jax.interpreters import batching

    try:
        from jax._src.lax.lax import optimization_barrier_p as _barrier_p
    except ImportError:
        # 0.4.31: lax.optimization_barrier is new in 0.4.32; the primitive
        # and its function are in ad_checkpoint.
        from jax._src import ad_checkpoint

        _barrier_p = ad_checkpoint.optimization_barrier_p
        optimization_barrier = ad_checkpoint._optimization_barrier
    else:
        optimization_barrier = lax.optimization_barrier

    def _batch_barrier(batched_args, batch_dims, **params):
        return _barrier_p.bind(*batched_args, **params), batch_dims

    batching.primitive_batchers.setdefault(_barrier_p, _batch_barrier)
