"""The live-cov rwalk replacement kernel and callable splitting for :mod:`tinyns`."""

from __future__ import annotations

import functools
import logging
import weakref
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, random

from tinyns.clusters import loo_frames, swap_step

try:  # jaxpr evaluation moved between JAX versions; hoisting is optional.
    from jax.extend.core import eval_jaxpr as _eval_jaxpr
except ImportError:
    try:
        from jax.core import eval_jaxpr as _eval_jaxpr
    except ImportError:
        _eval_jaxpr = None
try:
    from jax.core import Tracer as _Tracer
except ImportError:
    _Tracer = ()

_logger = logging.getLogger(__name__)

# Closure constants with at least this many elements become jit arguments on
# the compiled kernels (see _hoist_closure_consts); smaller ones stay embedded.
_HOIST_MIN_SIZE = 4096


class _CallableSpec:
    """Hashable, data-free stand-in for a callable of the compiled kernels.

    The compiled kernels are cached on specs, not on callables. A
    kernel takes the callable's ``nleaves`` dynamic leaves as jit arguments and
    ``rebuild(leaves)`` returns the callable inside the trace. The spec of a
    pytree callable is keyed on its structure, the static part from
    :func:`_partition_callable`: pytrees that differ only in their array leaves
    (``Partial(loglike_fn, data_i)`` over mock datasets) share one kernel, and
    JAX's jit cache keys the compiled program on the leaves' shapes and dtypes.
    Any other callable gets a spec of its own (``key=None``: compared by
    identity) that holds the callable only weakly.
    """

    __slots__ = ("key", "nleaves", "rebuild")

    def __init__(self, nleaves: int, rebuild, key=None):
        self.nleaves = int(nleaves)
        self.rebuild = rebuild
        self.key = object() if key is None else key

    def __hash__(self) -> int:
        return hash(self.key)

    def __eq__(self, other) -> bool:
        return isinstance(other, _CallableSpec) and self.key == other.key


def _weak_ref(obj, callback):
    """Return ``weakref.ref(obj, callback)``, or a strong reference if unsupported."""
    try:
        return weakref.ref(obj, callback)
    except TypeError:  # kept alive, as every callable was before v0.2.3
        return lambda: obj


def _is_dynamic_leaf(leaf) -> bool:
    return isinstance(leaf, (jax.Array, np.ndarray))


def _partition_callable(fn):
    """Split a pytree callable into ``(dynamic, static)`` (eqx.partition-style).

    ``dynamic`` is the tuple of array leaves (``jax.Array`` / ``np.ndarray``) of
    ``jax.tree_util.tree_flatten(fn)``; ``static`` holds the treedef and every
    other leaf. The compiled kernels take ``dynamic`` as explicit jit
    arguments and rebuild ``fn`` inside the trace with
    :func:`_combine_callable`, so large data are not embedded as constants.

    A plain function or closure flattens to ``[fn]``: ``dynamic`` is empty and
    :func:`_combine_callable` returns ``fn`` itself (closure semantics). Such
    callables have their large constants hoisted instead
    (:func:`_split_callable`).

    The jit boundaries that use this split are the rwalk kernel
    (:func:`_make_rwalk_jax_kernel_cached`), the block kernel and the initial
    live-point pass in :mod:`tinyns.run`.
    """
    leaves, treedef = jax.tree_util.tree_flatten(fn)
    mask = tuple(_is_dynamic_leaf(leaf) for leaf in leaves)
    dynamic = tuple(leaf for leaf in leaves if _is_dynamic_leaf(leaf))
    static_leaves = tuple(None if _is_dynamic_leaf(leaf) else leaf for leaf in leaves)
    return dynamic, (treedef, mask, static_leaves)


def _combine_callable(dynamic, static):
    """Rebuild the callable split by :func:`_partition_callable`."""
    treedef, mask, static_leaves = static
    dynamic = tuple(dynamic)
    if len(dynamic) != sum(mask):
        raise ValueError(
            f"expected {sum(mask)} dynamic callable leaves, got {len(dynamic)}"
        )
    it = iter(dynamic)
    leaves = [
        next(it) if is_dyn else leaf
        for leaf, is_dyn in zip(static_leaves, mask, strict=True)
    ]
    return jax.tree_util.tree_unflatten(treedef, leaves)


def _hoist_closure_consts(ref, example):
    """Trace the callable ``ref()`` on ``example`` and lift its large constants.

    Returns ``(hoisted, rebuild)`` or ``None``. ``hoisted`` holds the jaxpr
    constants with at least ``_HOIST_MIN_SIZE`` elements, as device arrays;
    ``rebuild(hoisted)`` returns a callable that evaluates the traced jaxpr
    with them (smaller constants stay embedded, so they still fold). A call
    whose arguments do not match ``example`` runs the callable itself.
    ``rebuild`` keeps the jaxpr, the small constants and ``ref`` (a weak
    reference), never the callable or the hoisted arrays. ``None`` (closure
    semantics) means there is nothing to hoist or tracing failed.
    """
    if _eval_jaxpr is None:
        return None
    fn = ref()
    try:
        closed, out_shape = jax.make_jaxpr(fn, return_shape=True)(example)
        consts = list(closed.consts)
        if any(isinstance(c, _Tracer) for c in consts):
            return None  # traced under an outer transformation: do not cache
        jaxpr = closed.jaxpr
        constvars = jaxpr.constvars
        big = [
            i
            for i, (c, var) in enumerate(zip(consts, constvars, strict=True))
            if np.size(c) >= _HOIST_MIN_SIZE and jnp.shape(c) == var.aval.shape
        ]
        hoisted = tuple(
            jnp.asarray(consts[i], dtype=constvars[i].aval.dtype) for i in big
        )
    except Exception as exc:  # noqa: BLE001 - any failure keeps closure semantics
        _logger.debug("tinyns: not hoisting constants of %r: %r", fn, exc)
        return None
    if not big:
        return None
    for i in big:
        consts[i] = None  # keep only the small constants alive here
    in_avals = [
        (leaf.shape, leaf.dtype) for leaf in jax.tree_util.tree_leaves(example)
    ]
    in_tree = jax.tree_util.tree_structure((example,))
    out_tree = jax.tree_util.tree_structure(out_shape)

    def rebuild(leaves):
        all_consts = list(consts)
        for i, leaf in zip(big, leaves, strict=True):
            all_consts[i] = leaf

        def hoisted_fn(*args, **kwargs):
            flat, tree = jax.tree_util.tree_flatten(args)
            if kwargs or tree != in_tree or [
                (jnp.shape(x), jnp.result_type(x)) for x in flat
            ] != in_avals:
                return ref()(*args, **kwargs)
            out = _eval_jaxpr(jaxpr, all_consts, *flat)
            return jax.tree_util.tree_unflatten(out_tree, out)

        return hoisted_fn

    return hoisted, rebuild


# Splits of callables that are not keyed on their structure (closures, plain
# functions), cached by identity: (id(fn), key) -> (weakref to fn, split).
# A split never references its callable and the weakref callback drops the
# entry when the callable is garbage collected, so no dataset outlives it.
_IDENTITY_SPLITS: dict = {}


def _split_callable(fn, key, example):
    """Return ``(dynamic, spec)`` with ``spec.rebuild(dynamic)`` the callable.

    A pytree callable with array leaves splits into those leaves
    (:func:`_partition_callable`) and a spec keyed on everything else,
    including the types of the static leaves (``1`` and ``1.0`` trace
    differently). Nothing is cached, so the leaves are not retained.

    A callable without array leaves (a plain function or closure) is traced on
    ``example()`` (unless it returns ``None``) and its large jaxpr constants
    become ``dynamic`` (:func:`_hoist_closure_consts`); otherwise ``dynamic``
    is empty and ``spec.rebuild`` returns the callable itself. That split is
    cached per callable identity and ``key`` (a tuple of plain values that
    determines ``example()``) for as long as the callable is alive, so the
    hoisted leaves and the spec are the same objects on every call.
    """
    dynamic, static = _partition_callable(fn)
    if dynamic:
        spec_key = (*static, tuple(type(leaf) for leaf in static[2]))
        try:
            hash(spec_key)
        except TypeError:
            pass  # unhashable static leaves: split by identity below
        else:
            rebuild = functools.partial(_combine_callable, static=static)
            return dynamic, _CallableSpec(len(dynamic), rebuild, spec_key)
    cache_key = (id(fn), key)
    entry = _IDENTITY_SPLITS.get(cache_key)
    if entry is not None and entry[0]() is fn:
        return entry[1]
    # Bound now: the callback may run at interpreter shutdown, after globals.
    ref = _weak_ref(fn, lambda _, pop=_IDENTITY_SPLITS.pop: pop(cache_key, None))
    if dynamic:
        rebuild = functools.partial(_combine_callable, static=static)
        split = dynamic, _CallableSpec(len(dynamic), rebuild)
    else:
        example = example()
        hoisted = None if example is None else _hoist_closure_consts(ref, example)
        if hoisted is None:
            split = (), _CallableSpec(0, lambda leaves: ref())
        else:
            split = hoisted[0], _CallableSpec(len(hoisted[0]), hoisted[1])
    _IDENTITY_SPLITS[cache_key] = (ref, split)
    return split


def _split_callables(loglike, prior_transform, ndim: int):
    """Return ``((dynamic, spec), (dynamic, spec))`` for the compiled kernels.

    ``loglike`` is traced on a theta of shape ``(ndim,)`` with the dtype
    ``prior_transform`` produces, ``prior_transform`` on a u of shape
    ``(ndim,)`` (see :func:`_split_callable`).
    """
    u_example = jax.ShapeDtypeStruct((int(ndim),), jnp.result_type(float))
    key = (int(ndim), u_example.dtype.name)

    def theta_example():
        try:
            theta_shape = jax.eval_shape(prior_transform, u_example)
            theta_dtype = jnp.result_type(*jax.tree_util.tree_leaves(theta_shape))
        except Exception:  # noqa: BLE001 - loglike then keeps closure semantics
            return None
        return jax.ShapeDtypeStruct((int(ndim),), theta_dtype)

    return (
        _split_callable(loglike, key, theta_example),
        _split_callable(prior_transform, key, lambda: u_example),
    )


_split_callables.cache_clear = _IDENTITY_SPLITS.clear


def _callable_leaves(loglike, prior_transform, ndim: int):
    """Return the dynamic leaves of ``loglike`` then ``prior_transform``."""
    (loglike_dynamic, _), (prior_dynamic, _) = _split_callables(
        loglike, prior_transform, ndim
    )
    return tuple(loglike_dynamic) + tuple(prior_dynamic)


def _callable_specs(loglike, prior_transform, ndim: int):
    """Return the :class:`_CallableSpec` of ``loglike`` and ``prior_transform``."""
    (_, loglike_spec), (_, prior_spec) = _split_callables(
        loglike, prior_transform, ndim
    )
    return loglike_spec, prior_spec


def _evaluate_jax_prior_batch(prior_transform, u_batch, ndim: int):
    """Evaluate a scalar JAX prior transform on a batch (``jax.vmap``)."""
    u_batch = jnp.asarray(u_batch)
    if u_batch.ndim != 2 or u_batch.shape[1] != ndim:
        raise ValueError(f"u_batch must have shape (batch, {ndim})")
    nbatch = int(u_batch.shape[0])
    theta_batch = jnp.asarray(jax.vmap(prior_transform)(u_batch))
    if ndim == 1 and theta_batch.shape == (nbatch,):
        theta_batch = theta_batch.reshape((nbatch, 1))
    if theta_batch.shape != (nbatch, ndim):
        raise ValueError(f"prior_transform must return shape ({ndim},)")
    return theta_batch


def _evaluate_jax_batch(loglike, prior_transform, u_batch, ndim):
    """Evaluate scalar JAX prior/likelihood functions on a unit-cube batch."""
    u_batch = jnp.asarray(u_batch)
    if u_batch.ndim != 2 or u_batch.shape[1] != ndim:
        raise ValueError(f"u_batch must have shape (batch, {ndim})")
    nbatch = int(u_batch.shape[0])
    theta_batch = _evaluate_jax_prior_batch(prior_transform, u_batch, ndim)
    logl_batch = jnp.asarray(jax.vmap(loglike)(theta_batch))
    if logl_batch.shape != (nbatch,):
        raise ValueError("loglike must return a scalar")
    return theta_batch, logl_batch


def live_cov_cholesky(live_u):
    """Return the Cholesky factor of the live-point covariance (unit cube).

    A small relative jitter keeps near-degenerate live sets factorable; if the
    factorization still fails, the per-axis standard deviations are used.
    """
    live_u = jnp.asarray(live_u)
    nlive, ndim = live_u.shape
    centered = live_u - jnp.mean(live_u, axis=0)
    cov = centered.T @ centered / max(nlive - 1, 1)
    mean_var = jnp.maximum(jnp.trace(cov) / ndim, jnp.finfo(cov.dtype).tiny)
    jitter = 10.0 * jnp.finfo(cov.dtype).eps * mean_var
    cov = cov + jitter * jnp.eye(ndim, dtype=cov.dtype)
    chol = jnp.linalg.cholesky(cov)
    fallback = jnp.diag(jnp.sqrt(jnp.diag(cov)))
    return jnp.where(jnp.all(jnp.isfinite(chol)), chol, fallback)


@lru_cache(maxsize=32)
def _make_rwalk_jax_kernel_cached(
    loglike_spec,
    prior_spec,
    ndim: int,
    walks: int,
    replacement_chains: int,
    cluster_swap: bool = False,
):
    """Return a cached compiled live-cov rwalk replacement kernel.

    The block kernel of :mod:`tinyns.run` calls it once per iteration. The
    returned jitted ``kernel`` always produces an eight-element tuple, in this
    order:

    1. ``key`` -- the advanced PRNG key.
    2. ``new_u`` -- the accepted replacement point in unit-cube coordinates.
    3. ``new_theta`` -- the prior-transformed replacement point.
    4. ``new_logl`` -- the log-likelihood of ``new_theta``.
    5. ``ncall`` -- likelihood evaluations actually made. This is batches x
       walks x ``replacement_chains``, except that a single chain skips (and
       does not count) proposals that leave the unit cube.
    6. ``accepted`` -- whether a chain ended inside the constraint before the
       ``max_batches`` budget was exhausted.
    7. ``accepted_move_count`` -- accepted proposals summed over all chains and
       batches (rwalk-acceptance numerator).
    8. ``total_proposal_count`` -- total proposals attempted (batches x walks x
       ``replacement_chains``), the rwalk-acceptance denominator.

    Each step is ``scale * L N(0, I)`` with ``L`` the Cholesky factor of the
    live-point covariance, so the step follows the contracting, correlated
    live set. Moves that leave the cube are rejected. Chains start from live
    points strictly above ``logl_min`` whenever any exist. A batch runs
    ``replacement_chains`` chains of ``walks`` steps and keeps one chain that
    ends inside the constraint; an unmoved chain is kept as a copy of its seed
    (its seed is strictly above ``logl_min``). Only if no chain qualifies does
    the kernel run another batch, up to ``max_batches``.

    The callables arrive as :class:`_CallableSpec` objects (``loglike_spec``,
    ``prior_spec``), so the cache holds no data. Their array leaves (pytree
    callables) or large constants (closures, see :func:`_split_callable`) are
    trailing arguments of ``kernel``: call it as ``kernel(key, ...,
    max_batches, *_callable_leaves(loglike, prior_transform, ndim))`` and the
    callables are rebuilt inside the trace, so the arrays are jit arguments
    rather than baked-in constants. Callables with neither take no trailing
    arguments.

    ``cluster_swap=True`` (single chain only) makes a fraction of the chain
    steps affine swaps between cluster frames (see :mod:`tinyns.clusters`).
    The kernel then takes the cluster frames as an extra argument before the
    callable leaves and returns a ninth output, ``[accepted swaps, proposed
    swaps]``; the swap steps are included in ``accepted_move_count`` and
    ``total_proposal_count``.
    """

    # With one chain, out-of-cube proposals skip the likelihood (lax.cond);
    # under vmap a cond would evaluate both branches.
    skip_out_of_cube = replacement_chains == 1
    if cluster_swap and not skip_out_of_cube:
        raise ValueError("cluster_swap needs a single replacement chain")
    nloglike_leaves = loglike_spec.nleaves
    nleaves = loglike_spec.nleaves + prior_spec.nleaves

    @jax.jit
    def kernel(
        key,
        logl_min,
        live_u,
        live_logl,
        scale,
        max_batches,
        *callable_leaves,
    ):
        if cluster_swap:
            clusters, *callable_leaves = callable_leaves
        if len(callable_leaves) != nleaves:
            raise ValueError(
                f"kernel expects {nleaves} callable array leaves, "
                f"got {len(callable_leaves)}"
            )
        loglike = loglike_spec.rebuild(callable_leaves[:nloglike_leaves])
        prior_transform = prior_spec.rebuild(callable_leaves[nloglike_leaves:])
        chol = live_cov_cholesky(live_u)
        # Never restart a chain from the point being replaced when others exist.
        above = live_logl > logl_min
        seed_logits = jnp.where(jnp.any(above), jnp.where(above, 0.0, -jnp.inf), 0.0)
        template_u = live_u[0]
        template_theta = _evaluate_jax_prior_batch(
            prior_transform, template_u[None, :], ndim
        )[0]
        initial_best_logl = jnp.asarray(-jnp.inf, dtype=live_logl.dtype)
        initial_ncall = jnp.asarray(0, dtype=jnp.int32)
        initial_done = jnp.asarray(False)
        initial_batch_index = jnp.asarray(0, dtype=jnp.int32)
        initial_accepted_move_count = jnp.asarray(0, dtype=jnp.int32)
        batch_ncall = jnp.asarray(walks * replacement_chains, dtype=jnp.int32)

        def cond(state):
            return (~state[2]) & (state[11] < max_batches)

        def body(state):
            (
                key,
                ncall,
                _done,
                _accepted,
                out_u,
                out_theta,
                out_logl,
                best_u,
                best_theta,
                best_logl,
                accepted_move_count,
                batch_index,
                n_evals,
                *swap_counts,
            ) = state

            key, seed_key = random.split(key)
            seed_idx = random.categorical(
                seed_key, seed_logits, shape=(replacement_chains,)
            )
            current_u = live_u[seed_idx]
            current_theta = _evaluate_jax_prior_batch(prior_transform, current_u, ndim)
            current_logl = live_logl[seed_idx]
            if cluster_swap:
                frames = loo_frames(clusters, seed_idx[0], current_u[0])
            attempt_best_u = current_u
            attempt_best_theta = current_theta
            attempt_best_logl = jnp.full(
                (replacement_chains,), -jnp.inf, live_logl.dtype
            )
            accepted_moves = jnp.zeros((replacement_chains,), dtype=jnp.int32)
            batch_evals = jnp.asarray(0, dtype=jnp.int32)

            def one_step(carry, _):
                (
                    key,
                    current_u,
                    current_theta,
                    current_logl,
                    attempt_best_u,
                    attempt_best_theta,
                    attempt_best_logl,
                    accepted_moves,
                    batch_evals,
                ) = carry
                key, proposal_key = random.split(key)
                z = random.normal(proposal_key, shape=(replacement_chains, ndim))
                u_raw = current_u + scale * (z @ chol.T)
                if cluster_swap:
                    swap_u, swap_ok, is_swap = swap_step(
                        random.fold_in(proposal_key, 1), current_u[0], frames
                    )
                    u_raw = jnp.where(is_swap, swap_u[None, :], u_raw)
                in_cube = jnp.all((u_raw >= 0.0) & (u_raw <= 1.0), axis=1)
                if cluster_swap:
                    # A swap that fails its likelihood-free tests is rejected
                    # without a call, like an out-of-cube step.
                    in_cube = in_cube & (swap_ok | ~is_swap)
                u_prop = jnp.clip(u_raw, 0.0, 1.0)
                if skip_out_of_cube:

                    def evaluate(u):
                        theta, logl = _evaluate_jax_batch(
                            loglike, prior_transform, u, ndim
                        )
                        return (
                            theta.astype(current_theta.dtype),
                            logl.astype(current_logl.dtype),
                        )

                    def skip(u):
                        del u
                        return current_theta, jnp.full_like(current_logl, -jnp.inf)

                    theta_prop, logl_prop = lax.cond(in_cube[0], evaluate, skip, u_prop)
                    batch_evals = batch_evals + in_cube[0].astype(jnp.int32)
                else:
                    theta_prop, logl_prop = _evaluate_jax_batch(
                        loglike, prior_transform, u_prop, ndim
                    )
                    batch_evals = batch_evals + jnp.asarray(
                        replacement_chains, dtype=jnp.int32
                    )

                # Out-of-cube moves never count as the fallback best point.
                logl_prop = jnp.where(in_cube, logl_prop, -jnp.inf)
                is_best = logl_prop > attempt_best_logl
                attempt_best_u = jnp.where(is_best[:, None], u_prop, attempt_best_u)
                attempt_best_theta = jnp.where(
                    is_best[:, None], theta_prop, attempt_best_theta
                )
                attempt_best_logl = jnp.where(is_best, logl_prop, attempt_best_logl)

                # Out-of-cube moves are rejected even when logl_min is -inf.
                accept = (logl_prop >= logl_min) & in_cube
                current_u = jnp.where(accept[:, None], u_prop, current_u)
                current_theta = jnp.where(accept[:, None], theta_prop, current_theta)
                current_logl = jnp.where(accept, logl_prop, current_logl)
                accepted_moves = accepted_moves + accept.astype(jnp.int32)
                return (
                    key,
                    current_u,
                    current_theta,
                    current_logl,
                    attempt_best_u,
                    attempt_best_theta,
                    attempt_best_logl,
                    accepted_moves,
                    batch_evals,
                ), (jnp.stack([is_swap & accept[0], is_swap]) if cluster_swap else None)

            (
                (
                    key,
                    current_u,
                    current_theta,
                    current_logl,
                    attempt_best_u,
                    attempt_best_theta,
                    attempt_best_logl,
                    accepted_moves,
                    batch_evals,
                ),
                swap_steps,
            ) = lax.scan(
                one_step,
                (
                    key,
                    current_u,
                    current_theta,
                    current_logl,
                    attempt_best_u,
                    attempt_best_theta,
                    attempt_best_logl,
                    accepted_moves,
                    batch_evals,
                ),
                xs=None,
                length=walks,
            )

            batch_best_idx = jnp.argmax(attempt_best_logl)
            batch_best_logl = attempt_best_logl[batch_best_idx]
            is_global_best = batch_best_logl > best_logl
            best_u = jnp.where(is_global_best, attempt_best_u[batch_best_idx], best_u)
            best_theta = jnp.where(
                is_global_best, attempt_best_theta[batch_best_idx], best_theta
            )
            best_logl = jnp.where(is_global_best, batch_best_logl, best_logl)

            # A chain succeeds if it ends inside the constraint. An unmoved
            # chain is kept as a copy of its seed, provided the seed is
            # strictly above logl_min; discarding unmoved chains would
            # under-sample hard-to-move regions.
            success_mask = jnp.where(
                accepted_moves > 0, current_logl >= logl_min, current_logl > logl_min
            )
            any_success = jnp.any(success_mask)
            key, select_key = random.split(key)
            selection_scores = jnp.where(
                success_mask, random.uniform(select_key, (replacement_chains,)), -1.0
            )
            selected_idx = jnp.argmax(selection_scores)
            out_u = jnp.where(any_success, current_u[selected_idx], out_u)
            out_theta = jnp.where(any_success, current_theta[selected_idx], out_theta)
            out_logl = jnp.where(any_success, current_logl[selected_idx], out_logl)
            # dtype-pinned accumulation: under JAX_ENABLE_X64 the bare
            # jnp.sum promotes to int64 while the while_loop carry was
            # initialized int32, which aborts the kernel with a carry-type
            # mismatch (state[10] int32 vs int64).
            ncall = ncall + jnp.asarray(batch_ncall, dtype=ncall.dtype)
            accepted_move_count = accepted_move_count + jnp.sum(
                accepted_moves, dtype=accepted_move_count.dtype
            )
            batch_index = batch_index + jnp.asarray(1, dtype=jnp.int32)
            n_evals = n_evals + batch_evals.astype(n_evals.dtype)
            if cluster_swap:
                swap_counts = [
                    swap_counts[0] + jnp.sum(swap_steps, axis=0, dtype=jnp.int32)
                ]
            return (
                key,
                ncall,
                any_success,
                any_success,
                out_u,
                out_theta,
                out_logl,
                best_u,
                best_theta,
                best_logl,
                accepted_move_count,
                batch_index,
                n_evals,
                *swap_counts,
            )

        (
            key,
            ncall,
            done,
            accepted,
            out_u,
            out_theta,
            out_logl,
            best_u,
            best_theta,
            best_logl,
            accepted_move_count,
            _batch_index,
            n_evals,
            *swap_counts,
        ) = lax.while_loop(
            cond,
            body,
            (
                key,
                initial_ncall,
                initial_done,
                initial_done,
                template_u,
                template_theta,
                initial_best_logl,
                template_u,
                template_theta,
                initial_best_logl,
                initial_accepted_move_count,
                initial_batch_index,
                jnp.asarray(0, dtype=jnp.int32),
                *([jnp.zeros(2, dtype=jnp.int32)] if cluster_swap else []),
            ),
        )
        new_u = jnp.where(done, out_u, best_u)
        new_theta = jnp.where(done, out_theta, best_theta)
        new_logl = jnp.where(done, out_logl, best_logl)
        return (
            key,
            new_u,
            new_theta,
            new_logl,
            n_evals,
            accepted,
            accepted_move_count,
            ncall,
            *swap_counts,
        )

    return kernel
