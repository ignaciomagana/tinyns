"""Callables of the compiled kernels: pytree partition, constant hoisting, caches.

``loglike`` and ``prior_transform`` enter the compiled kernels of
:mod:`tinyns.core` as a data-free :class:`_CallableSpec` (part of the kernel
cache key) plus their array leaves (jit arguments). Pytree callables split into
their array leaves and their static structure; closures and plain functions
have their large jaxpr constants hoisted, and are cached by identity through a
weak reference. The kernel builders are cached on the specs
(:func:`_kernel_cache`), so a cache never holds a dataset.
"""

from __future__ import annotations

import functools
import logging
import weakref

import jax
import jax.numpy as jnp
import numpy as np

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

    The jit boundaries that use this split are the kernels of
    :mod:`tinyns.core`: the live-point pass of ``init``, the block of ``step``
    and the live-cov chain inside it.
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
    in_avals = [(leaf.shape, leaf.dtype) for leaf in jax.tree_util.tree_leaves(example)]
    in_tree = jax.tree_util.tree_structure((example,))
    out_tree = jax.tree_util.tree_structure(out_shape)

    def rebuild(leaves):
        all_consts = list(consts)
        for i, leaf in zip(big, leaves, strict=True):
            all_consts[i] = leaf

        def hoisted_fn(*args, **kwargs):
            flat, tree = jax.tree_util.tree_flatten(args)
            if (
                kwargs
                or tree != in_tree
                or [(jnp.shape(x), jnp.result_type(x)) for x in flat] != in_avals
            ):
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


# The compiled-kernel builders of tinyns.core, cached on the callables' specs
# and their static arguments. JAX's own jit cache keys the compiled programs on
# the shapes and dtypes of the array arguments.
_KERNEL_CACHES: list = []


def _kernel_cache(build):
    """Cache a kernel builder ``build(loglike_spec, prior_spec, *static)``."""
    cached = functools.lru_cache(maxsize=32)(build)
    _KERNEL_CACHES.append(cached)
    return cached


def _clear_caches() -> None:
    """Drop every compiled kernel and every identity split."""
    for cached in _KERNEL_CACHES:
        cached.cache_clear()
    _IDENTITY_SPLITS.clear()


def _device_leaves(fn):
    """Return ``fn`` with its numpy array leaves placed on the device.

    The host loop calls this once per run, so that the compiled kernels are
    not handed a host array (and a transfer) at every block.
    """
    leaves, treedef = jax.tree_util.tree_flatten(fn)
    if not any(isinstance(leaf, np.ndarray) for leaf in leaves):
        return fn
    leaves = [jnp.asarray(x) if isinstance(x, np.ndarray) else x for x in leaves]
    return jax.tree_util.tree_unflatten(treedef, leaves)
