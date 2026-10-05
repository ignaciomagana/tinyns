"""Shared test helpers."""

from __future__ import annotations

from tinyns import NestedSampler

_SAMPLER_OPTIONS = ("walks", "replacement_chains", "block_size", "cluster_swap")


def run_ns(key, loglike, prior_transform, ndim, nlive, **kwargs):
    """``NestedSampler(...).run(key, ...)``, with the options split by kind."""
    options = {name: kwargs.pop(name) for name in _SAMPLER_OPTIONS if name in kwargs}
    sampler = NestedSampler(loglike, prior_transform, ndim, nlive, **options)
    return sampler.run(key, **kwargs)
