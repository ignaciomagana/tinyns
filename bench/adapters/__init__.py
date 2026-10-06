"""One adapter per sampler, all with ``run(target, seed, cfg) -> dict``.

``cfg`` holds ``nlive``, ``dlogz``, ``variant`` and ``opts`` (free-form
sampler options from ``run.py --opt key=value``). The returned dict has:

``samples``      (n, d) physical-space samples (weighted, or equal-weight)
``logwt``        (n,) log weights of ``samples``, or None for equal weights
``logz``, ``logzerr``  the sampler's own estimates (``logzerr`` may be None)
``ncall``        likelihood evaluations the sampler reports (or the adapter counts)
``ncall_valid``  evaluations that were not wasted lanes, when known
``wall_s``       wall time of the whole run, compilation included
``compile_s``    compilation time inside ``wall_s`` when it can be separated
``sampler_version``, ``config`` (the resolved sampler settings)

A sampler spec is ``name`` or ``name:variant`` (``dynesty:rwalk100``).
``DEVICE`` is where a sampler runs by default: ``gpu`` samplers use the default
JAX backend, ``cpu`` samplers get ``JAX_PLATFORMS=cpu`` (their likelihood calls
are one point or one small batch at a time, so a device round trip would
dominate).
"""

from __future__ import annotations

import importlib
import importlib.util

# name: (module, default device, distribution the adapter imports)
REGISTRY = {
    "tinyns_v02": ("bench.adapters.tinyns_v02", "gpu", "tinyns"),
    "tinyns_v1": ("bench.adapters.tinyns_v1", "gpu", "tinyns"),
    "blackjax_nss": ("bench.adapters.blackjax_nss", "gpu", "blackjax"),
    "dynesty": ("bench.adapters.dynesty_sampler", "cpu", "dynesty"),
    "ultranest": ("bench.adapters.ultranest_sampler", "cpu", "ultranest"),
    "nautilus": ("bench.adapters.nautilus_sampler", "cpu", "nautilus"),
    "jaxns": ("bench.adapters.jaxns_sampler", "gpu", "jaxns"),
}


class AdapterUnavailable(RuntimeError):
    """The sampler (or the right version of it) is not installed."""


def parse_spec(spec: str) -> tuple[str, str]:
    name, _, variant = spec.partition(":")
    if name not in REGISTRY:
        raise ValueError(f"unknown sampler {name!r}; known: {sorted(REGISTRY)}")
    module = load(name)
    variant = variant or "default"
    if variant not in module.VARIANTS:
        raise ValueError(f"{name} has variants {module.VARIANTS}, not {variant!r}")
    return name, variant


def load(name: str):
    return importlib.import_module(REGISTRY[name][0])


def device(name: str) -> str:
    return REGISTRY[name][1]


def unavailable_reason(name: str) -> str | None:
    """None when the adapter can run here, else why not."""
    package = REGISTRY[name][2]
    if importlib.util.find_spec(package) is None:
        return f"{package} is not installed"
    return load(name).unavailable_reason()
