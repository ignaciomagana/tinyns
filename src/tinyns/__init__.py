"""tinyns: a tiny, full-JAX nested sampler."""

from tinyns.api import NestedSampler
from tinyns.core import STATUS, Config, delta_logz, finalise, init, step
from tinyns.result import LogZBootstrap, NestedSamplingResult

__all__ = [
    "NestedSampler",
    "Config",
    "init",
    "step",
    "finalise",
    "delta_logz",
    "STATUS",
    "NestedSamplingResult",
    "LogZBootstrap",
]

# The single source of the version: pyproject.toml reads it (hatch dynamic).
__version__ = "1.0.0"
