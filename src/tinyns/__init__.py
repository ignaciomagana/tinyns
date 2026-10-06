"""tinyns: a tiny, full-JAX nested sampler."""

from tinyns.api import NestedSampler
from tinyns.core import Config, finalise, init, step
from tinyns.result import LogZBootstrap, NestedSamplingResult

__all__ = [
    "NestedSampler",
    "Config",
    "init",
    "step",
    "finalise",
    "NestedSamplingResult",
    "LogZBootstrap",
]

__version__ = "1.0.0.dev0"
