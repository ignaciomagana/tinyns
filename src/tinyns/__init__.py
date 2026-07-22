"""tinyns: a tiny dynesty-style nested sampler for JAX likelihoods."""

from tinyns.api import NestedSampler
from tinyns.result import LogZBootstrap, NestedSamplingResult
from tinyns.run import run_static_nested

__all__ = [
    "NestedSampler",
    "NestedSamplingResult",
    "LogZBootstrap",
    "run_static_nested",
]

__version__ = "0.1.0"
