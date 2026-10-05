"""tinyns: a tiny dynesty-style nested sampler for JAX likelihoods."""

from tinyns.api import NestedSampler
from tinyns.result import LogZBootstrap, NestedSamplingResult

__all__ = [
    "NestedSampler",
    "NestedSamplingResult",
    "LogZBootstrap",
]

__version__ = "0.2.5"
