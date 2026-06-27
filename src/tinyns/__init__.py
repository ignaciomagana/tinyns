"""tinyns: a tiny dynesty-style nested sampler for JAX likelihoods."""

from tinyns.api import NestedSampler
from tinyns.result import NestedSamplingResult

__all__ = ["NestedSampler", "NestedSamplingResult"]

__version__ = "0.1.0"
