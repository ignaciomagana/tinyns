"""tinyns: a tiny dynesty-style nested sampler for JAX likelihoods."""

from tinyns.api import NestedSampler
from tinyns.result import NestedSamplerResult

__all__ = ["NestedSampler", "NestedSamplerResult"]

__version__ = "0.1.0"
