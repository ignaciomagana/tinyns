"""Adapter timing and stopping: what counts as compilation, where BlackJAX
stops, the padded batches of the CPU likelihood, and the headline's cells."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def test_vectorized_likelihood_compiles_once_per_power_of_two():
    """A batch is padded to the next power of two, so 1 to 33 points take 7
    compiles (UltraNest asks for 56 to 59 batch sizes in one run), and the
    padding never reaches the result."""
    import jax.numpy as jnp
    from bench.adapters._common import numpy_callables

    traces = []

    def loglike(x):
        traces.append(x.shape)  # runs once per compile
        return -0.5 * jnp.sum(x**2)

    target = SimpleNamespace(loglike=loglike, prior_transform_np=lambda u: u)
    f, _ = numpy_callables(target, vectorized=True)
    rng = np.random.default_rng(0)
    for n in range(1, 34):
        x = rng.normal(size=(n, 3))
        assert np.allclose(f(x), -0.5 * np.sum(x**2, axis=1))
    assert len(traces) == 7
    assert f(np.ones(3)).shape == (1,)  # one point, as UltraNest sends it
