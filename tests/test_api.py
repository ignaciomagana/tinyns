"""The public surface: exports, NestedSampler arguments, the result it returns."""

from __future__ import annotations

import inspect
import math

import jax.numpy as jnp
import numpy as np
import pytest

import tinyns
from tinyns import Config, NestedSampler, NestedSamplingResult


def loglike(x):
    return -0.5 * jnp.sum(((x - 0.5) / 0.1) ** 2)


def identity(u):
    return u


def test_public_exports() -> None:
    for name in ("NestedSampler", "Config", "init", "step", "finalise",
                 "NestedSamplingResult"):
        assert name in tinyns.__all__ and hasattr(tinyns, name)


def test_signatures() -> None:
    init = inspect.signature(NestedSampler.__init__).parameters
    assert list(init) == ["self", "loglike", "prior_transform", "ndim", "nlive",
                          "num_delete", "walks"]
    assert init["nlive"].default == 1000
    assert init["num_delete"].kind is inspect.Parameter.KEYWORD_ONLY
    run = inspect.signature(NestedSampler.run).parameters
    assert list(run) == ["self", "key", "dlogz", "maxiter", "maxcall", "progress"]
    assert run["dlogz"].default == 0.1


@pytest.mark.parametrize(
    "kwargs", [{"block_size": 8}, {"replacement_chains": 2}, {"bound": "multi"}]
)
def test_unknown_options_raise_type_error(kwargs) -> None:
    with pytest.raises(TypeError):
        NestedSampler(loglike, identity, 2, **kwargs)
    with pytest.raises(TypeError):
        NestedSampler(loglike, identity, 2, 20).run(0, checkpoint_path="x", **kwargs)


def test_sampler_validates_its_arguments() -> None:
    with pytest.raises(TypeError):
        NestedSampler(None, identity, 2)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        NestedSampler(loglike, None, 2)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        NestedSampler(loglike, identity, 0)
    with pytest.raises(ValueError):
        NestedSampler(loglike, identity, 2, nlive=20, num_delete=11)
    sampler = NestedSampler(loglike, identity, 2, 20)
    with pytest.raises(ValueError):
        sampler.run(0, dlogz=-1.0)
    with pytest.raises(ValueError):
        sampler.run(0, maxiter=-1)
    assert sampler.config == Config(2, 20, 2, 25)


def test_run_returns_a_complete_result(capsys) -> None:
    result = NestedSampler(loglike, lambda u: 2.0 * u, 2, 60).run(0, progress=True)
    assert isinstance(result, NestedSamplingResult)
    n = result.niter + 60
    assert result.samples_u.shape == (n, 2) and result.samples.shape == (n, 2)
    np.testing.assert_allclose(result.samples, 2.0 * np.asarray(result.samples_u),
                               rtol=1e-6)
    for name in ("logl", "logwt", "logl_birth", "nlive_i"):
        assert np.asarray(getattr(result, name)).shape == (n,)
    assert (result.nlive, result.num_delete, result.ndim) == (60, 6, 2)
    assert result.niter % 6 == 0 and math.isfinite(result.logz)
    truth = math.log(2 * math.pi * 0.01 / 4)  # the prior box is [0, 2]^2
    assert abs(result.logz - truth) < 4 * result.logzerr + 0.05
    md = result.metadata
    assert md["status"] == "converged" and md["walks"] == 25
    assert md["ncall_valid"] <= result.ncall and 0 < md["acceptance"] < 1
    assert "tinyns: ndead=" in capsys.readouterr().out
    assert "num_delete: 6" in result.summary()
    assert result.resample_equal(np.asarray([0, 1], np.uint32), 10).shape == (10, 2)


def test_scalar_prior_transform_in_one_dimension() -> None:
    result = NestedSampler(lambda x: -0.5 * ((x - 0.5) / 0.1) ** 2,
                           lambda u: u[0], 1, 40).run(1)
    assert result.samples.shape == (result.niter + 40, 1)
    assert math.isfinite(result.logz)
