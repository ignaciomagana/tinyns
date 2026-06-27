from __future__ import annotations

import importlib.util
from pathlib import Path


def test_gaussian_2d_rwalk_jax_example_imports() -> None:
    path = Path(__file__).resolve().parents[1] / "examples" / "gaussian_2d_rwalk_jax.py"
    spec = importlib.util.spec_from_file_location("gaussian_2d_rwalk_jax", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.NestedSampler is not None
    assert module.prior_transform(module.jnp.array([0.0, 1.0])).shape == (2,)
