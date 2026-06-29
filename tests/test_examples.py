from __future__ import annotations

import importlib.util
from pathlib import Path


def _import_example(module_name: str):
    path = Path(__file__).resolve().parents[1] / "examples" / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gaussian_2d_rwalk_jax_example_imports() -> None:
    module = _import_example("gaussian_2d_rwalk_jax")

    assert module.NestedSampler is not None
    assert module.prior_transform(module.jnp.array([0.0, 1.0])).shape == (2,)


def test_gaussian_2d_rwalk_jax_block_example_imports() -> None:
    module = _import_example("gaussian_2d_rwalk_jax_block")

    assert module.NestedSampler is not None
    assert module.prior_transform(module.jnp.array([0.0, 1.0])).shape == (2,)
