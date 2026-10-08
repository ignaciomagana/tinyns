"""Run the scripts in ``examples/`` and the code blocks of the README."""

from __future__ import annotations

import importlib.util
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _run_example(name: str):
    path = ROOT / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"tinyns_example_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module.main()


def _near_truth(result, truth: float) -> bool:
    return result.success and abs(result.logz - truth) < 5.0 * result.logzerr


def test_examples_are_the_documented_five() -> None:
    names = sorted(path.stem for path in (ROOT / "examples").glob("*.py"))
    assert names == [
        "checkpoint_resume",
        "functional_core",
        "multimodal",
        "quickstart",
        "sbc_batched",
    ]


def test_quickstart_example() -> None:
    module, result = _run_example("quickstart")
    assert _near_truth(result, module.TRUE_LOGZ)


def test_multimodal_example() -> None:
    module, result = _run_example("multimodal")
    assert _near_truth(result, module.TRUE_LOGZ)
    modes = result.modes()
    assert len(modes) == 2
    assert abs(modes[0]["mass"] - 0.75) < 0.1
    assert not any(mode["unresolved"] for mode in modes)


def test_checkpoint_resume_example() -> None:
    _, (first, resumed, straight) = _run_example("checkpoint_resume")
    assert first.niter == 3000 and not first.success
    assert resumed.success and resumed.metadata["resumed"]
    assert resumed.logz == straight.logz
    np.testing.assert_array_equal(resumed.samples_u, straight.samples_u)


def test_sbc_batched_example() -> None:
    module, (results, quantiles) = _run_example("sbc_batched")
    assert len(results) == module.B
    assert all(result.success for result in results)
    assert quantiles.shape == (module.B, 2)
    # 64 lanes: the 90% coverage is 0.90 +- 0.04 for a calibrated sampler.
    inside = np.abs(np.asarray(quantiles) - 0.5) < 0.45
    assert inside.mean() > 0.7


def test_functional_core_example() -> None:
    module, result = _run_example("functional_core")
    assert _near_truth(result, module.TRUE_LOGZ)


def test_readme_code_blocks_run(tmp_path, monkeypatch, capsys) -> None:
    """Every ``python`` block of the README runs, in order, in one namespace."""
    text = (ROOT / "README.md").read_text()
    blocks = re.findall(r"^```python\n(.*?)^```", text, flags=re.M | re.S)
    assert len(blocks) >= 6
    monkeypatch.chdir(tmp_path)  # the snippets write result.npz and run.npz
    namespace: dict = {}
    for block in blocks:
        exec(compile(block, "README.md", "exec"), namespace)
    capsys.readouterr()

    result = namespace["result"]
    assert abs(result.logz + math.log(400.0)) < 5.0 * result.logzerr
    assert len(namespace["results"]) == 16
    assert (tmp_path / "result.npz").exists() and (tmp_path / "run.npz").exists()
