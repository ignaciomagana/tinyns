"""The package metadata and the sdist's file list."""

from __future__ import annotations

from pathlib import Path

import pytest

import tinyns

ROOT = Path(__file__).resolve().parents[1]


def test_sdist_ships_only_the_listed_paths(tmp_path) -> None:
    """Untracked files in the build tree (agent worktrees under .claude/, run
    outputs) stay out of the sdist."""
    pytest.importorskip("hatchling")
    from hatchling.builders.sdist import SdistBuilder

    (tmp_path / "pyproject.toml").write_bytes((ROOT / "pyproject.toml").read_bytes())
    shipped = [
        "src/tinyns/__init__.py",
        "tests/test_api.py",
        "bench/validate.py",
        "bench/tests/test_bench.py",
        "examples/quickstart.py",
        "README.md",
        "CHANGELOG.md",
        "LICENSE",
    ]
    junk = [
        ".claude/magana_ci_env.tgz",
        ".claude/worktrees/wt/src/tinyns/__init__.py",
        "result.npz",
        "run.npz",
        "results.jsonl",
        "gate.json",
        "scratch/notes.py",
    ]
    for rel in shipped + junk:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'__version__ = "{tinyns.__version__}"\n')
    builder = SdistBuilder(str(tmp_path))
    files = {Path(f.relative_path).as_posix() for f in builder.recurse_included_files()}
    assert set(shipped) | {"pyproject.toml"} <= files
    assert not files & set(junk)
