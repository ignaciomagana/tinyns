"""The package metadata, the sdist's file list and the CI matrix agree."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import pytest

import tinyns

tomllib = pytest.importorskip("tomllib")  # Python 3.11+

ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / ".github" / "workflows" / "ci.yml"
# The oldest Python that the newest JAX supports (JAX >= 0.11 needs 3.12).
NEWEST_JAX_MIN_PYTHON = 12


def _pyproject() -> dict:
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)


def _python_minors(classifiers: list[str]) -> list[int]:
    pattern = r"Programming Language :: Python :: 3\.(\d+)"
    return sorted(
        int(m.group(1)) for c in classifiers if (m := re.fullmatch(pattern, c))
    )


def _ci_matrix() -> list[tuple[int, str, str]]:
    """The test job's (python minor, x64, jax) entries."""
    if not CI.exists():  # an unpacked sdist has no .github/
        pytest.skip("no .github/workflows/ci.yml")
    entries = re.findall(
        r'- python-version: "3\.(\d+)"\s+x64: "([01])"\s+jax: "([^"]+)"',
        CI.read_text(),
    )
    assert entries, "no matrix entries found in ci.yml"
    return [(int(minor), x64, jax) for minor, x64, jax in entries]


def _ci_step_script(name: str) -> str:
    """The ``run: |`` script of the ci.yml step called ``name``."""
    lines = CI.read_text().splitlines()
    (start,) = [i for i, line in enumerate(lines) if line.strip() == f"- name: {name}"]
    end = start + 1
    while lines[end].strip() != "run: |":
        assert not lines[end].strip().startswith("- name:"), f"{name!r} has no script"
        end += 1
    body = lines[end + 1 :]
    indent = len(body[0]) - len(body[0].lstrip())
    script = []
    for line in body:
        if line.strip() and len(line) - len(line.lstrip()) < indent:
            break
        script.append(line[indent:])
    return "\n".join(script).rstrip("\n") + "\n"


def test_version_has_one_source() -> None:
    """The wheel and the sdist read their version from ``tinyns.__version__``."""
    config = _pyproject()
    assert "version" not in config["project"]
    assert config["project"]["dynamic"] == ["version"]
    assert config["tool"]["hatch"]["version"]["path"] == "src/tinyns/__init__.py"


def test_installed_metadata_has_the_version() -> None:
    """An installed wheel (the CI wheel job) carries ``tinyns.__version__``."""
    try:
        dist = metadata.distribution("tinyns")
    except metadata.PackageNotFoundError:
        pytest.skip("tinyns is not installed")
    installed = Path(dist.locate_file("tinyns/__init__.py"))
    if not (installed.exists() and installed.samefile(tinyns.__file__)):
        pytest.skip("the imported tinyns is not an installed wheel")
    assert dist.version == tinyns.__version__


def test_metadata() -> None:
    """A license expression, a 1.x status, and no redundant jaxlib pin."""
    config = _pyproject()
    project = config["project"]
    # An SPDX expression with the file listed (PEP 639); hatchling reads it
    # from 1.27 on, and a License classifier next to it is deprecated.
    assert project["license"] == "MIT"
    assert project["license-files"] == ["LICENSE"]
    assert config["build-system"]["requires"] == ["hatchling>=1.27"]
    classifiers = project["classifiers"]
    assert not [c for c in classifiers if c.startswith("License ::")]
    assert "Development Status :: 5 - Production/Stable" in classifiers  # 1.x
    # jax pins its own matching jaxlib.
    names = [re.split(r"[<>=!~; \[]", d)[0] for d in project["dependencies"]]
    assert names == ["jax", "numpy"]


def test_python_classifiers_match_requires_python() -> None:
    project = _pyproject()["project"]
    minors = _python_minors(project["classifiers"])
    assert minors == list(range(minors[0], minors[-1] + 1))
    assert project["requires-python"] == f">=3.{minors[0]}"


def test_ci_tests_every_python_and_the_newest_jax() -> None:
    """CI runs each Python the classifiers list, the JAX floor, and the newest
    JAX unpinned on every Python that the newest JAX supports."""
    matrix = _ci_matrix()
    minors = _python_minors(_pyproject()["project"]["classifiers"])
    assert sorted({minor for minor, _, _ in matrix}) == minors
    for minor in minors:
        if minor >= NEWEST_JAX_MIN_PYTHON:
            assert (minor, "0", "latest") in matrix, f"no py3.{minor} jax latest job"
        else:
            # JAX >= 0.7 needs Python 3.11 and >= 0.11 needs 3.12, so "latest"
            # on these is an old JAX: such jobs are pinned and named for it.
            assert all(jax != "latest" for m, _, jax in matrix if m == minor)
    floor = re.fullmatch(r"jax>=(\S+)", _pyproject()["project"]["dependencies"][0])
    assert floor is not None
    for x64 in "01":
        assert (minors[0], x64, floor.group(1)) in matrix


def test_ci_runs_the_wheel_and_a_weekly_schedule() -> None:
    _ci_matrix()  # skips without ci.yml
    text = CI.read_text()
    test_job, wheel_job = re.split(r"^  wheel:$", text, flags=re.M)
    assert "python -m build" in wheel_job
    assert "python -m pytest -o pythonpath=" in wheel_job  # not src/: the wheel
    for job in (test_job, wheel_job):  # a failure log names the JAX it ran on
        assert "jaxlib.__version__" in job
    assert re.search(r'schedule:\s+- cron: "[^"]+"', text)


def test_ci_newest_jax_check_only_warns(tmp_path) -> None:
    """The "latest is the newest JAX" step warns when the JAX is not the newest
    or PyPI cannot be read, and never fails the job (GitHub runs it with
    ``bash -e``)."""
    _ci_matrix()  # skips without ci.yml
    bash, curl = shutil.which("bash"), shutil.which("curl")
    if not (bash and curl):
        pytest.skip("needs bash and curl")
    import jax

    url = "https://pypi.org/pypi/jax/json"
    script = _ci_step_script('Check that "latest" is the newest JAX')
    assert url in script
    script = script.replace("${{ matrix.python-version }}", "3.99")
    assert "${{" not in script
    # `python` on PATH is this interpreter, as in the job; curl reads file://
    # URLs, so nothing goes to the network.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    (bin_dir / "python").chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}

    def run(body: str | None, flags: str = "-e") -> str:
        source = tmp_path / "jax.json"
        source.unlink(missing_ok=True)
        if body is not None:
            source.write_text(body)
        proc = subprocess.run(
            [bash, *flags.split(), "-c", script.replace(url, source.as_uri())],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        return proc.stdout

    assert "::warning::" not in run(f'{{"info": {{"version": "{jax.__version__}"}}}}')
    stale = run('{"info": {"version": "999.0"}}')
    assert f"resolves jax {jax.__version__}, but the newest JAX is 999.0" in stale
    # PyPI down (curl fails) or an error page instead of JSON, also with the
    # `shell: bash` flags.
    for flags in ("-e", "-e -o pipefail"):
        for body in (None, "<html>503</html>"):
            assert "::warning::could not read the newest JAX" in run(body, flags)


def test_sdist_ships_only_the_listed_paths(tmp_path) -> None:
    """Files outside the listed top-level paths (agent worktrees under
    .claude/, run outputs at the root) stay out of a sdist built from a working
    tree."""
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
