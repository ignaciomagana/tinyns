# tinyns release checklist

A release X.Y.Z is built from one commit on `main`. Files on PyPI can never be
replaced, not even after a delete, so every check runs on the exact files that
get uploaded, and the tag goes on last.

## 1. Version and docs

- [ ] `__version__` in `src/tinyns/__init__.py` is X.Y.Z. It is the only copy:
      `pyproject.toml` reads it (hatch dynamic version), so the wheel, the sdist
      and `tinyns.__version__` agree
- [ ] the CHANGELOG heading is `## vX.Y.Z (YYYY-MM-DD)`, dated the release day
- [ ] the README install line pins the tag:
      `git+https://github.com/ignaciomagana/tinyns.git@vX.Y.Z` (once tinyns is
      on PyPI, it leads with `pip install tinyns` instead)
- [ ] the Python classifiers in `pyproject.toml` match the CI matrix
      (`tests/test_packaging.py` checks it)

## 2. Tests

- [ ] `ruff check .`
- [ ] `pytest` with `JAX_ENABLE_X64=0` and with `JAX_ENABLE_X64=1`
- [ ] `pytest -m slow` (the heavy statistical gates of the mode moves)
- [ ] `python bench/validate.py`: the quick tier of the validation gate passes
      (about 6 minutes on 4 CPU cores), and again with `--no-x64`
- [ ] `python bench/validate.py --tier full` passes on a GPU or a Slurm node,
      and again with `--no-x64`
- [ ] every script in `examples/` runs
- [ ] CI is green on the release commit, every job: the newest Python on
      unpinned JAX and the `wheel` job too. If a JAX release came out after the
      last run, rerun it (Actions, CI, "Run workflow")

`bench/README.md` describes the gate's cases and criteria. A failed case is a
3-sigma event under a correct sampler about once in a hundred cases: rerun it
with another key and more seeds (`--cases NAME --seed 1 --seeds 128`) before
calling it a regression.

## 3. Build and inspect

- [ ] `make dist` (needs `python -m pip install build twine`). It builds the
      sdist and the wheel from `git archive HEAD`, never from the working tree,
      into `dist/`, and runs `twine check --strict` on them. It refuses to run
      with uncommitted changes
- [ ] `tar tzf dist/tinyns-X.Y.Z.tar.gz`: only `src/`, `tests/`, `bench/`,
      `examples/`, `README.md`, `CHANGELOG.md`, `LICENSE`, `pyproject.toml`,
      `.gitignore` and `PKG-INFO` (about 60 files, about 250 KB)
- [ ] `python -m zipfile -l dist/tinyns-X.Y.Z-py3-none-any.whl`: only
      `tinyns/` and `tinyns-X.Y.Z.dist-info/` (about 85 KB)
- [ ] in a fresh venv on Python 3.12 or newer (so the newest JAX), install the
      wheel and run the quickstart and the tests against it from the unpacked
      sdist, without its `src/`:

      python3.13 -m venv /tmp/tinyns-rel && . /tmp/tinyns-rel/bin/activate
      python -m pip install dist/tinyns-X.Y.Z-py3-none-any.whl pytest hatchling
      mkdir /tmp/tinyns-sdist
      tar xzf dist/tinyns-X.Y.Z.tar.gz -C /tmp/tinyns-sdist --strip-components=1
      rm -r /tmp/tinyns-sdist/src && cd /tmp/tinyns-sdist
      python examples/quickstart.py
      python -m pytest -o pythonpath=

## 4. Tag

- [ ] tag the commit the files were built from, and push the tag:
      `git tag -a vX.Y.Z -m "tinyns X.Y.Z" <sha>`, then
      `git push origin vX.Y.Z`
- [ ] a GitHub Release for the tag, with the CHANGELOG entry as its notes and
      the sdist and the wheel attached

### Moving a tag

A tag may move only while nothing of that version is on PyPI (v1.0.0 was pushed
at 8851058 before the review fixes, and moves to the fixed commit):

    git fetch origin --tags
    git tag -f -a vX.Y.Z -m "tinyns X.Y.Z" <new sha>
    git push --force origin refs/tags/vX.Y.Z
    git ls-remote origin 'vX.Y.Z^{}'   # prints <new sha>, the tagged commit

If a GitHub Release exists for the tag, check that it shows the new commit and
replace its attached files. A clone that fetched the old tag keeps it until
`git fetch --tags --force`. `pip install git+...@vX.Y.Z` resolves the tag
again on every install: pip caches wheels built from git only for commit
hashes.

## 5. PyPI (later)

If the upload comes long after the tag, rebuild from the tag (`git checkout
vX.Y.Z`, `make dist`) and repeat section 3 on the new files.

- [ ] rehearse on TestPyPI: `python -m twine upload --repository testpypi
      dist/*`, then in a fresh venv
      `python -m pip install --index-url https://test.pypi.org/simple/
      --extra-index-url https://pypi.org/simple/ tinyns==X.Y.Z` and run the
      quickstart. TestPyPI refuses a second upload of a version too: rehearse
      a retry as X.Y.Zrc1, rc2, ...
- [ ] `python -m twine upload dist/*`: the same files that section 3 checked
- [ ] in a fresh venv, `python -m pip install tinyns==X.Y.Z` and run the
      quickstart
- [ ] README: lead with `pip install tinyns`; the git line becomes the
      development version (`git+https://github.com/ignaciomagana/tinyns.git`)

A tag-triggered workflow with PyPI trusted publishing would move the upload
from a laptop to CI; it needs a pending publisher configured on PyPI first.
