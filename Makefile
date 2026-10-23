.PHONY: test dist

test:
	ruff check .
	pytest

# The release build (RELEASE_CHECKLIST.md): the sdist and the wheel, built from
# a clean export of HEAD so that no untracked file ships.
dist:
	@git diff --quiet HEAD || { echo "make dist: uncommitted changes; commit them first" >&2; exit 1; }
	rm -rf build/release-src dist
	mkdir -p build/release-src
	git archive HEAD | tar -x -C build/release-src
	python -m build --outdir dist build/release-src
	python -m twine check --strict dist/*
