# tinyns release checklist

- [ ] `ruff check .`
- [ ] `pytest` with `JAX_ENABLE_X64=0` and with `JAX_ENABLE_X64=1`
- [ ] `pytest -m slow` (the heavy statistical gates of the mode moves)
- [ ] `python bench/validate.py`: the quick tier of the validation gate passes
      (about 6 minutes on 4 CPU cores)
- [ ] `python bench/validate.py --tier full` passes on a GPU or a Slurm node,
      and again with `--no-x64`
- [ ] every script in `examples/` runs
- [ ] the version in `pyproject.toml` and the CHANGELOG entry agree

`bench/README.md` describes the gate's cases and criteria. A failed case is a
3-sigma event under a correct sampler about once in a hundred cases: rerun it
with another key and more seeds (`--cases NAME --seed 1 --seeds 128`) before
calling it a regression.
