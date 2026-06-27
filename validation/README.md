# tinyns validation harness

This directory contains a small validation harness for `tinyns`. It is intended
for scientific reliability checks, not speed benchmarking.

The analytic targets check whether repeated nested-sampling runs produce
reasonable evidence estimates and posterior moments. The stress targets, such as
banana and eggbox likelihoods, are qualitative probes that can reveal obvious
sampler failures, mixing problems, or suspicious diagnostics.

Run multiple seeds whenever possible. A single run can look good or bad by
chance, and the reported `logzerr` is only a lightweight nested-sampling
diagnostic; it should not be blindly trusted from one run.

Suggested workflow:

```bash
python validation/run_validation.py --output validation_results.json
python validation/summarize_validation.py validation_results.json
```

## Interpreting validation summaries

The validation harness is meant to detect reliability problems across repeated
seeds. A single `success=True` run does not guarantee calibrated evidence
estimates.

Useful warning signs:

- repeated `|z| > 2` or any `|z| > 3` on analytic targets
- coverage much lower than expected
- high final live-point weight fraction
- high maximum posterior weight fraction
- low posterior weight entropy
- large posterior mean or covariance errors

The recommendation column is heuristic and should be treated as a debugging aid,
not a formal statistical test.

`ring2d` is a qualitative annulus target. It is useful for checking whether constrained-replacement samplers can move around curved shell-like likelihood regions. If no analytic evidence is provided, use posterior diagnostics, insertion-rank behavior, and repeated-run stability rather than z-scores.


`rslice` is available in the validation harness and is a useful comparison
against coordinate-wise `slice`, especially on correlated targets. It remains a
local, simple random-direction constrained slice sampler in the unit cube, not a
full PolyChord-style slice sampler.

## Comparing sampler settings

Use `validation/compare_validation.py` to compare settings such as
`min_accepts=1` and `min_accepts=3`.

A setting is not better merely because it performs more accepted local moves.
Prefer settings that improve coverage, reduce large evidence z-scores, and keep
likelihood-call cost reasonable.

To compare two validation runs, for example `min_accepts=1` versus
`min_accepts=3`:

```bash
python validation/compare_validation.py \
  validation_min_accepts1.json validation_min_accepts3.json \
  --labels min1 min3
```

The comparison table reports changes in coverage, absolute evidence error,
large-z outlier fraction, maximum z-score, and likelihood-call cost. The verdict
column is heuristic and should be treated as a debugging aid.
