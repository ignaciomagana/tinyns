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


`rslice` is available in the validation harness and is a useful comparison against coordinate-wise `slice`, especially on correlated targets. It remains a local, simple random-direction constrained slice sampler in the unit cube, not a full PolyChord-style slice sampler.
