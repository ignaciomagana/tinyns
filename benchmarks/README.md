# Benchmarks

## Performance benchmarks

A lightweight benchmark harness is available:

```bash
python benchmarks/bench_static.py \
  --targets gaussian2d correlated_gaussian2d \
  --samplers rwalk slice rslice \
  --seeds 0 1 2 \
  --nlive 200 \
  --dlogz 0.1 \
  --output bench.json
```

The benchmark reports wall time, iterations/sec, likelihood calls/sec, replacement cost, and basic diagnostics. It is intended to guide optimization, not to replace validation.
