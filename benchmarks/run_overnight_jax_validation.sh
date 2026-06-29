#!/usr/bin/env bash
set -euo pipefail

NLIVE=${NLIVE:-500}
DLOGZ=${DLOGZ:-0.1}
SEEDS=${SEEDS:-"0 1 2 3 4"}
TARGETS=${TARGETS:-"gaussian2d correlated_gaussian2d banana2d ring2d"}
MAXITER=${MAXITER:-10000}
OUTPUT=${OUTPUT:-"benchmarks/results/overnight_jax_validation_$(date +%Y%m%d_%H%M%S).json"}

mkdir -p "$(dirname "${OUTPUT}")"

# Intentionally split SEEDS and TARGETS on shell whitespace so users can pass
# space-separated lists through the environment, e.g. SEEDS="0 1 2".
# shellcheck disable=SC2086
python benchmarks/overnight_jax_validation.py \
  --nlive "${NLIVE}" \
  --dlogz "${DLOGZ}" \
  --maxiter "${MAXITER}" \
  --seeds ${SEEDS} \
  --targets ${TARGETS} \
  --output "${OUTPUT}"

python benchmarks/summarize_overnight_jax_validation.py "${OUTPUT}"
