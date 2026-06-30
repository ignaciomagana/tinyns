.PHONY: test quick-validation overnight-b32 overnight-b16 overnight-comparison summarize-overnight

test:
	ruff check .
	pytest

quick-validation:
	python benchmarks/overnight_jax_validation.py --quick --include-block --output quick_validation.json

overnight-b32:
	python benchmarks/overnight_jax_validation.py \
	  --targets gaussian2d correlated_gaussian2d ring2d banana2d eggbox2d \
	  --seeds 0 1 2 3 4 5 6 7 8 9 \
	  --nlive 500 \
	  --dlogz 0.1 \
	  --maxiter 10000 \
	  --include-block \
	  --jax-block-size 32 \
	  --output overnight_jax_validation_block_B32.json

overnight-b16:
	python benchmarks/overnight_jax_validation.py \
	  --targets gaussian2d correlated_gaussian2d ring2d banana2d eggbox2d \
	  --seeds 0 1 2 3 4 5 6 7 8 9 \
	  --nlive 500 \
	  --dlogz 0.1 \
	  --maxiter 10000 \
	  --include-block \
	  --jax-block-size 16 \
	  --output overnight_jax_validation_block_B16.json

overnight-comparison:
	python benchmarks/overnight_jax_validation.py \
	  --targets gaussian2d correlated_gaussian2d ring2d banana2d eggbox2d \
	  --seeds 0 1 2 3 4 5 6 7 8 9 \
	  --nlive 500 \
	  --dlogz 0.1 \
	  --maxiter 10000 \
	  --include-bounds \
	  --output overnight_jax_validation_no_block.json

summarize-overnight:
	python benchmarks/summarize_overnight_jax_validation.py \
	  overnight_jax_validation_no_block.json \
	  overnight_jax_validation_block_B16.json \
	  overnight_jax_validation_block_B32.json
