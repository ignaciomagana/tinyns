.PHONY: test

test:
	ruff check .
	pytest
