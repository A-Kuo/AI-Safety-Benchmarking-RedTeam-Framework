.PHONY: install install-dev test lint format run-nb1 run-nb2 run-all clean help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime dependencies via uv
	uv sync

install-dev:  ## Install all dependencies including dev tools
	uv sync --group dev

test:  ## Run full test suite with coverage
	uv run pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

test-fast:  ## Run tests excluding detection (no torch startup cost)
	uv run pytest tests/test_evaluation.py tests/test_judge.py -v --tb=short

lint:  ## Lint src/ and tests/ with ruff
	uv run ruff check src/ tests/

format:  ## Auto-format src/ and tests/ with ruff
	uv run ruff format src/ tests/

format-check:  ## Check formatting without applying changes
	uv run ruff format --check src/ tests/

run-nb1:  ## Execute notebook 1 (embedding anomaly detection)
	uv run jupyter nbconvert --to notebook --execute notebooks/01_embedding_anomaly_detection.ipynb \
		--output-dir=notebooks --ExecutePreprocessor.timeout=600

run-nb2:  ## Execute notebook 2 (calibrated safety judge) — requires GCP creds
	uv run jupyter nbconvert --to notebook --execute notebooks/02_calibrated_safety_judge.ipynb \
		--output-dir=notebooks --ExecutePreprocessor.timeout=900

run-all: run-nb1 run-nb2  ## Execute both notebooks in order

clean:  ## Remove generated artifacts
	rm -rf chroma_security/ .pytest_cache .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
