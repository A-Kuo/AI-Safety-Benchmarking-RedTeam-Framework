.PHONY: install run-nb1 run-nb2 run-all lint clean help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies via uv
	uv sync

run-nb1:  ## Execute notebook 1 (embedding anomaly detection)
	jupyter nbconvert --to notebook --execute notebooks/01_embedding_anomaly_detection.ipynb \
		--output-dir=notebooks --ExecutePreprocessor.timeout=600

run-nb2:  ## Execute notebook 2 (calibrated safety judge) — requires GCP creds
	jupyter nbconvert --to notebook --execute notebooks/02_calibrated_safety_judge.ipynb \
		--output-dir=notebooks --ExecutePreprocessor.timeout=900

run-all: run-nb1 run-nb2  ## Execute both notebooks in order

lint:  ## Lint src/ modules
	uv run ruff check src/

clean:  ## Remove generated artifacts
	rm -rf chroma_security/ __pycache__ .ipynb_checkpoints
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
