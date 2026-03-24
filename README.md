# AI Security & Safety Benchmarking

[![CI](https://github.com/A-Kuo/AI-Security-Benchmarking/actions/workflows/ci.yml/badge.svg)](https://github.com/A-Kuo/AI-Security-Benchmarking/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Research-grade evaluation framework for adversarial robustness, embedding-based anomaly detection, and calibrated LLM safety judging.

Inspired by the [MadData 2026 AmFam Workshop](https://github.com/zachzhou777/MadData-2026-AmFam-Workshop), this project applies the same RAG and LLM evaluation patterns to the domain of AI security.

## Project structure

```
├── notebooks/
│   ├── 01_embedding_anomaly_detection.ipynb   # Embedding-space adversarial detection
│   └── 02_calibrated_safety_judge.ipynb       # LLM-as-judge + defense pipeline ablation
├── src/
│   ├── detection.py       # Centroid, Mahalanobis, isolation forest detectors
│   ├── judge.py           # Safety judge and defense pipeline
│   └── evaluation.py      # Bootstrap CI, Wilson CI, Cohen's kappa
├── .github/workflows/
│   └── ci.yml             # Lint and notebook validation
├── Makefile               # Common tasks: install, run, lint, clean
├── CITATION.cff           # Machine-readable citation metadata
└── pyproject.toml         # Dependencies (uv / Python ≥3.12)
```

**Run order:** notebook 01 first — it creates the ChromaDB collection that notebook 02 extends.

## Quick start

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/A-Kuo/AI-Security-Benchmarking.git
cd AI-Security-Benchmarking
uv sync
cp .env.example .env   # fill in GCP credentials for notebook 02
```

```bash
make run-nb1   # execute notebook 01
make run-nb2   # execute notebook 02 (requires GCP)
```

### Environment variables

Notebook 02 requires Google Cloud (Vertex AI) credentials:

```
GCP_PROJECT=your-project-id
GCP_LOCATION=us-central1
```

## Key techniques

| Technique | Notebook |
|-----------|----------|
| Embedding-based anomaly detection (centroid, Mahalanobis, isolation forest) | 01 |
| ROC/AUC with bootstrap confidence intervals, stratified k-fold CV | 01 |
| Adversarial robustness testing (obfuscation, paraphrase, multilingual) | 01 |
| Automated red teaming — single-turn and multi-turn | 02 |
| LLM-as-a-safety-judge with Cohen's κ calibration | 02 |
| Defense-in-depth pipeline with per-layer ablation (9 configurations) | 02 |
| CWE-aligned vulnerability reports | 02 |
| Wilson score confidence intervals on all benchmark metrics | 02 |

## Methodology

### Notebook 01 — Embedding-Space Anomaly Detection

Formalizes a four-level threat model (L0–L3) and evaluates three unsupervised detection methods against adversarial prompts spanning four attack categories. Evaluation includes Fisher's discriminant ratio, KL divergence per category, and a full ablation across all method combinations.

### Notebook 02 — Calibrated Safety Evaluation

Implements a configurable four-layer defense pipeline (input screening → prompt hardening → LLM judge → retry loop) and measures the Safety Rate / False Refusal Rate tradeoff across 9 ablation configurations. Judge reliability is quantified via multi-invocation Cohen's κ, and vulnerability findings are mapped to the CWE taxonomy.

## References

- Greshake et al. (2023). *Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection.* [arXiv:2302.12173](https://arxiv.org/abs/2302.12173)
- Perez et al. (2022). *Red Teaming Language Models with Language Models.* [arXiv:2202.03286](https://arxiv.org/abs/2202.03286)
- Zheng et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS 2023
- Mazeika et al. (2024). *HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal.* ICML 2024
- Jain et al. (2023). *Baseline Defenses for Adversarial Attacks Against Aligned Language Models.* [arXiv:2309.00614](https://arxiv.org/abs/2309.00614)
- Alon & Kamfonas (2023). *Detecting Language Model Attacks with Perplexity.* [arXiv:2308.14132](https://arxiv.org/abs/2308.14132)
- Landis & Koch (1977). *The Measurement of Observer Agreement for Categorical Data.* Biometrics 33(1)

## Citation

See [`CITATION.cff`](CITATION.cff) for machine-readable citation metadata.

## License

MIT
