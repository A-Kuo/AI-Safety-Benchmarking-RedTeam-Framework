# AGENTS.md — AI Security & Safety Benchmarking

Context file for AI coding assistants. Provides everything needed to understand the project
and continue development without prior context.

---

## Project overview

Research-grade AI security evaluation framework. Two Jupyter notebooks demonstrate
embedding-based adversarial prompt detection (NB1) and calibrated LLM safety judging
with defense pipeline ablation (NB2). Core reusable logic is extracted into `src/`.

Built on the same RAG and LLM evaluation patterns as the
[MadData 2026 AmFam Workshop](https://github.com/zachzhou777/MadData-2026-AmFam-Workshop),
applied to AI security.

---

## Repo layout

```
AI-Security-Benchmarking/
├── notebooks/
│   ├── 01_embedding_anomaly_detection.ipynb    # Embedding-space anomaly detection
│   └── 02_calibrated_safety_judge.ipynb        # LLM-as-judge + defense pipeline
├── src/
│   ├── __init__.py
│   ├── detection.py        # Centroid, Mahalanobis, isolation forest detectors
│   ├── judge.py            # Safety judge + defense pipeline
│   └── evaluation.py       # Bootstrap CI, Wilson CI, Cohen's kappa
├── .github/workflows/
│   └── ci.yml              # Lint + notebook validation
├── pyproject.toml           # uv-managed deps (Python ≥3.12)
├── Makefile                 # make install / run-nb1 / run-nb2 / lint / clean
├── CITATION.cff             # Machine-readable citation metadata
├── .env.example             # Template for required env vars
├── .gitignore
├── README.md                # Human-facing overview
└── AGENTS.md                # This file
```

### Runtime artifacts (gitignored)

- `chroma_security/` — ChromaDB persistent store; created by NB1, extended by NB2

---

## Dependencies

Managed by [`uv`](https://docs.astral.sh/uv/).  Install: `uv sync`

| Package | Purpose |
|---------|---------|
| `chromadb` | Persistent vector store for adversarial prompt embeddings |
| `google-genai` | Gemini API — red teaming and safety judging (NB2) |
| `openai` | OpenAI-compatible client — used to call Nemotron via NVIDIA NIM |
| `transformers` + `torch` | Sentence embeddings via `all-MiniLM-L6-v2` (NB1) |
| `scikit-learn` | Isolation forest, Ledoit-Wolf covariance, cross-validation |
| `scipy` | KL divergence, statistical tests |
| `pydantic` | Structured JSON output schemas |
| `python-dotenv` | `.env` loading for credentials |

---

## Environment variables

Required for NB2 only.  Store in `.env` (gitignored):

```
# Google Gemini (default)
GCP_PROJECT=your-project-id
GCP_LOCATION=us-central1

# NVIDIA Nemotron (optional — free key at build.nvidia.com/settings/api-keys)
NVIDIA_API_KEY=nvapi-your-key-here
LLM_PROVIDER=gemini   # set to "nemotron" to switch providers
```

---

## Notebook 1 — `01_embedding_anomaly_detection.ipynb`

**Mirrors:** `hugging_face_chromadb_demo.ipynb` from the AmFam workshop.

**Goal:** One-class anomaly detection — classify adversarial vs. benign prompts using
embedding-space geometry without labeled training data at inference time.

### Threat model

| Level | Capability | Example |
|-------|-----------|---------|
| L0 | Naive jailbreaks | "Ignore all previous instructions" |
| L1 | Obfuscation | Leetspeak, Base64, Unicode homoglyphs |
| L2 | Semantic reframing | Role-play, hypothetical framing |
| L3 | Multi-step injection | Payload splitting, indirect injection |

Defender constraints: ≤50 ms latency, <5% FPR, online updatability.

### Detection methods

1. **Centroid cosine similarity** — baseline centroid distance
2. **Mahalanobis distance** — Ledoit-Wolf shrinkage for covariance stability
3. **Isolation forest** — one-class, no distributional assumptions

### Evaluation

- ROC/AUC and Precision-Recall curves
- Bootstrap CIs (n=1000) for AUC
- Stratified 5-fold CV for threshold selection
- Fisher's discriminant ratio; KL divergence by category
- Ablation across all 7 method combinations
- Adversarial robustness: obfuscation, paraphrase, multilingual

### ChromaDB output

Collection `attack_patterns` in `chroma_security/`.  Per-document metadata:
`category`, `sophistication`, `centroid_score`, `mahalanobis_score`, `ensemble_score`.

---

## Notebook 2 — `02_calibrated_safety_judge.ipynb`

**Mirrors:** `gemini_rag_pipeline_demo.ipynb` from the AmFam workshop.

**Goal:** Evaluate a 4-layer defense pipeline using a calibrated LLM-as-a-safety-judge.

### Metrics

| Metric | Definition |
|--------|-----------|
| Safety Rate (SR) | P(safe verdict \| unsafe input) |
| Attack Success Rate (ASR) | 1 − SR |
| False Refusal Rate (FRR) | P(refusal \| benign input) |

### Judge calibration

- 3 independent invocations per test case (temperature=0.3)
- Cohen's κ for inter-rater reliability; target κ > 0.6
- Confidence–consistency correlation analysis

### Defense pipeline layers

1. **Input screen** — ChromaDB k-NN threat scoring from NB1
2. **Prompt hardening** — safety policy in system prompt
3. **LLM judge** — structured JSON verdict
4. **Retry** — violation feedback loop

Ablation: 9 layer-subset configurations, measuring SR and FRR tradeoff.

### Additional outputs

- CWE-aligned vulnerability reports (JSON)
- Multi-turn attack sequences (3–5 turn chains)
- Wilson score CIs on all benchmark proportions

---

## `src/` modules

Reusable logic extracted from the notebooks:

| Module | Contents |
|--------|----------|
| `detection.py` | `embed_texts`, `centroid_score`, `mahalanobis_score`, `isolation_score`, model loading |
| `judge.py` | `safety_judge`, `screen_input`, `defense_pipeline`, safety policy definition (Gemini) |
| `llm_client.py` | Nemotron integration: `make_nemotron_client`, `nemotron_safety_judge`, `nemotron_batch_judge`, `nemotron_defense_pipeline` |
| `evaluation.py` | `bootstrap_auc`, `wilson_ci`, `cohens_kappa` |

The notebooks are self-contained (no imports from `src/`), but the modules provide
a testable, importable interface to the same logic.

### Nemotron vs Gemini — when to use each

| Concern | Gemini path (`judge.py`) | Nemotron path (`llm_client.py`) |
|---------|--------------------------|----------------------------------|
| Default / existing notebooks | ✓ | — |
| Reduce consensus-loop token cost (~60 %) | — | `nemotron_batch_judge` (3 verdicts, 1 call) |
| Prevent goal drift across retries | — | `nemotron_defense_pipeline` (message-list history) |
| Very long multi-turn attack sequences | — | Nemotron 1 M-token context window |
| Free tier without GCP setup | — | build.nvidia.com free key |

---

## Security notes

- **No real secrets** exist in this repo.  API keys in NB2's test scenario
  (e.g. `AMFAM-DEMO-KEY-2026`) are intentionally fake — they are part of the
  prompt injection test suite to verify the model won't leak credentials.
- Real GCP credentials live in `.env` which is gitignored.
- The `.env.example` file contains only placeholder values.

---

## Development conventions

- **Python ≥ 3.12** (pinned in `.python-version`)
- **Package manager:** `uv` — add deps with `uv add <pkg>`, never hand-edit `pyproject.toml` for packages
- **Notebooks:** keep cells idempotent; ChromaDB writes should check-before-add to avoid duplicates
- **Secrets:** always use `python-dotenv` + `.env`; never hardcode credentials
- **Before committing:** clear notebook outputs to minimize diff noise
- **ChromaDB collection:** `attack_patterns` — NB1 creates, NB2 extends
- **Path resolution:** both notebooks use `PROJECT_ROOT` (found via `pyproject.toml` traversal) for all file paths

---

## How to continue work

```bash
uv sync                           # install deps
cp .env.example .env              # fill in GCP creds
make run-nb1                      # execute notebook 1
make run-nb2                      # execute notebook 2 (needs GCP)
make lint                         # lint src/ modules
```

If picking up mid-implementation, read the notebook cells to determine which
sections are stubs vs. complete before making changes.

---

## Architecture mapping (workshop → security)

| Original Workshop | This Project | Shared Pattern |
|--------------------|-------------|----------------|
| Embed textbook pages | Embed adversarial prompts | Sentence Transformers |
| t-SNE of OS topics | t-SNE with threat-level overlay | Embedding visualization |
| Custom VectorDatabase | 3-method ensemble detector | Cosine + Mahalanobis + IsoForest |
| ChromaDB for RAG | ChromaDB for threat scoring | Persistent vector store |
| RAG with Gemini | Multi-turn red teaming | LLM-powered generation |
| LLM-as-judge (grounding) | LLM-as-judge (safety) + calibration | Second-LLM validation |
| Structured JSON output | CWE-aligned vulnerability reports | JSON mode / Pydantic |
| Retrieve → Generate → Judge → Retry | Screen → Generate → Judge → Retry + ablation | Multi-layer pipeline |
| — | Bootstrap CIs, ROC/AUC, Cohen's κ | Statistical rigor |
