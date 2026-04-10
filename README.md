# AI Safety Benchmarking & Red Team Framework

**Systematic adversarial evaluation for production AI systems**

[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange.svg)](https://jupyter.org)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Status](https://img.shields.io/badge/Status-Beta-green.svg)]()

> *"AI safety is not a checklist you complete before shipping. It's a continuous adversarial process — and someone needs to play the adversary."*

---

## What This Is

A structured framework for **red-teaming AI systems** — discovering failure modes, measuring robustness, and documenting vulnerabilities before they affect users.

Built initially in collaboration with American Family Insurance, this framework generalizes to:
- Language model safety evaluation (toxicity, bias, jailbreaks)
- Structured output validation (API schemas, JSON generation)
- Decision system robustness (classification, scoring, ranking)
- Multi-agent system stability

---

## Core Philosophy

**Red teaming is not about finding edge cases. It's about finding *characteristic* failures — the kinds of inputs that reliably expose model limitations.**

This framework provides:

| Component | Purpose |
|-----------|---------|
| **Attack Taxonomy** | Structured categorization of failure types (not just "prompt injection") |
| **Automated Probing** | Systematic generation of adversarial inputs |
| **Failure Clustering** | Group similar failures to identify root causes |
| **Severity Scoring** | Calibrated assessment of exploitability and impact |
| **Regression Testing** | Re-run evaluations as models update |

---

## Attack Taxonomy

Instead of ad-hoc testing, we categorize attacks by mechanism:

```
┌─────────────────────────────────────────────────────────────────┐
│                    ATTACK TAXONOMY                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. SEMANTIC MANIPULATION                                       │
│     • Instruction override ("Ignore previous instructions...")  │
│     • Context window poisoning                                  │
│     • Encoding obfuscation (base64, rot13, etc.)                │
│     • Semantic gap exploitation                                 │
│                                                                 │
│  2. STRUCTURE ATTACKS                                           │
│     • Schema injection in structured outputs                    │
│     • Delimiter confusion                                       │
│     • Recursive formatting attacks                             │
│                                                                 │
│  3. KNOWLEDGE EXPLOITATION                                      │
│     • Hallucination amplification                               │
│     • Confidence manipulation                                   │
│     • Training data extraction                                  │
│                                                                 │
│  4. MULTI-TURN COERCION                                         │
│     • Gradual escalation chains                                 │
│     • Persona adoption                                        │
│     • Reframing and refractory attacks                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Framework Structure

```
ai-safety-redteam/
├── taxonomy/               # Attack categorization and definitions
├── generators/             # Adversarial input generation
│   ├── prompt_based/       # Template-driven attacks
│   ├── gradient_based/     # Optimization-based (if white-box)
│   └── llm_driven/         # Recursive LLM-generated attacks
├── evaluators/             # Success/failure detection
│   ├── content_policy/     # Toxicity, PII, restricted content
│   ├── structural/         # JSON schema validation
│   └── semantic/           # Output meaning verification
├── clustering/             # Failure pattern analysis
├── reporting/              # Severity scoring and documentation
└── regression/             # Continuous evaluation pipelines
```

---

## Usage Example

```python
from redteam.framework import RedTeamEvaluator
from redteam.taxonomy import SemanticManipulation

# Initialize evaluator for your model endpoint
evaluator = RedTeamEvaluator(
    model_endpoint="https://api.my-llm.com/v1/completions",
    taxonomy=SemanticManipulation(),
    evaluator_suite=["content_policy", "structural"]
)

# Run evaluation
results = evaluator.evaluate(
    num_probes=1000,
    attack_types=["instruction_override", "encoding_obfuscation"],
    severity_threshold="medium"
)

# Analyze failures
failure_clusters = results.cluster_failures(method="semantic_embedding")
for cluster in failure_clusters:
    print(f"Cluster {cluster.id}: {cluster.description}")
    print(f"Severity: {cluster.severity}")
    print(f"Example: {cluster.canonical_example}")
```

---

## Severity Scoring

Not all failures are equal. We score on two dimensions:

**Exploitability (E):** How easy is it to trigger?
- E1: Requires insider knowledge or model weights
- E2: Requires crafted input with domain expertise
- E3: Simple prompt engineering
- E4: Accidental triggering (common input)

**Impact (I):** What happens if triggered?
- I1: Minor output quality degradation
- I2: Incorrect but benign output
- I3: Harmful output (toxicity, misinformation)
- I4: System compromise or data exfiltration

**Overall Severity:** Matrix product E × I, mapped to Critical/High/Medium/Low.

---

## Integration with MLOps

```python
# As part of CI/CD pipeline
from redteam.regression import RegressionTestSuite

suite = RegressionTestSuite.from_baseline("v1.2.3_baseline.json")
new_results = suite.run_against("https://staging-api.my-llm.com")

delta = suite.compare(new_results)
if delta.has_new_critical_failures():
    raise DeploymentBlocked("New critical vulnerabilities detected")
```

---

## Research Context

This work connects to broader AI safety efforts:

- **Anthropic's Constitutional AI** — Our taxonomy extends their harm categories with structural attack vectors
- **OpenAI's Evals framework** — Complementary: they focus on capability evaluation, we focus on adversarial robustness
- **Purple Llama (Meta)** — Similar goals, different methodology: we emphasize systematic taxonomy and severity scoring

Related work in this portfolio:
- [Language-Model-Hallucination-Detection-via-Entropy-Divergence](https://github.com/A-Kuo/Language-Model-Hallucination-Detection-via-Entropy-Divergence) — Uncertainty quantification for safety
- [CIPHER](https://github.com/A-Kuo/CIPHER) — Cryptographic analysis of AI-generated content

---

## Current Status

**Beta (April 2026)**

- ✅ Core taxonomy defined (4 major categories, 12 subcategories)
- ✅ Automated prompt-based attack generation
- ✅ Content policy evaluators (toxicity, PII detection)
- ✅ JSON schema validation evaluators
- ✅ Basic clustering using semantic embeddings
- 🔄 Gradient-based attacks (requires white-box access)
- 🔄 Multi-turn conversation evaluation
- ⏸️ Automated severity scoring calibration (requires human rater data)

---

## Citation

```bibtex
@software{ai_safety_redteam_2026,
  author = {A-Kuo},
  title = {AI Safety Benchmarking and Red Team Framework},
  url = {https://github.com/A-Kuo/AI-Safety-Benchmarking-RedTeam-Framework},
  year = {2026},
  note = {Developed in collaboration with American Family Insurance}
}
```

---

*Find failures before they find you. April 2026.*
