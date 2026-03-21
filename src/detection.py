"""Embedding-based adversarial prompt detection methods.

Three detection approaches with increasing statistical sophistication:
  1. Centroid cosine similarity
  2. Mahalanobis distance (Ledoit-Wolf covariance shrinkage)
  3. Isolation forest (one-class anomaly detection)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from transformers import AutoModel, AutoTokenizer


def load_embedding_model(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> tuple[AutoTokenizer, AutoModel]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def mean_pooling(
    model_output: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    token_embeddings = model_output[0]
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask_expanded, 1) / torch.clamp(
        mask_expanded.sum(1), min=1e-9
    )


def embed_texts(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    batch_size: int = 16,
) -> np.ndarray:
    all_embeddings: list[torch.Tensor] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model(**encoded)
        embeddings = mean_pooling(outputs, encoded["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings)
    return torch.cat(all_embeddings, dim=0).numpy()


# ---------------------------------------------------------------------------
# Detection methods
# ---------------------------------------------------------------------------


def centroid_score(
    embedding: np.ndarray,
    centroid_benign: np.ndarray,
    centroid_malicious: np.ndarray,
) -> float:
    """Higher value → more likely malicious."""
    return float(embedding @ centroid_malicious - embedding @ centroid_benign)


def fit_mahalanobis(embeddings: np.ndarray) -> LedoitWolf:
    return LedoitWolf().fit(embeddings)


def mahalanobis_score(
    embedding: np.ndarray,
    cov_benign: LedoitWolf,
    cov_malicious: LedoitWolf,
) -> float:
    """Relative Mahalanobis distance.  Higher value → more likely malicious."""
    d_benign = cov_benign.mahalanobis(embedding.reshape(1, -1))[0]
    d_malicious = cov_malicious.mahalanobis(embedding.reshape(1, -1))[0]
    return float(d_benign - d_malicious)


def fit_isolation_forest(
    benign_embeddings: np.ndarray,
    n_estimators: int = 200,
    contamination: float = 0.1,
    random_state: int = 42,
) -> IsolationForest:
    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )
    iso.fit(benign_embeddings)
    return iso


def isolation_score(embedding: np.ndarray, iso_forest: IsolationForest) -> float:
    """Negated anomaly score so higher → more likely malicious."""
    return float(-iso_forest.score_samples(embedding.reshape(1, -1))[0])


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
