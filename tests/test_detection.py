"""Unit tests for src.detection — embedding and scoring methods."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from src.detection import (
    centroid_score,
    embed_texts,
    fit_isolation_forest,
    fit_mahalanobis,
    isolation_score,
    mahalanobis_score,
    mean_pooling,
    normalize_scores,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def rng():
    return np.random.default_rng(42)


@pytest.fixture()
def benign_embeddings(rng):
    """Small synthetic benign cluster."""
    return rng.normal(loc=0.5, scale=0.1, size=(30, 32)).astype(np.float32)


@pytest.fixture()
def malicious_embeddings(rng):
    """Small synthetic malicious cluster, clearly separated."""
    return rng.normal(loc=-0.5, scale=0.1, size=(20, 32)).astype(np.float32)


@pytest.fixture()
def centroids(benign_embeddings, malicious_embeddings):
    cb = benign_embeddings.mean(axis=0)
    cm = malicious_embeddings.mean(axis=0)
    cb /= np.linalg.norm(cb)
    cm /= np.linalg.norm(cm)
    return cb, cm


# ---------------------------------------------------------------------------
# mean_pooling
# ---------------------------------------------------------------------------


def test_mean_pooling_shape():
    batch, seq_len, hidden = 4, 10, 32
    token_emb = torch.ones(batch, seq_len, hidden)
    model_output = MagicMock()
    model_output.__getitem__ = lambda self, idx: token_emb
    mask = torch.ones(batch, seq_len)
    result = mean_pooling(model_output, mask)
    assert result.shape == (batch, hidden)


def test_mean_pooling_masked_tokens():
    """Masked-out tokens should not contribute to the mean."""
    token_emb = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    model_output = MagicMock()
    model_output.__getitem__ = lambda self, idx: token_emb
    mask = torch.tensor([[1, 0]])
    result = mean_pooling(model_output, mask)
    assert result.shape == (1, 2)
    assert float(result[0, 0]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# embed_texts
# ---------------------------------------------------------------------------


def test_embed_texts_shape_and_normalization():
    """embed_texts should return L2-normalized vectors of the right shape."""
    texts = ["hello world", "adversarial prompt test"]
    hidden = 32

    # Build a mock BatchEncoding-like object that supports ** unpacking
    attention_mask = torch.ones(len(texts), 5, dtype=torch.long)
    mock_encoded = MagicMock()
    mock_encoded.keys.return_value = ["input_ids", "attention_mask"]
    mock_encoded.__getitem__ = lambda self, k: (
        torch.zeros(len(texts), 5, dtype=torch.long) if k == "input_ids" else attention_mask
    )
    # Support **unpacking via keys() + __getitem__
    mock_encoded.__iter__ = lambda self: iter(["input_ids", "attention_mask"])

    # Mock model output: shape (batch, seq_len, hidden)
    fake_token_emb = torch.rand(len(texts), 5, hidden)
    mock_output = MagicMock()
    mock_output.__getitem__ = lambda self, idx: fake_token_emb

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = mock_encoded

    mock_model = MagicMock()
    mock_model.return_value = mock_output
    mock_model.eval = MagicMock()

    result = embed_texts(texts, mock_tokenizer, mock_model)

    assert result.shape[0] == len(texts)
    assert result.shape[1] == hidden
    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(norms, np.ones(len(texts)), atol=1e-5)


# ---------------------------------------------------------------------------
# centroid_score
# ---------------------------------------------------------------------------


def test_centroid_score_ordering(benign_embeddings, malicious_embeddings, centroids):
    """Benign embeddings should score lower (less malicious) than malicious ones."""
    cb, cm = centroids
    benign_scores = [centroid_score(e, cb, cm) for e in benign_embeddings]
    malicious_scores = [centroid_score(e, cb, cm) for e in malicious_embeddings]
    assert np.mean(benign_scores) < np.mean(malicious_scores)


def test_centroid_score_returns_float(benign_embeddings, centroids):
    cb, cm = centroids
    result = centroid_score(benign_embeddings[0], cb, cm)
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Mahalanobis distance
# ---------------------------------------------------------------------------


def test_fit_mahalanobis_returns_fitted_model(benign_embeddings):
    cov = fit_mahalanobis(benign_embeddings)
    assert hasattr(cov, "mahalanobis")
    assert hasattr(cov, "shrinkage_")


def test_mahalanobis_score_ordering(benign_embeddings, malicious_embeddings):
    """Malicious embeddings should score higher than benign."""
    cov_b = fit_mahalanobis(benign_embeddings)
    cov_m = fit_mahalanobis(malicious_embeddings)
    benign_scores = [mahalanobis_score(e, cov_b, cov_m) for e in benign_embeddings]
    malicious_scores = [mahalanobis_score(e, cov_b, cov_m) for e in malicious_embeddings]
    assert np.mean(benign_scores) < np.mean(malicious_scores)


def test_mahalanobis_score_returns_float(benign_embeddings, malicious_embeddings):
    cov_b = fit_mahalanobis(benign_embeddings)
    cov_m = fit_mahalanobis(malicious_embeddings)
    result = mahalanobis_score(benign_embeddings[0], cov_b, cov_m)
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Isolation forest
# ---------------------------------------------------------------------------


def test_fit_isolation_forest_returns_fitted_model(benign_embeddings):
    iso = fit_isolation_forest(benign_embeddings)
    assert hasattr(iso, "score_samples")


def test_isolation_score_ordering(benign_embeddings, malicious_embeddings):
    """Malicious embeddings (out-of-distribution) should score higher than benign."""
    iso = fit_isolation_forest(benign_embeddings)
    benign_scores = [isolation_score(e, iso) for e in benign_embeddings]
    malicious_scores = [isolation_score(e, iso) for e in malicious_embeddings]
    assert np.mean(benign_scores) < np.mean(malicious_scores)


def test_isolation_score_returns_float(benign_embeddings):
    iso = fit_isolation_forest(benign_embeddings)
    result = isolation_score(benign_embeddings[0], iso)
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# normalize_scores
# ---------------------------------------------------------------------------


def test_normalize_scores_range():
    scores = np.array([-5.0, 0.0, 3.0, 10.0])
    normalized = normalize_scores(scores)
    assert float(normalized.min()) == pytest.approx(0.0)
    assert float(normalized.max()) == pytest.approx(1.0)


def test_normalize_scores_constant_input():
    """All-same input should not raise; output should be ~0."""
    scores = np.array([3.0, 3.0, 3.0])
    normalized = normalize_scores(scores)
    assert np.all(normalized < 1e-5)


def test_normalize_scores_preserves_order():
    scores = np.array([1.0, 5.0, 3.0, 2.0])
    normalized = normalize_scores(scores)
    assert np.all(np.diff(normalized[np.argsort(scores)]) >= 0)
