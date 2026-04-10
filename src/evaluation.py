"""Statistical evaluation utilities for safety benchmarking.

Provides bootstrap confidence intervals, Wilson score intervals,
Cohen's kappa, permutation-based AUC significance tests, and
related metrics used across both notebooks.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import auc, roc_curve


def bootstrap_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> dict[str, float | np.ndarray]:
    """Compute AUC with bootstrap confidence intervals."""
    n = len(labels)
    aucs: list[float] = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        if len(np.unique(labels[idx])) < 2:
            continue
        fpr, tpr, _ = roc_curve(labels[idx], scores[idx])
        aucs.append(auc(fpr, tpr))
    arr = np.array(aucs)
    alpha = (1 - ci) / 2
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "ci_low": float(np.percentile(arr, alpha * 100)),
        "ci_high": float(np.percentile(arr, (1 - alpha) * 100)),
        "distribution": arr,
    }


def wilson_ci(
    successes: int,
    total: int,
    z: float = 1.96,
) -> tuple[float, float, float]:
    """Wilson score confidence interval for a binomial proportion.

    Returns (center, lower, upper).
    """
    if total == 0:
        return 0.0, 0.0, 0.0
    p_hat = successes / total
    denom = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denom
    spread = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total) / denom
    return float(center), max(0.0, center - spread), min(1.0, center + spread)


def mannwhitney_auc_pvalue(
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:
    """
    Compute AUROC and its p-value via Mann-Whitney U test.

    The Mann-Whitney U statistic is equivalent to AUROC:
        AUROC = U / (n_pos * n_neg)

    The p-value tests H₀: AUROC = 0.5 (random classifier).
    Uses scipy.stats.mannwhitneyu with the 'greater' alternative,
    which matches the one-sided hypothesis that positive scores are
    stochastically greater than negative scores.

    Returns
    -------
    dict with keys: auroc, u_statistic, p_value, significant_at_05
    """
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return {"auroc": 0.5, "u_statistic": 0.0, "p_value": 1.0, "significant_at_05": False}

    result = stats.mannwhitneyu(pos_scores, neg_scores, alternative="greater")
    auroc = result.statistic / (len(pos_scores) * len(neg_scores))
    significant = bool(result.pvalue < 0.05)

    return {
        "auroc": float(auroc),
        "u_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "significant_at_05": significant,
    }


def ks_distribution_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> dict[str, float]:
    """
    Kolmogorov-Smirnov test for distributional shift between two score sets.

    Used to detect when the distribution of safety scores drifts between
    evaluation runs (e.g., before vs. after a model update).

    Returns
    -------
    dict with keys: ks_statistic, p_value, shifted (bool at α=0.05)
    """
    result = stats.ks_2samp(scores_a, scores_b)
    return {
        "ks_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "shifted": bool(result.pvalue < 0.05),
    }


def cohens_kappa(labels1: list[bool], labels2: list[bool]) -> float:
    """Cohen's kappa for two sets of binary labels."""
    n = len(labels1)
    assert n == len(labels2), "Label lists must have equal length"
    agreement = sum(1 for a, b in zip(labels1, labels2, strict=True) if a == b)
    p_o = agreement / n
    p1 = sum(labels1) / n
    p2 = sum(labels2) / n
    p_e = p1 * p2 + (1 - p1) * (1 - p2)
    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)
