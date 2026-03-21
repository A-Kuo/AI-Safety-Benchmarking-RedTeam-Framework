"""Statistical evaluation utilities for safety benchmarking.

Provides bootstrap confidence intervals, Wilson score intervals,
Cohen's kappa, and related metrics used across both notebooks.
"""

from __future__ import annotations

import numpy as np
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


def cohens_kappa(labels1: list[bool], labels2: list[bool]) -> float:
    """Cohen's kappa for two sets of binary labels."""
    n = len(labels1)
    assert n == len(labels2), "Label lists must have equal length"
    agreement = sum(1 for a, b in zip(labels1, labels2) if a == b)
    p_o = agreement / n
    p1 = sum(labels1) / n
    p2 = sum(labels2) / n
    p_e = p1 * p2 + (1 - p1) * (1 - p2)
    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)
