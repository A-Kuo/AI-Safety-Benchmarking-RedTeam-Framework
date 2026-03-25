"""Unit tests for src.evaluation — statistical metrics."""

import numpy as np
import pytest

from src.evaluation import bootstrap_auc, cohens_kappa, wilson_ci

# ---------------------------------------------------------------------------
# bootstrap_auc
# ---------------------------------------------------------------------------


def test_bootstrap_auc_perfect_classifier():
    """A perfect classifier should yield AUC near 1.0."""
    labels = np.array([0] * 50 + [1] * 50)
    scores = np.array([0.0] * 50 + [1.0] * 50)
    result = bootstrap_auc(labels, scores, n_bootstrap=200)
    assert result["mean"] == pytest.approx(1.0, abs=1e-3)
    assert result["ci_low"] > 0.95


def test_bootstrap_auc_random_classifier():
    """A random classifier should yield AUC near 0.5."""
    rng = np.random.default_rng(42)
    labels = rng.integers(0, 2, size=200)
    scores = rng.random(200)
    result = bootstrap_auc(labels, scores, n_bootstrap=500)
    assert 0.35 < result["mean"] < 0.65


def test_bootstrap_auc_ci_ordering():
    """ci_low <= mean <= ci_high must always hold."""
    labels = np.array([0] * 30 + [1] * 30)
    scores = np.concatenate([np.random.default_rng(0).random(30) * 0.5,
                              np.random.default_rng(0).random(30) * 0.5 + 0.5])
    result = bootstrap_auc(labels, scores, n_bootstrap=200)
    assert result["ci_low"] <= result["mean"] <= result["ci_high"]


def test_bootstrap_auc_distribution_length():
    labels = np.array([0] * 20 + [1] * 20)
    scores = np.random.default_rng(1).random(40)
    result = bootstrap_auc(labels, scores, n_bootstrap=300)
    assert len(result["distribution"]) > 0
    assert len(result["distribution"]) <= 300


def test_bootstrap_auc_same_seed_is_reproducible():
    """Two calls with the same random_state must return identical distributions."""
    labels = np.array([0] * 30 + [1] * 30)
    scores = np.random.default_rng(7).random(60)
    r1 = bootstrap_auc(labels, scores, n_bootstrap=200, random_state=99)
    r2 = bootstrap_auc(labels, scores, n_bootstrap=200, random_state=99)
    np.testing.assert_array_equal(r1["distribution"], r2["distribution"])


def test_bootstrap_auc_different_seeds_differ():
    """Two calls with different seeds should (with overwhelming probability) differ."""
    labels = np.array([0] * 30 + [1] * 30)
    scores = np.random.default_rng(7).random(60)
    r1 = bootstrap_auc(labels, scores, n_bootstrap=200, random_state=1)
    r2 = bootstrap_auc(labels, scores, n_bootstrap=200, random_state=2)
    assert not np.array_equal(r1["distribution"], r2["distribution"])


# ---------------------------------------------------------------------------
# wilson_ci
# ---------------------------------------------------------------------------


def test_wilson_ci_all_successes():
    center, lo, hi = wilson_ci(10, 10)
    assert 0.7 < center <= 1.0
    assert lo >= 0.0
    assert hi == pytest.approx(1.0)


def test_wilson_ci_no_successes():
    center, lo, hi = wilson_ci(0, 10)
    assert 0.0 <= center < 0.3
    assert lo == pytest.approx(0.0)


def test_wilson_ci_zero_total():
    center, lo, hi = wilson_ci(0, 0)
    assert center == 0.0
    assert lo == 0.0
    assert hi == 0.0


def test_wilson_ci_bounds_valid():
    """Lower bound <= center <= upper bound for any inputs."""
    for n, k in [(1, 1), (5, 3), (100, 47), (1000, 200)]:
        center, lo, hi = wilson_ci(k, n)
        assert lo <= center <= hi
        assert 0.0 <= lo
        assert hi <= 1.0


def test_wilson_ci_half_point():
    """50% success rate should give a center near 0.5."""
    center, lo, hi = wilson_ci(50, 100)
    assert center == pytest.approx(0.5, abs=0.02)


# ---------------------------------------------------------------------------
# cohens_kappa
# ---------------------------------------------------------------------------


def test_cohens_kappa_perfect_agreement():
    labels = [True, False, True, False, True]
    kappa = cohens_kappa(labels, labels)
    assert kappa == pytest.approx(1.0)


def test_cohens_kappa_no_agreement():
    """Opposite labels → kappa should be negative."""
    labels1 = [True, True, False, False]
    labels2 = [False, False, True, True]
    kappa = cohens_kappa(labels1, labels2)
    assert kappa < 0.0


def test_cohens_kappa_all_same_label():
    """When both raters always agree trivially, kappa = 1.0."""
    labels = [True] * 10
    kappa = cohens_kappa(labels, labels)
    assert kappa == pytest.approx(1.0)


def test_cohens_kappa_known_value():
    """Manually computed: 3/4 agreement, balanced classes → κ = 0.5."""
    labels1 = [True, True, False, False]
    labels2 = [True, False, False, False]
    kappa = cohens_kappa(labels1, labels2)
    assert isinstance(kappa, float)
    # p_o = 0.75, p1=0.5, p2=0.25 => p_e = 0.5*0.25 + 0.5*0.75 = 0.5
    # kappa = (0.75 - 0.5) / (1 - 0.5) = 0.5
    assert kappa == pytest.approx(0.5)


def test_cohens_kappa_length_mismatch_raises():
    with pytest.raises(AssertionError):
        cohens_kappa([True, False], [True])
