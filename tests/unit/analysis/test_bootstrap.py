"""
Unit tests for bootstrap confidence interval module.
"""

import numpy as np
import pytest

from vartrustml.analysis.bootstrap import (
    BootstrapAnalyzer,
    BootstrapCIResult,
    format_ci,
)


class TestBootstrapAnalyzer:
    """Tests for BootstrapAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a BootstrapAnalyzer with default settings."""
        return BootstrapAnalyzer(n_iterations=500, ci_level=0.95, seed=42)

    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions for testing."""
        np.random.seed(42)
        n_samples = 200
        y_true = np.random.randint(0, 2, n_samples)
        y_pred = y_true.copy()
        flip_idx = np.random.choice(n_samples, size=int(n_samples * 0.2), replace=False)
        y_pred[flip_idx] = 1 - y_pred[flip_idx]
        y_prob = np.clip(y_pred + np.random.normal(0, 0.1, n_samples), 0.05, 0.95)
        return y_true, y_pred, y_prob

    def test_init_valid(self):
        """Test valid initialization."""
        analyzer = BootstrapAnalyzer(n_iterations=1000, ci_level=0.95, seed=42)
        assert analyzer.n_iterations == 1000
        assert analyzer.ci_level == 0.95
        assert analyzer.seed == 42

    def test_init_invalid_n_iterations(self):
        """Test initialization with too few iterations."""
        with pytest.raises(ValueError, match="at least 100"):
            BootstrapAnalyzer(n_iterations=50)

    def test_init_invalid_ci_level(self):
        """Test initialization with invalid CI level."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            BootstrapAnalyzer(ci_level=1.5)
        with pytest.raises(ValueError, match="between 0 and 1"):
            BootstrapAnalyzer(ci_level=0)

    def test_compute_ci_from_predictions(self, analyzer, sample_predictions):
        """Test computing CI from predictions."""
        y_true, y_pred, y_prob = sample_predictions

        def accuracy(y_t, y_p):
            return np.mean(y_t == y_p)

        result = analyzer.compute_ci_from_predictions(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            metric_func=accuracy,
            metric_name="Accuracy",
            requires_prob=False,
        )

        assert isinstance(result, BootstrapCIResult)
        assert result.ci_lower <= result.point_estimate <= result.ci_upper

    def test_compute_all_cis_from_predictions(self, analyzer, sample_predictions):
        """Test computing all CIs from predictions."""
        y_true, y_pred, y_prob = sample_predictions

        results = analyzer.compute_all_cis_from_predictions(y_true, y_pred, y_prob)

        expected_metrics = [
            "Precision (Class 0)",
            "Precision (Class 1)",
            "Recall (Class 0)",
            "Recall (Class 1)",
            "F1 Score (Class 0)",
            "F1 Score (Class 1)",
            "F1 Score (Weighted)",
            "Matthews Corr. Coef.",
            "Balanced Accuracy",
            "AUROC",
        ]

        for metric in expected_metrics:
            assert metric in results
            assert isinstance(results[metric], BootstrapCIResult)

    def test_compute_ci_reproducibility(self, sample_predictions):
        """Test that results are reproducible with same seed."""
        y_true, y_pred, y_prob = sample_predictions

        analyzer1 = BootstrapAnalyzer(n_iterations=500, ci_level=0.95, seed=42)
        analyzer2 = BootstrapAnalyzer(n_iterations=500, ci_level=0.95, seed=42)

        result1 = analyzer1.compute_all_cis_from_predictions(y_true, y_pred, y_prob)
        result2 = analyzer2.compute_all_cis_from_predictions(y_true, y_pred, y_prob)

        mcc1 = result1["Matthews Corr. Coef."]
        mcc2 = result2["Matthews Corr. Coef."]

        assert mcc1.ci_lower == mcc2.ci_lower
        assert mcc1.ci_upper == mcc2.ci_upper

    def test_ci_width_increases_with_variance(self):
        """Test that CI width is larger for more variable predictions."""
        analyzer = BootstrapAnalyzer(n_iterations=1000, seed=42)

        # Low variance scenario: predictions mostly correct
        np.random.seed(42)
        n = 100
        y_true_low = np.random.randint(0, 2, n)
        y_pred_low = y_true_low.copy()
        flip_low = np.random.choice(n, size=int(n * 0.05), replace=False)
        y_pred_low[flip_low] = 1 - y_pred_low[flip_low]

        # High variance scenario: predictions more random
        y_pred_high = y_true_low.copy()
        flip_high = np.random.choice(n, size=int(n * 0.30), replace=False)
        y_pred_high[flip_high] = 1 - y_pred_high[flip_high]

        result_low = analyzer.compute_all_cis_from_predictions(
            y_true_low, y_pred_low, y_prob=None
        )
        result_high = analyzer.compute_all_cis_from_predictions(
            y_true_low, y_pred_high, y_prob=None
        )

        # MCC CI should be narrower for low variance
        width_low = (
            result_low["Matthews Corr. Coef."].ci_upper
            - result_low["Matthews Corr. Coef."].ci_lower
        )
        width_high = (
            result_high["Matthews Corr. Coef."].ci_upper
            - result_high["Matthews Corr. Coef."].ci_lower
        )

        # Higher variance should give wider CI
        assert width_high > width_low

    def test_ci_level_affects_width(self, sample_predictions):
        """Test that higher CI level gives wider intervals."""
        y_true, y_pred, y_prob = sample_predictions

        analyzer_90 = BootstrapAnalyzer(n_iterations=1000, ci_level=0.90, seed=42)
        analyzer_95 = BootstrapAnalyzer(n_iterations=1000, ci_level=0.95, seed=42)
        analyzer_99 = BootstrapAnalyzer(n_iterations=1000, ci_level=0.99, seed=42)

        result_90 = analyzer_90.compute_all_cis_from_predictions(y_true, y_pred, y_prob)
        result_95 = analyzer_95.compute_all_cis_from_predictions(y_true, y_pred, y_prob)
        result_99 = analyzer_99.compute_all_cis_from_predictions(y_true, y_pred, y_prob)

        # Compare MCC CI widths
        width_90 = (
            result_90["Matthews Corr. Coef."].ci_upper
            - result_90["Matthews Corr. Coef."].ci_lower
        )
        width_95 = (
            result_95["Matthews Corr. Coef."].ci_upper
            - result_95["Matthews Corr. Coef."].ci_lower
        )
        width_99 = (
            result_99["Matthews Corr. Coef."].ci_upper
            - result_99["Matthews Corr. Coef."].ci_lower
        )

        assert width_90 < width_95 < width_99


class TestFormatCI:
    """Tests for format_ci function."""

    def test_format_ci(self):
        """Test formatting CI results with various precisions."""
        result = BootstrapCIResult(
            metric_name="Accuracy",
            point_estimate=0.8567,
            ci_lower=0.8234,
            ci_upper=0.8901,
            ci_level=0.95,
            n_iterations=1000,
            std=0.02,
        )

        # Default precision (3)
        assert format_ci(result) == "0.857 [0.823, 0.890]"
        # Custom precision
        assert format_ci(result, precision=2) == "0.86 [0.82, 0.89]"


class TestBootstrapCIResult:
    """Tests for BootstrapCIResult dataclass."""

    def test_dataclass_creation(self):
        """Test creating a BootstrapCIResult."""
        result = BootstrapCIResult(
            metric_name="Accuracy",
            point_estimate=0.85,
            ci_lower=0.82,
            ci_upper=0.88,
            ci_level=0.95,
            n_iterations=1000,
            std=0.02,
        )

        assert result.metric_name == "Accuracy"
        assert result.point_estimate == 0.85
        assert result.ci_lower == 0.82
        assert result.ci_upper == 0.88
        assert result.ci_level == 0.95
        assert result.n_iterations == 1000
        assert result.std == 0.02
        assert result.ci_method == "bca"  # default


class TestBCaConfidenceInterval:
    """Tests for the bias-corrected and accelerated (BCa) interval."""

    @pytest.fixture
    def predictions(self):
        rng = np.random.default_rng(0)
        n = 1500
        y_true = rng.integers(0, 2, n)
        y_prob = np.clip(0.5 * y_true + 0.25 * rng.standard_normal(n) + 0.5, 0, 1)
        y_pred = (y_prob >= 0.5).astype(int)
        return y_true, y_pred, y_prob

    def test_default_method_is_bca(self):
        assert BootstrapAnalyzer().ci_method == "bca"

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            BootstrapAnalyzer(ci_method="nonsense")

    def test_bca_result_records_method(self, predictions):
        from sklearn.metrics import matthews_corrcoef

        y_true, y_pred, _ = predictions
        analyzer = BootstrapAnalyzer(n_iterations=500, seed=1, ci_method="bca")
        res = analyzer.compute_ci_from_predictions(
            y_true, y_pred, None, matthews_corrcoef, "MCC"
        )
        assert res.ci_method == "bca"
        assert res.ci_lower <= res.point_estimate <= res.ci_upper

    def test_bca_and_percentile_brackets_point_estimate(self, predictions):
        from sklearn.metrics import matthews_corrcoef

        y_true, y_pred, _ = predictions
        for method in ("bca", "percentile"):
            analyzer = BootstrapAnalyzer(n_iterations=500, seed=1, ci_method=method)
            res = analyzer.compute_ci_from_predictions(
                y_true, y_pred, None, matthews_corrcoef, "MCC"
            )
            assert res.ci_lower < res.ci_upper
            assert res.ci_lower <= res.point_estimate <= res.ci_upper

    def test_bca_falls_back_to_percentile_at_boundary(self):
        """A perfect predictor puts the point estimate at the distribution edge,
        where BCa is undefined and must fall back to the percentile interval."""
        from sklearn.metrics import recall_score

        y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1] * 100)
        y_pred = y_true.copy()  # perfect -> recall == 1.0 everywhere
        analyzer = BootstrapAnalyzer(n_iterations=300, seed=2, ci_method="bca")
        res = analyzer.compute_ci_from_predictions(
            y_true,
            y_pred,
            None,
            lambda t, p: recall_score(t, p, pos_label=1, zero_division=0),
            "Recall",
        )
        assert res.ci_method == "percentile"
