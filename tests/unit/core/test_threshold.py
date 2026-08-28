"""
Unit tests for ThresholdOptimizer class.
"""

import numpy as np
import pytest

from vartrustml.core.threshold import (
    ThresholdMethod,
    ThresholdOptimizer,
    ThresholdResult,
)


class TestThresholdOptimizer:
    """Tests for ThresholdOptimizer class."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer with default settings."""
        return ThresholdOptimizer(method=ThresholdMethod.AUTO)

    @pytest.fixture
    def perfect_predictions(self):
        """Create perfect predictions."""
        np.random.seed(42)
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.1, 0.2, 0.1, 0.2, 0.9, 0.9, 0.8, 0.9, 0.8])
        return y_true, y_prob

    @pytest.fixture
    def noisy_predictions(self):
        """Create noisy but separable predictions."""
        np.random.seed(42)
        y_true = np.array([0] * 100 + [1] * 100)
        y_prob = np.concatenate(
            [
                np.clip(np.random.normal(0.3, 0.1, 100), 0.05, 0.95),
                np.clip(np.random.normal(0.7, 0.1, 100), 0.05, 0.95),
            ]
        )
        return y_true, y_prob

    def test_init_default(self):
        """Test default initialization."""
        optimizer = ThresholdOptimizer()
        assert optimizer.method == ThresholdMethod.AUTO
        assert optimizer.auto_threshold_n_samples == 1000

    def test_init_custom(self):
        """Test custom initialization."""
        optimizer = ThresholdOptimizer(
            method=ThresholdMethod.OOF, auto_threshold_n_samples=500
        )
        assert optimizer.method == ThresholdMethod.OOF
        assert optimizer.auto_threshold_n_samples == 500

    def test_find_optimal_threshold(self, optimizer, noisy_predictions):
        """Test finding optimal threshold."""
        y_true, y_prob = noisy_predictions
        threshold, youden_j, sensitivity, specificity = (
            optimizer.find_optimal_threshold(y_true, y_prob)
        )

        assert 0 <= threshold <= 1
        assert -1 <= youden_j <= 1
        assert 0 <= sensitivity <= 1
        assert 0 <= specificity <= 1

    def test_find_optimal_threshold_perfect(self, optimizer, perfect_predictions):
        """Test finding optimal threshold with perfect predictions."""
        y_true, y_prob = perfect_predictions
        threshold, youden_j, sensitivity, specificity = (
            optimizer.find_optimal_threshold(y_true, y_prob)
        )

        assert youden_j > 0.9
        assert 0.2 < threshold <= 0.8

    def test_optimize_from_oof(self, optimizer, noisy_predictions):
        """Test OOF optimization."""
        y_true, y_prob = noisy_predictions
        result = optimizer.optimize_from_oof(y_true, y_prob)

        assert isinstance(result, ThresholdResult)
        assert result.method_used == ThresholdMethod.OOF
        assert result.n_samples == len(y_true)
        assert 0 <= result.optimal_threshold <= 1
        assert result.fold_thresholds is None

    def test_optimize_from_folds(self, optimizer):
        """Test fold-based optimization."""
        np.random.seed(42)
        fold_results = []
        for _ in range(5):
            y_true = np.array([0] * 20 + [1] * 20)
            y_prob = np.concatenate(
                [
                    np.clip(np.random.normal(0.3, 0.1, 20), 0.05, 0.95),
                    np.clip(np.random.normal(0.7, 0.1, 20), 0.05, 0.95),
                ]
            )
            fold_results.append((y_true, y_prob))

        result = optimizer.optimize_from_folds(fold_results)

        assert isinstance(result, ThresholdResult)
        assert result.method_used == ThresholdMethod.CV
        assert len(result.fold_thresholds) == 5
        assert result.n_samples == 200

    def test_select_method_small_dataset(self, optimizer):
        """Test auto method selection for small datasets."""
        method = optimizer.select_method(n_samples=500)
        assert method == ThresholdMethod.OOF

    def test_select_method_at_boundary(self, optimizer):
        """Test auto method selection at boundary."""
        method_below = optimizer.select_method(n_samples=999)
        method_at = optimizer.select_method(n_samples=1000)

        assert method_below == ThresholdMethod.OOF
        assert method_at == ThresholdMethod.CV

    def test_edge_case_random_predictions(self, optimizer):
        """Test with random (non-informative) predictions."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.uniform(0, 1, 100)

        result = optimizer.optimize_from_oof(y_true, y_prob)

        assert result.youden_j < 0.5


class TestThresholdResult:
    """Tests for ThresholdResult dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = ThresholdResult(
            optimal_threshold=0.5,
            youden_j=0.8,
            method_used=ThresholdMethod.OOF,
            sensitivity_at_threshold=0.9,
            specificity_at_threshold=0.85,
            n_samples=100,
        )

        d = result.to_dict()

        assert d["optimal_threshold"] == 0.5
        assert d["youden_j"] == 0.8
        assert d["method_used"] == "oof"
        assert d["sensitivity_at_threshold"] == 0.9
        assert d["specificity_at_threshold"] == 0.85
        assert d["n_samples"] == 100

    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {
            "optimal_threshold": 0.45,
            "youden_j": 0.75,
            "method_used": "cv",
            "sensitivity_at_threshold": 0.88,
            "specificity_at_threshold": 0.82,
            "fold_thresholds": [0.4, 0.45, 0.5],
            "n_samples": 150,
        }

        result = ThresholdResult.from_dict(d)

        assert result.optimal_threshold == 0.45
        assert result.youden_j == 0.75
        assert result.method_used == ThresholdMethod.CV
        assert result.sensitivity_at_threshold == 0.88
        assert result.specificity_at_threshold == 0.82
        assert result.fold_thresholds == [0.4, 0.45, 0.5]
        assert result.n_samples == 150


class TestThresholdMethod:
    """Tests for ThresholdMethod enum."""

    def test_enum_values(self):
        """Test enum has expected values."""
        assert ThresholdMethod.OOF.value == "oof"
        assert ThresholdMethod.CV.value == "cv"
        assert ThresholdMethod.AUTO.value == "auto"
