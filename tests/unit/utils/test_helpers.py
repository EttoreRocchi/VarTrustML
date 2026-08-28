"""
Unit tests for helper functions.
"""

import json
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from vartrustml.utils.reporting import (
    create_feature_importance_report,
    create_summary_report,
)
from vartrustml.utils.serialization import np_encoder
from vartrustml.utils.validation import (
    calculate_minimum_samples_for_cv,
    validate_target_for_cv,
)


class TestCalculateMinimumSamples:
    """Tests for calculate_minimum_samples_for_cv function."""

    def test_basic_calculation(self):
        """Test basic calculation with typical values."""
        result = calculate_minimum_samples_for_cv(n_outer_splits=10, n_inner_splits=5)
        assert result == 10

    def test_small_folds(self):
        """Test calculation with small fold numbers."""
        result = calculate_minimum_samples_for_cv(n_outer_splits=2, n_inner_splits=2)
        assert result == 4

    def test_more_inner_than_outer(self):
        """Test calculation when inner folds > outer folds."""
        result = calculate_minimum_samples_for_cv(n_outer_splits=3, n_inner_splits=10)
        assert result == 15

    def test_invalid_outer_splits_one(self):
        """Test that n_outer_splits = 1 raises error."""
        with pytest.raises(ValueError, match="must be > 1"):
            calculate_minimum_samples_for_cv(n_outer_splits=1, n_inner_splits=3)


class TestValidateTargetForCV:
    """Tests for validate_target_for_cv function."""

    def test_valid_target(self):
        """Test valid target variable."""
        y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 5)
        is_valid, msg = validate_target_for_cv(y, n_outer_splits=5, n_inner_splits=3)
        assert is_valid
        assert msg == ""

    def test_single_class(self):
        """Test target with single class."""
        y = pd.Series([0] * 50)
        is_valid, msg = validate_target_for_cv(y, n_outer_splits=5, n_inner_splits=3)
        assert not is_valid
        assert "1 unique" in msg

    def test_insufficient_minority_samples(self):
        """Test target with insufficient minority class samples."""
        y = pd.Series([0] * 47 + [1] * 3)
        is_valid, msg = validate_target_for_cv(y, n_outer_splits=5, n_inner_splits=3)
        assert not is_valid
        assert "Insufficient" in msg or "samples" in msg.lower()

    def test_edge_case_exact_minimum(self):
        """Test target with exactly the minimum required samples."""
        y = pd.Series([0] * 45 + [1] * 5)
        is_valid, msg = validate_target_for_cv(y, n_outer_splits=5, n_inner_splits=3)
        assert is_valid, f"Should be valid with exact minimum. Error: {msg}"


class TestNpEncoder:
    """Tests for np_encoder JSON serialization helper."""

    def test_encode_numpy_integer(self):
        """Test encoding numpy integers."""
        result = np_encoder(np.int64(42))
        assert result == 42
        assert isinstance(result, int)

    def test_encode_numpy_float(self):
        """Test encoding numpy floats."""
        result = np_encoder(np.float64(3.14))
        assert result == 3.14
        assert isinstance(result, float)

    def test_encode_numpy_array(self):
        """Test encoding numpy arrays."""
        arr = np.array([1, 2, 3])
        result = np_encoder(arr)
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_encode_numpy_bool(self):
        """Test encoding numpy booleans."""
        result = np_encoder(np.bool_(True))
        assert result is True
        assert isinstance(result, bool)

    def test_unsupported_type_raises(self):
        """Test that unsupported types raise TypeError."""
        with pytest.raises(TypeError, match="not JSON serializable"):
            np_encoder(object())

    def test_json_dumps_integration(self):
        """Test np_encoder works with json.dumps."""
        data = {
            "int": np.int32(10),
            "float": np.float64(2.5),
            "array": np.array([1, 2]),
            "bool": np.bool_(False),
        }
        result = json.dumps(data, default=np_encoder)
        parsed = json.loads(result)
        assert parsed["int"] == 10
        assert parsed["float"] == 2.5
        assert parsed["array"] == [1, 2]
        assert parsed["bool"] is False


class TestCreateSummaryReport:
    """Tests for create_summary_report function."""

    @pytest.fixture
    def mock_fold_metrics(self):
        """Create mock FoldMetrics objects."""
        fold = MagicMock()
        fold.metrics = {
            "AUROC": 0.85,
            "F1 Score (Weighted)": 0.80,
            "Balanced Accuracy": 0.75,
            "Matthews Corr. Coef.": 0.70,
            "Precision (Class 1)": 0.78,
            "Recall (Class 1)": 0.72,
            "Precision (Class 0)": 0.82,
            "Recall (Class 0)": 0.77,
        }
        fold.misclassified_samples = pd.DataFrame()
        fold.error_analysis = {}
        return fold

    def test_empty_results(self, tmp_path):
        """Test with empty results dict."""
        create_summary_report({}, tmp_path)
        # Should not create files when results are empty
        assert not (tmp_path / "summary_report.txt").exists()

    def test_none_results(self, tmp_path):
        """Test with None results."""
        create_summary_report(None, tmp_path)
        assert not (tmp_path / "summary_report.txt").exists()

    def test_creates_report_files(self, tmp_path, mock_fold_metrics):
        """Test that report files are created."""
        results = {"TestModel": [mock_fold_metrics, mock_fold_metrics]}

        create_summary_report(results, tmp_path)

        assert (tmp_path / "summary_report.txt").exists()
        assert (tmp_path / "model_comparison.csv").exists()

    def test_report_content(self, tmp_path, mock_fold_metrics):
        """Test report contains expected content."""
        results = {"TestModel": [mock_fold_metrics]}

        create_summary_report(results, tmp_path)

        report_content = (tmp_path / "summary_report.txt").read_text()
        assert "SUMMARY REPORT" in report_content
        assert "TestModel" in report_content
        assert "BEST MODEL BY METRIC" in report_content


class TestCreateFeatureImportanceReport:
    """Tests for create_feature_importance_report function."""

    @pytest.fixture
    def mock_fold_with_importance(self):
        """Create mock FoldMetrics with feature importances."""
        fold = MagicMock()
        fold.feature_importances = np.array([0.5, 0.3, 0.2])
        return fold

    def test_empty_results(self, tmp_path):
        """Test with empty results."""
        result = create_feature_importance_report({}, ["f1", "f2"], tmp_path / "fi.csv")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_none_results(self, tmp_path):
        """Test with None results."""
        result = create_feature_importance_report(
            None, ["f1", "f2"], tmp_path / "fi.csv"
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_creates_importance_report(self, tmp_path, mock_fold_with_importance):
        """Test feature importance report creation."""
        results = {"Model1": [mock_fold_with_importance, mock_fold_with_importance]}
        feature_names = ["feature_a", "feature_b", "feature_c"]
        output_path = tmp_path / "importance.csv"

        result = create_feature_importance_report(results, feature_names, output_path)

        assert output_path.exists()
        assert len(result) == 3  # 3 features
        assert "Feature" in result.columns
        assert "Mean_Importance" in result.columns
        assert "Rank" in result.columns

    def test_no_importances_available(self, tmp_path):
        """Test when folds have no feature importances."""
        fold = MagicMock()
        fold.feature_importances = None
        results = {"Model1": [fold]}

        result = create_feature_importance_report(results, ["f1"], tmp_path / "fi.csv")

        assert len(result) == 0
