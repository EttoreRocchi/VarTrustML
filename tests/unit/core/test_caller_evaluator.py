"""
Unit tests for caller_evaluator module.
"""

import numpy as np
import pandas as pd
import pytest

from vartrustml.core.caller_evaluator import (
    CallerEvaluator,
    CallerResult,
    validate_caller_columns,
)


class TestCallerEvaluator:
    """Tests for CallerEvaluator class."""

    @pytest.fixture
    def evaluator(self):
        """Create a CallerEvaluator with sample caller columns."""
        return CallerEvaluator(["MANTA", "DELLY", "SMOOVE"])

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 100
        return pd.DataFrame(
            {
                "MANTA": np.random.randint(0, 2, n_samples),
                "DELLY": np.random.randint(0, 2, n_samples),
                "SMOOVE": np.random.randint(0, 2, n_samples),
            }
        )

    @pytest.fixture
    def sample_y_true(self):
        """Create sample ground truth labels."""
        np.random.seed(42)
        return np.random.randint(0, 2, 100)

    def test_init_valid(self):
        """Test valid initialization."""
        evaluator = CallerEvaluator(["MANTA", "DELLY"])
        assert evaluator.caller_columns == ["MANTA", "DELLY"]

    def test_init_empty_columns(self):
        """Test initialization with empty columns raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            CallerEvaluator([])

    def test_evaluate_single_caller(self, evaluator, sample_data, sample_y_true):
        """Test evaluating a single caller."""
        result = evaluator.evaluate_single_caller(
            caller_name="MANTA",
            y_true=sample_y_true,
            caller_predictions=sample_data["MANTA"].values,
            fold_id=0,
        )

        assert isinstance(result, CallerResult)
        assert result.name == "MANTA"
        assert result.fold_id == 0
        assert "Recall (Class 0)" in result.metrics
        assert "Recall (Class 1)" in result.metrics
        assert "Matthews Corr. Coef." in result.metrics
        assert result.confusion_matrix.shape == (2, 2)
        assert len(result.y_true) == len(sample_y_true)
        assert len(result.y_pred) == len(sample_y_true)

    def test_evaluate_single_caller_invalid_name(self, evaluator, sample_y_true):
        """Test evaluating with invalid caller name raises error."""
        with pytest.raises(ValueError, match="not in caller_columns"):
            evaluator.evaluate_single_caller(
                caller_name="INVALID",
                y_true=sample_y_true,
                caller_predictions=np.zeros(100),
                fold_id=0,
            )

    def test_evaluate_combination_and(self, evaluator, sample_data, sample_y_true):
        """Test AND combination of callers."""
        result = evaluator.evaluate_combination(
            caller_names=["MANTA", "DELLY"],
            operation="AND",
            y_true=sample_y_true,
            caller_data=sample_data,
            fold_id=0,
        )

        assert result.name == "MANTA AND DELLY"
        expected = (sample_data["MANTA"] & sample_data["DELLY"]).values
        np.testing.assert_array_equal(result.y_pred, expected)

    def test_evaluate_combination_or(self, evaluator, sample_data, sample_y_true):
        """Test OR combination of callers."""
        result = evaluator.evaluate_combination(
            caller_names=["MANTA", "DELLY"],
            operation="OR",
            y_true=sample_y_true,
            caller_data=sample_data,
            fold_id=0,
        )

        assert result.name == "MANTA OR DELLY"
        expected = (sample_data["MANTA"] | sample_data["DELLY"]).values
        np.testing.assert_array_equal(result.y_pred, expected)

    def test_evaluate_combination_invalid_operation(
        self, evaluator, sample_data, sample_y_true
    ):
        """Test invalid operation raises error."""
        with pytest.raises(ValueError, match="must be 'AND' or 'OR'"):
            evaluator.evaluate_combination(
                caller_names=["MANTA", "DELLY"],
                operation="XOR",
                y_true=sample_y_true,
                caller_data=sample_data,
                fold_id=0,
            )

    def test_parse_combination_expression_and(self, evaluator):
        """Test parsing AND expression."""
        callers, op = evaluator.parse_combination_expression("MANTA AND DELLY")
        assert callers == ["MANTA", "DELLY"]
        assert op == "AND"

    def test_parse_combination_expression_mixed(self, evaluator):
        """Test mixed AND/OR raises error."""
        with pytest.raises(ValueError, match="Mixed AND/OR not supported"):
            evaluator.parse_combination_expression("MANTA AND DELLY OR SMOOVE")

    def test_get_default_combinations_two_callers(self):
        """Test default combinations with 2 callers."""
        evaluator = CallerEvaluator(["MANTA", "DELLY"])
        combos = evaluator.get_default_combinations()

        assert "MANTA AND DELLY" in combos
        assert "MANTA OR DELLY" in combos
        assert len(combos) == 2

    def test_evaluate_from_expression(self, evaluator, sample_data, sample_y_true):
        """Test evaluate_from_expression method."""
        result = evaluator.evaluate_from_expression(
            expression="MANTA AND DELLY",
            y_true=sample_y_true,
            caller_data=sample_data,
            fold_id=0,
        )

        assert result.name == "MANTA AND DELLY"
        assert "Matthews Corr. Coef." in result.metrics

    def test_metrics_consistency(self, evaluator, sample_data, sample_y_true):
        """Test that metrics are computed correctly."""
        result = evaluator.evaluate_single_caller(
            caller_name="MANTA",
            y_true=sample_y_true,
            caller_predictions=sample_data["MANTA"].values,
            fold_id=0,
        )

        assert 0 <= result.metrics["Recall (Class 0)"] <= 1
        assert 0 <= result.metrics["Recall (Class 1)"] <= 1
        assert 0 <= result.metrics["Balanced Accuracy"] <= 1
        assert -1 <= result.metrics["Matthews Corr. Coef."] <= 1


class TestValidateCallerColumns:
    """Tests for validate_caller_columns function."""

    def test_valid_columns(self):
        """Test validation passes with valid columns."""
        df = pd.DataFrame(
            {
                "MANTA": [0, 1, 0, 1],
                "DELLY": [1, 1, 0, 0],
                "state": [0, 1, 0, 1],
            }
        )
        validate_caller_columns(df, ["MANTA", "DELLY"], "state")

    def test_missing_column(self):
        """Test validation fails when column is missing."""
        df = pd.DataFrame(
            {
                "MANTA": [0, 1, 0, 1],
                "state": [0, 1, 0, 1],
            }
        )
        with pytest.raises(ValueError, match="not found in dataset"):
            validate_caller_columns(df, ["MANTA", "DELLY"], "state")
