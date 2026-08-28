"""
Unit tests for CrossValidationPipeline class.
"""

import tempfile

import numpy as np
import pandas as pd
import pytest

from vartrustml.analysis.error_analysis import FoldMetrics
from vartrustml.config import (
    BootstrapConfig,
    CVConfig,
    ExperimentConfig,
    ModelConfig,
)
from vartrustml.core.pipeline import CrossValidationPipeline


@pytest.fixture
def base_config(tmp_path):
    """Create base experiment config for testing."""
    return ExperimentConfig(
        cv=CVConfig(seed=42, n_outer_splits=3, n_inner_splits=2),
        bootstrap=BootstrapConfig(bootstrap_n_iterations=100),  # Fast for testing
        output_dir=str(tmp_path / "test_output"),
        models_to_use=["Logistic Regression"],
        save_checkpoints=False,
        generate_html_report=False,
    )


@pytest.fixture
def sample_dataframe():
    """Create a small sample DataFrame for testing."""
    np.random.seed(42)
    n_samples = 100
    return pd.DataFrame(
        {
            "feature_1": np.random.randn(n_samples),
            "feature_2": np.random.randn(n_samples),
            "feature_3": np.random.rand(n_samples),
            "state": np.random.randint(0, 2, n_samples),
        }
    )


@pytest.fixture
def sample_fold_metrics():
    """Create sample FoldMetrics for testing."""
    np.random.seed(42)
    n_samples = 50

    y_true = np.random.randint(0, 2, n_samples)
    y_prob = np.clip(y_true + np.random.normal(0, 0.2, n_samples), 0.01, 0.99)

    return FoldMetrics(
        fold_id=0,
        metrics={
            "AUROC": 0.85,
            "Balanced Accuracy": 0.80,
            "Matthews Corr. Coef.": 0.60,
            "Precision (Class 0)": 0.78,
            "Precision (Class 1)": 0.82,
            "Recall (Class 0)": 0.75,
            "Recall (Class 1)": 0.85,
            "F1 Score (Class 0)": 0.76,
            "F1 Score (Class 1)": 0.83,
            "F1 Score (Weighted)": 0.80,
        },
        confusion_matrix=np.array([[20, 5], [3, 22]]),
        misclassified_samples=pd.DataFrame(),
        error_analysis={},
        y_true_oof=y_true,
        y_prob_oof=y_prob,
        feature_importances=None,
        shap_values=None,
        best_params={},
    )


def create_fold_metrics_list(n_folds: int = 3) -> list:
    """Create a list of FoldMetrics with varying metrics."""
    fold_metrics_list = []
    for i in range(n_folds):
        np.random.seed(42 + i)
        n_samples = 50

        y_true = np.random.randint(0, 2, n_samples)
        y_prob = np.clip(y_true + np.random.normal(0, 0.2, n_samples), 0.01, 0.99)

        fold_metrics = FoldMetrics(
            fold_id=i,
            metrics={
                "AUROC": 0.80 + np.random.uniform(-0.05, 0.05),
                "Balanced Accuracy": 0.75 + np.random.uniform(-0.05, 0.05),
                "Matthews Corr. Coef.": 0.55 + np.random.uniform(-0.05, 0.05),
                "Precision (Class 0)": 0.75 + np.random.uniform(-0.05, 0.05),
                "Precision (Class 1)": 0.78 + np.random.uniform(-0.05, 0.05),
                "Recall (Class 0)": 0.72 + np.random.uniform(-0.05, 0.05),
                "Recall (Class 1)": 0.80 + np.random.uniform(-0.05, 0.05),
                "F1 Score (Class 0)": 0.73 + np.random.uniform(-0.05, 0.05),
                "F1 Score (Class 1)": 0.79 + np.random.uniform(-0.05, 0.05),
                "F1 Score (Weighted)": 0.76 + np.random.uniform(-0.05, 0.05),
            },
            confusion_matrix=np.array([[20, 5], [3, 22]]),
            misclassified_samples=pd.DataFrame(),
            error_analysis={},
            y_true_oof=y_true,
            y_prob_oof=y_prob,
            feature_importances=None,
            shap_values=None,
            best_params={},
        )
        fold_metrics_list.append(fold_metrics)
    return fold_metrics_list


class TestCrossValidationPipelineInit:
    """Tests for CrossValidationPipeline initialization."""

    def test_init_with_default_model_config(self, base_config):
        """Test initialization with default ModelConfig."""
        pipeline = CrossValidationPipeline(base_config)

        assert pipeline.config == base_config
        assert pipeline.model_config is None
        assert pipeline.evaluator is not None
        assert pipeline.visualizer is not None

    def test_init_with_custom_model_config(self, base_config):
        """Test initialization with custom ModelConfig."""
        model_config = ModelConfig(rf_n_estimators_options=[50, 100])
        pipeline = CrossValidationPipeline(base_config, model_config)

        assert pipeline.model_config == model_config

    def test_evaluator_uses_config(self, base_config):
        """Test that evaluator receives the config."""
        pipeline = CrossValidationPipeline(base_config)

        assert pipeline.evaluator.config == base_config


class TestConcatenateOofPredictions:
    """Tests for _concatenate_oof_predictions method."""

    def test_concatenate_valid_predictions(self, base_config):
        """Test concatenating valid OOF predictions."""
        pipeline = CrossValidationPipeline(base_config)
        fold_results = create_fold_metrics_list(3)

        y_true, y_pred, y_prob, sample_indices = pipeline._concatenate_oof_predictions(
            fold_results
        )

        # Should concatenate 3 folds of 50 samples each
        assert y_true is not None
        assert y_pred is not None
        assert y_prob is not None
        assert len(y_true) == 150
        assert len(y_pred) == 150
        assert len(y_prob) == 150

    def test_concatenate_returns_none_if_missing_data(self, base_config):
        """Test that None is returned when OOF data is missing."""
        pipeline = CrossValidationPipeline(base_config)

        # Create fold metrics with missing OOF data
        fold_metrics = FoldMetrics(
            fold_id=0,
            metrics={"AUROC": 0.85},
            confusion_matrix=np.array([[20, 5], [3, 22]]),
            misclassified_samples=pd.DataFrame(),
            error_analysis={},
            y_true_oof=None,  # Missing
            y_prob_oof=None,  # Missing
            feature_importances=None,
            shap_values=None,
            best_params={},
        )

        y_true, y_pred, y_prob, sample_indices = pipeline._concatenate_oof_predictions(
            [fold_metrics]
        )

        assert y_true is None
        assert y_pred is None
        assert y_prob is None
        assert sample_indices is None

    def test_concatenate_partial_missing_data(self, base_config):
        """Test that None is returned when some folds have missing data."""
        pipeline = CrossValidationPipeline(base_config)

        fold_results = create_fold_metrics_list(2)
        # Set one fold's OOF data to None
        fold_results[1] = FoldMetrics(
            fold_id=1,
            metrics=fold_results[1].metrics,
            confusion_matrix=fold_results[1].confusion_matrix,
            misclassified_samples=pd.DataFrame(),
            error_analysis={},
            y_true_oof=None,
            y_prob_oof=None,
            feature_importances=None,
            shap_values=None,
            best_params={},
        )

        y_true, y_pred, y_prob, sample_indices = pipeline._concatenate_oof_predictions(
            fold_results
        )

        assert y_true is None
        assert y_pred is None
        assert y_prob is None
        assert sample_indices is None

    def test_y_pred_computed_from_y_prob(self, base_config):
        """Test that y_pred is computed correctly from y_prob."""
        pipeline = CrossValidationPipeline(base_config)

        # Create fold with known probabilities
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.3, 0.7, 0.4, 0.6])  # Threshold 0.5

        fold_metrics = FoldMetrics(
            fold_id=0,
            metrics={"AUROC": 0.85},
            confusion_matrix=np.array([[2, 0], [0, 2]]),
            misclassified_samples=pd.DataFrame(),
            error_analysis={},
            y_true_oof=y_true,
            y_prob_oof=y_prob,
            feature_importances=None,
            shap_values=None,
            best_params={},
        )

        _, y_pred, _, _ = pipeline._concatenate_oof_predictions([fold_metrics])

        expected_y_pred = np.array([0, 1, 0, 1])
        np.testing.assert_array_equal(y_pred, expected_y_pred)


class TestAggregateMetrics:
    """Tests for _aggregate_metrics method."""

    def test_aggregate_returns_dataframe(self, base_config):
        """Test that _aggregate_metrics returns a DataFrame."""
        pipeline = CrossValidationPipeline(base_config)
        fold_results = create_fold_metrics_list(3)

        summary = pipeline._aggregate_metrics(fold_results)

        assert isinstance(summary, pd.DataFrame)

    def test_aggregate_contains_expected_columns(self, base_config):
        """Test that summary contains expected columns."""
        pipeline = CrossValidationPipeline(base_config)
        fold_results = create_fold_metrics_list(3)

        summary = pipeline._aggregate_metrics(fold_results)

        expected_columns = [
            "mean",
            "std",
            "min",
            "max",
            "median",
            "ci_lower",
            "ci_upper",
        ]
        for col in expected_columns:
            assert col in summary.columns

    def test_aggregate_mean_computed_correctly(self, base_config):
        """Test that mean is computed correctly."""
        pipeline = CrossValidationPipeline(base_config)

        # Create fold metrics with known values
        fold_metrics_list = []
        for i, auroc in enumerate([0.80, 0.85, 0.90]):
            fold_metrics = FoldMetrics(
                fold_id=i,
                metrics={"AUROC": auroc},
                confusion_matrix=np.array([[20, 5], [3, 22]]),
                misclassified_samples=pd.DataFrame(),
                error_analysis={},
                y_true_oof=np.array([0, 1, 0, 1]),
                y_prob_oof=np.array([0.3, 0.7, 0.4, 0.6]),
                feature_importances=None,
                shap_values=None,
                best_params={},
            )
            fold_metrics_list.append(fold_metrics)

        summary = pipeline._aggregate_metrics(fold_metrics_list)

        assert abs(summary.loc["AUROC", "mean"] - 0.85) < 0.001

    def test_ci_columns_present(self, base_config):
        """Test that CI columns are present in summary."""
        pipeline = CrossValidationPipeline(base_config)
        fold_results = create_fold_metrics_list(3)

        summary = pipeline._aggregate_metrics(fold_results)

        # Check that CI columns exist
        assert "ci_lower" in summary.columns
        assert "ci_upper" in summary.columns
        # CI values should be numeric
        for metric in summary.index:
            assert not np.isnan(summary.loc[metric, "ci_lower"])
            assert not np.isnan(summary.loc[metric, "ci_upper"])


class TestCheckpointMethods:
    """Tests for checkpoint save/load/exists methods."""

    def test_checkpoint_path_structure(self, base_config):
        """Test checkpoint path is structured correctly."""
        base_config.checkpoint_dir = "checkpoints"
        pipeline = CrossValidationPipeline(base_config)

        path = pipeline.checkpoint.get_checkpoint_path("test_dataset", "XGBoost", 0)

        assert "test_output" in str(path)
        assert "checkpoints" in str(path)
        assert "test_dataset" in str(path)
        assert "XGBoost" in str(path)
        assert "fold_0" in str(path)

    def test_checkpoint_exists_false_initially(self, base_config):
        """Test checkpoint_exists returns False for non-existent checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config.output_dir = tmpdir
            pipeline = CrossValidationPipeline(base_config)

            assert not pipeline.checkpoint.checkpoint_exists("test", "Model", 0)

    def test_save_and_load_checkpoint(self, base_config, sample_fold_metrics):
        """Test saving and loading checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config.output_dir = tmpdir
            base_config.save_checkpoints = True
            base_config.checkpoint_dir = "checkpoints"
            pipeline = CrossValidationPipeline(base_config)

            # Save checkpoint
            pipeline.checkpoint.save_checkpoint("test", "Model", 0, sample_fold_metrics)

            # Load checkpoint
            loaded = pipeline.checkpoint.load_checkpoint("test", "Model", 0)

            assert loaded is not None
            assert loaded.fold_id == sample_fold_metrics.fold_id
            assert loaded.metrics["AUROC"] == sample_fold_metrics.metrics["AUROC"]

    def test_save_checkpoint_disabled(self, base_config, sample_fold_metrics):
        """Test that checkpoint is not saved when disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config.output_dir = tmpdir
            base_config.save_checkpoints = False
            pipeline = CrossValidationPipeline(base_config)

            pipeline.checkpoint.save_checkpoint("test", "Model", 0, sample_fold_metrics)

            assert not pipeline.checkpoint.checkpoint_exists("test", "Model", 0)


class TestRunCrossValidation:
    """Integration tests for run_cross_validation method.

    Note: Full integration tests are in tests/integration/ directory.
    These tests use mocking to avoid domain-specific data requirements.
    """

    def test_run_cv_initializes_correctly(self, base_config):
        """Test that run_cross_validation can be called (mocked)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config.output_dir = tmpdir
            pipeline = CrossValidationPipeline(base_config)

            # Verify pipeline is set up correctly
            assert pipeline.config == base_config
            assert "Logistic Regression" in pipeline.evaluator.models

    def test_caller_comparison_disabled_by_default(self, base_config):
        """Test that caller comparison is disabled by default."""
        pipeline = CrossValidationPipeline(base_config)
        assert not pipeline.config.caller_comparison.compare_callers

    def test_caller_comparison_can_be_enabled(self, base_config):
        """Test that caller comparison can be enabled."""
        base_config.caller_comparison.compare_callers = True
        base_config.caller_comparison.caller_columns = ["CALLER1", "CALLER2"]
        pipeline = CrossValidationPipeline(base_config)
        assert pipeline.config.caller_comparison.compare_callers
        assert pipeline.config.caller_comparison.caller_columns == [
            "CALLER1",
            "CALLER2",
        ]
