"""
Unit tests for CrossDatasetEvaluator class.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from vartrustml.config import CVConfig, ExperimentConfig
from vartrustml.core.cross_dataset import CrossDatasetEvaluator


@pytest.fixture
def experiment_config(tmp_path):
    """Create a basic experiment configuration for testing."""
    return ExperimentConfig(
        cv=CVConfig(seed=42, n_outer_splits=3, n_inner_splits=2),
        target_column="state",
        models_to_use=["Logistic Regression"],
        output_dir=str(tmp_path / "test_output"),
        save_checkpoints=False,
        verbose=0,
    )


@pytest.fixture
def sample_datasets():
    """Create sample datasets for cross-dataset testing."""
    np.random.seed(42)

    df1 = pd.DataFrame(
        {
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "feature3": np.random.randn(100),
            "state": np.random.randint(0, 2, 100),
        }
    )

    df2 = pd.DataFrame(
        {
            "feature1": np.random.randn(80),
            "feature2": np.random.randn(80),
            "feature3": np.random.randn(80),
            "state": np.random.randint(0, 2, 80),
        }
    )

    df3 = pd.DataFrame(
        {
            "feature1": np.random.randn(90),
            "feature2": np.random.randn(90),
            "feature3": np.random.randn(90),
            "state": np.random.randint(0, 2, 90),
        }
    )

    return [
        (df1, "Dataset1"),
        (df2, "Dataset2"),
        (df3, "Dataset3"),
    ]


@pytest.fixture
def imbalanced_datasets():
    """Create datasets with imbalanced classes."""
    np.random.seed(42)

    df1 = pd.DataFrame(
        {"feature1": np.random.randn(100), "state": np.array([0] * 50 + [1] * 50)}
    )

    df2 = pd.DataFrame(
        {"feature1": np.random.randn(50), "state": np.array([0] * 48 + [1] * 2)}
    )

    return [
        (df1, "Balanced"),
        (df2, "Imbalanced"),
    ]


class TestCrossDatasetEvaluatorInit:
    """Tests for CrossDatasetEvaluator initialization."""

    def test_init_basic(self, experiment_config):
        """Test basic initialization."""
        evaluator = CrossDatasetEvaluator(experiment_config)

        assert evaluator.config == experiment_config
        assert evaluator.model_config is None
        assert evaluator.evaluator is not None

    def test_init_with_model_config(self, experiment_config):
        """Test initialization with custom model config."""
        from vartrustml.config import ModelConfig

        model_config = ModelConfig()

        evaluator = CrossDatasetEvaluator(experiment_config, model_config)

        assert evaluator.model_config == model_config

    def test_metrics_to_track(self, experiment_config):
        """Test that required metrics are tracked."""
        evaluator = CrossDatasetEvaluator(experiment_config)

        assert "AUROC" in evaluator.METRICS_TO_TRACK
        assert "F1 Score (Weighted)" in evaluator.METRICS_TO_TRACK
        assert "Balanced Accuracy" in evaluator.METRICS_TO_TRACK
        assert "Matthews Corr. Coef." in evaluator.METRICS_TO_TRACK


class TestValidateDatasets:
    """Tests for dataset validation."""

    def test_validate_valid_datasets(self, experiment_config, sample_datasets):
        """Test validation with valid datasets."""
        evaluator = CrossDatasetEvaluator(experiment_config)
        valid = evaluator._validate_datasets(sample_datasets)

        assert len(valid) == 3

    def test_validate_filters_invalid(self, experiment_config, imbalanced_datasets):
        """Test that invalid datasets are filtered out."""
        evaluator = CrossDatasetEvaluator(experiment_config)
        valid = evaluator._validate_datasets(imbalanced_datasets)

        assert len(valid) == 1
        assert valid[0][1] == "Balanced"


class TestInitializeResultMatrices:
    """Tests for result matrix initialization."""

    def test_initialize_matrices(self, experiment_config):
        """Test matrix initialization."""
        evaluator = CrossDatasetEvaluator(experiment_config)
        dataset_names = ["D1", "D2", "D3"]

        results, results_std = evaluator._initialize_result_matrices(dataset_names)

        for model_name in evaluator.evaluator.models.keys():
            assert model_name in results
            assert model_name in results_std

            for metric in evaluator.METRICS_TO_TRACK:
                assert metric in results[model_name]
                assert metric in results_std[model_name]

                matrix = results[model_name][metric]
                assert list(matrix.index) == dataset_names
                assert list(matrix.columns) == dataset_names


class TestPrepareCVSplits:
    """Tests for CV split preparation."""

    def test_prepare_cv_splits(self, experiment_config, sample_datasets):
        """Test CV split preparation."""
        evaluator = CrossDatasetEvaluator(experiment_config)
        outer_splits = evaluator._prepare_cv_splits(sample_datasets)

        assert len(outer_splits) == 3

        for name in ["Dataset1", "Dataset2", "Dataset3"]:
            assert name in outer_splits
            assert len(outer_splits[name]) == experiment_config.cv.n_outer_splits

            for train_idx, test_idx in outer_splits[name]:
                assert len(train_idx) > 0
                assert len(test_idx) > 0
                assert len(set(train_idx) & set(test_idx)) == 0

    def test_cv_splits_reproducibility(self, experiment_config, sample_datasets):
        """Test that CV splits are reproducible with same seed."""
        evaluator1 = CrossDatasetEvaluator(experiment_config)
        evaluator2 = CrossDatasetEvaluator(experiment_config)

        splits1 = evaluator1._prepare_cv_splits(sample_datasets)
        splits2 = evaluator2._prepare_cv_splits(sample_datasets)

        for name in ["Dataset1", "Dataset2", "Dataset3"]:
            for i in range(experiment_config.cv.n_outer_splits):
                np.testing.assert_array_equal(splits1[name][i][0], splits2[name][i][0])
                np.testing.assert_array_equal(splits1[name][i][1], splits2[name][i][1])


class TestGetCommonFeatures:
    """Tests for feature alignment."""

    def test_common_features_same_columns(self, experiment_config):
        """Test feature alignment with identical columns."""
        evaluator = CrossDatasetEvaluator(experiment_config)

        X_src = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})

        tgt_df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6], "state": [0, 1]})

        common = evaluator._get_common_features(X_src, tgt_df)

        assert set(common) == {"a", "b", "c"}

    def test_common_features_partial_overlap(self, experiment_config):
        """Test feature alignment with partial overlap."""
        evaluator = CrossDatasetEvaluator(experiment_config)

        X_src = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})

        tgt_df = pd.DataFrame({"a": [1, 2], "c": [3, 4], "d": [5, 6]})

        common = evaluator._get_common_features(X_src, tgt_df)

        assert set(common) == {"a", "c"}


class TestAggregateAndStoreResults:
    """Tests for result aggregation."""

    def test_aggregate_results(self, experiment_config):
        """Test result aggregation from CV folds."""
        evaluator = CrossDatasetEvaluator(experiment_config)
        dataset_names = ["D1", "D2"]

        results, results_std = evaluator._initialize_result_matrices(dataset_names)

        # Per-fold values are keyed by fold id, not stored positionally
        cv_results = {
            "Logistic Regression": {
                metric: dict(enumerate(values))
                for metric, values in {
                    "AUROC": [0.8, 0.85, 0.9],
                    "F1 Score (Weighted)": [0.75, 0.8, 0.85],
                    "Balanced Accuracy": [0.7, 0.75, 0.8],
                    "Matthews Corr. Coef.": [0.6, 0.65, 0.7],
                    "Precision (Class 0)": [0.72, 0.77, 0.82],
                    "Precision (Class 1)": [0.7, 0.75, 0.8],
                    "Recall (Class 0)": [0.68, 0.73, 0.78],
                    "Recall (Class 1)": [0.65, 0.7, 0.75],
                }.items()
            }
        }

        evaluator._aggregate_and_store_results(
            cv_results, results, results_std, "D1", "D2"
        )

        expected_mean = np.mean([0.8, 0.85, 0.9])
        expected_std = np.std([0.8, 0.85, 0.9])

        assert (
            abs(results["Logistic Regression"]["AUROC"].loc["D1", "D2"] - expected_mean)
            < 0.001
        )
        assert (
            abs(
                results_std["Logistic Regression"]["AUROC"].loc["D1", "D2"]
                - expected_std
            )
            < 0.001
        )


class TestEvaluateCrossDatasetMinimumDatasets:
    """Tests for minimum dataset requirement."""

    def test_insufficient_datasets_error(self, experiment_config):
        """Test error when fewer than 2 datasets are provided."""
        np.random.seed(42)
        df = pd.DataFrame(
            {"feature1": np.random.randn(100), "state": np.random.randint(0, 2, 100)}
        )

        evaluator = CrossDatasetEvaluator(experiment_config)

        with pytest.raises(ValueError, match="Insufficient"):
            evaluator.evaluate_cross_dataset([(df, "SingleDataset")])


class TestIntegrationCrossDatasetEvaluator:
    """Integration tests for full cross-dataset evaluation."""

    @pytest.fixture
    def minimal_config(self, tmp_path):
        """Create minimal config for integration tests."""
        return ExperimentConfig(
            cv=CVConfig(seed=42, n_outer_splits=2, n_inner_splits=2),
            target_column="state",
            models_to_use=["Logistic Regression"],
            output_dir=str(tmp_path / "output"),
            save_checkpoints=False,
            verbose=0,
            generate_html_report=False,
        )

    @pytest.fixture
    def minimal_datasets(self):
        """Create minimal datasets for quick testing."""
        np.random.seed(42)

        df1 = pd.DataFrame(
            {
                "f1": np.random.randn(40),
                "f2": np.random.randn(40),
                "state": np.array([0] * 20 + [1] * 20),
            }
        )

        df2 = pd.DataFrame(
            {
                "f1": np.random.randn(40),
                "f2": np.random.randn(40),
                "state": np.array([0] * 20 + [1] * 20),
            }
        )

        return [(df1, "D1"), (df2, "D2")]

    @patch("vartrustml.core.cross_dataset.CrossDatasetEvaluator._save_results")
    @patch("vartrustml.core.cross_dataset.ModelEvaluator.evaluate_model")
    def test_evaluate_cross_dataset_structure(
        self, mock_evaluate, mock_save, minimal_config, minimal_datasets
    ):
        """Test the structure of cross-dataset evaluation results."""
        mock_evaluate.return_value = (
            {
                "AUROC": 0.85,
                "F1 Score (Weighted)": 0.80,
                "Balanced Accuracy": 0.75,
                "Matthews Corr. Coef.": 0.70,
                "Precision (Class 1)": 0.78,
                "Recall (Class 1)": 0.72,
            },
            None,  # y_prob
            None,  # optimal_threshold
        )

        evaluator = CrossDatasetEvaluator(minimal_config)
        results = evaluator.evaluate_cross_dataset(minimal_datasets)

        assert "Logistic Regression" in results
        assert "AUROC" in results["Logistic Regression"]

        auroc_matrix = results["Logistic Regression"]["AUROC"]
        assert list(auroc_matrix.index) == ["D1", "D2"]
        assert list(auroc_matrix.columns) == ["D1", "D2"]

        for i in ["D1", "D2"]:
            for j in ["D1", "D2"]:
                assert not pd.isna(auroc_matrix.loc[i, j])


class TestNpEncoderUsage:
    """Tests to verify np_encoder is used correctly in JSON output."""

    def test_json_output_uses_np_encoder(self, experiment_config, tmp_path):
        """Test that JSON output correctly handles numpy types."""
        import json

        from vartrustml.utils.serialization import np_encoder

        assert np_encoder(np.int64(1)) == 1
        assert np_encoder(np.float64(1.5)) == 1.5
        assert np_encoder(np.array([1, 2, 3])) == [1, 2, 3]
        assert np_encoder(np.bool_(True)) is True

        data = {
            "int": np.int64(42),
            "float": np.float64(3.14),
            "array": np.array([1, 2, 3]),
            "bool": np.bool_(False),
        }

        serialized = json.dumps(data, default=np_encoder)
        deserialized = json.loads(serialized)

        assert deserialized["int"] == 42
        assert abs(deserialized["float"] - 3.14) < 0.001
        assert deserialized["array"] == [1, 2, 3]
        assert deserialized["bool"] is False
