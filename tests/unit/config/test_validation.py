"""
Unit tests for configuration validation module.
"""

import numpy as np
import pandas as pd

from vartrustml.config.experiment import (
    CalibrationConfig,
    CVConfig,
    ExperimentConfig,
    ThresholdConfig,
)
from vartrustml.config.validation import (
    validate_dataset_for_cv,
    validate_experiment_config,
)


class TestValidateExperimentConfig:
    """Tests for validate_experiment_config function."""

    def test_valid_config_default(self):
        """Test that default config is valid."""
        config = ExperimentConfig()
        is_valid, errors = validate_experiment_config(config)
        assert is_valid
        assert len(errors) == 0

    def test_valid_config_custom(self):
        """Test valid custom configuration."""
        config = ExperimentConfig(
            cv=CVConfig(seed=123, n_outer_splits=5, n_inner_splits=5),
            models_to_use=["XGBoost", "Random Forest"],
            calibration=CalibrationConfig(
                calibrate_models=True,
                calibration_method="sigmoid",
                calibration_cv=3,
            ),
            verbose=2,
        )
        is_valid, errors = validate_experiment_config(config)
        assert is_valid
        assert len(errors) == 0

    def test_invalid_seed_negative(self):
        """Test that negative seed is invalid."""
        config = ExperimentConfig(cv=CVConfig(seed=-1))
        is_valid, errors = validate_experiment_config(config)
        assert not is_valid
        assert any("seed" in e for e in errors)

    def test_invalid_outer_splits_too_small(self):
        """Test that n_outer_splits < 2 is invalid."""
        config = ExperimentConfig(cv=CVConfig(n_outer_splits=1))
        is_valid, errors = validate_experiment_config(config)
        assert not is_valid
        assert any("n_outer_splits" in e for e in errors)

    def test_invalid_calibration_cv_too_small(self):
        """Test that calibration_cv < 2 is invalid when calibration enabled."""
        config = ExperimentConfig(
            calibration=CalibrationConfig(calibrate_models=True, calibration_cv=1)
        )
        is_valid, errors = validate_experiment_config(config)
        assert not is_valid
        assert any("calibration_cv" in e for e in errors)

    def test_invalid_calibration_method(self):
        """Test that unknown calibration method is invalid."""
        config = ExperimentConfig(
            calibration=CalibrationConfig(
                calibrate_models=True, calibration_method="unknown_method"
            )
        )
        is_valid, errors = validate_experiment_config(config)
        assert not is_valid
        assert any("calibration_method" in e for e in errors)

    def test_invalid_models_empty(self):
        """Test that empty models_to_use is invalid."""
        config = ExperimentConfig(models_to_use=[])
        is_valid, errors = validate_experiment_config(config)
        assert not is_valid
        assert any("models_to_use" in e.lower() for e in errors)

    def test_invalid_model_name(self):
        """Test that unknown model name is invalid."""
        config = ExperimentConfig(models_to_use=["UnknownModel"])
        is_valid, errors = validate_experiment_config(config)
        assert not is_valid
        assert any("UnknownModel" in e for e in errors)

    def test_invalid_threshold_method(self):
        """Test that unknown threshold method is invalid."""
        config = ExperimentConfig(
            threshold=ThresholdConfig(
                optimize_threshold=True, threshold_method="unknown"
            )
        )
        is_valid, errors = validate_experiment_config(config)
        assert not is_valid
        assert any("threshold_method" in e for e in errors)


class TestValidateDatasetForCV:
    """Tests for validate_dataset_for_cv function."""

    def test_valid_dataset(self):
        """Test valid dataset passes validation."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "state": np.random.randint(0, 2, 100),
            }
        )
        is_valid, errors = validate_dataset_for_cv(
            df, target_column="state", n_outer_splits=5, n_inner_splits=3
        )
        assert is_valid
        assert len(errors) == 0

    def test_empty_dataset(self):
        """Test empty dataset is invalid."""
        df = pd.DataFrame()
        is_valid, errors = validate_dataset_for_cv(
            df, target_column="state", n_outer_splits=5, n_inner_splits=3
        )
        assert not is_valid
        assert any("empty" in e.lower() for e in errors)

    def test_missing_target_column(self):
        """Test missing target column is invalid."""
        df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        is_valid, errors = validate_dataset_for_cv(
            df, target_column="state", n_outer_splits=5, n_inner_splits=3
        )
        assert not is_valid
        assert any("target" in e.lower() for e in errors)

    def test_single_class_target(self):
        """Test single class target is invalid."""
        df = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "state": [0, 0, 0, 0, 0]})
        is_valid, errors = validate_dataset_for_cv(
            df, target_column="state", n_outer_splits=2, n_inner_splits=2
        )
        assert not is_valid
        assert any("class" in e.lower() for e in errors)

    def test_insufficient_samples_per_class(self):
        """Test insufficient samples per class for CV."""
        df = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "state": [0, 0, 0, 1, 1]})
        is_valid, errors = validate_dataset_for_cv(
            df, target_column="state", n_outer_splits=5, n_inner_splits=3
        )
        assert not is_valid
        assert any("sample" in e.lower() for e in errors)
