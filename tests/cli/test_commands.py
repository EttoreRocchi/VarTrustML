"""
Consolidated CLI tests for all VarTrustML commands.

Tests the essential functionality of: version, list-models, train, compare-models,
cross-dataset, predict, evaluate, and ablation commands.
"""

import json
import re
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from vartrustml import __version__
from vartrustml.cli._shared import _parse_multi_value
from vartrustml.cli.main import app

runner = CliRunner()


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


# =============================================================================
# Parsing Functions Tests
# =============================================================================


class TestParseMultiValue:
    """Tests for _parse_multi_value function (used by --continuous and --categorical)."""

    def test_none_returns_none(self):
        """Test that None input returns None."""
        assert _parse_multi_value(None) is None

    def test_comma_separated_list(self):
        """Test parsing comma-separated values."""
        result = _parse_multi_value("col1,col2,col3")
        assert result == ["col1", "col2", "col3"]

    def test_comma_separated_with_spaces(self):
        """Test parsing comma-separated values with spaces."""
        result = _parse_multi_value("col1, col2 , col3")
        assert result == ["col1", "col2", "col3"]

    def test_long_comma_separated_list_no_error(self):
        """Test that long comma-separated lists don't cause 'File name too long' error."""
        # This is the bug fix test - long lists were being interpreted as file paths
        long_list = ",".join([f"feature_{i}" for i in range(100)])
        assert len(long_list) > 255  # Exceeds Linux filename limit

        result = _parse_multi_value(long_list)
        assert result is not None
        assert len(result) == 100
        assert result[0] == "feature_0"
        assert result[99] == "feature_99"

    def test_txt_file_one_per_line(self, tmp_path):
        """Test reading columns from .txt file (one per line)."""
        txt_file = tmp_path / "columns.txt"
        txt_file.write_text("col1\ncol2\ncol3\n")

        result = _parse_multi_value(str(txt_file))
        assert result == ["col1", "col2", "col3"]

    def test_txt_file_comma_separated(self, tmp_path):
        """Test reading columns from .txt file (comma-separated)."""
        txt_file = tmp_path / "columns.txt"
        txt_file.write_text("col1,col2,col3")

        result = _parse_multi_value(str(txt_file))
        assert result == ["col1", "col2", "col3"]

    def test_txt_file_mixed_format(self, tmp_path):
        """Test reading columns from .txt file (mixed format)."""
        txt_file = tmp_path / "columns.txt"
        txt_file.write_text("col1,col2\ncol3,col4\n")

        result = _parse_multi_value(str(txt_file))
        assert result == ["col1", "col2", "col3", "col4"]

    def test_nonexistent_txt_file_treated_as_value(self, tmp_path):
        """Test that non-existent .txt file path is treated as comma-separated."""
        result = _parse_multi_value("nonexistent.txt")
        # Since file doesn't exist, it's parsed as a single value
        assert result == ["nonexistent.txt"]

    def test_non_txt_extension_not_treated_as_file(self, tmp_path):
        """Test that non-.txt files are not read as column files."""
        csv_file = tmp_path / "columns.csv"
        csv_file.write_text("col1,col2,col3")

        # Even though file exists, it's not .txt so treated as comma-separated string
        result = _parse_multi_value(str(csv_file))
        # The path string is parsed as comma-separated (likely returns the path components)
        assert result is not None


class TestContinuousCategoricalFlags:
    """Integration tests for --continuous and --categorical flags."""

    def test_continuous_as_comma_list(self, temp_data_files):
        """Test --continuous with comma-separated list."""
        data_dir, datasets = temp_data_files

        result = runner.invoke(
            app,
            [
                "compare-models",
                datasets[0],
                "-d",
                str(data_dir),
                "--target-column",
                "state",
                "--continuous",
                "feature1,feature2",
                "--dry-run",
            ],
        )

        assert result.exit_code == 0

    def test_continuous_as_txt_file(self, temp_data_files, tmp_path):
        """Test --continuous with .txt file."""
        data_dir, datasets = temp_data_files

        cols_file = tmp_path / "continuous_cols.txt"
        cols_file.write_text("feature1\nfeature2\n")

        result = runner.invoke(
            app,
            [
                "compare-models",
                datasets[0],
                "-d",
                str(data_dir),
                "--target-column",
                "state",
                "--continuous",
                str(cols_file),
                "--dry-run",
            ],
        )

        assert result.exit_code == 0

    def test_categorical_as_comma_list(self, temp_data_files):
        """Test --categorical with comma-separated list."""
        data_dir, datasets = temp_data_files

        result = runner.invoke(
            app,
            [
                "compare-models",
                datasets[0],
                "-d",
                str(data_dir),
                "--target-column",
                "state",
                "--categorical",
                "cat1,cat2",
                "--dry-run",
            ],
        )

        assert result.exit_code == 0

    def test_categorical_as_txt_file(self, temp_data_files, tmp_path):
        """Test --categorical with .txt file."""
        data_dir, datasets = temp_data_files

        cols_file = tmp_path / "categorical_cols.txt"
        cols_file.write_text("cat1\ncat2\n")

        result = runner.invoke(
            app,
            [
                "compare-models",
                datasets[0],
                "-d",
                str(data_dir),
                "--target-column",
                "state",
                "--categorical",
                str(cols_file),
                "--dry-run",
            ],
        )

        assert result.exit_code == 0

    def test_long_continuous_list_no_oserror(self, temp_data_files):
        """Test that long --continuous list doesn't cause OSError."""
        data_dir, datasets = temp_data_files

        # Create a long list of column names (exceeds 255 char filename limit)
        long_cols = ",".join([f"feature_{i}" for i in range(50)])

        result = runner.invoke(
            app,
            [
                "compare-models",
                datasets[0],
                "-d",
                str(data_dir),
                "--target-column",
                "state",
                "--continuous",
                long_cols,
                "--dry-run",
            ],
        )

        # Should not fail with OSError: File name too long
        assert "File name too long" not in result.output
        assert result.exit_code == 0


# =============================================================================
# Basic Commands (version, list-models, help)
# =============================================================================


class TestBasicCommands:
    """Tests for basic CLI commands."""

    def test_version_command(self):
        """Test that version command shows version."""
        result = runner.invoke(app, ["version"])

        assert result.exit_code == 0
        assert __version__ in result.output
        assert "VarTrustML" in result.output

    def test_list_models_command(self):
        """Test that list-models command works."""
        result = runner.invoke(app, ["list-models"])

        assert result.exit_code == 0
        assert "XGBoost" in result.output or "Logistic Regression" in result.output

    def test_main_help(self):
        """Test main help display."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "compare-models" in result.output
        assert "cross-dataset" in result.output
        assert "train" in result.output


# =============================================================================
# Train Command
# =============================================================================


class TestTrainCommand:
    """Tests for the train command."""

    def test_dry_run_basic(self, sample_dataset):
        """Test basic dry run."""
        data_dir, dataset = sample_dataset

        result = runner.invoke(
            app, ["train", str(data_dir / dataset), "--target", "state", "--dry-run"]
        )

        assert result.exit_code == 0
        assert "DRY RUN MODE" in result.output
        assert "Configuration is valid" in result.output

    def test_calibration_options(self, sample_dataset):
        """Test calibration options."""
        data_dir, dataset = sample_dataset

        result = runner.invoke(
            app,
            [
                "train",
                str(data_dir / dataset),
                "--target",
                "state",
                "--calibrate-model",
                "--calibration-method",
                "sigmoid",
                "--dry-run",
            ],
        )

        assert result.exit_code == 0
        assert "sigmoid" in result.output.lower()

    def test_threshold_options(self, sample_dataset):
        """Test threshold optimization options."""
        data_dir, dataset = sample_dataset

        result = runner.invoke(
            app,
            [
                "train",
                str(data_dir / dataset),
                "--target",
                "state",
                "--optimize-threshold",
                "--threshold-method",
                "cv",
                "--dry-run",
            ],
        )

        assert result.exit_code == 0
        assert "cv" in result.output.lower()

    def test_custom_target(self, tmp_path):
        """Test custom target column."""
        np.random.seed(42)
        df = pd.DataFrame(
            {"feature1": np.random.randn(50), "label": np.array([0] * 25 + [1] * 25)}
        )
        path = tmp_path / "custom_target.csv"
        df.to_csv(path, index=False)

        result = runner.invoke(
            app, ["train", str(path), "--target", "label", "--dry-run"]
        )

        assert result.exit_code == 0
        assert "label" in result.output

    def test_param_config_file(self, sample_dataset, tmp_path):
        """Test loading parameter config from JSON file."""
        data_dir, dataset = sample_dataset

        config_path = tmp_path / "params.json"
        config = {"rf_n_estimators_options": [50, 100], "rf_max_depth_options": [5, 10]}
        config_path.write_text(json.dumps(config))

        result = runner.invoke(
            app,
            [
                "train",
                str(data_dir / dataset),
                "--target",
                "state",
                "--model",
                "Random Forest",
                "--param-config",
                str(config_path),
                "--dry-run",
            ],
        )

        assert result.exit_code == 0

    @patch("vartrustml.core.train_model.ModelTrainer.fit")
    def test_train_execution(self, mock_fit, sample_dataset, tmp_path):
        """Test train execution with mocked trainer."""
        data_dir, dataset = sample_dataset
        output_dir = tmp_path / "output"

        mock_fit.return_value = {
            "best_params": {"n_estimators": 100},
            "best_score": 0.85,
            "model_path": str(output_dir / "model.joblib"),
        }

        result = runner.invoke(
            app,
            [
                "train",
                str(data_dir / dataset),
                "--target",
                "state",
                "--model",
                "Random Forest",
                "--output-dir",
                str(output_dir),
                "--cv-folds",
                "2",
            ],
        )

        assert result.exit_code == 0
        assert mock_fit.called


# =============================================================================
# Compare-Models Command
# =============================================================================


class TestCompareModelsCommand:
    """Tests for the compare-models command."""

    def test_dry_run_with_dataset(self, temp_data_files):
        """Test dry run with a dataset."""
        data_dir, datasets = temp_data_files

        result = runner.invoke(
            app,
            [
                "compare-models",
                datasets[0],
                "-d",
                str(data_dir),
                "--target-column",
                "state",
                "--dry-run",
            ],
        )

        assert result.exit_code == 0
        assert "DRY RUN MODE" in result.output

    def test_calibration_options(self, temp_data_files):
        """Test calibration options."""
        data_dir, datasets = temp_data_files

        result = runner.invoke(
            app,
            [
                "compare-models",
                datasets[0],
                "-d",
                str(data_dir),
                "--target-column",
                "state",
                "--calibrate-model",
                "--calibration",
                "isotonic",
                "--dry-run",
            ],
        )

        assert result.exit_code == 0
        assert "isotonic" in result.output.lower()

    def test_hpo_options(self, temp_data_files):
        """Test HPO options."""
        data_dir, datasets = temp_data_files

        result = runner.invoke(
            app,
            [
                "compare-models",
                datasets[0],
                "-d",
                str(data_dir),
                "--target-column",
                "state",
                "--hpo-method",
                "optuna",
                "--optuna-trials",
                "25",
                "--dry-run",
            ],
        )

        assert result.exit_code == 0
        assert "optuna" in result.output.lower()

    def test_compare_callers_requires_callers_option(self, temp_data_files):
        """Test that --compare-callers requires --callers."""
        data_dir, datasets = temp_data_files

        result = runner.invoke(
            app,
            [
                "compare-models",
                datasets[0],
                "-d",
                str(data_dir),
                "--target-column",
                "state",
                "--compare-callers",
                "--dry-run",
            ],
        )

        # Validation errors now use exit code 2
        assert result.exit_code == 2
        assert (
            "--callers" in result.output.lower() or "required" in result.output.lower()
        )

    def test_load_config_file(self, temp_data_files, tmp_path):
        """Test loading config from JSON file."""
        data_dir, datasets = temp_data_files

        config_path = tmp_path / "config.json"
        config = {
            "seed": 123,
            "n_outer_splits": 3,
            "n_inner_splits": 2,
            "models_to_use": ["Logistic Regression"],
            "target_column": "state",
        }
        config_path.write_text(json.dumps(config))

        result = runner.invoke(
            app,
            [
                "compare-models",
                datasets[0],
                "-d",
                str(data_dir),
                "--target-column",
                "state",
                "--config",
                str(config_path),
                "--dry-run",
            ],
        )

        assert result.exit_code == 0
        assert "Loaded ExperimentConfig" in result.output

    def test_multiple_datasets(self, temp_data_files):
        """Test with multiple datasets."""
        data_dir, datasets = temp_data_files

        result = runner.invoke(
            app,
            [
                "compare-models",
                datasets[0],
                datasets[1],
                "-d",
                str(data_dir),
                "--target-column",
                "state",
                "--dry-run",
            ],
        )

        assert result.exit_code == 0
        assert datasets[0] in result.output
        assert datasets[1] in result.output


# =============================================================================
# Cross-Dataset Command
# =============================================================================


class TestCrossDatasetCommand:
    """Tests for the cross-dataset command."""

    def test_single_dataset_error(self, temp_data_files):
        """Test that single dataset produces error."""
        data_dir, datasets = temp_data_files

        result = runner.invoke(
            app, ["cross-dataset", datasets[0], "-d", str(data_dir), "--dry-run"]
        )

        # Validation errors now use exit code 2
        assert result.exit_code == 2
        assert "requires at least 2 datasets" in result.output.lower()

    def test_two_datasets_minimum(self, temp_data_files):
        """Test that two datasets work."""
        data_dir, datasets = temp_data_files

        result = runner.invoke(
            app,
            [
                "cross-dataset",
                datasets[0],
                datasets[1],
                "-d",
                str(data_dir),
                "--dry-run",
            ],
        )

        assert result.exit_code == 0

    def test_dry_run_displays_config(self, temp_data_files):
        """Test that dry run displays configuration."""
        data_dir, datasets = temp_data_files

        result = runner.invoke(
            app,
            [
                "cross-dataset",
                datasets[0],
                datasets[1],
                "-d",
                str(data_dir),
                "--dry-run",
            ],
        )

        assert result.exit_code == 0
        assert "DRY RUN MODE" in result.output
        assert "Configuration is valid" in result.output

    def test_load_config_file(self, temp_data_files, tmp_path):
        """Test loading config from JSON file."""
        data_dir, datasets = temp_data_files

        config_path = tmp_path / "config.json"
        config = {
            "seed": 123,
            "n_outer_splits": 3,
            "n_inner_splits": 2,
            "models_to_use": ["Logistic Regression"],
        }
        config_path.write_text(json.dumps(config))

        result = runner.invoke(
            app,
            [
                "cross-dataset",
                datasets[0],
                datasets[1],
                "-d",
                str(data_dir),
                "--config",
                str(config_path),
                "--dry-run",
            ],
        )

        assert result.exit_code == 0
        assert "Loaded ExperimentConfig" in result.output

    def test_save_config(self, temp_data_files, tmp_path):
        """Test saving effective config."""
        data_dir, datasets = temp_data_files
        save_path = tmp_path / "saved_config.json"

        result = runner.invoke(
            app,
            [
                "cross-dataset",
                datasets[0],
                datasets[1],
                "-d",
                str(data_dir),
                "--save-config",
                str(save_path),
                "--dry-run",
            ],
        )

        assert result.exit_code == 0
        assert save_path.exists()

    def test_incompatible_datasets_error(self, tmp_path):
        """Test error for incompatible datasets."""
        df1 = pd.DataFrame(
            {"feature_a": np.random.randn(50), "state": np.array([0] * 25 + [1] * 25)}
        )
        df2 = pd.DataFrame(
            {"feature_b": np.random.randn(50), "state": np.array([0] * 25 + [1] * 25)}
        )

        path1 = tmp_path / "ds1.csv"
        path2 = tmp_path / "ds2.csv"
        df1.to_csv(path1, index=False)
        df2.to_csv(path2, index=False)

        result = runner.invoke(
            app,
            [
                "cross-dataset",
                "ds1.csv",
                "ds2.csv",
                "-d",
                str(tmp_path),
                "--n-outer-splits",
                "2",
                "--n-inner-splits",
                "2",
                "--models",
                "Logistic Regression",
            ],
        )

        assert "compatible" in result.output.lower() or result.exit_code != 0

    @patch("vartrustml.core.cross_dataset.CrossDatasetEvaluator.evaluate_cross_dataset")
    @patch("vartrustml.io.data_loader.DataLoader.validate_datasets_compatibility")
    def test_full_execution(
        self, mock_validate, mock_evaluate, temp_data_files, tmp_path
    ):
        """Test full execution with mocked evaluator."""
        data_dir, datasets = temp_data_files
        output_dir = tmp_path / "output"

        mock_validate.return_value = {"compatible": True, "issues": []}
        mock_evaluate.return_value = {}

        result = runner.invoke(
            app,
            [
                "cross-dataset",
                datasets[0],
                datasets[1],
                "-d",
                str(data_dir),
                "-o",
                str(output_dir),
                "--n-outer-splits",
                "2",
                "--n-inner-splits",
                "2",
                "--models",
                "Logistic Regression",
                "--no-checkpoints",
                "--no-html-report",
            ],
        )

        assert result.exit_code == 0
        assert "COMPLETE" in result.output

    def test_custom_target(self, tmp_path):
        """Test custom target column."""
        np.random.seed(42)

        df = pd.DataFrame(
            {"feature1": np.random.randn(50), "label": np.array([0] * 25 + [1] * 25)}
        )
        path1 = tmp_path / "custom_target1.csv"
        path2 = tmp_path / "custom_target2.csv"
        df.to_csv(path1, index=False)
        df.to_csv(path2, index=False)

        result = runner.invoke(
            app,
            [
                "cross-dataset",
                "custom_target1.csv",
                "custom_target2.csv",
                "-d",
                str(tmp_path),
                "--target-column",
                "label",
                "--dry-run",
            ],
        )

        assert result.exit_code == 0
        assert "label" in result.output


# =============================================================================
# Predict and Evaluate Commands
# =============================================================================


class TestPredictCommand:
    """Tests for the predict command."""

    @patch("vartrustml.core.train_model.ModelTrainer.load_model")
    def test_predict_basic(self, mock_load, sample_dataset, tmp_path):
        """Test basic prediction."""
        data_dir, dataset = sample_dataset

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0, 1, 0, 1, 0] * 20)
        # load_model returns a dict with model and metadata
        mock_load.return_value = {
            "model": mock_model,
            "optimal_threshold": 0.5,
            "threshold_metadata": None,
            "config": None,
        }

        model_path = tmp_path / "model.joblib"
        model_path.touch()

        output_path = tmp_path / "predictions.csv"

        result = runner.invoke(
            app,
            [
                "predict",
                str(model_path),
                str(data_dir / dataset),
                "--output",
                str(output_path),
            ],
        )

        assert result.exit_code == 0
        assert output_path.exists()
        assert mock_model.predict.called

    @patch("vartrustml.core.train_model.ModelTrainer.load_model")
    def test_predict_with_proba(self, mock_load, sample_dataset, tmp_path):
        """Test prediction with probabilities."""
        data_dir, dataset = sample_dataset

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.random.rand(100, 2)
        # load_model returns a dict with model and metadata
        mock_load.return_value = {
            "model": mock_model,
            "optimal_threshold": 0.72,
            "threshold_metadata": {"method_used": "oof"},
            "config": None,
        }

        model_path = tmp_path / "model.joblib"
        model_path.touch()

        output_path = tmp_path / "predictions.csv"

        result = runner.invoke(
            app,
            [
                "predict",
                str(model_path),
                str(data_dir / dataset),
                "--proba",
                "--output",
                str(output_path),
            ],
        )

        assert result.exit_code == 0
        assert mock_model.predict_proba.called


class TestEvaluateCommand:
    """Tests for the evaluate command."""

    @patch("vartrustml.core.train_model.ModelTrainer.load_model")
    def test_evaluate_basic(self, mock_load, sample_dataset, tmp_path):
        """Test basic evaluation."""
        data_dir, dataset = sample_dataset

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array(
            [0] * 40 + [1] * 40 + [0] * 10 + [1] * 10
        )
        mock_model.predict_proba.return_value = np.random.rand(100, 2)
        # load_model returns a dict with model and metadata
        mock_load.return_value = {
            "model": mock_model,
            "optimal_threshold": 0.5,
            "threshold_metadata": None,
            "config": None,
        }

        model_path = tmp_path / "model.joblib"
        model_path.touch()

        output_path = tmp_path / "evaluation.txt"

        result = runner.invoke(
            app,
            [
                "evaluate",
                str(model_path),
                str(data_dir / dataset),
                "--target",
                "state",
                "--output",
                str(output_path),
            ],
        )

        assert result.exit_code == 0

    @patch("vartrustml.core.train_model.ModelTrainer.load_model")
    def test_evaluate_custom_target(self, mock_load, tmp_path):
        """Test evaluation with custom target column."""
        np.random.seed(42)
        df = pd.DataFrame(
            {"feature1": np.random.randn(50), "label": np.array([0] * 25 + [1] * 25)}
        )
        data_path = tmp_path / "custom.csv"
        df.to_csv(data_path, index=False)

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0] * 25 + [1] * 25)
        mock_model.predict_proba.return_value = np.random.rand(50, 2)
        # load_model returns a dict with model and metadata
        mock_load.return_value = {
            "model": mock_model,
            "optimal_threshold": 0.5,
            "threshold_metadata": None,
            "config": None,
        }

        model_path = tmp_path / "model.joblib"
        model_path.touch()

        output_path = tmp_path / "evaluation.txt"

        result = runner.invoke(
            app,
            [
                "evaluate",
                str(model_path),
                str(data_path),
                "--target",
                "label",
                "--output",
                str(output_path),
            ],
        )

        assert result.exit_code == 0


# =============================================================================
# Ablation Command
# =============================================================================


class TestAblationCommand:
    """Tests for the ablation command."""

    def test_help(self):
        """Test ablation help display."""
        result = runner.invoke(app, ["ablation", "--help"])
        output = strip_ansi(result.output)

        assert result.exit_code == 0
        assert "ablation" in output.lower()
        assert "--target-column" in output or "-t" in output
        assert "--model" in output or "-m" in output
        assert "--metric" in output

    def test_missing_required_options(self, sample_dataset):
        """Test that missing required options produces error."""
        data_dir, dataset = sample_dataset

        # Missing --target-column
        result = runner.invoke(
            app,
            ["ablation", str(data_dir / dataset)],
        )
        assert result.exit_code != 0

    def test_invalid_model(self, sample_dataset):
        """Test that invalid model name produces error."""
        data_dir, dataset = sample_dataset

        result = runner.invoke(
            app,
            [
                "ablation",
                str(data_dir / dataset),
                "--target-column",
                "state",
                "--model",
                "InvalidModel",
            ],
        )

        assert result.exit_code == 2  # Validation error
        assert "unsupported model" in result.output.lower()

    def test_invalid_metric(self, sample_dataset):
        """Test that invalid metric produces error."""
        data_dir, dataset = sample_dataset

        result = runner.invoke(
            app,
            [
                "ablation",
                str(data_dir / dataset),
                "--target-column",
                "state",
                "--model",
                "XGBoost",
                "--metric",
                "invalid_metric",
            ],
        )

        assert result.exit_code == 2  # Validation error
        assert "invalid metric" in result.output.lower()

    def test_invalid_target_column(self, sample_dataset):
        """Test that invalid target column produces error."""
        data_dir, dataset = sample_dataset

        result = runner.invoke(
            app,
            [
                "ablation",
                str(data_dir / dataset),
                "-d",
                str(data_dir),
                "--target-column",
                "nonexistent_column",
                "--model",
                "XGBoost",
            ],
        )

        assert result.exit_code == 2  # Validation error
        assert "not found" in result.output.lower()

    def test_dry_run(self, sample_dataset):
        """Test dry run mode."""
        data_dir, dataset = sample_dataset

        result = runner.invoke(
            app,
            [
                "ablation",
                dataset,
                "-d",
                str(data_dir),
                "--target-column",
                "state",
                "--model",
                "XGBoost",
                "--dry-run",
            ],
        )

        assert result.exit_code == 0
        assert "DRY RUN MODE" in result.output
        assert "Configuration is valid" in result.output

    @patch(
        "vartrustml.analysis.ablation_config.ConfigAblationAnalyzer.ablation_from_config"
    )
    def test_basic_execution(self, mock_ablation, sample_dataset, tmp_path):
        """Test basic ablation execution."""
        data_dir, dataset = sample_dataset
        output_dir = tmp_path / "ablation_output"

        # Mock ablation results
        from vartrustml.analysis.ablation import AblationResult, AblationStudyResult

        mock_result = AblationStudyResult(
            study_type="feature",
            results=[
                AblationResult(
                    ablation_name="feature1",
                    baseline_score=0.85,
                    baseline_std=0.02,
                    ablated_score=0.80,
                    ablated_std=0.03,
                    delta=-0.05,
                    delta_pct=-5.9,
                    p_value=0.01,
                    is_significant=True,
                    effect_size=0.8,
                    baseline_scores=[0.84, 0.85, 0.86, 0.85, 0.85],
                    ablated_scores=[0.79, 0.80, 0.81, 0.80, 0.80],
                    metric_name="balanced_accuracy",
                ),
            ],
            baseline_score=0.85,
            metric_name="balanced_accuracy",
            n_splits=5,
            seed=42,
        )
        mock_ablation.return_value = mock_result

        result = runner.invoke(
            app,
            [
                "ablation",
                dataset,
                "-d",
                str(data_dir),
                "--target-column",
                "state",
                "--model",
                "XGBoost",
                "--output-dir",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0
        assert mock_ablation.called
        assert "Ablation study complete" in result.output

    @patch(
        "vartrustml.analysis.ablation_config.ConfigAblationAnalyzer.group_ablation_from_config"
    )
    def test_feature_groups(self, mock_group_ablation, sample_dataset, tmp_path):
        """Test ablation with feature groups."""
        data_dir, dataset = sample_dataset
        output_dir = tmp_path / "ablation_output"

        # Create feature groups YAML file
        groups_file = tmp_path / "groups.yaml"
        groups_file.write_text(
            """
group1:
  - feature1
  - feature2
group2:
  - feature3
"""
        )

        # Mock ablation results
        from vartrustml.analysis.ablation import AblationResult, AblationStudyResult

        mock_result = AblationStudyResult(
            study_type="feature_group",
            results=[
                AblationResult(
                    ablation_name="group1",
                    baseline_score=0.85,
                    baseline_std=0.02,
                    ablated_score=0.75,
                    ablated_std=0.04,
                    delta=-0.10,
                    delta_pct=-11.8,
                    p_value=0.001,
                    is_significant=True,
                    effect_size=1.2,
                    baseline_scores=[0.84, 0.85, 0.86, 0.85, 0.85],
                    ablated_scores=[0.74, 0.75, 0.76, 0.75, 0.75],
                    metric_name="balanced_accuracy",
                ),
            ],
            baseline_score=0.85,
            metric_name="balanced_accuracy",
            n_splits=5,
            seed=42,
        )
        mock_group_ablation.return_value = mock_result

        result = runner.invoke(
            app,
            [
                "ablation",
                dataset,
                "-d",
                str(data_dir),
                "--target-column",
                "state",
                "--model",
                "XGBoost",
                "--feature-groups",
                str(groups_file),
                "--output-dir",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0
        assert mock_group_ablation.called
        assert "feature group" in result.output.lower()

    @patch(
        "vartrustml.analysis.ablation_config.ConfigAblationAnalyzer.ablation_from_config"
    )
    def test_specific_features(self, mock_ablation, sample_dataset, tmp_path):
        """Test ablation with specific features."""
        data_dir, dataset = sample_dataset
        output_dir = tmp_path / "ablation_output"

        # Mock ablation results
        from vartrustml.analysis.ablation import AblationResult, AblationStudyResult

        mock_result = AblationStudyResult(
            study_type="feature",
            results=[
                AblationResult(
                    ablation_name="feature1",
                    baseline_score=0.85,
                    baseline_std=0.02,
                    ablated_score=0.82,
                    ablated_std=0.03,
                    delta=-0.03,
                    delta_pct=-3.5,
                    p_value=0.05,
                    is_significant=True,
                    effect_size=0.5,
                    baseline_scores=[0.84, 0.85, 0.86, 0.85, 0.85],
                    ablated_scores=[0.81, 0.82, 0.83, 0.82, 0.82],
                    metric_name="balanced_accuracy",
                ),
            ],
            baseline_score=0.85,
            metric_name="balanced_accuracy",
            n_splits=5,
            seed=42,
        )
        mock_ablation.return_value = mock_result

        result = runner.invoke(
            app,
            [
                "ablation",
                dataset,
                "-d",
                str(data_dir),
                "--target-column",
                "state",
                "--model",
                "XGBoost",
                "--features",
                "feature1,feature2",
                "--output-dir",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0
        assert mock_ablation.called

    def test_different_metrics(self, sample_dataset):
        """Test ablation with different metrics."""
        data_dir, dataset = sample_dataset

        for metric in ["balanced_accuracy", "f1", "mcc", "roc_auc"]:
            result = runner.invoke(
                app,
                [
                    "ablation",
                    dataset,
                    "-d",
                    str(data_dir),
                    "--target-column",
                    "state",
                    "--model",
                    "XGBoost",
                    "--metric",
                    metric,
                    "--dry-run",
                ],
            )

            assert result.exit_code == 0
            assert metric in result.output.lower() or "Metric" in result.output

    @patch(
        "vartrustml.analysis.ablation_config.ConfigAblationAnalyzer.ablation_from_config"
    )
    def test_output_files_created(self, mock_ablation, sample_dataset, tmp_path):
        """Test that output files are created."""
        data_dir, dataset = sample_dataset
        output_dir = tmp_path / "ablation_output"

        # Mock ablation results with summary_df
        from vartrustml.analysis.ablation import AblationResult, AblationStudyResult

        mock_result = AblationStudyResult(
            study_type="feature",
            results=[
                AblationResult(
                    ablation_name="feature1",
                    baseline_score=0.85,
                    baseline_std=0.02,
                    ablated_score=0.80,
                    ablated_std=0.03,
                    delta=-0.05,
                    delta_pct=-5.9,
                    p_value=0.01,
                    is_significant=True,
                    effect_size=0.8,
                    baseline_scores=[0.84, 0.85, 0.86, 0.85, 0.85],
                    ablated_scores=[0.79, 0.80, 0.81, 0.80, 0.80],
                    metric_name="balanced_accuracy",
                ),
            ],
            baseline_score=0.85,
            metric_name="balanced_accuracy",
            n_splits=5,
            seed=42,
        )
        mock_ablation.return_value = mock_result

        result = runner.invoke(
            app,
            [
                "ablation",
                dataset,
                "-d",
                str(data_dir),
                "--target-column",
                "state",
                "--model",
                "XGBoost",
                "--output-dir",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0
        assert output_dir.exists()
        assert (output_dir / "ablation_results.csv").exists()
        assert (output_dir / "ablation_report.txt").exists()
