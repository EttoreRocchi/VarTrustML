"""
Unit tests for checkpoint utilities.
"""

import joblib
import pandas as pd

from vartrustml.io.checkpoint import (
    cleanup_checkpoints,
    get_checkpoint_summary,
    list_checkpoints,
    load_checkpoint_model,
)


class TestListCheckpoints:
    """Tests for list_checkpoints function."""

    def test_nonexistent_directory(self, tmp_path):
        """Test with non-existent directory returns empty dict."""
        result = list_checkpoints(str(tmp_path / "nonexistent"))
        assert result == {}

    def test_empty_directory(self, tmp_path):
        """Test with empty directory returns empty dict."""
        result = list_checkpoints(str(tmp_path))
        assert result == {}

    def test_directory_with_files_only(self, tmp_path):
        """Test directory with only files (no subdirs) returns empty dict."""
        (tmp_path / "some_file.txt").touch()
        result = list_checkpoints(str(tmp_path))
        assert result == {}

    def test_valid_checkpoint_structure(self, tmp_path):
        """Test with valid checkpoint structure."""
        # Create checkpoint structure: dataset/model/fold_N/fold_N_results.joblib
        dataset_dir = tmp_path / "test_dataset"
        model_dir = dataset_dir / "Logistic_Regression"
        fold_dir = model_dir / "fold_0"
        fold_dir.mkdir(parents=True)

        # Create the results file
        joblib.dump({"test": "data"}, fold_dir / "fold_0_results.joblib")

        result = list_checkpoints(str(tmp_path))

        assert "test_dataset" in result
        assert "Logistic_Regression" in result["test_dataset"]
        assert 0 in result["test_dataset"]["Logistic_Regression"]

    def test_multiple_folds(self, tmp_path):
        """Test with multiple folds."""
        dataset_dir = tmp_path / "dataset1"
        model_dir = dataset_dir / "XGBoost"

        for fold_id in [0, 1, 2]:
            fold_dir = model_dir / f"fold_{fold_id}"
            fold_dir.mkdir(parents=True)
            joblib.dump({}, fold_dir / f"fold_{fold_id}_results.joblib")

        result = list_checkpoints(str(tmp_path))

        assert result["dataset1"]["XGBoost"] == [0, 1, 2]

    def test_multiple_models_and_datasets(self, tmp_path):
        """Test with multiple datasets and models."""
        for dataset in ["dataset_A", "dataset_B"]:
            for model in ["RF", "LR"]:
                fold_dir = tmp_path / dataset / model / "fold_0"
                fold_dir.mkdir(parents=True)
                joblib.dump({}, fold_dir / "fold_0_results.joblib")

        result = list_checkpoints(str(tmp_path))

        assert len(result) == 2
        assert "dataset_A" in result
        assert "dataset_B" in result
        assert "RF" in result["dataset_A"]
        assert "LR" in result["dataset_A"]


class TestCleanupCheckpoints:
    """Tests for cleanup_checkpoints function."""

    def test_nonexistent_directory(self, tmp_path):
        """Test cleanup on non-existent directory (should not raise)."""
        cleanup_checkpoints(str(tmp_path / "nonexistent"))

    def test_cleanup_all(self, tmp_path):
        """Test cleaning all checkpoints."""
        # Create some files
        (tmp_path / "dataset" / "model").mkdir(parents=True)
        (tmp_path / "dataset" / "model" / "file1.txt").touch()
        (tmp_path / "dataset" / "model" / "file2.joblib").touch()

        cleanup_checkpoints(str(tmp_path))

        # Files should be deleted
        assert not (tmp_path / "dataset" / "model" / "file1.txt").exists()
        assert not (tmp_path / "dataset" / "model" / "file2.joblib").exists()

    def test_cleanup_specific_dataset(self, tmp_path):
        """Test cleaning specific dataset checkpoints."""
        # Create files in two datasets
        (tmp_path / "dataset1" / "model").mkdir(parents=True)
        (tmp_path / "dataset2" / "model").mkdir(parents=True)
        (tmp_path / "dataset1" / "model" / "file.txt").touch()
        (tmp_path / "dataset2" / "model" / "file.txt").touch()

        cleanup_checkpoints(str(tmp_path), dataset_name="dataset1")

        assert not (tmp_path / "dataset1" / "model" / "file.txt").exists()
        assert (tmp_path / "dataset2" / "model" / "file.txt").exists()

    def test_cleanup_specific_model(self, tmp_path):
        """Test cleaning specific model checkpoints."""
        # Create files for two models
        (tmp_path / "dataset" / "Random_Forest").mkdir(parents=True)
        (tmp_path / "dataset" / "XGBoost").mkdir(parents=True)
        (tmp_path / "dataset" / "Random_Forest" / "file.txt").touch()
        (tmp_path / "dataset" / "XGBoost" / "file.txt").touch()

        cleanup_checkpoints(
            str(tmp_path), dataset_name="dataset", model_name="Random Forest"
        )

        assert not (tmp_path / "dataset" / "Random_Forest" / "file.txt").exists()
        assert (tmp_path / "dataset" / "XGBoost" / "file.txt").exists()


class TestLoadCheckpointModel:
    """Tests for load_checkpoint_model function."""

    def test_load_model_with_metadata(self, tmp_path):
        """Test loading model dict with metadata."""
        model_path = tmp_path / "model.joblib"
        model_data = {
            "model": "mock_model",
            "metadata": {"version": "1.0"},
        }
        joblib.dump(model_data, model_path)

        result = load_checkpoint_model(str(model_path))
        assert result["model"] == "mock_model"
        assert result["metadata"]["version"] == "1.0"

    def test_load_nonexistent_file(self, tmp_path):
        """Test loading non-existent file returns None."""
        result = load_checkpoint_model(str(tmp_path / "nonexistent.joblib"))
        assert result is None

    def test_load_corrupted_file(self, tmp_path):
        """Test loading corrupted file returns None."""
        model_path = tmp_path / "corrupted.joblib"
        model_path.write_text("not a valid joblib file")

        result = load_checkpoint_model(str(model_path))
        assert result is None


class TestGetCheckpointSummary:
    """Tests for get_checkpoint_summary function."""

    def test_empty_checkpoints(self, tmp_path):
        """Test summary with no checkpoints."""
        result = get_checkpoint_summary(str(tmp_path))
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_summary_with_checkpoints(self, tmp_path):
        """Test summary generation with valid checkpoints."""
        # Create checkpoint structure
        for dataset in ["data1", "data2"]:
            for model in ["Model_A"]:
                for fold_id in [0, 1]:
                    fold_dir = tmp_path / dataset / model / f"fold_{fold_id}"
                    fold_dir.mkdir(parents=True)
                    joblib.dump({}, fold_dir / f"fold_{fold_id}_results.joblib")

        result = get_checkpoint_summary(str(tmp_path))

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "Dataset" in result.columns
        assert "Model" in result.columns
        assert "Completed Folds" in result.columns
        assert result["Completed Folds"].iloc[0] == 2
