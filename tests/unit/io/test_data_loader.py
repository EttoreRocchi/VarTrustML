"""
Unit tests for DataLoader class.
"""

import pandas as pd
import pytest

from vartrustml.io.data_loader import DataLoader


class TestDataLoaderInit:
    """Tests for DataLoader initialization."""

    def test_init_valid_directory(self, temp_data_dir):
        """Test initialization with valid directory."""
        loader = DataLoader(str(temp_data_dir))
        assert loader.data_directory == temp_data_dir

    def test_init_invalid_directory(self):
        """Test initialization with non-existent directory."""
        with pytest.raises(ValueError, match="does not exist"):
            DataLoader("/nonexistent/path")


class TestLoadDataset:
    """Tests for load_dataset method."""

    def test_load_csv(self, temp_data_dir):
        """Test loading CSV file."""
        loader = DataLoader(str(temp_data_dir))
        df = loader.load_dataset("test_data.csv")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert "feature1" in df.columns
        assert "state" in df.columns

    def test_load_tsv(self, temp_data_dir):
        """Test loading TSV file."""
        loader = DataLoader(str(temp_data_dir))
        df = loader.load_dataset("test_data.tsv")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5

    def test_load_nonexistent_file(self, temp_data_dir):
        """Test loading non-existent file raises error."""
        loader = DataLoader(str(temp_data_dir))
        with pytest.raises(FileNotFoundError):
            loader.load_dataset("nonexistent.csv")

    def test_load_unsupported_format(self, temp_data_dir):
        """Test loading unsupported file format raises error."""
        (temp_data_dir / "test.xyz").write_text("data")

        loader = DataLoader(str(temp_data_dir))
        with pytest.raises(ValueError, match="Unsupported file type"):
            loader.load_dataset("test.xyz")

    def test_drop_duplicates(self, temp_data_dir):
        """Test that duplicates are dropped by default."""
        loader = DataLoader(str(temp_data_dir))
        df = loader.load_dataset("test_duplicates.csv", drop_duplicates=True)

        assert len(df) == 5

    def test_subset_columns(self, temp_data_dir):
        """Test subsetting to specific columns."""
        loader = DataLoader(str(temp_data_dir))
        df = loader.load_dataset("test_data.csv", subset_cols=["feature1", "state"])

        assert len(df.columns) == 2
        assert "feature1" in df.columns
        assert "state" in df.columns
        assert "feature2" not in df.columns


class TestLoadMultipleDatasets:
    """Tests for load_multiple_datasets method."""

    def test_load_multiple(self, temp_data_dir):
        """Test loading multiple datasets."""
        loader = DataLoader(str(temp_data_dir))
        configs = [
            {"filename": "test_data.csv", "name": "Dataset1"},
            {"filename": "test_data.tsv", "name": "Dataset2"},
        ]
        datasets = loader.load_multiple_datasets(configs)

        assert len(datasets) == 2
        assert datasets[0][1] == "Dataset1"
        assert datasets[1][1] == "Dataset2"
        assert isinstance(datasets[0][0], pd.DataFrame)


class TestValidateDatasetsCompatibility:
    """Tests for validate_datasets_compatibility method."""

    def test_compatible_datasets(self, temp_data_dir):
        """Test that identical datasets are compatible."""
        loader = DataLoader(str(temp_data_dir))

        df1 = pd.DataFrame(
            {
                "f1": [1, 2, 3],
                "f2": [4, 5, 6],
                "f3": [7, 8, 9],
                "f4": [10, 11, 12],
                "f5": [13, 14, 15],
                "state": [0, 1, 0],
            }
        )
        df2 = df1.copy()

        report = loader.validate_datasets_compatibility(
            [(df1, "Dataset1"), (df2, "Dataset2")], target_col="state"
        )

        assert report["compatible"]
        assert len(report["issues"]) == 0

    def test_missing_target_column(self, temp_data_dir):
        """Test detection of missing target column."""
        loader = DataLoader(str(temp_data_dir))

        df1 = pd.DataFrame({"feature1": [1, 2, 3]})
        df2 = pd.DataFrame({"feature1": [1, 2, 3], "state": [0, 1, 0]})

        report = loader.validate_datasets_compatibility(
            [(df1, "NoTarget"), (df2, "WithTarget")], target_col="state"
        )

        assert not report["compatible"]
        assert any("target" in issue.lower() for issue in report["issues"])

    def test_different_target_classes(self, temp_data_dir):
        """Test detection of different target classes."""
        loader = DataLoader(str(temp_data_dir))

        df1 = pd.DataFrame({"f": [1, 2, 3], "state": [0, 1, 0]})
        df2 = pd.DataFrame({"f": [1, 2, 3], "state": [0, 1, 2]})

        report = loader.validate_datasets_compatibility(
            [(df1, "Binary"), (df2, "MultiClass")], target_col="state"
        )

        assert not report["compatible"]
        assert any("class" in issue.lower() for issue in report["issues"])


class TestCreateFeatureReport:
    """Tests for create_feature_report method."""

    def test_basic_report(self, temp_data_dir):
        """Test creating a basic feature report."""
        loader = DataLoader(str(temp_data_dir))
        df = loader.load_dataset("test_data.csv")

        report = loader.create_feature_report(df, target_col="state")

        assert "shape" in report
        assert "columns" in report
        assert "feature_stats" in report
        assert "target_distribution" in report
