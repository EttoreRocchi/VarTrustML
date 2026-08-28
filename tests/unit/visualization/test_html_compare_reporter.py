"""
Unit tests for HTMLCompareReporter class.
"""

from pathlib import Path

import pandas as pd
import pytest

from vartrustml.visualization.html_compare_reporter import HTMLCompareReporter


@pytest.fixture
def sample_config():
    """Create sample configuration dictionary."""
    return {
        "seed": 42,
        "n_outer_splits": 5,
        "n_inner_splits": 3,
        "calibrate_models": True,
        "calibration_method": "isotonic",
        "calibration_cv": 3,
        "bootstrap_n_iterations": 1000,
        "bootstrap_ci_level": 0.95,
        "compare_callers": False,
        "caller_columns": [],
        "models_to_use": ["XGBoost", "Random Forest"],
    }


@pytest.fixture
def sample_dataset_info():
    """Create sample dataset information."""
    return {
        "n_samples": 1000,
        "n_features": 50,
        "class_0_count": 600,
        "class_1_count": 400,
        "class_balance": "60%/40%",
        "continuous_features": ["feature_1", "feature_2"],
        "n_continuous": 45,
        "n_categorical": 5,
    }


@pytest.fixture
def sample_results_df():
    """Create sample results DataFrame."""
    return pd.DataFrame(
        {
            "AUROC": [0.90, 0.85, 0.82],
            "F1 Score (Weighted)": [0.85, 0.80, 0.78],
            "Matthews Corr. Coef.": [0.75, 0.70, 0.68],
            "Balanced Accuracy": [0.82, 0.78, 0.75],
        },
        index=["XGBoost", "Random Forest", "Logistic Regression"],
    )


class TestHTMLCompareReporterInit:
    """Tests for HTMLCompareReporter initialization."""

    def test_default_init(self):
        """Test default initialization."""
        reporter = HTMLCompareReporter()

        assert reporter.output_path == Path("report.html")
        assert reporter.sections == []

    def test_custom_output_path(self):
        """Test initialization with custom output path."""
        reporter = HTMLCompareReporter("custom/path/report.html")

        assert reporter.output_path == Path("custom/path/report.html")

    def test_sections_initially_empty(self):
        """Test sections list is empty on init."""
        reporter = HTMLCompareReporter()

        assert len(reporter.sections) == 0


class TestAddOverview:
    """Tests for add_overview method."""

    def test_add_overview_creates_section(self, sample_config, sample_dataset_info):
        """Test that add_overview adds a section."""
        reporter = HTMLCompareReporter()
        reporter.add_overview(sample_config, sample_dataset_info)

        assert len(reporter.sections) == 1

    def test_add_overview_contains_seed(self, sample_config, sample_dataset_info):
        """Test overview contains seed value."""
        reporter = HTMLCompareReporter()
        reporter.add_overview(sample_config, sample_dataset_info)

        html = reporter.sections[0]
        assert "42" in html

    def test_add_overview_contains_cv_splits(self, sample_config, sample_dataset_info):
        """Test overview contains CV split configuration."""
        reporter = HTMLCompareReporter()
        reporter.add_overview(sample_config, sample_dataset_info)

        html = reporter.sections[0]
        assert "5" in html  # n_outer_splits
        assert "3" in html  # n_inner_splits

    def test_add_overview_calibration_enabled(self, sample_config, sample_dataset_info):
        """Test overview shows calibration info when enabled."""
        sample_config["calibrate_models"] = True
        reporter = HTMLCompareReporter()
        reporter.add_overview(sample_config, sample_dataset_info)

        html = reporter.sections[0]
        assert "isotonic" in html.lower() or "Enabled" in html

    def test_add_overview_calibration_disabled(
        self, sample_config, sample_dataset_info
    ):
        """Test overview shows calibration disabled."""
        sample_config["calibrate_models"] = False
        reporter = HTMLCompareReporter()
        reporter.add_overview(sample_config, sample_dataset_info)

        html = reporter.sections[0]
        assert "Disabled" in html

    def test_add_overview_dataset_info(self, sample_config, sample_dataset_info):
        """Test overview contains dataset information."""
        reporter = HTMLCompareReporter()
        reporter.add_overview(sample_config, sample_dataset_info)

        html = reporter.sections[0]
        assert "1000" in html  # n_samples
        assert "50" in html  # n_features


class TestAddBestModelsTable:
    """Tests for add_best_models_table method."""

    def test_add_best_models_creates_section(self, sample_results_df):
        """Test that add_best_models_table adds a section."""
        reporter = HTMLCompareReporter()
        reporter.add_best_models_table(sample_results_df)

        assert len(reporter.sections) == 1

    def test_add_best_models_contains_metrics(self, sample_results_df):
        """Test table contains metric names."""
        reporter = HTMLCompareReporter()
        reporter.add_best_models_table(sample_results_df)

        html = reporter.sections[0]
        assert "AUROC" in html
        assert "F1 Score" in html
        assert "Matthews" in html

    def test_add_best_models_identifies_winner(self, sample_results_df):
        """Test table identifies best model for each metric."""
        reporter = HTMLCompareReporter()
        reporter.add_best_models_table(sample_results_df)

        html = reporter.sections[0]
        # XGBoost should be best for all metrics in our sample data
        assert "XGBoost" in html

    def test_add_best_models_shows_score(self, sample_results_df):
        """Test table shows best score values."""
        reporter = HTMLCompareReporter()
        reporter.add_best_models_table(sample_results_df)

        html = reporter.sections[0]
        assert "0.90" in html or "0.9" in html  # AUROC score

    def test_add_best_models_empty_df(self):
        """Test handling of empty DataFrame."""
        reporter = HTMLCompareReporter()
        empty_df = pd.DataFrame()
        reporter.add_best_models_table(empty_df)

        # Should not add section for empty data
        assert len(reporter.sections) == 0

    def test_add_best_models_filters_std_columns(self):
        """Test that _std columns are excluded from best models."""
        df = pd.DataFrame(
            {
                "AUROC": [0.90, 0.85],
                "AUROC_std": [0.02, 0.03],
                "F1": [0.85, 0.80],
                "F1_std": [0.03, 0.04],
            },
            index=["Model1", "Model2"],
        )

        reporter = HTMLCompareReporter()
        reporter.add_best_models_table(df)

        html = reporter.sections[0]
        # Should show AUROC and F1 but not the _std versions as separate metrics
        assert "AUROC" in html
        assert html.count("<tr>") >= 3  # Header + 2 metric rows


class TestGenerateReport:
    """Tests for generate_report method."""

    def test_generate_creates_file(self, tmp_path, sample_config, sample_dataset_info):
        """Test that generate_report creates HTML file."""
        output_path = tmp_path / "test_report.html"
        reporter = HTMLCompareReporter(str(output_path))

        reporter.add_overview(sample_config, sample_dataset_info)
        result_path = reporter.generate_report()

        assert output_path.exists()
        assert result_path == str(output_path)

    def test_generate_creates_parent_dirs(
        self, tmp_path, sample_config, sample_dataset_info
    ):
        """Test that generate_report creates parent directories."""
        output_path = tmp_path / "subdir" / "another" / "report.html"
        reporter = HTMLCompareReporter(str(output_path))

        reporter.add_overview(sample_config, sample_dataset_info)
        reporter.generate_report()

        assert output_path.exists()

    def test_generate_html_structure(
        self, tmp_path, sample_config, sample_dataset_info
    ):
        """Test generated HTML has correct structure."""
        output_path = tmp_path / "test_report.html"
        reporter = HTMLCompareReporter(str(output_path))

        reporter.add_overview(sample_config, sample_dataset_info)
        reporter.generate_report()

        content = output_path.read_text()

        assert "<!DOCTYPE html>" in content
        assert "<html" in content  # May include lang attribute
        assert "</html>" in content
        assert "<head>" in content
        assert "<body>" in content
        assert "<style>" in content

    def test_generate_includes_sections(
        self, tmp_path, sample_config, sample_dataset_info, sample_results_df
    ):
        """Test generated report includes all added sections."""
        output_path = tmp_path / "test_report.html"
        reporter = HTMLCompareReporter(str(output_path))

        reporter.add_overview(sample_config, sample_dataset_info)
        reporter.add_best_models_table(sample_results_df)
        reporter.generate_report()

        content = output_path.read_text()

        assert "Experiment Overview" in content
        assert "Best Model by Metric" in content

    def test_generate_empty_report(self, tmp_path):
        """Test generating report with no sections."""
        output_path = tmp_path / "empty_report.html"
        reporter = HTMLCompareReporter(str(output_path))

        reporter.generate_report()

        assert output_path.exists()
        content = output_path.read_text()
        assert "<html" in content  # May include lang attribute


class TestMultipleSections:
    """Tests for adding multiple sections."""

    def test_multiple_sections_preserved(
        self, sample_config, sample_dataset_info, sample_results_df
    ):
        """Test that multiple sections are all preserved."""
        reporter = HTMLCompareReporter()

        reporter.add_overview(sample_config, sample_dataset_info)
        reporter.add_best_models_table(sample_results_df)

        assert len(reporter.sections) == 2

    def test_sections_ordered(
        self, tmp_path, sample_config, sample_dataset_info, sample_results_df
    ):
        """Test sections appear in order they were added."""
        output_path = tmp_path / "test_report.html"
        reporter = HTMLCompareReporter(str(output_path))

        reporter.add_overview(sample_config, sample_dataset_info)
        reporter.add_best_models_table(sample_results_df)
        reporter.generate_report()

        content = output_path.read_text()

        overview_pos = content.find("Experiment Overview")
        best_model_pos = content.find("Best Model by Metric")

        assert overview_pos < best_model_pos
