"""
Unit tests for HTMLCrossDatasetReporter class.
"""

from pathlib import Path

import pandas as pd
import pytest

from vartrustml.visualization.html_cross_dataset_reporter import (
    HTMLCrossDatasetReporter,
)


@pytest.fixture
def sample_config():
    """Create sample configuration for testing."""
    return {
        "seed": 42,
        "n_outer_splits": 5,
        "n_inner_splits": 3,
        "calibrate_models": True,
        "calibration_method": "isotonic",
        "calibration_cv": 3,
        "optimize_threshold": True,
        "threshold_method": "oof",
        "hpo_method": "grid",
        "models_to_use": ["XGBoost", "Random Forest", "Logistic Regression"],
    }


@pytest.fixture
def sample_datasets_info():
    """Create sample dataset information."""
    return [
        {
            "name": "Dataset1",
            "n_samples": 1000,
            "n_features": 50,
            "class_distribution": {0: 600, 1: 400},
        },
        {
            "name": "Dataset2",
            "n_samples": 800,
            "n_features": 50,
            "class_distribution": {0: 500, 1: 300},
        },
        {
            "name": "Dataset3",
            "n_samples": 1200,
            "n_features": 50,
            "class_distribution": {0: 700, 1: 500},
        },
    ]


@pytest.fixture
def sample_results():
    """Create sample cross-dataset results."""
    dataset_names = ["Dataset1", "Dataset2", "Dataset3"]

    results_mean = {
        "XGBoost": {
            "AUROC": pd.DataFrame(
                [[0.90, 0.85, 0.82], [0.84, 0.88, 0.80], [0.83, 0.81, 0.87]],
                index=dataset_names,
                columns=dataset_names,
            ),
            "F1 Score (Weighted)": pd.DataFrame(
                [[0.85, 0.80, 0.78], [0.79, 0.83, 0.76], [0.78, 0.77, 0.82]],
                index=dataset_names,
                columns=dataset_names,
            ),
            "Matthews Corr. Coef.": pd.DataFrame(
                [[0.75, 0.70, 0.68], [0.69, 0.73, 0.66], [0.68, 0.67, 0.72]],
                index=dataset_names,
                columns=dataset_names,
            ),
        },
        "Random Forest": {
            "AUROC": pd.DataFrame(
                [[0.88, 0.82, 0.79], [0.81, 0.86, 0.78], [0.80, 0.79, 0.85]],
                index=dataset_names,
                columns=dataset_names,
            ),
            "F1 Score (Weighted)": pd.DataFrame(
                [[0.83, 0.77, 0.75], [0.76, 0.81, 0.74], [0.75, 0.74, 0.80]],
                index=dataset_names,
                columns=dataset_names,
            ),
            "Matthews Corr. Coef.": pd.DataFrame(
                [[0.72, 0.67, 0.65], [0.66, 0.70, 0.63], [0.65, 0.64, 0.69]],
                index=dataset_names,
                columns=dataset_names,
            ),
        },
    }

    results_std = {
        "XGBoost": {
            "AUROC": pd.DataFrame(
                [[0.03, 0.04, 0.05], [0.04, 0.03, 0.05], [0.05, 0.05, 0.03]],
                index=dataset_names,
                columns=dataset_names,
            ),
            "F1 Score (Weighted)": pd.DataFrame(
                [[0.04, 0.05, 0.06], [0.05, 0.04, 0.06], [0.06, 0.06, 0.04]],
                index=dataset_names,
                columns=dataset_names,
            ),
            "Matthews Corr. Coef.": pd.DataFrame(
                [[0.05, 0.06, 0.07], [0.06, 0.05, 0.07], [0.07, 0.07, 0.05]],
                index=dataset_names,
                columns=dataset_names,
            ),
        },
        "Random Forest": {
            "AUROC": pd.DataFrame(
                [[0.04, 0.05, 0.06], [0.05, 0.04, 0.06], [0.06, 0.06, 0.04]],
                index=dataset_names,
                columns=dataset_names,
            ),
            "F1 Score (Weighted)": pd.DataFrame(
                [[0.05, 0.06, 0.07], [0.06, 0.05, 0.07], [0.07, 0.07, 0.05]],
                index=dataset_names,
                columns=dataset_names,
            ),
            "Matthews Corr. Coef.": pd.DataFrame(
                [[0.06, 0.07, 0.08], [0.07, 0.06, 0.08], [0.08, 0.08, 0.06]],
                index=dataset_names,
                columns=dataset_names,
            ),
        },
    }

    return results_mean, results_std, dataset_names


class TestHTMLCrossDatasetReporterInit:
    """Tests for HTMLCrossDatasetReporter initialization."""

    def test_init_default(self):
        """Test default initialization."""
        reporter = HTMLCrossDatasetReporter()
        assert reporter.output_path == Path("cross_dataset_report.html")
        assert reporter.sections == []


class TestAddOverview:
    """Tests for add_overview method."""

    def test_add_overview(self, sample_config, sample_datasets_info):
        """Test adding overview section."""
        reporter = HTMLCrossDatasetReporter()
        reporter.add_overview(sample_config, sample_datasets_info)

        assert len(reporter.sections) == 1
        html = reporter.sections[0]

        assert "Experiment Overview" in html
        assert "42" in html
        assert "5" in html
        assert "Dataset1" in html
        assert "Dataset2" in html
        assert "Dataset3" in html


class TestAddPerformanceMatrices:
    """Tests for add_performance_matrices method."""

    def test_add_performance_matrices(self, sample_results):
        """Test adding performance matrices."""
        results_mean, results_std, dataset_names = sample_results

        reporter = HTMLCrossDatasetReporter()
        reporter.add_performance_matrices(results_mean, results_std, dataset_names)

        assert len(reporter.sections) == 1
        html = reporter.sections[0]

        assert "Performance Matrices" in html
        assert "XGBoost" in html
        assert "Random Forest" in html
        assert "AUROC" in html

    def test_add_performance_matrices_empty(self):
        """Test adding empty performance matrices."""
        reporter = HTMLCrossDatasetReporter()
        reporter.add_performance_matrices({}, {}, [])

        assert len(reporter.sections) == 0


class TestAddGeneralizationGapAnalysis:
    """Tests for add_generalization_gap_analysis method."""

    def test_add_generalization_gap(self, sample_results):
        """Test adding generalization gap analysis."""
        results_mean, _, dataset_names = sample_results

        reporter = HTMLCrossDatasetReporter()
        reporter.add_generalization_gap_analysis(results_mean, dataset_names)

        assert len(reporter.sections) == 1
        html = reporter.sections[0]

        assert "Generalization Gap" in html
        assert "Within-Dataset" in html
        assert "Cross-Dataset" in html


class TestAddBestWorstCombinations:
    """Tests for add_best_worst_combinations method."""

    def test_add_best_worst(self, sample_results):
        """Test adding best/worst combinations."""
        results_mean, _, dataset_names = sample_results

        reporter = HTMLCrossDatasetReporter()
        reporter.add_best_worst_combinations(results_mean, dataset_names)

        assert len(reporter.sections) == 1
        html = reporter.sections[0]

        assert "Top" in html
        assert "Bottom" in html


class TestAddCrossDatasetSummary:
    """Tests for add_cross_dataset_summary method."""

    def test_add_summary(self, sample_results):
        """Test adding executive summary."""
        results_mean, _, dataset_names = sample_results

        reporter = HTMLCrossDatasetReporter()
        reporter.add_cross_dataset_summary(results_mean, dataset_names)

        assert len(reporter.sections) == 1
        html = reporter.sections[0]

        assert "Executive Summary" in html
        assert "Model Summary" in html


class TestGenerateReport:
    """Tests for generate_report method."""

    def test_generate_report_creates_file(
        self, tmp_path, sample_config, sample_datasets_info, sample_results
    ):
        """Test that generate_report creates HTML file."""
        results_mean, results_std, dataset_names = sample_results

        output_path = tmp_path / "test_report.html"
        reporter = HTMLCrossDatasetReporter(output_path=str(output_path))

        reporter.add_overview(sample_config, sample_datasets_info)
        reporter.add_performance_matrices(results_mean, results_std, dataset_names)
        reporter.add_generalization_gap_analysis(results_mean, dataset_names)

        result_path = reporter.generate_report()

        assert output_path.exists()
        assert result_path == str(output_path)

    def test_generate_report_html_structure(
        self, tmp_path, sample_config, sample_datasets_info, sample_results
    ):
        """Test that generated HTML has correct structure."""
        results_mean, results_std, dataset_names = sample_results

        output_path = tmp_path / "test_report.html"
        reporter = HTMLCrossDatasetReporter(output_path=str(output_path))

        reporter.add_overview(sample_config, sample_datasets_info)
        reporter.generate_report()

        content = output_path.read_text()

        assert "<!DOCTYPE html>" in content
        assert "<html" in content  # May include lang attribute
        assert "</html>" in content
        assert "<head>" in content
        assert "<body>" in content
        assert "<style>" in content
        assert "Cross-Dataset Generalizability Report" in content


class TestFullReportGeneration:
    """Integration tests for full report generation."""

    def test_full_report_workflow(
        self, tmp_path, sample_config, sample_datasets_info, sample_results
    ):
        """Test complete report generation workflow."""
        results_mean, results_std, dataset_names = sample_results

        output_path = tmp_path / "full_report.html"
        reporter = HTMLCrossDatasetReporter(output_path=str(output_path))

        reporter.add_overview(sample_config, sample_datasets_info)
        reporter.add_performance_matrices(results_mean, results_std, dataset_names)
        reporter.add_generalization_gap_analysis(results_mean, dataset_names)
        reporter.add_best_worst_combinations(results_mean, dataset_names)
        reporter.add_cross_dataset_summary(results_mean, dataset_names)

        reporter.generate_report()

        assert output_path.exists()
        content = output_path.read_text()

        assert len(content) > 10000

        assert "Experiment Overview" in content
        assert "Performance Matrices" in content
        assert "Generalization Gap" in content
        assert "Top" in content
        assert "Executive Summary" in content
