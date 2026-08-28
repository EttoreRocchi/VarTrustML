"""
Unit tests for Visualizer class and plotting functions.
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from vartrustml.config import ExperimentConfig, VisualizationConfig
from vartrustml.config.experiment import CVConfig
from vartrustml.config.experiment import VisualizationConfig as ExpVisualizationConfig
from vartrustml.visualization.plots import Visualizer


@pytest.fixture
def experiment_config(tmp_path):
    """Create test experiment config."""
    return ExperimentConfig(
        cv=CVConfig(seed=42, n_outer_splits=3, n_inner_splits=2),
        visualization=ExpVisualizationConfig(figure_dpi=100, plot_top_n_features=10),
        output_dir=str(tmp_path / "test_output"),
    )


@pytest.fixture
def vis_config():
    """Create test visualization config."""
    return VisualizationConfig(
        figure_dpi=150,
        plot_top_n_features=5,
    )


@pytest.fixture
def mock_fold_metrics():
    """Create mock FoldMetrics object."""
    fold = MagicMock()
    fold.confusion_matrix = np.array([[0.8, 0.2], [0.1, 0.9]])
    fold.metrics = {"AUROC": 0.85, "F1 Score (Weighted)": 0.80}
    fold.feature_importances = np.array([0.3, 0.5, 0.2])
    fold.shap_values = None
    fold.X_test_transformed = None
    return fold


class TestVisualizerInit:
    """Tests for Visualizer initialization."""

    def test_init_with_experiment_config(self, experiment_config):
        """Test initialization with ExperimentConfig only."""
        viz = Visualizer(experiment_config)

        assert viz.config == experiment_config
        assert viz.vis_config.figure_dpi == experiment_config.visualization.figure_dpi
        assert (
            viz.vis_config.plot_top_n_features
            == experiment_config.visualization.plot_top_n_features
        )

    def test_init_with_vis_config(self, experiment_config, vis_config):
        """Test initialization with both configs."""
        viz = Visualizer(experiment_config, vis_config)

        assert viz.config == experiment_config
        assert viz.vis_config == vis_config
        assert viz.vis_config.figure_dpi == 150

    def test_set_style_called(self, experiment_config):
        """Test that set_style is called during initialization."""
        viz = Visualizer(experiment_config)
        # Just verify it doesn't raise - style is set internally
        assert viz is not None


class TestVisualizerConfusionMatrix:
    """Tests for confusion matrix plotting."""

    def test_plot_confusion_matrix(
        self, experiment_config, mock_fold_metrics, tmp_path
    ):
        """Test confusion matrix plot generation."""
        viz = Visualizer(experiment_config)
        fold_results = [mock_fold_metrics, mock_fold_metrics]

        viz.plot_confusion_matrix(fold_results, "TestModel", tmp_path)

        assert (tmp_path / "TestModel_confusion_matrix.png").exists()

    def test_plot_confusion_matrix_with_spaces(
        self, experiment_config, mock_fold_metrics, tmp_path
    ):
        """Test model name with spaces is handled correctly."""
        viz = Visualizer(experiment_config)

        viz.plot_confusion_matrix([mock_fold_metrics], "Random Forest", tmp_path)

        assert (tmp_path / "Random_Forest_confusion_matrix.png").exists()


class TestVisualizerFeatureImportances:
    """Tests for feature importance plotting."""

    def test_plot_feature_importances(
        self, experiment_config, mock_fold_metrics, tmp_path
    ):
        """Test feature importance plot generation."""
        viz = Visualizer(experiment_config)
        feature_names = ["feature_a", "feature_b", "feature_c"]

        viz.plot_feature_importances(
            [mock_fold_metrics, mock_fold_metrics],
            feature_names,
            "TestModel",
            tmp_path,
        )

        assert (tmp_path / "TestModel_feature_importances.png").exists()

    def test_plot_feature_importances_no_data(self, experiment_config, tmp_path):
        """Test handling when no feature importances available."""
        viz = Visualizer(experiment_config)
        fold = MagicMock()
        fold.feature_importances = None

        # Should not raise, just log warning
        viz.plot_feature_importances([fold], ["f1", "f2"], "Model", tmp_path)

        # No file created when no data
        assert not (tmp_path / "Model_feature_importances.png").exists()


class TestVisualizerErrorAnalysis:
    """Tests for error analysis plotting."""

    def test_plot_error_analysis(self, experiment_config, tmp_path):
        """Test error analysis plot generation."""
        viz = Visualizer(experiment_config)

        error_report = pd.DataFrame(
            {
                "confidence_threshold": [0.8, 0.9, 0.95],
                "mean_n_errors": [10, 5, 2],
                "std_n_errors": [2, 1, 0.5],
                "mean_pct_errors": [5.0, 2.5, 1.0],
                "std_pct_errors": [1.0, 0.5, 0.2],
                "mean_pct_of_all_errors": [50.0, 25.0, 10.0],
            }
        )

        viz.plot_error_analysis(error_report, "TestModel", tmp_path)

        assert (tmp_path / "TestModel_error_analysis.png").exists()


class TestVisualizationConfig:
    """Tests for VisualizationConfig defaults and customization."""

    def test_default_values(self):
        """Test default configuration values."""
        config = VisualizationConfig()

        assert config.figure_dpi == 300
        assert config.plot_top_n_features == 20
        assert config.default_font_size == 10

    def test_custom_values(self):
        """Test custom configuration values."""
        config = VisualizationConfig(
            figure_dpi=150,
            plot_top_n_features=10,
        )

        assert config.figure_dpi == 150
        assert config.plot_top_n_features == 10
