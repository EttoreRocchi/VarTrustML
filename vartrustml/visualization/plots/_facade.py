"""
Thin facade class that preserves the original Visualizer public API.

Each method delegates to the corresponding standalone function,
passing configuration values as explicit parameters.

Classes
-------
Visualizer
    Handle all static visualization tasks using Matplotlib/Seaborn.

Notes
-----
Uses Agg backend for thread-safe non-interactive plotting.

See Also
--------
vartrustml.visualization.html_compare_reporter.HTMLCompareReporter : Interactive reports.
vartrustml.config.VisualizationConfig : Plot styling configuration.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import numpy as np

matplotlib.use("Agg")
import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from vartrustml.analysis.error_analysis import FoldMetrics
from vartrustml.config import ExperimentConfig, VisualizationConfig
from vartrustml.visualization.plots.comparison import (
    plot_fold_consistency,
    plot_metrics_comparison,
    plot_reliability_diagram,
)
from vartrustml.visualization.plots.confusion import plot_confusion_matrix
from vartrustml.visualization.plots.errors import (
    plot_confidence_distribution,
    plot_error_analysis,
    plot_error_by_feature,
    plot_error_by_features,
)
from vartrustml.visualization.plots.features import (
    plot_feature_importances,
    plot_shap_summary,
)

logger = logging.getLogger(__name__)


class Visualizer:
    """Handle all static visualization tasks for model evaluation.

    Creates plots for confusion matrices, ROC curves, feature importance,
    SHAP values, and error analysis using Matplotlib and Seaborn.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration containing output paths and settings.
    vis_config : VisualizationConfig, optional
        Visualization-specific settings. If None, creates one from
        ExperimentConfig values.

    Attributes
    ----------
    config : ExperimentConfig
        Experiment configuration.
    vis_config : VisualizationConfig
        Visualization settings (figure sizes, DPI, colors).

    See Also
    --------
    VisualizationConfig : Configure plot appearance.
    HTMLCompareReporter : Generate interactive HTML reports.

    Examples
    --------
    >>> from vartrustml import ExperimentConfig
    >>> from vartrustml.visualization import Visualizer
    >>> config = ExperimentConfig(figure_dpi=150)
    >>> viz = Visualizer(config)
    >>> viz.plot_confusion_matrix(fold_results, "XGBoost", output_dir)
    """

    def __init__(
        self, config: ExperimentConfig, vis_config: Optional[VisualizationConfig] = None
    ):
        self.config = config
        if vis_config is None:
            self.vis_config = VisualizationConfig(
                figure_dpi=config.visualization.figure_dpi,
                plot_top_n_features=config.visualization.plot_top_n_features,
            )
        else:
            self.vis_config = vis_config
        self.set_style()

    def set_style(self):
        """Set matplotlib and seaborn visual style."""
        sns.set_theme(style="darkgrid", palette="husl")
        plt.rcParams["figure.dpi"] = self.vis_config.figure_dpi
        plt.rcParams["savefig.dpi"] = self.vis_config.figure_dpi
        plt.rcParams["font.size"] = self.vis_config.default_font_size

    def plot_error_analysis(
        self, error_report: pd.DataFrame, model_name: str, output_dir: Path
    ):
        """Plot error analysis results across confidence thresholds.

        Parameters
        ----------
        error_report : pandas.DataFrame
            Error analysis summary from ErrorAnalyzer.generate_error_report().
        model_name : str
            Name of the model for plot titles.
        output_dir : pathlib.Path
            Directory to save the plot.
        """
        plot_error_analysis(
            error_report=error_report,
            model_name=model_name,
            output_dir=output_dir,
            figsize=self.vis_config.figsize_error_analysis,
        )

    def plot_confusion_matrix(
        self, fold_results: List[FoldMetrics], model_name: str, output_dir: Path
    ):
        """Plot average confusion matrix across CV folds.

        Parameters
        ----------
        fold_results : list of FoldMetrics
            Results from each CV fold.
        model_name : str
            Name of the model for plot title.
        output_dir : pathlib.Path
            Directory to save the plot.
        """
        plot_confusion_matrix(
            fold_results=fold_results,
            model_name=model_name,
            output_dir=output_dir,
            figsize=self.vis_config.figsize_confusion_matrix,
        )

    def plot_feature_importances(
        self,
        fold_results: List[FoldMetrics],
        feature_names: List[str],
        model_name: str,
        output_dir: Path,
    ):
        """Plot feature importances with error bars across CV folds.

        Parameters
        ----------
        fold_results : list of FoldMetrics
            Results from each CV fold.
        feature_names : list of str
            Names of features in order.
        model_name : str
            Name of the model for plot title.
        output_dir : pathlib.Path
            Directory to save the plot.
        """
        plot_feature_importances(
            fold_results=fold_results,
            feature_names=feature_names,
            model_name=model_name,
            output_dir=output_dir,
            top_n=self.vis_config.plot_top_n_features,
            figsize=self.vis_config.figsize_feature_importance,
        )

    def plot_confidence_distribution(
        self, fold_results: List[FoldMetrics], model_name: str, output_dir: Path
    ):
        """Plot confidence score distribution for misclassified samples.

        Parameters
        ----------
        fold_results : list of FoldMetrics
            Results from each CV fold.
        model_name : str
            Name of the model for plot title.
        output_dir : pathlib.Path
            Directory to save the plot.
        """
        plot_confidence_distribution(
            fold_results=fold_results,
            model_name=model_name,
            output_dir=output_dir,
            confidence_thresholds=self.config.confidence_thresholds,
            figsize=self.vis_config.figsize_confidence_dist,
        )

    def plot_error_by_feature(
        self,
        fold_results: List[FoldMetrics],
        feature_name: str,
        model_name: str,
        output_dir: Path,
        is_categorical: bool = False,
    ):
        """Plot error distribution for a specific feature.

        Parameters
        ----------
        fold_results : list of FoldMetrics
            Results from each CV fold.
        feature_name : str
            Name of feature to analyze.
        model_name : str
            Name of the model for plot title.
        output_dir : pathlib.Path
            Directory to save the plot.
        is_categorical : bool, default=False
            Whether the feature is categorical.
        """
        plot_error_by_feature(
            fold_results=fold_results,
            feature_name=feature_name,
            model_name=model_name,
            output_dir=output_dir,
            is_categorical=is_categorical,
            dpi=self.config.visualization.figure_dpi,
        )

    def plot_error_by_features(
        self,
        fold_results: List[FoldMetrics],
        feature_names: List[str],
        model_name: str,
        output_dir: Path,
        continuous_cols: List[str],
    ):
        """Plot error distributions for multiple features.

        Parameters
        ----------
        fold_results : list of FoldMetrics
            Results from each CV fold.
        feature_names : list of str
            Names of features to analyze.
        model_name : str
            Name of the model for plot titles.
        output_dir : pathlib.Path
            Directory to save plots.
        continuous_cols : list of str
            List of continuous feature names (others treated as categorical).
        """
        plot_error_by_features(
            fold_results=fold_results,
            feature_names=feature_names,
            model_name=model_name,
            output_dir=output_dir,
            continuous_cols=continuous_cols,
            dpi=self.config.visualization.figure_dpi,
        )

    def plot_shap_summary(
        self,
        shap_values_list: List[np.ndarray],
        X_test_list: List[np.ndarray],
        feature_names: List[str],
        model_name: str,
        output_dir: Path,
    ):
        """Create SHAP summary plot from all CV folds.

        Parameters
        ----------
        shap_values_list : list of numpy.ndarray
            SHAP values from each fold.
        X_test_list : list of numpy.ndarray
            Test features from each fold.
        feature_names : list of str
            Names of features in order.
        model_name : str
            Name of the model for plot title.
        output_dir : pathlib.Path
            Directory to save the plot.
        """
        plot_shap_summary(
            shap_values_list=shap_values_list,
            X_test_list=X_test_list,
            feature_names=feature_names,
            model_name=model_name,
            output_dir=output_dir,
            top_n=self.vis_config.plot_top_n_features,
            figsize=self.vis_config.figsize_shap_summary,
        )

    def plot_metrics_comparison(
        self, all_results: Dict[str, List[FoldMetrics]], output_dir: Path
    ):
        """Plot comparison of key metrics across all models.

        Parameters
        ----------
        all_results : dict of str to list of FoldMetrics
            Results from all models, keyed by model name.
        output_dir : pathlib.Path
            Directory to save the plot.
        """
        plot_metrics_comparison(
            all_results=all_results,
            output_dir=output_dir,
            figsize=self.vis_config.figsize_model_comparison,
        )

    def plot_fold_consistency(
        self, fold_results: List[FoldMetrics], model_name: str, output_dir: Path
    ):
        """Analyze and plot consistency of errors across CV folds.

        Parameters
        ----------
        fold_results : list of FoldMetrics
            Results from each CV fold.
        model_name : str
            Name of the model for plot title.
        output_dir : pathlib.Path
            Directory to save the plot and statistics CSV.
        """
        plot_fold_consistency(
            fold_results=fold_results,
            model_name=model_name,
            output_dir=output_dir,
            figsize=self.vis_config.figsize_fold_consistency,
        )

    def plot_reliability_diagram(
        self,
        fold_results: List[FoldMetrics],
        model_name: str,
        output_dir: Path,
        n_bins: int = 10,
    ):
        """Plot reliability diagram (calibration curve) across CV folds.

        Parameters
        ----------
        fold_results : list of FoldMetrics
            Results from each CV fold containing y_true_oof and y_prob_oof.
        model_name : str
            Name of the model for plot title.
        output_dir : pathlib.Path
            Directory to save the plot.
        n_bins : int, default=10
            Number of bins for calibration curve.
        """
        plot_reliability_diagram(
            fold_results=fold_results,
            model_name=model_name,
            output_dir=output_dir,
            n_bins=n_bins,
            dpi=self.vis_config.figure_dpi,
        )
