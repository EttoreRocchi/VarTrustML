"""
Visualization configuration for VarTrustML.

:class:`VisualizationConfig` centralises the plotting settings.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings.

    This dataclass centralizes all visualization-related parameters,
    allowing consistent styling across all plots.

    Attributes:
        figure_dpi (int): DPI for saved figures. Defaults to 300.
        default_font_size (int): Default font size for labels. Defaults to 10.
        title_font_size (int): Font size for plot titles. Defaults to 14.
        figsize_error_analysis (Tuple[int, int]): Size for error analysis plots. Defaults to (12, 5).
        figsize_confusion_matrix (Tuple[int, int]): Size for confusion matrices. Defaults to (6, 5).
        figsize_feature_importance (Tuple[int, int]): Base size for feature importance plots. Defaults to (10, 6).
        figsize_confidence_dist (Tuple[int, int]): Size for confidence distribution. Defaults to (10, 6).
        figsize_shap_summary (Tuple[int, int]): Size for SHAP summary plots. Defaults to (10, 8).
        figsize_model_comparison (Tuple[int, int]): Size for model comparison. Defaults to (14, 10).
        figsize_fold_consistency (Tuple[int, int]): Size for fold consistency plots. Defaults to (12, 10).
        plot_top_n_features (int): Number of top features to display. Defaults to 20.
    """

    # DPI settings
    figure_dpi: int = 300

    # Font sizes
    default_font_size: int = 10
    title_font_size: int = 14

    # Figure sizes (width, height) in inches
    figsize_error_analysis: Tuple[int, int] = (12, 5)
    figsize_confusion_matrix: Tuple[int, int] = (6, 5)
    figsize_feature_importance: Tuple[int, int] = (10, 6)
    figsize_confidence_dist: Tuple[int, int] = (10, 6)
    figsize_shap_summary: Tuple[int, int] = (10, 8)
    figsize_model_comparison: Tuple[int, int] = (14, 10)
    figsize_fold_consistency: Tuple[int, int] = (12, 10)

    # Feature display
    plot_top_n_features: int = 20
