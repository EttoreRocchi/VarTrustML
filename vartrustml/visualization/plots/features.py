"""Feature importance and SHAP summary plotting."""

import logging
import warnings
from pathlib import Path
from typing import List, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from vartrustml.analysis.error_analysis import FoldMetrics

logger = logging.getLogger(__name__)


def plot_feature_importances(
    fold_results: List[FoldMetrics],
    feature_names: List[str],
    model_name: str,
    output_dir: Path,
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 6),
) -> None:
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
    top_n : int, optional
        Number of top features to display. Defaults to 20.
    figsize : tuple of int, optional
        Base figure size in inches. Defaults to (10, 6).
    """
    importances = np.array(
        [
            fold.feature_importances
            for fold in fold_results
            if fold.feature_importances is not None
        ]
    )

    if len(importances) == 0:
        logger.warning(f"No feature importances available for {model_name}")
        return

    mean_importances = importances.mean(axis=0)
    std_importances = importances.std(axis=0)

    indices = np.argsort(mean_importances)[::-1][:top_n]

    base_w, base_h = figsize
    plt.figure(figsize=(base_w, max(base_h, len(indices) * 0.3)))

    y_pos = np.arange(len(indices))
    plt.barh(
        y_pos,
        mean_importances[indices],
        xerr=std_importances[indices],
        capsize=5,
        color="steelblue",
        alpha=0.8,
    )

    plt.yticks(y_pos, [feature_names[i] for i in indices])
    plt.xlabel("Importance")
    plt.title(f"{model_name}: Top {len(indices)} Feature Importances")
    plt.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / f"{model_name.replace(' ', '_')}_feature_importances.png",
        bbox_inches="tight",
    )
    plt.close()


def plot_shap_summary(
    shap_values_list: List[np.ndarray],
    X_test_list: List[np.ndarray],
    feature_names: List[str],
    model_name: str,
    output_dir: Path,
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 8),
) -> None:
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
    top_n : int, optional
        Number of top features to display. Defaults to 20.
    figsize : tuple of int, optional
        Figure size in inches. Defaults to (10, 8).
    """
    if not shap_values_list or shap_values_list[0] is None:
        logger.warning(f"No SHAP values available for {model_name}")
        return

    combined_shap = np.vstack(shap_values_list)
    combined_data = np.vstack(X_test_list)

    plt.figure(figsize=figsize)

    # Suppress FutureWarning from SHAP's internal numpy random seed usage
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=FutureWarning, message=".*NumPy global RNG.*"
        )
        shap.summary_plot(
            combined_shap,
            combined_data,
            feature_names=feature_names,
            show=False,
            max_display=top_n,
        )

    plt.title(f"{model_name}: SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(
        output_dir / f"{model_name.replace(' ', '_')}_shap_summary.png",
        bbox_inches="tight",
    )
    plt.close()
