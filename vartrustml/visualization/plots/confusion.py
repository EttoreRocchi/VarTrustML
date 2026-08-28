"""Confusion matrix plotting."""

from pathlib import Path
from typing import List, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from vartrustml.analysis.error_analysis import FoldMetrics


def plot_confusion_matrix(
    fold_results: List[FoldMetrics],
    model_name: str,
    output_dir: Path,
    figsize: Tuple[int, int] = (6, 5),
) -> None:
    """Plot average confusion matrix across CV folds.

    Parameters
    ----------
    fold_results : list of FoldMetrics
        Results from each CV fold.
    model_name : str
        Name of the model for plot title.
    output_dir : pathlib.Path
        Directory to save the plot.
    figsize : tuple of int, optional
        Figure size in inches. Defaults to (6, 5).
    """
    conf_matrices = [fold.confusion_matrix for fold in fold_results]
    mean_conf_matrix = np.mean(conf_matrices, axis=0)

    plt.figure(figsize=figsize)

    sns.heatmap(
        mean_conf_matrix,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        xticklabels=["Predicted 0", "Predicted 1"],
        yticklabels=["True 0", "True 1"],
        cbar_kws={"label": "Proportion"},
        vmin=0.0,
        vmax=1.0,
    )
    plt.title(f"{model_name}: Confusion Matrix Analysis", fontsize=14)
    plt.tight_layout()
    plt.savefig(
        output_dir / f"{model_name.replace(' ', '_')}_confusion_matrix.png",
        bbox_inches="tight",
    )
    plt.close()
