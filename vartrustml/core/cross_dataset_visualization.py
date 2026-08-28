"""
Cross-dataset visualization helpers for VarTrustML.

Plotting functions split out of ``CrossDatasetEvaluator`` so that the
evaluator itself only orchestrates.

Functions
---------
plot_heatmap_with_uncertainty
    Render a heatmap showing mean +/- std for a single model/metric.
create_comparison_plots
    Render a 2x2 panel comparing same-dataset vs cross-dataset performance.
"""

import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_heatmap_with_uncertainty(
    mean_matrix: pd.DataFrame,
    std_matrix: pd.DataFrame,
    model_name: str,
    metric_name: str,
    dataset_names: List[str],
    output_dir: Path,
    n_outer_splits: int,
    figure_dpi: int,
) -> None:
    """Create a heatmap showing mean +/- std for a model/metric.

    This is a standalone version of the former
    ``CrossDatasetEvaluator._plot_heatmap_with_uncertainty`` method.

    Parameters
    ----------
    mean_matrix : pandas.DataFrame
        DataFrame of mean values.
    std_matrix : pandas.DataFrame
        DataFrame of std values.
    model_name : str
        Name of the model.
    metric_name : str
        Name of the metric.
    dataset_names : list of str
        List of dataset names.
    output_dir : pathlib.Path
        Directory to save the plot.
    n_outer_splits : int
        Number of outer CV folds (used in the plot title).
    figure_dpi : int
        DPI for saved figure.
    """
    plt.figure(figsize=(10, 8))

    # Create annotation matrix with mean +/- std
    annot_matrix = mean_matrix.copy().astype(object)
    for i in range(len(dataset_names)):
        for j in range(len(dataset_names)):
            mean_val = mean_matrix.iloc[i, j]
            std_val = std_matrix.iloc[i, j]
            if pd.notna(mean_val):
                annot_matrix.iloc[i, j] = f"{mean_val:.3f}\n+/-{std_val:.3f}"

    vmin = 0 if metric_name == "Matthews Corr. Coef." else 0.5
    vmax = 1
    cmap = "viridis"

    sns.heatmap(
        mean_matrix.astype(float),
        annot=annot_matrix,
        fmt="",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": f"{metric_name} (mean)"},
    )

    plt.title(
        f"{model_name}: {metric_name}\n(Mean +/- Std from {n_outer_splits}-fold CV)"
    )
    plt.ylabel("Training Dataset")
    plt.xlabel("Test Dataset")

    # Highlight diagonal (within-dataset performance)
    for i in range(len(dataset_names)):
        plt.gca().add_patch(
            plt.Rectangle((i, i), 1, 1, fill=False, edgecolor="red", lw=2)
        )

    plt.tight_layout()
    safe_metric = metric_name.replace(" ", "_").lower()
    plt.savefig(
        output_dir / f"{safe_metric}_heatmap_cv.png",
        dpi=figure_dpi,
        bbox_inches="tight",
    )
    plt.close()


def create_comparison_plots(
    results: Dict[str, Dict[str, pd.DataFrame]],
    results_std: Dict[str, Dict[str, pd.DataFrame]],
    dataset_names: List[str],
    output_dir: Path,
    figure_dpi: int,
) -> None:
    """Create plots comparing same-dataset vs cross-dataset performance.

    This is a standalone version of the former
    ``CrossDatasetEvaluator._create_comparison_plots`` method.

    Parameters
    ----------
    results : dict
        Mean results matrices.
    results_std : dict
        Std results matrices.
    dataset_names : list of str
        List of dataset names.
    output_dir : pathlib.Path
        Directory to save plots.
    figure_dpi : int
        DPI for saved figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    key_metrics = [
        "AUROC",
        "F1 Score (Weighted)",
        "Balanced Accuracy",
        "Matthews Corr. Coef.",
    ]

    for idx, metric in enumerate(key_metrics):
        ax = axes[idx]
        model_performance = []

        for model_name, model_results in results.items():
            if metric not in model_results:
                continue

            matrix = model_results[metric]
            std_matrix = results_std[model_name][metric]

            # Collect diagonal (same-dataset) and off-diagonal (cross-dataset) values
            diagonal_means: List[float] = []
            diagonal_stds: List[float] = []
            off_diagonal_means: List[float] = []
            off_diagonal_stds: List[float] = []

            for i, ds in enumerate(dataset_names):
                if pd.notna(matrix.loc[ds, ds]):
                    diagonal_means.append(matrix.loc[ds, ds])
                    diagonal_stds.append(std_matrix.loc[ds, ds])

            for i, train_ds in enumerate(dataset_names):
                for j, test_ds in enumerate(dataset_names):
                    if i != j and pd.notna(matrix.loc[train_ds, test_ds]):
                        off_diagonal_means.append(matrix.loc[train_ds, test_ds])
                        off_diagonal_stds.append(std_matrix.loc[train_ds, test_ds])

            if diagonal_means and off_diagonal_means:
                model_performance.append(
                    {
                        "Model": model_name,
                        "Same_Mean": np.mean(diagonal_means),
                        "Same_Std": np.mean(diagonal_stds),
                        "Cross_Mean": np.mean(off_diagonal_means),
                        "Cross_Std": np.mean(off_diagonal_stds),
                    }
                )

        if model_performance:
            perf_df = pd.DataFrame(model_performance)
            x = np.arange(len(perf_df))
            width = 0.35

            ax.bar(
                x - width / 2,
                perf_df["Same_Mean"],
                width,
                yerr=perf_df["Same_Std"],
                capsize=5,
                label="Same Dataset",
                alpha=0.8,
            )
            ax.bar(
                x + width / 2,
                perf_df["Cross_Mean"],
                width,
                yerr=perf_df["Cross_Std"],
                capsize=5,
                label="Cross Dataset",
                alpha=0.8,
            )

            ax.set_xlabel("Model")
            ax.set_ylabel(metric)
            ax.set_title(f"{metric}: Generalization")
            ax.set_xticks(x)
            ax.set_xticklabels(perf_df["Model"], rotation=45, ha="right")
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Cross-Dataset Generalization Analysis", fontsize=14)
    plt.tight_layout()
    plt.savefig(
        output_dir / "generalization_comparison.png",
        dpi=figure_dpi,
        bbox_inches="tight",
    )
    plt.close()
