"""Model comparison, fold consistency, and reliability diagram plotting."""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from vartrustml.analysis.error_analysis import FoldMetrics
from vartrustml.visualization.colors import model_palette

logger = logging.getLogger(__name__)


def plot_metrics_comparison(
    all_results: Dict[str, List[FoldMetrics]],
    output_dir: Path,
    figsize: Tuple[int, int] = (14, 10),
) -> None:
    """Plot comparison of key metrics across all models.

    Parameters
    ----------
    all_results : dict of str to list of FoldMetrics
        Results from all models, keyed by model name.
    output_dir : pathlib.Path
        Directory to save the plot.
    figsize : tuple of int, optional
        Figure size in inches. Defaults to (14, 10).
    """
    metrics_data = []

    for model_name, fold_results in all_results.items():
        for fold in fold_results:
            for metric_name, value in fold.metrics.items():
                metrics_data.append(
                    {
                        "Model": model_name,
                        "Metric": metric_name,
                        "Value": value,
                        "Fold": fold.fold_id,
                    }
                )

    df = pd.DataFrame(metrics_data)

    # Alphabetical model order + fixed per-model colours, so the layout is
    # unambiguous and colours match the other figures.
    model_order = sorted(df["Model"].unique())
    palette = model_palette(model_order)

    key_metrics = [
        "F1 Score (Weighted)",
        "AUROC",
        "Balanced Accuracy",
        "Matthews Corr. Coef.",
    ]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.ravel()

    for idx, metric in enumerate(key_metrics):
        ax = axes[idx]
        metric_df = df[df["Metric"] == metric]

        sns.boxplot(
            x="Model",
            y="Value",
            data=metric_df,
            ax=ax,
            hue="Model",
            order=model_order,
            hue_order=model_order,
            palette=palette,
            legend=False,
            orient="v",
        )

        ax.set_title(metric)
        ax.set_xlabel("")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        ax.grid(True, alpha=0.3, axis="y")

        # Set y-limits based on metric type
        if metric == "Matthews Corr. Coef.":
            ax.set_ylim([-0.05, 1.05])
        else:  # AUROC, F1 Score, Balanced Accuracy
            ax.set_ylim([0.45, 1.05])

    plt.suptitle("Model Performance Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", bbox_inches="tight")
    plt.close()


def plot_fold_consistency(
    fold_results: List[FoldMetrics],
    model_name: str,
    output_dir: Path,
    figsize: Tuple[int, int] = (12, 10),
) -> None:
    """Analyze and plot consistency of errors across CV folds.

    Parameters
    ----------
    fold_results : list of FoldMetrics
        Results from each CV fold.
    model_name : str
        Name of the model for plot title.
    output_dir : pathlib.Path
        Directory to save the plot and statistics CSV.
    figsize : tuple of int, optional
        Figure size in inches. Defaults to (12, 10).
    """
    fold_data = []

    for fold in fold_results:
        fold_dict = {
            "fold_id": fold.fold_id,
            "total_errors": len(fold.misclassified_samples),
        }

        for threshold, analysis in fold.error_analysis.items():
            fold_dict[f"errors_{threshold}"] = analysis["n_high_conf_errors"]
            fold_dict[f"pct_{threshold}"] = analysis["pct_high_conf_errors"]

        fold_data.append(fold_dict)

    fold_df = pd.DataFrame(fold_data)

    plt.figure(figsize=figsize)

    error_cols = [
        col
        for col in fold_df.columns
        if col.startswith("errors_") and not col == "errors_total"
    ]

    for col in error_cols:
        threshold = float(col.split("_")[1])
        plt.plot(
            fold_df["fold_id"],
            fold_df[col],
            marker="o",
            label=f"Threshold {threshold}",
        )

    plt.xlabel("Fold ID")
    plt.ylabel("Number of High-Confidence Errors")
    plt.title(f"{model_name}: High-Confidence Errors by Fold")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        output_dir / f"{model_name.replace(' ', '_')}_fold_consistency.png",
        bbox_inches="tight",
    )
    plt.close()

    fold_df.to_csv(
        output_dir / f"{model_name.replace(' ', '_')}_fold_statistics.csv",
        index=False,
    )


def plot_reliability_diagram(
    fold_results: List[FoldMetrics],
    model_name: str,
    output_dir: Path,
    n_bins: int = 10,
    dpi: int = 300,
) -> None:
    """Plot reliability diagram (calibration curve) across CV folds.

    A reliability diagram shows how well predicted probabilities match
    actual frequencies. A perfectly calibrated model lies on the diagonal.

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
    dpi : int, optional
        DPI for saved figure. Defaults to 300.

    Notes
    -----
    The reliability diagram shows:
    - The diagonal line representing perfect calibration
    - The actual calibration curve (mean probability vs fraction positive)
    - A histogram showing the distribution of predictions across bins
    """
    from sklearn.calibration import calibration_curve

    # Aggregate predictions from all folds
    y_true_all = []
    y_prob_all = []

    for fold in fold_results:
        if hasattr(fold, "y_true_oof") and fold.y_true_oof is not None:
            y_true_all.extend(fold.y_true_oof)
        if hasattr(fold, "y_prob_oof") and fold.y_prob_oof is not None:
            y_prob_all.extend(fold.y_prob_oof)

    if not y_true_all or not y_prob_all:
        logger.warning(
            f"No prediction data available for reliability diagram: {model_name}"
        )
        return

    y_true_all = np.array(y_true_all)
    y_prob_all = np.array(y_prob_all)

    # Compute calibration curve
    try:
        prob_true, prob_pred = calibration_curve(
            y_true_all, y_prob_all, n_bins=n_bins, strategy="uniform"
        )
    except ValueError as e:
        logger.warning(f"Could not compute calibration curve for {model_name}: {e}")
        return

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]}
    )

    # Top plot: Reliability diagram
    ax1.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated", linewidth=2)
    ax1.plot(
        prob_pred,
        prob_true,
        "s-",
        color="steelblue",
        label=model_name,
        markersize=8,
        linewidth=2,
    )

    # Calculate and display calibration metrics
    from vartrustml.core.calibration import (
        expected_calibration_error,
        maximum_calibration_error,
    )

    ece = expected_calibration_error(y_true_all, y_prob_all, n_bins=n_bins)
    mce = maximum_calibration_error(y_true_all, y_prob_all, n_bins=n_bins)

    # Calculate Brier score
    from sklearn.metrics import brier_score_loss

    brier = brier_score_loss(y_true_all, y_prob_all)

    ax1.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax1.set_ylabel("Fraction of Positives", fontsize=11)
    ax1.set_title(
        f"{model_name}: Reliability Diagram\n"
        f"Brier = {brier:.4f}, ECE = {ece:.4f}, MCE = {mce:.4f}",
        fontsize=12,
    )
    ax1.legend(loc="lower right")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    ax1.grid(True, alpha=0.3)

    # Fill area between perfect calibration and actual curve
    ax1.fill_between(
        prob_pred,
        prob_pred,  # Perfect calibration line y=x
        prob_true,
        alpha=0.2,
        color="steelblue",
    )

    # Bottom plot: Histogram of predictions
    ax2.hist(
        y_prob_all,
        bins=n_bins,
        range=(0, 1),
        color="steelblue",
        alpha=0.7,
        edgecolor="black",
    )
    ax2.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_xlim([0.0, 1.0])
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(
        output_dir / f"{model_name.replace(' ', '_')}_reliability_diagram.png",
        bbox_inches="tight",
        dpi=dpi,
    )
    plt.close()

    logger.info(f"Reliability diagram created for {model_name}")
