"""Error analysis and confidence distribution plotting."""

import logging
from collections import Counter
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vartrustml.analysis.error_analysis import FoldMetrics

logger = logging.getLogger(__name__)


def plot_error_analysis(
    error_report: pd.DataFrame,
    model_name: str,
    output_dir: Path,
    figsize: Tuple[int, int] = (12, 5),
) -> None:
    """Plot error analysis results across confidence thresholds.

    Parameters
    ----------
    error_report : pandas.DataFrame
        Error analysis summary from ErrorAnalyzer.generate_error_report().
    model_name : str
        Name of the model for plot titles.
    output_dir : pathlib.Path
        Directory to save the plot.
    figsize : tuple of int, optional
        Figure size in inches. Defaults to (12, 5).
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    ax = axes[0]
    ax.errorbar(
        error_report["confidence_threshold"],
        error_report["mean_n_errors"],
        yerr=error_report["std_n_errors"],
        marker="o",
        capsize=5,
        linestyle="none",
        markersize=8,
    )
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Number of High-Confidence Errors")
    ax.set_title("High-Confidence Errors by Threshold")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.errorbar(
        error_report["confidence_threshold"],
        error_report["mean_pct_errors"],
        yerr=error_report["std_pct_errors"],
        marker="o",
        capsize=5,
        color="orange",
        linestyle="none",
        markersize=8,
    )
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Percentage of Total Samples (%)")
    ax.set_title("High-Confidence Error Rate")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.errorbar(
        error_report["confidence_threshold"],
        error_report["mean_pct_of_all_errors"],
        marker="s",
        color="green",
        linestyle="none",
        markersize=8,
    )
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Percentage of All Errors (%)")
    ax.set_title("High-Conf Errors as % of All Errors")
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"{model_name}: Error Analysis Summary", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(
        output_dir / f"{model_name.replace(' ', '_')}_error_analysis.png",
        bbox_inches="tight",
    )
    plt.close()


def plot_confidence_distribution(
    fold_results: List[FoldMetrics],
    model_name: str,
    output_dir: Path,
    confidence_thresholds: Sequence[float] = (),
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """Plot confidence score distribution for misclassified samples.

    Parameters
    ----------
    fold_results : list of FoldMetrics
        Results from each CV fold.
    model_name : str
        Name of the model for plot title.
    output_dir : pathlib.Path
        Directory to save the plot.
    confidence_thresholds : sequence of float, optional
        Confidence thresholds to draw as vertical lines.
    figsize : tuple of int, optional
        Figure size in inches. Defaults to (10, 6).
    """
    all_confidences = []

    for fold in fold_results:
        if len(fold.misclassified_samples) > 0:
            all_confidences.extend(fold.misclassified_samples["confidence"].values)

    if not all_confidences:
        logger.warning(f"No misclassified samples for {model_name}")
        return

    plt.figure(figsize=figsize)

    n, bins, patches = plt.hist(
        all_confidences,
        bins=30,
        alpha=0.7,
        color="red",
        density=True,
        edgecolor="black",
    )

    mean_conf = np.mean(all_confidences)
    median_conf = np.median(all_confidences)

    plt.axvline(
        mean_conf,
        color="darkred",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_conf:.3f}",
    )
    plt.axvline(
        median_conf,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_conf:.3f}",
    )

    for threshold in confidence_thresholds:
        plt.axvline(threshold, color="gray", linestyle=":", alpha=0.5)

    plt.xlabel("Confidence Score")
    plt.ylabel("Density")
    plt.title(f"{model_name}: Confidence Distribution of Misclassified Samples")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)

    textstr = f"Total Errors: {len(all_confidences)}\n"
    textstr += f"Std Dev: {np.std(all_confidences):.3f}"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    plt.text(
        0.02,
        0.98,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )

    plt.tight_layout()
    plt.savefig(
        output_dir / f"{model_name.replace(' ', '_')}_confidence_distribution.png",
        bbox_inches="tight",
    )
    plt.close()


def plot_error_by_feature(
    fold_results: List[FoldMetrics],
    feature_name: str,
    model_name: str,
    output_dir: Path,
    is_categorical: bool = False,
    dpi: int = 300,
) -> None:
    """Plot error distribution for a specific feature.

    For categorical features, shows percentage of misclassifications by
    category for each fold. For continuous features, shows boxplots of
    feature values across folds.

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
    dpi : int, optional
        DPI for saved figure. Defaults to 300.
    """
    if is_categorical:
        # For categorical features: percentage of misclassifications by category per fold
        fold_data = []
        all_categories = set()

        for fold in fold_results:
            if feature_name in fold.misclassified_samples.columns:
                misclassified_values = fold.misclassified_samples[feature_name].values
                if len(misclassified_values) > 0:
                    counts = Counter(misclassified_values)
                    total = sum(counts.values())
                    # Convert to percentages
                    percentages = {
                        cat: (count / total) * 100 for cat, count in counts.items()
                    }
                    fold_data.append(
                        {
                            "fold_id": fold.fold_id,
                            "percentages": percentages,
                            "total": total,
                        }
                    )
                    all_categories.update(percentages.keys())

        if not fold_data:
            logger.warning(
                f"No misclassified data for categorical feature {feature_name}"
            )
            return

        # Create grouped bar chart
        categories = sorted(all_categories)
        n_folds = len(fold_data)
        n_cats = len(categories)

        fig, ax = plt.subplots(figsize=(max(10, n_cats * 1.5), 6))

        # Bar width and positions
        bar_width = 0.8 / n_folds if n_folds > 1 else 0.8
        x = np.arange(n_cats)

        # Plot bars for each fold
        for i, fold_info in enumerate(fold_data):
            percentages = [fold_info["percentages"].get(cat, 0) for cat in categories]
            offset = (i - n_folds / 2) * bar_width + bar_width / 2
            ax.bar(
                x + offset,
                percentages,
                bar_width,
                label=f"Fold {fold_info['fold_id']} (n={fold_info['total']})",
                alpha=0.8,
            )

        ax.set_xlabel(feature_name)
        ax.set_ylabel("Percentage of Misclassifications (%)")
        ax.set_title(f"{model_name}: Misclassification Distribution by {feature_name}")
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha="right")
        ax.legend(loc="best", fontsize="small")
        ax.grid(True, alpha=0.3, axis="y")

    else:
        # For continuous features: boxplot per fold
        fold_data = []
        fold_labels = []

        for fold in fold_results:
            if feature_name in fold.misclassified_samples.columns:
                misclassified_values = fold.misclassified_samples[feature_name].values
                if len(misclassified_values) > 0:
                    fold_data.append(misclassified_values)
                    fold_labels.append(
                        f"Fold {fold.fold_id}\n(n={len(misclassified_values)})"
                    )

        if not fold_data:
            logger.warning(
                f"No misclassified data for continuous feature {feature_name}"
            )
            return

        fig, ax = plt.subplots(figsize=(max(10, len(fold_data) * 1.5), 6))

        bp = ax.boxplot(
            fold_data,
            labels=fold_labels,
            patch_artist=True,
            showmeans=True,
            meanline=True,
        )

        # Color the boxplots
        for patch in bp["boxes"]:
            patch.set_facecolor("lightcoral")
            patch.set_alpha(0.7)

        ax.set_xlabel("Fold")
        ax.set_ylabel(feature_name)
        ax.set_title(f"{model_name}: {feature_name} Distribution in Misclassifications")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    safe_feature_name = feature_name.replace(" ", "_").replace("/", "_")
    plt.savefig(
        output_dir
        / f"{model_name.replace(' ', '_')}_{safe_feature_name}_error_dist.png",
        bbox_inches="tight",
        dpi=dpi,
    )
    plt.close()


def plot_error_by_features(
    fold_results: List[FoldMetrics],
    feature_names: List[str],
    model_name: str,
    output_dir: Path,
    continuous_cols: List[str],
    dpi: int = 300,
) -> None:
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
    dpi : int, optional
        DPI for saved figures. Defaults to 300.
    """
    if not feature_names:
        logger.info(f"No error analysis features specified for {model_name}")
        return

    logger.info(f"Creating error distribution plots for {len(feature_names)} features")

    for feature_name in feature_names:
        # Check if feature exists in misclassified samples
        feature_exists = False
        for fold in fold_results:
            if (
                len(fold.misclassified_samples) > 0
                and feature_name in fold.misclassified_samples.columns
            ):
                feature_exists = True
                break

        if not feature_exists:
            logger.warning(
                f"Feature '{feature_name}' not found in misclassified samples"
            )
            continue

        # Determine if feature is categorical or continuous
        is_categorical = feature_name not in continuous_cols

        try:
            plot_error_by_feature(
                fold_results=fold_results,
                feature_name=feature_name,
                model_name=model_name,
                output_dir=output_dir,
                is_categorical=is_categorical,
                dpi=dpi,
            )
            logger.info(f"Created error distribution plot for {feature_name}")
        except Exception as e:
            logger.warning(f"Failed to create error plot for {feature_name}: {e}")
