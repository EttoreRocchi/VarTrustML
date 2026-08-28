"""
Report generation utilities for ML pipeline results.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from vartrustml.analysis.error_analysis import (
    FoldMetrics,
    resolve_importance_feature_names,
)

logger = logging.getLogger(__name__)


def create_summary_report(results: Dict[str, List[FoldMetrics]], output_dir: Path):
    """Create an overall summary report across all models.

    Parameters
    ----------
    results : dict
        Dictionary mapping model names to lists of FoldMetrics.
    output_dir : pathlib.Path
        Directory to save report.
    """
    # Handle empty results (e.g., from validation failure)
    if not results or len(results) == 0:
        logger.warning(
            "No results provided to create_summary_report. Skipping report generation."
        )
        return

    report_lines = []
    report_lines.append("MACHINE LEARNING PIPELINE SUMMARY REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    comparison_data = []

    for model_name, fold_results in results.items():
        all_metrics = [fold.metrics for fold in fold_results]
        metrics_df = pd.DataFrame(all_metrics)

        key_metrics = [
            "AUROC",
            "F1 Score (Weighted)",
            "Balanced Accuracy",
            "Matthews Corr. Coef.",
            "Precision (Class 1)",
            "Recall (Class 1)",
            "Precision (Class 0)",
            "Recall (Class 0)",
        ]

        model_summary = {"Model": model_name}

        for metric in key_metrics:
            if metric in metrics_df.columns:
                mean_val = metrics_df[metric].mean()
                std_val = metrics_df[metric].std()
                model_summary[metric] = f"{mean_val:.3f} ± {std_val:.3f}"

        comparison_data.append(model_summary)

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)

    report_lines.append("MODEL PERFORMANCE COMPARISON")
    report_lines.append("-" * 40)
    report_lines.append(comparison_df.to_string(index=False))
    report_lines.append("")

    report_lines.append("\nBEST MODEL BY METRIC")
    report_lines.append("-" * 40)

    # Metrics where lower is better
    lower_is_better = {"Brier Score", "ECE", "MCE"}

    for metric in key_metrics:
        best_model = None
        # Initialize based on whether lower or higher is better
        best_score = np.inf if metric in lower_is_better else -np.inf

        for model_name, fold_results in results.items():
            all_metrics = [fold.metrics for fold in fold_results]
            metrics_df = pd.DataFrame(all_metrics)

            if metric in metrics_df.columns:
                mean_score = metrics_df[metric].mean()
                # Compare based on whether lower or higher is better
                if metric in lower_is_better:
                    is_better = mean_score < best_score
                else:
                    is_better = mean_score > best_score

                if is_better:
                    best_score = mean_score
                    best_model = model_name

        if best_model:
            report_lines.append(f"{metric}: {best_model} ({best_score:.3f})")

    report_lines.append("\n\nERROR ANALYSIS SUMMARY")
    report_lines.append("-" * 40)

    for model_name, fold_results in results.items():
        report_lines.append(f"\n{model_name}:")

        total_errors = sum(len(fold.misclassified_samples) for fold in fold_results)
        report_lines.append(f"  Total misclassified samples: {total_errors}")

        high_conf_errors = []
        for fold in fold_results:
            if 0.9 in fold.error_analysis:
                high_conf_errors.append(fold.error_analysis[0.9]["n_high_conf_errors"])

        if high_conf_errors:
            report_lines.append(
                f"  Avg high-conf errors (≥0.9): {np.mean(high_conf_errors):.2f}"
            )

    with open(output_dir / "summary_report.txt", "w") as f:
        f.write("\n".join(report_lines))

    logger.info(f"Summary report saved to {output_dir / 'summary_report.txt'}")


def create_feature_importance_report(
    results: Dict[str, List[FoldMetrics]], feature_names: List[str], output_path: Path
) -> pd.DataFrame:
    """Create a feature importance report.

    Parameters
    ----------
    results : dict
        Model results mapping model names to lists of FoldMetrics.
    feature_names : list of str
        List of feature names.
    output_path : pathlib.Path
        Path to save report.

    Returns
    -------
    pandas.DataFrame
        DataFrame with feature importance rankings. Returns empty
        DataFrame if no results provided.
    """
    # Handle empty results (e.g., from validation failure)
    if not results or len(results) == 0:
        logger.warning(
            "No results provided to create_feature_importance_report. Returning empty DataFrame."
        )
        return pd.DataFrame()

    importance_data = []

    for model_name, fold_results in results.items():
        importances = [
            fold.feature_importances
            for fold in fold_results
            if fold.feature_importances is not None
        ]

        if importances:
            mean_importance = np.mean(importances, axis=0)
            std_importance = np.std(importances, axis=0)

            # Importances are in preprocessor output order, not input order
            names = resolve_importance_feature_names(
                fold_results, feature_names, expected_length=len(mean_importance)
            )

            for idx, (mean_val, std_val) in enumerate(
                zip(mean_importance, std_importance)
            ):
                importance_data.append(
                    {
                        "Model": model_name,
                        "Feature": names[idx],
                        "Mean_Importance": mean_val,
                        "Std_Importance": std_val,
                    }
                )

    importance_df = pd.DataFrame(importance_data)

    # Handle case where no models have feature importances
    if importance_df.empty:
        logger.warning("No feature importances available. Returning empty DataFrame.")
        return pd.DataFrame()

    importance_df["Rank"] = importance_df.groupby("Model")["Mean_Importance"].rank(
        ascending=False, method="dense"
    )

    importance_df = importance_df.sort_values(["Model", "Rank"])

    importance_df.to_csv(output_path, index=False)

    top_features = importance_df[importance_df["Rank"] <= 10].groupby("Feature").size()
    top_features = top_features.sort_values(ascending=False)

    logger.info(f"Top features appearing in top 10 across models:\n{top_features}")

    return importance_df
