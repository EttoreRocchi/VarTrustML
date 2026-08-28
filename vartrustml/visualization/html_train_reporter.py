"""HTML report generator for single model training experiments."""

import logging
from datetime import datetime
from html import escape as h
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, PackageLoader, select_autoescape

from vartrustml.visualization._html_mixins import render_confusion_matrix_html
from vartrustml.visualization._html_styles import REPORT_JS, get_report_css

logger = logging.getLogger(__name__)


class HTMLTrainReporter:
    """Generate interactive HTML reports for single model training.

    Creates HTML reports with Plotly visualizations for standalone model
    training experiments with hyperparameter tuning results, test metrics,
    and feature importance.

    Parameters
    ----------
    output_path : str, default="train_report.html"
        Path to save the HTML report.

    Attributes
    ----------
    output_path : pathlib.Path
        Path for the HTML report output.
    sections : list of str
        List of report section HTML strings.

    See Also
    --------
    HTMLCompareReporter : Reports for multi-model comparison.
    ModelTrainer : Generates training results for this reporter.

    Examples
    --------
    >>> reporter = HTMLTrainReporter("results/train_report.html")
    >>> reporter.add_training_overview(config, dataset_info, "XGBoost")
    >>> reporter.add_test_results(test_metrics)
    >>> reporter.generate_report()
    """

    def __init__(self, output_path: str = "train_report.html"):
        self.output_path = Path(output_path)
        self.sections = []
        self.env = Environment(
            loader=PackageLoader("vartrustml.visualization", "templates"),
            autoescape=select_autoescape(["html"]),
        )

    def add_training_overview(
        self,
        config: Dict[str, Any],
        dataset_info: Dict[str, Any],
        model_name: str,
        run_metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Add training overview section with optional run metadata.

        Parameters
        ----------
        config : Dict[str, Any]
            Training configuration
        dataset_info : Dict[str, Any]
            Dataset information (shape, class distribution, etc.)
        model_name : str
            Name of the model being trained
        run_metadata : Optional[Dict[str, Any]]
            Optional dict with vartrustml_version, input_file, output_dir, python_version
        """
        # Format calibration info
        calibration_enabled = config.get("calibrate_model", False)
        if calibration_enabled:
            calibration_method = config.get("calibration_method", "N/A")
            calibration_cv = config.get("calibration_cv", "N/A")
            calibration_info = (
                f"Enabled ({calibration_method}, {calibration_cv}-fold CV)"
            )
        else:
            calibration_info = "Disabled"

        # Format test set info
        test_set_info = "None"
        if dataset_info.get("has_test_set", False):
            test_set_info = f"{dataset_info.get('test_size', 'N/A')} samples"
            if dataset_info.get("test_split_ratio"):
                test_set_info += f" ({dataset_info.get('test_split_ratio'):.1%} split)"

        # Build run metadata section if provided
        run_metadata_html = ""
        if run_metadata:
            run_metadata_html = """
            <h3>Run Metadata</h3>
            <table class="info-table">
                <tr><th>Property</th><th>Value</th></tr>
        """
            if run_metadata.get("vartrustml_version"):
                run_metadata_html += f"<tr><td>VarTrustML Version</td><td>{h(str(run_metadata['vartrustml_version']))}</td></tr>"
            if run_metadata.get("python_version"):
                run_metadata_html += f"<tr><td>Python Version</td><td>{h(str(run_metadata['python_version']))}</td></tr>"
            if run_metadata.get("input_file"):
                run_metadata_html += f"<tr><td>Input Data File</td><td>{h(str(run_metadata['input_file']))}</td></tr>"
            if run_metadata.get("output_dir"):
                run_metadata_html += f"<tr><td>Output Directory</td><td>{h(str(run_metadata['output_dir']))}</td></tr>"
            run_metadata_html += "</table>"

        html = f"""
        <div class="section">
            <h2>Training Overview</h2>
            <table class="info-table">
                <tr><th>Parameter</th><th>Value</th></tr>
                <tr><td>Model</td><td><strong>{h(model_name)}</strong></td></tr>
                <tr><td>Random Seed</td><td>{h(str(config.get("seed", "N/A")))}</td></tr>
                <tr><td>CV Folds</td><td>{h(str(config.get("n_cv_folds", "N/A")))}</td></tr>
                <tr><td>Scoring Metric</td><td>{h(str(config.get("scoring", "N/A")))}</td></tr>
                <tr><td>Calibration</td><td>{h(calibration_info)}</td></tr>
                <tr><td>Test Set</td><td>{h(test_set_info)}</td></tr>
            </table>

            <h3>Dataset Information</h3>
            <table class="info-table">
                <tr><th>Property</th><th>Value</th></tr>
                <tr><td>Training Samples</td><td>{dataset_info.get("n_train_samples", "N/A")}</td></tr>
                <tr><td>Total Features</td><td>{dataset_info.get("n_features", "N/A")}</td></tr>
                <tr><td>Continuous Features</td><td>{dataset_info.get("n_continuous", "N/A")}</td></tr>
                <tr><td>Train Class 0 Count</td><td>{dataset_info.get("train_class_0_count", "N/A")}</td></tr>
                <tr><td>Train Class 1 Count</td><td>{dataset_info.get("train_class_1_count", "N/A")}</td></tr>
                <tr><td>Train Class Balance</td><td>{dataset_info.get("train_class_balance", "N/A")}</td></tr>
            </table>
            {run_metadata_html}
        </div>
        """
        self.sections.append(html)

    def add_hyperparameter_results(
        self,
        best_params: Dict[str, Any],
        best_score: float,
        cv_results: Optional[pd.DataFrame] = None,
    ):
        """
        Add hyperparameter tuning results.

        Parameters
        ----------
        best_params : Dict[str, Any]
            Best parameters found
        best_score : float
            Best CV score
        cv_results : Optional[pd.DataFrame]
            DataFrame with all CV results (optional)
        """
        html_parts = ['<div class="section">', "<h2>Hyperparameter Tuning Results</h2>"]

        html_parts.append(f"<p><strong>Best CV Score:</strong> {best_score:.4f}</p>")

        html_parts.append("<h3>Best Parameters</h3>")
        html_parts.append('<table class="info-table">')
        html_parts.append("<tr><th>Parameter</th><th>Value</th></tr>")

        for param, value in best_params.items():
            html_parts.append(
                f"<tr><td>{h(str(param))}</td><td>{h(str(value))}</td></tr>"
            )

        html_parts.append("</table>")

        html_parts.append("</div>")
        self.sections.append("".join(html_parts))

    def add_test_results(self, test_metrics: Dict[str, float]):
        """
        Add test set evaluation results.

        Parameters
        ----------
        test_metrics : Dict[str, float]
            Dictionary of test metrics
        """
        if not test_metrics:
            return

        html_parts = ['<div class="section">', "<h2>Test Set Performance</h2>"]
        html_parts.append('<table class="info-table">')
        html_parts.append("<tr><th>Metric</th><th>Score</th></tr>")

        # Order metrics for better display
        metric_order = [
            "AUROC",
            "Balanced Accuracy",
            "Matthews Corr. Coef.",
            "F1 (Weighted)",
            "Precision",
            "Recall",
            "Accuracy",
        ]

        # Display ordered metrics first
        for metric in metric_order:
            for key, value in test_metrics.items():
                if metric.lower() in key.lower():
                    html_parts.append(f"<tr><td>{h(key)}</td><td>{value:.4f}</td></tr>")
                    break

        # Display any remaining metrics
        displayed = set()
        for metric in metric_order:
            for key in test_metrics.keys():
                if metric.lower() in key.lower():
                    displayed.add(key)

        for key, value in test_metrics.items():
            if key not in displayed:
                if isinstance(value, (int, float)):
                    html_parts.append(f"<tr><td>{h(key)}</td><td>{value:.4f}</td></tr>")

        html_parts.append("</table>")

        # Add metrics visualization
        metric_names = []
        metric_values = []
        for key, value in test_metrics.items():
            if isinstance(value, (int, float)) and not key.endswith("_count"):
                metric_names.append(key)
                metric_values.append(value)

        if metric_names:
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=metric_names,
                    y=metric_values,
                    marker_color="darkturquoise",
                    text=[f"{v:.3f}" for v in metric_values],
                    textposition="outside",
                )
            )

            fig.update_layout(
                title="Test Set Metrics",
                title_font_size=18,
                xaxis_title="Metric",
                yaxis_title="Score",
                height=500,
                yaxis_range=[0, 1],
                font=dict(size=14),
            )

            fig.update_xaxes(title_font_size=16, tickfont=dict(size=13), tickangle=45)
            fig.update_yaxes(title_font_size=16, tickfont=dict(size=13))

            html_parts.append(
                fig.to_html(include_plotlyjs="cdn", div_id="test_metrics")
            )

        html_parts.append("</div>")
        self.sections.append("".join(html_parts))

    def add_confusion_matrix(
        self, confusion_matrix: np.ndarray, normalize: bool = True
    ):
        """Add confusion matrix visualization."""
        if confusion_matrix is None or confusion_matrix.size == 0:
            return
        self.sections.append(
            render_confusion_matrix_html(
                confusion_matrix, normalize, div_id="confusion_matrix"
            )
        )

    def add_feature_importance(
        self, feature_importances: np.ndarray, feature_names: List[str], top_n: int = 20
    ):
        """
        Add feature importance visualization.

        Parameters
        ----------
        feature_importances : np.ndarray
            Array of feature importances
        feature_names : List[str]
            List of feature names
        top_n : int
            Number of top features to display
        """
        if feature_importances is None or len(feature_importances) == 0:
            return

        # Get top N features
        indices = np.argsort(feature_importances)[-top_n:][::-1]
        top_features = [feature_names[idx] for idx in indices]
        top_importances = feature_importances[indices]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=top_importances,
                y=top_features,
                orientation="h",
                marker_color="darkturquoise",
            )
        )

        fig.update_layout(
            title=f"Top {top_n} Feature Importances",
            title_font_size=18,
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=max(500, top_n * 25),
            font=dict(size=14),
        )

        fig.update_xaxes(title_font_size=16, tickfont=dict(size=13))
        fig.update_yaxes(tickfont=dict(size=13))

        html = f"""
        <div class="section">
            <h2>Feature Importance</h2>
            {fig.to_html(include_plotlyjs="cdn", div_id="feature_importance")}
        </div>
        """
        self.sections.append(html)

    def add_data_summary(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        continuous_cols: Optional[List[str]] = None,
    ):
        """
        Add the data summary section.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features DataFrame
        y_train : pd.Series
            Training labels Series
        continuous_cols : Optional[List[str]]
            List of continuous feature column names
        """
        html_parts = ['<div class="section">', "<h2>Data Summary</h2>"]

        # Dataset dimensions
        n_samples, n_features = X_train.shape
        html_parts.append("<h3>Dataset Dimensions</h3>")
        html_parts.append('<table class="info-table">')
        html_parts.append("<tr><th>Property</th><th>Value</th></tr>")
        html_parts.append(f"<tr><td>Total Samples</td><td>{n_samples:,}</td></tr>")
        html_parts.append(f"<tr><td>Total Features</td><td>{n_features:,}</td></tr>")
        html_parts.append("</table>")

        # Class distribution
        class_counts = y_train.value_counts().sort_index()
        html_parts.append("<h3>Class Distribution</h3>")
        html_parts.append('<table class="info-table">')
        html_parts.append("<tr><th>Class</th><th>Count</th><th>Percentage</th></tr>")
        for cls, count in class_counts.items():
            pct = count / len(y_train) * 100
            html_parts.append(
                f"<tr><td>{h(str(cls))}</td><td>{count:,}</td><td>{pct:.2f}%</td></tr>"
            )
        html_parts.append("</table>")

        # Feature statistics for continuous columns
        if continuous_cols:
            cont_features = [col for col in continuous_cols if col in X_train.columns]
            if cont_features:
                html_parts.append("<h3>Continuous Feature Statistics</h3>")
                html_parts.append('<table class="info-table">')
                html_parts.append(
                    "<tr><th>Feature</th><th>Mean</th><th>Std</th><th>Min</th><th>Median</th><th>Max</th><th>Missing</th></tr>"
                )

                for col in cont_features:
                    col_data = X_train[col]
                    missing = col_data.isna().sum()
                    mean = col_data.mean()
                    std = col_data.std()
                    min_val = col_data.min()
                    median = col_data.median()
                    max_val = col_data.max()

                    html_parts.append(
                        f"<tr><td>{h(str(col))}</td>"
                        f"<td>{mean:.4f}</td>"
                        f"<td>{std:.4f}</td>"
                        f"<td>{min_val:.4f}</td>"
                        f"<td>{median:.4f}</td>"
                        f"<td>{max_val:.4f}</td>"
                        f"<td>{missing}</td></tr>"
                    )
                html_parts.append("</table>")

        # Missing values summary
        missing_counts = X_train.isna().sum()
        total_missing = missing_counts.sum()
        if total_missing > 0:
            html_parts.append("<h3>Missing Values Summary</h3>")
            html_parts.append('<table class="info-table">')
            html_parts.append(
                "<tr><th>Feature</th><th>Missing Count</th><th>Percentage</th></tr>"
            )
            for col, count in missing_counts[missing_counts > 0].items():
                pct = count / n_samples * 100
                html_parts.append(
                    f"<tr><td>{h(str(col))}</td><td>{count:,}</td><td>{pct:.2f}%</td></tr>"
                )
            html_parts.append("</table>")
        else:
            html_parts.append(
                "<p><strong>No missing values in the dataset.</strong></p>"
            )

        html_parts.append("</div>")
        self.sections.append("".join(html_parts))

    def add_cv_fold_details(
        self, cv_results_df: pd.DataFrame, scoring_metric: str, n_cv_folds: int
    ):
        """
        Add detailed CV fold results section.

        Parameters
        ----------
        cv_results_df : pd.DataFrame
            DataFrame from ``GridSearchCV.cv_results_``
        scoring_metric : str
            Name of the scoring metric used
        n_cv_folds : int
            Number of CV folds
        """
        if cv_results_df is None or cv_results_df.empty:
            return

        html_parts = ['<div class="section">', "<h2>Cross-Validation Fold Details</h2>"]

        # Find the best configuration index
        if "rank_test_score" in cv_results_df.columns:
            best_idx = cv_results_df["rank_test_score"].idxmin()
        else:
            best_idx = cv_results_df["mean_test_score"].idxmax()

        # Extract per-fold scores for the best configuration
        fold_cols = [
            col
            for col in cv_results_df.columns
            if col.startswith("split") and col.endswith("_test_score")
        ]

        if fold_cols:
            html_parts.append("<h3>Per-Fold Scores (Best Configuration)</h3>")
            html_parts.append('<table class="info-table">')
            html_parts.append("<tr><th>Fold</th><th>Score</th></tr>")

            fold_scores = []
            for i, col in enumerate(sorted(fold_cols)):
                score = cv_results_df.loc[best_idx, col]
                fold_scores.append(score)
                html_parts.append(f"<tr><td>Fold {i}</td><td>{score:.4f}</td></tr>")
            html_parts.append("</table>")

            # Variance analysis
            fold_scores = np.array(fold_scores)
            mean_score = fold_scores.mean()
            std_score = fold_scores.std()
            cv_coeff = (std_score / mean_score * 100) if mean_score != 0 else 0

            html_parts.append("<h3>Fold Variance Analysis</h3>")
            html_parts.append('<table class="info-table">')
            html_parts.append("<tr><th>Statistic</th><th>Value</th></tr>")
            html_parts.append(f"<tr><td>Mean Score</td><td>{mean_score:.4f}</td></tr>")
            html_parts.append(f"<tr><td>Std Score</td><td>{std_score:.4f}</td></tr>")
            html_parts.append(
                f"<tr><td>Coefficient of Variation</td><td>{cv_coeff:.2f}%</td></tr>"
            )
            html_parts.append(
                f"<tr><td>Min Score</td><td>{fold_scores.min():.4f}</td></tr>"
            )
            html_parts.append(
                f"<tr><td>Max Score</td><td>{fold_scores.max():.4f}</td></tr>"
            )
            html_parts.append(
                f"<tr><td>Score Range</td><td>{fold_scores.max() - fold_scores.min():.4f}</td></tr>"
            )
            html_parts.append("</table>")

        # Timing information
        if "mean_fit_time" in cv_results_df.columns:
            html_parts.append("<h3>Timing Information</h3>")
            html_parts.append('<table class="info-table">')
            html_parts.append("<tr><th>Metric</th><th>Value</th></tr>")

            mean_fit_time = cv_results_df.loc[best_idx, "mean_fit_time"]
            std_fit_time = (
                cv_results_df.loc[best_idx, "std_fit_time"]
                if "std_fit_time" in cv_results_df.columns
                else 0
            )
            html_parts.append(
                f"<tr><td>Mean Fit Time (s)</td><td>{mean_fit_time:.3f} ± {std_fit_time:.3f}</td></tr>"
            )

            if "mean_score_time" in cv_results_df.columns:
                mean_score_time = cv_results_df.loc[best_idx, "mean_score_time"]
                std_score_time = (
                    cv_results_df.loc[best_idx, "std_score_time"]
                    if "std_score_time" in cv_results_df.columns
                    else 0
                )
                html_parts.append(
                    f"<tr><td>Mean Score Time (s)</td><td>{mean_score_time:.3f} ± {std_score_time:.3f}</td></tr>"
                )

            html_parts.append("</table>")

        html_parts.append("</div>")
        self.sections.append("".join(html_parts))

    def add_full_search_history(self, cv_results_df: pd.DataFrame, scoring_metric: str):
        """
        Add complete hyperparameter search history section.

        Parameters
        ----------
        cv_results_df : pd.DataFrame
            DataFrame from ``GridSearchCV.cv_results_``
        scoring_metric : str
            Name of the scoring metric used
        """
        if cv_results_df is None or cv_results_df.empty:
            return

        html_parts = [
            '<div class="section">',
            "<h2>Full Hyperparameter Search History</h2>",
        ]

        # Get parameter columns
        param_cols = [col for col in cv_results_df.columns if col.startswith("param_")]

        # Sort by mean test score
        sorted_results = cv_results_df.sort_values(
            "mean_test_score", ascending=False
        ).reset_index(drop=True)

        # Build table
        html_parts.append(
            f"<p><strong>Total configurations tested:</strong> {len(sorted_results)}</p>"
        )
        html_parts.append('<div style="max-height: 500px; overflow-y: auto;">')
        html_parts.append('<table class="info-table">')

        # Header
        header = "<tr><th>Rank</th><th>Mean Score</th><th>Std Score</th>"
        for col in param_cols:
            param_name = (
                col.replace("param_", "")
                .replace("clf__", "")
                .replace("estimator__", "")
            )
            header += f"<th>{h(param_name)}</th>"
        header += "</tr>"
        html_parts.append(header)

        # Rows (all configurations)
        for idx, row in sorted_results.iterrows():
            rank = idx + 1
            mean_score = row["mean_test_score"]
            std_score = row.get("std_test_score", 0)

            row_html = (
                f"<tr><td>{rank}</td><td>{mean_score:.4f}</td><td>{std_score:.4f}</td>"
            )
            for col in param_cols:
                value = row[col]
                if pd.isna(value):
                    value = "N/A"
                elif isinstance(value, float):
                    value = f"{value:.4g}"
                row_html += f"<td>{h(str(value))}</td>"
            row_html += "</tr>"
            html_parts.append(row_html)

        html_parts.append("</table>")
        html_parts.append("</div>")

        # Score distribution summary
        html_parts.append("<h3>Score Distribution Summary</h3>")
        html_parts.append('<table class="info-table">')
        html_parts.append("<tr><th>Statistic</th><th>Value</th></tr>")
        scores = sorted_results["mean_test_score"]
        html_parts.append(f"<tr><td>Best Score</td><td>{scores.max():.4f}</td></tr>")
        html_parts.append(f"<tr><td>Worst Score</td><td>{scores.min():.4f}</td></tr>")
        html_parts.append(f"<tr><td>Mean Score</td><td>{scores.mean():.4f}</td></tr>")
        html_parts.append(f"<tr><td>Std Score</td><td>{scores.std():.4f}</td></tr>")
        html_parts.append(
            f"<tr><td>Score Range</td><td>{scores.max() - scores.min():.4f}</td></tr>"
        )
        html_parts.append("</table>")

        html_parts.append("</div>")
        self.sections.append("".join(html_parts))

    def add_threshold_results(
        self,
        threshold_result: Optional[Dict[str, Any]] = None,
        test_metrics_comparison: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        """
        Add threshold optimization results section.

        Parameters
        ----------
        threshold_result : Optional[Dict[str, Any]]
            Dictionary from ThresholdResult.to_dict() containing:
            - optimal_threshold: float
            - youden_j: float
            - method_used: str ('oof', 'cv', or 'auto')
            - sensitivity_at_threshold: float
            - specificity_at_threshold: float
            - fold_thresholds: list of float (if CV method)
            - n_samples: int
        test_metrics_comparison : Optional[Dict[str, Dict[str, float]]]
            Optional dict with 'default' and 'optimized' keys,
            each containing metric values for comparison
        """
        if threshold_result is None:
            return

        html_parts = ['<div class="section">', "<h2>Threshold Optimization</h2>"]

        # Main threshold info
        optimal_threshold = threshold_result.get("optimal_threshold", 0.5)
        youden_j = threshold_result.get("youden_j", 0)
        method_used = threshold_result.get("method_used", "unknown")
        sensitivity = threshold_result.get("sensitivity_at_threshold", 0)
        specificity = threshold_result.get("specificity_at_threshold", 0)
        n_samples = threshold_result.get("n_samples", 0)
        fold_thresholds = threshold_result.get("fold_thresholds")

        # Method description
        method_descriptions = {
            "oof": "Out-of-Fold (single threshold from pooled OOF predictions)",
            "cv": "Cross-Validation (average of per-fold optimal thresholds)",
            "auto": "Auto-selected based on sample size",
        }
        method_desc = method_descriptions.get(method_used, method_used)

        html_parts.append("<h3>Optimization Results</h3>")
        html_parts.append('<table class="info-table">')
        html_parts.append("<tr><th>Property</th><th>Value</th></tr>")
        html_parts.append(
            f"<tr><td>Optimal Threshold</td><td><strong>{optimal_threshold:.4f}</strong></td></tr>"
        )
        html_parts.append(
            f"<tr><td>Youden's J Statistic</td><td>{youden_j:.4f}</td></tr>"
        )
        html_parts.append(f"<tr><td>Method Used</td><td>{h(method_desc)}</td></tr>")
        html_parts.append(
            f"<tr><td>Sensitivity at Threshold</td><td>{sensitivity:.4f}</td></tr>"
        )
        html_parts.append(
            f"<tr><td>Specificity at Threshold</td><td>{specificity:.4f}</td></tr>"
        )
        html_parts.append(f"<tr><td>Samples Used</td><td>{n_samples:,}</td></tr>")
        html_parts.append("</table>")

        # Per-fold thresholds (if CV method was used)
        if fold_thresholds and len(fold_thresholds) > 0:
            html_parts.append("<h3>Per-Fold Thresholds</h3>")
            html_parts.append('<table class="info-table">')
            html_parts.append("<tr><th>Fold</th><th>Threshold</th></tr>")
            for i, t in enumerate(fold_thresholds):
                html_parts.append(f"<tr><td>Fold {i}</td><td>{t:.4f}</td></tr>")
            html_parts.append("</table>")

            # Statistics
            fold_arr = np.array(fold_thresholds)
            html_parts.append("<h4>Threshold Statistics Across Folds</h4>")
            html_parts.append('<table class="info-table">')
            html_parts.append("<tr><th>Statistic</th><th>Value</th></tr>")
            html_parts.append(f"<tr><td>Mean</td><td>{fold_arr.mean():.4f}</td></tr>")
            html_parts.append(f"<tr><td>Std</td><td>{fold_arr.std():.4f}</td></tr>")
            html_parts.append(f"<tr><td>Min</td><td>{fold_arr.min():.4f}</td></tr>")
            html_parts.append(f"<tr><td>Max</td><td>{fold_arr.max():.4f}</td></tr>")
            html_parts.append("</table>")

        # Test metrics comparison (if provided)
        if test_metrics_comparison:
            default_metrics = test_metrics_comparison.get("default", {})
            optimized_metrics = test_metrics_comparison.get("optimized", {})

            if default_metrics and optimized_metrics:
                html_parts.append("<h3>Test Set Metrics Comparison</h3>")
                html_parts.append(
                    "<p>Comparison of metrics at default threshold (0.5) vs optimized threshold:</p>"
                )
                html_parts.append('<table class="info-table">')
                html_parts.append(
                    "<tr><th>Metric</th><th>Default (0.5)</th><th>Optimized</th><th>Difference</th></tr>"
                )

                for metric in default_metrics:
                    if metric in optimized_metrics:
                        default_val = default_metrics[metric]
                        opt_val = optimized_metrics[metric]
                        diff = opt_val - default_val
                        diff_class = (
                            "positive" if diff > 0 else "negative" if diff < 0 else ""
                        )
                        diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
                        html_parts.append(
                            f"<tr><td>{h(metric)}</td><td>{default_val:.4f}</td>"
                            f"<td>{opt_val:.4f}</td><td class='{diff_class}'>{diff_str}</td></tr>"
                        )

                html_parts.append("</table>")

        html_parts.append("</div>")
        self.sections.append("".join(html_parts))

    def generate_report(self):
        """
        Generate and save the complete HTML report.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Use base template with shared CSS/JS
        base_tpl = self.env.get_template("base.html.j2")
        html_content = base_tpl.render(
            title="VarTrustML Training Report",
            subtitle="Generated automatically by VarTrustML (train command)",
            timestamp=timestamp,
            css_content=get_report_css("teal"),
            js_content=REPORT_JS,
            sections=self.sections,
        )

        # Save to file
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"HTML training report saved to: {self.output_path}")
        return str(self.output_path)
