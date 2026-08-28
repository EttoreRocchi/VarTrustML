"""Metric aggregation for cross-validation results.

:class:`MetricAggregator` folds per-fold metrics into summaries carrying
bootstrap confidence intervals, threshold optimization results, and model
comparison data.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from vartrustml.analysis.bootstrap import BootstrapAnalyzer
from vartrustml.analysis.error_analysis import FoldMetrics

logger = logging.getLogger(__name__)


class MetricAggregator:
    """Aggregate fold-level metrics into summaries with bootstrap CIs.

    Handles concatenation of out-of-fold predictions, metric aggregation
    across folds, threshold result aggregation, and model result
    summarization for reporting.

    Parameters
    ----------
    bootstrap_n_iterations : int, default=1000
        Number of bootstrap resamples for confidence interval estimation.
    bootstrap_ci_level : float, default=0.95
        Confidence level for bootstrap CIs.
    seed : int, default=42
        Random seed for bootstrap reproducibility.
    optimize_threshold : bool, default=False
        Whether threshold optimization is enabled (affects OOF prediction
        concatenation behavior).
    bootstrap_ci_method : str, default="bca"
        Bootstrap CI method: ``"bca"`` (bias-corrected and accelerated) or
        ``"percentile"``.
    """

    def __init__(
        self,
        bootstrap_n_iterations: int = 1000,
        bootstrap_ci_level: float = 0.95,
        seed: int = 42,
        optimize_threshold: bool = False,
        bootstrap_ci_method: str = "bca",
    ):
        self.bootstrap_n_iterations = bootstrap_n_iterations
        self.bootstrap_ci_level = bootstrap_ci_level
        self.seed = seed
        self.optimize_threshold = optimize_threshold
        self.bootstrap_ci_method = bootstrap_ci_method

    def concatenate_oof_predictions(
        self,
        fold_results: List[FoldMetrics],
        use_optimized_threshold: bool = False,
    ) -> Tuple[
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """Concatenate out-of-fold predictions from all folds.

        Parameters
        ----------
        fold_results : list of FoldMetrics
            List of per-fold evaluation results containing OOF predictions.
        use_optimized_threshold : bool, default=False
            If True and threshold optimization was used, apply each fold's
            optimal threshold to convert probabilities to predictions.
            If False, use the default 0.5 threshold.

        Returns
        -------
        tuple of (y_true, y_pred, y_prob, sample_indices)
            Concatenated arrays if all folds have OOF data, otherwise
            (None, None, None, None). ``sample_indices`` is ``None``
            when any fold lacks index information.
        """
        y_true_list, y_prob_list, y_pred_list = [], [], []
        sample_indices_list: List[np.ndarray] = []

        for fold in fold_results:
            if fold.y_true_oof is None or fold.y_prob_oof is None:
                return None, None, None, None
            y_true_list.append(fold.y_true_oof)
            y_prob_list.append(fold.y_prob_oof)

            if fold.sample_indices is not None:
                sample_indices_list.append(fold.sample_indices)

            # Apply appropriate threshold
            if use_optimized_threshold and fold.fold_optimal_threshold is not None:
                threshold = fold.fold_optimal_threshold
            else:
                threshold = 0.5
            y_pred_list.append((fold.y_prob_oof >= threshold).astype(int))

        y_true_all = np.concatenate(y_true_list)
        y_prob_all = np.concatenate(y_prob_list)
        y_pred_all = np.concatenate(y_pred_list)
        sample_indices_all = (
            np.concatenate(sample_indices_list)
            if len(sample_indices_list) == len(fold_results)
            else None
        )

        return y_true_all, y_pred_all, y_prob_all, sample_indices_all

    def aggregate_metrics(self, fold_results: List[FoldMetrics]) -> pd.DataFrame:
        """Aggregate metrics across folds with bootstrap confidence intervals.

        Uses prediction-level bootstrap by
        resampling individual predictions rather than fold-level metrics.

        Parameters
        ----------
        fold_results : list of FoldMetrics
            List of per-fold evaluation results.

        Returns
        -------
        pandas.DataFrame
            Summary statistics with columns: mean, std, min, max, median,
            mean_bootstrap, ci_lower, ci_upper. Index contains metric names.
            The ci_lower/ci_upper bracket mean_bootstrap (OOF-based estimate).
        """
        all_metrics = [fold.metrics for fold in fold_results]
        metrics_df = pd.DataFrame(all_metrics)

        summary = pd.DataFrame(
            {
                "mean": metrics_df.mean(),
                "std": metrics_df.std(),
                "min": metrics_df.min(),
                "max": metrics_df.max(),
                "median": metrics_df.median(),
            }
        )

        # Prediction-level bootstrap: thousands of observations, not 5-10 folds
        bootstrap = BootstrapAnalyzer(
            n_iterations=self.bootstrap_n_iterations,
            ci_level=self.bootstrap_ci_level,
            seed=self.seed,
            ci_method=self.bootstrap_ci_method,
        )

        # Try to concatenate OOF predictions for prediction-level bootstrap
        y_true_all, y_pred_all, y_prob_all, _ = self.concatenate_oof_predictions(
            fold_results,
            use_optimized_threshold=self.optimize_threshold,
        )

        ci_lower = []
        ci_upper = []
        bootstrap_means = []

        if y_true_all is not None:
            # Prediction-level bootstrap
            ci_results = bootstrap.compute_all_cis_from_predictions(
                y_true_all, y_pred_all, y_prob_all
            )

            for metric_name in metrics_df.columns:
                if metric_name in ci_results:
                    ci_lower.append(ci_results[metric_name].ci_lower)
                    ci_upper.append(ci_results[metric_name].ci_upper)
                    bootstrap_means.append(ci_results[metric_name].point_estimate)
                else:
                    # Metrics without prediction-level CI (e.g., Brier, ECE)
                    # Use fold mean as fallback
                    ci_lower.append(summary.loc[metric_name, "mean"])
                    ci_upper.append(summary.loc[metric_name, "mean"])
                    bootstrap_means.append(summary.loc[metric_name, "mean"])
        else:
            # Fallback: OOF predictions unavailable
            logger.warning(
                "OOF predictions unavailable for bootstrap CI computation. "
                "Setting CI bounds to point estimates."
            )
            ci_lower = summary["mean"].tolist()
            ci_upper = summary["mean"].tolist()
            bootstrap_means = summary["mean"].tolist()

        summary["mean_bootstrap"] = bootstrap_means
        summary["ci_lower"] = ci_lower
        summary["ci_upper"] = ci_upper

        return summary

    def aggregate_threshold_results(
        self,
        all_results: Dict[str, List[FoldMetrics]],
        dataset_name: str,
        output_dir_base: Path,
    ):
        """Aggregate per-fold threshold optimization results for all models.

        Since threshold optimization is now performed within each fold's inner CV,
        this method aggregates the per-fold thresholds into a summary.

        Parameters
        ----------
        all_results : dict of {str: list of FoldMetrics}
            Dictionary mapping model names to fold results.
        dataset_name : str
            Name of the dataset.
        output_dir_base : Path
            Base output directory (``output_dir / dataset_name``).
        """
        output_dir = output_dir_base
        threshold_summary = {}

        for model_name, fold_results in all_results.items():
            # Collect per-fold threshold results
            fold_thresholds = []
            fold_youden_js = []
            fold_sensitivities = []
            fold_specificities = []

            for fold in fold_results:
                if fold.fold_optimal_threshold is not None:
                    fold_thresholds.append(fold.fold_optimal_threshold)
                if fold.fold_youden_j is not None:
                    fold_youden_js.append(fold.fold_youden_j)
                if fold.fold_sensitivity_at_threshold is not None:
                    fold_sensitivities.append(fold.fold_sensitivity_at_threshold)
                if fold.fold_specificity_at_threshold is not None:
                    fold_specificities.append(fold.fold_specificity_at_threshold)

            if not fold_thresholds:
                logger.warning(f"No threshold optimization results for {model_name}")
                continue

            # Compute aggregated statistics (convert to Python floats for JSON serialization)
            fold_thresholds_native = [float(t) for t in fold_thresholds]
            mean_threshold = float(np.mean(fold_thresholds))
            std_threshold = float(np.std(fold_thresholds))
            mean_youden_j = float(np.mean(fold_youden_js)) if fold_youden_js else None
            mean_sensitivity = (
                float(np.mean(fold_sensitivities)) if fold_sensitivities else None
            )
            mean_specificity = (
                float(np.mean(fold_specificities)) if fold_specificities else None
            )

            threshold_data = {
                "fold_thresholds": fold_thresholds_native,
                "mean_threshold": mean_threshold,
                "std_threshold": std_threshold,
                "recommended_threshold": mean_threshold,
                "mean_youden_j": mean_youden_j,
                "mean_sensitivity_at_threshold": mean_sensitivity,
                "mean_specificity_at_threshold": mean_specificity,
                "n_folds": len(fold_thresholds),
            }
            threshold_summary[model_name] = threshold_data

            # Save per-model threshold file
            model_dir = output_dir / model_name.replace(" ", "_")
            model_dir.mkdir(parents=True, exist_ok=True)
            threshold_path = model_dir / "threshold.joblib"
            joblib.dump(threshold_data, threshold_path)
            logger.debug(f"Threshold data saved to: {threshold_path}")

            youden_str = f"{mean_youden_j:.4f}" if mean_youden_j else "N/A"
            logger.info(
                f"{model_name}: Mean threshold = {mean_threshold:.4f} "
                f"(std = {std_threshold:.4f}, mean J = {youden_str})"
            )

    def aggregate_model_results(
        self,
        results: Dict[str, List[FoldMetrics]],
        output_dir: Path,
    ) -> Tuple:
        """Aggregate per-fold results into model summaries and comparison data.

        Parameters
        ----------
        results : dict of {str: list of FoldMetrics}
            Dictionary mapping model names to fold results.
        output_dir : Path
            Output directory for saving comparison CSV.

        Returns
        -------
        tuple
            (model_summaries, cv_results, confusion_matrices,
             feature_importances, oof_predictions, results_df)
        """
        model_summaries = {}
        cv_results = {}
        confusion_matrices = {}
        feature_importances = {}
        oof_predictions = {}

        for model_name, fold_results in results.items():
            if not fold_results:
                continue

            summary = self.aggregate_metrics(fold_results)
            model_summaries[model_name] = summary

            metrics_list = [fold.metrics for fold in fold_results]
            cv_results[model_name] = pd.DataFrame(metrics_list)

            y_true_oof, y_pred_oof, y_prob_oof, sample_indices_oof = (
                self.concatenate_oof_predictions(
                    fold_results,
                    use_optimized_threshold=self.optimize_threshold,
                )
            )
            if y_true_oof is not None and y_prob_oof is not None:
                # y_pred_oof uses each fold's operating point (optimized
                # threshold when enabled, else 0.5); needed for paired
                # operating-point comparisons (McNemar) against callers.
                oof_entry = {
                    "y_true": y_true_oof,
                    "y_pred": y_pred_oof,
                    "y_prob": y_prob_oof,
                }
                if sample_indices_oof is not None:
                    oof_entry["sample_indices"] = sample_indices_oof
                oof_predictions[model_name] = oof_entry

            confusion_matrices[model_name] = np.mean(
                [fold.confusion_matrix for fold in fold_results], axis=0
            )

            importances = [
                fold.feature_importances
                for fold in fold_results
                if fold.feature_importances is not None
            ]
            if importances:
                feature_importances[model_name] = np.mean(importances, axis=0)

        # Create results DataFrame
        results_df = pd.DataFrame(
            {
                model_name: summary["mean"]
                for model_name, summary in model_summaries.items()
            }
        ).T

        for model_name, summary in model_summaries.items():
            for metric in summary.index:
                results_df.loc[model_name, f"{metric}_std"] = summary.loc[metric, "std"]

        # Save consolidated comparison CSV
        comparison_rows = []
        for model_name, summary in model_summaries.items():
            row = {"model": model_name}
            for metric in summary.index:
                row[f"{metric}_mean"] = summary.loc[metric, "mean"]
                row[f"{metric}_mean_bootstrap"] = summary.loc[metric, "mean_bootstrap"]
                row[f"{metric}_std"] = summary.loc[metric, "std"]
                row[f"{metric}_ci_lower"] = summary.loc[metric, "ci_lower"]
                row[f"{metric}_ci_upper"] = summary.loc[metric, "ci_upper"]
            comparison_rows.append(row)

        comparison_df = pd.DataFrame(comparison_rows)
        comparison_df.to_csv(output_dir / "model_metrics_comparison.csv", index=False)
        logger.info(
            f"Model metrics comparison saved to: {output_dir / 'model_metrics_comparison.csv'}"
        )

        return (
            model_summaries,
            cv_results,
            confusion_matrices,
            feature_importances,
            oof_predictions,
            results_df,
        )
