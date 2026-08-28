"""Report generation for cross-validation experiments.

:class:`ReportGenerator` was split out of ``CrossValidationPipeline`` so
that report generation and pipeline orchestration stay separate.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from vartrustml.analysis.bootstrap import BootstrapAnalyzer
from vartrustml.analysis.error_analysis import (
    ErrorAnalyzer,
    FoldMetrics,
    resolve_importance_feature_names,
)
from vartrustml.config import ExperimentConfig
from vartrustml.core.metric_aggregator import MetricAggregator
from vartrustml.io.checkpoint import save_fold_results
from vartrustml.utils.reporting import create_summary_report
from vartrustml.visualization.plots import Visualizer

if TYPE_CHECKING:
    from vartrustml.core.caller_evaluator import CallerResult

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate reports and visualizations for cross-validation results.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration.
    visualizer : Visualizer
        Visualization generation handler.
    metric_aggregator : MetricAggregator
        Metric aggregation handler.
    error_analyzer : ErrorAnalyzer
        Error analysis handler.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        visualizer: Visualizer,
        metric_aggregator: MetricAggregator,
        error_analyzer: ErrorAnalyzer,
    ):
        self.config = config
        self.visualizer = visualizer
        self.metrics = metric_aggregator
        self.error_analyzer = error_analyzer

    def generate_reports(
        self,
        results: Dict[str, List[FoldMetrics]],
        dataset_name: str,
        feature_names: List[str],
        df: pd.DataFrame,
        caller_results: Optional[Dict[str, List[CallerResult]]] = None,
    ):
        """Generate all reports and visualizations.

        Parameters
        ----------
        results : dict
            Dictionary mapping model names to lists of FoldMetrics.
        dataset_name : str
            Name of the dataset being evaluated.
        feature_names : list of str
            Feature column names.
        df : pandas.DataFrame
            Full dataset DataFrame.
        caller_results : dict, optional
            Caller evaluation results keyed by caller name.
        """
        output_dir = Path(self.config.output_dir) / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)

        self.config.save(output_dir / "experiment_config.json")

        for model_name, fold_results in results.items():
            logger.info(f"\nGenerating reports for {model_name}")

            model_dir = output_dir / model_name.replace(" ", "_")
            model_dir.mkdir(parents=True, exist_ok=True)

            metrics_summary = self.metrics.aggregate_metrics(fold_results)
            metrics_summary.to_csv(model_dir / "metrics_summary.csv")

            error_analyses = [fold.error_analysis for fold in fold_results]

            if self.error_analyzer.confidence_thresholds:
                error_report = self.error_analyzer.generate_error_report(
                    error_analyses, model_name
                )
                error_report.to_csv(model_dir / "error_analysis_summary.csv")

            save_fold_results(fold_results, model_dir / "folds")

            all_misclassified = pd.concat(
                [fold.misclassified_samples for fold in fold_results]
            )
            if len(all_misclassified) > 0:
                all_misclassified.to_csv(
                    model_dir / "all_misclassified_samples.csv", index=True
                )
                if self.error_analyzer.confidence_thresholds:
                    error_summary = self.error_analyzer.create_detailed_error_summary(
                        all_misclassified, model_name
                    )
                    joblib.dump(
                        error_summary, model_dir / "detailed_error_summary.joblib"
                    )

            best_params_df = pd.DataFrame(
                [
                    {**{"fold": fold.fold_id}, **(fold.best_params or {})}
                    for fold in fold_results
                ]
            )
            best_params_df.to_csv(model_dir / "best_parameters.csv", index=False)

            self.create_visualizations(
                fold_results, model_name, model_dir, feature_names
            )

        create_summary_report(results, output_dir)

        # Generate HTML report if requested
        if self.config.generate_html_report:
            self.generate_html_report(
                results, dataset_name, feature_names, output_dir, df, caller_results
            )

    def save_caller_results(
        self, caller_results: Dict[str, List[CallerResult]], dataset_name: str
    ) -> None:
        """Save caller evaluation results to CSV files.

        Parameters
        ----------
        caller_results : dict
            Dictionary mapping caller names to lists of CallerResult.
        dataset_name : str
            Name of the dataset.
        """
        output_dir = Path(self.config.output_dir) / dataset_name / "caller_comparison"
        output_dir.mkdir(parents=True, exist_ok=True)

        bootstrap = BootstrapAnalyzer(
            n_iterations=self.config.bootstrap.bootstrap_n_iterations,
            ci_level=self.config.bootstrap.bootstrap_ci_level,
            seed=self.config.cv.seed,
            ci_method=self.config.bootstrap.bootstrap_ci_method,
        )

        individual_metrics = []
        combination_metrics = []

        for name, results in caller_results.items():
            y_true_all = np.concatenate([r.y_true for r in results])
            y_pred_all = np.concatenate([r.y_pred for r in results])

            bootstrap_cis = bootstrap.compute_all_cis_from_predictions(
                y_true_all, y_pred_all, y_prob=None
            )

            row: Dict[str, Any] = {"name": name}
            for metric_name, ci_result in bootstrap_cis.items():
                row[f"{metric_name}_mean"] = ci_result.point_estimate
                row[f"{metric_name}_ci_lower"] = ci_result.ci_lower
                row[f"{metric_name}_ci_upper"] = ci_result.ci_upper
                row[f"{metric_name}_std"] = ci_result.std

            if " AND " in name or " OR " in name:
                combination_metrics.append(row)
            else:
                individual_metrics.append(row)

        if individual_metrics:
            df_individual = pd.DataFrame(individual_metrics)
            df_individual.to_csv(
                output_dir / "individual_caller_metrics.csv", index=False
            )
            logger.info(
                f"Individual caller metrics saved to {output_dir / 'individual_caller_metrics.csv'}"
            )

        if combination_metrics:
            df_combinations = pd.DataFrame(combination_metrics)
            df_combinations.to_csv(output_dir / "combination_metrics.csv", index=False)
            logger.info(
                f"Combination metrics saved to {output_dir / 'combination_metrics.csv'}"
            )

    def create_visualizations(
        self,
        fold_results: List[FoldMetrics],
        model_name: str,
        output_dir: Path,
        feature_names: List[str],
    ):
        """Create all visualizations for a model.

        Parameters
        ----------
        fold_results : list of FoldMetrics
            Per-fold evaluation results.
        model_name : str
            Display name of the model.
        output_dir : pathlib.Path
            Directory to save plots.
        feature_names : list of str
            Feature column names.
        """
        error_analyses = [fold.error_analysis for fold in fold_results]
        if self.error_analyzer.confidence_thresholds:
            error_report = self.error_analyzer.generate_error_report(
                error_analyses, model_name
            )
            self.visualizer.plot_error_analysis(error_report, model_name, output_dir)

        self.visualizer.plot_confusion_matrix(fold_results, model_name, output_dir)

        # Importances and SHAP values are indexed in preprocessor output order
        first_importances = fold_results[0].feature_importances
        importance_names = resolve_importance_feature_names(
            fold_results,
            feature_names,
            expected_length=(
                len(first_importances) if first_importances is not None else None
            ),
        )

        if fold_results[0].feature_importances is not None:
            self.visualizer.plot_feature_importances(
                fold_results, importance_names, model_name, output_dir
            )

        self.visualizer.plot_confidence_distribution(
            fold_results, model_name, output_dir
        )

        self.visualizer.plot_fold_consistency(fold_results, model_name, output_dir)

        if self.config.visualization.error_analysis_features:
            self.visualizer.plot_error_by_features(
                fold_results=fold_results,
                feature_names=self.config.visualization.error_analysis_features,
                model_name=model_name,
                output_dir=output_dir,
                continuous_cols=self.config.continuous_cols,
            )

        # Plot SHAP values if available
        shap_values = [
            fold.shap_values for fold in fold_results if fold.shap_values is not None
        ]
        X_test_transformed = [
            fold.X_test_transformed
            for fold in fold_results
            if fold.X_test_transformed is not None
        ]

        if (
            shap_values
            and X_test_transformed
            and len(shap_values) == len(X_test_transformed)
        ):
            try:
                self.visualizer.plot_shap_summary(
                    shap_values,
                    X_test_transformed,
                    importance_names,
                    model_name,
                    output_dir,
                )
                logger.info(f"SHAP summary plot created for {model_name}")
            except Exception as e:
                logger.warning(
                    f"Failed to create SHAP visualization for {model_name}: {e}"
                )

        # Plot reliability diagram (calibration curve)
        try:
            self.visualizer.plot_reliability_diagram(
                fold_results, model_name, output_dir
            )
        except Exception as e:
            logger.warning(
                f"Failed to create reliability diagram for {model_name}: {e}"
            )

    def generate_html_report(
        self,
        results: Dict[str, List[FoldMetrics]],
        dataset_name: str,
        feature_names: List[str],
        output_dir: Path,
        df: pd.DataFrame,
        caller_results: Optional[Dict[str, List[CallerResult]]] = None,
    ):
        """Generate interactive HTML report.

        Parameters
        ----------
        results : dict
            Dictionary mapping model names to lists of FoldMetrics.
        dataset_name : str
            Name of the dataset.
        feature_names : list of str
            Feature column names.
        output_dir : pathlib.Path
            Output directory for the report.
        df : pandas.DataFrame
            Full dataset DataFrame.
        caller_results : dict, optional
            Caller evaluation results.
        """
        logger.info("Generating HTML report...")

        if self.config.html_report_path:
            html_path = output_dir / self.config.html_report_path
        else:
            html_path = output_dir / "report.html"

        from vartrustml.visualization.html_compare_reporter import HTMLCompareReporter

        reporter = HTMLCompareReporter(output_path=str(html_path))

        self._add_overview_section(reporter, df, feature_names)

        correlation_matrix, feature_target_corr = self._compute_correlations(
            df, output_dir
        )

        (
            model_summaries,
            cv_results,
            confusion_matrices,
            feature_importances,
            oof_predictions,
            results_df,
        ) = self.metrics.aggregate_model_results(results, output_dir)

        reporter.add_feature_correlation(correlation_matrix)
        reporter.add_feature_target_correlation(feature_target_corr)
        reporter.add_best_models_table(results_df)
        reporter.add_cross_validation_results(cv_results)
        reporter.add_confusion_matrices(confusion_matrices)

        if feature_importances:
            importance_names = feature_names
            for fold_results in results.values():
                resolved = resolve_importance_feature_names(fold_results, None)
                if resolved:
                    importance_names = resolved
                    break
            reporter.add_feature_importance(
                feature_importances,
                importance_names,
                top_n=self.config.visualization.plot_top_n_features,
            )

        if caller_results:
            reporter.add_caller_comparison(
                caller_results=caller_results,
                ml_results=results,
                bootstrap_n_iterations=self.config.bootstrap.bootstrap_n_iterations,
                bootstrap_ci_level=self.config.bootstrap.bootstrap_ci_level,
                bootstrap_ci_method=self.config.bootstrap.bootstrap_ci_method,
                seed=self.config.cv.seed,
            )

        self._add_statistical_sections(
            reporter, caller_results, oof_predictions, output_dir
        )

        self._add_threshold_section(reporter, cv_results, output_dir)

        report_path = reporter.generate_report()
        logger.info(f"HTML report generated: {report_path}")

    def _add_overview_section(self, reporter, df, feature_names):
        """Add experiment overview with metadata to the HTML reporter.

        Parameters
        ----------
        reporter : HTMLCompareReporter
            Report builder instance.
        df : pandas.DataFrame
            Full dataset DataFrame.
        feature_names : list of str
            Feature column names.
        """
        config_dict = {
            "seed": self.config.cv.seed,
            "n_outer_splits": self.config.cv.n_outer_splits,
            "n_inner_splits": self.config.cv.n_inner_splits,
            "hpo_method": self.config.hpo_method,
            "calibrate_models": self.config.calibration.calibrate_models,
            "calibration_method": self.config.calibration.calibration_method,
            "calibration_cv": self.config.calibration.calibration_cv,
            "optimize_threshold": self.config.threshold.optimize_threshold,
            "threshold_method": self.config.threshold.threshold_method,
            "models_to_use": self.config.models_to_use,
            "bootstrap_n_iterations": self.config.bootstrap.bootstrap_n_iterations,
            "bootstrap_ci_level": self.config.bootstrap.bootstrap_ci_level,
            "bootstrap_ci_method": self.config.bootstrap.bootstrap_ci_method,
            "compare_callers": self.config.caller_comparison.compare_callers,
            "caller_columns": self.config.caller_comparison.caller_columns,
        }

        X = df.drop(columns=[self.config.target_column])
        y = df[self.config.target_column]

        n_continuous = (
            len(self.config.continuous_cols) if self.config.continuous_cols else 0
        )
        n_categorical = len(X.columns) - n_continuous

        class_counts = y.value_counts()
        class_0_count = class_counts.get(0, 0)
        class_1_count = class_counts.get(1, 0)
        class_0_pct = (class_0_count / len(y) * 100) if len(y) > 0 else 0.0
        class_1_pct = (class_1_count / len(y) * 100) if len(y) > 0 else 0.0
        class_balance = (
            f"{class_0_pct:.2f}% : {class_1_pct:.2f}%" if len(y) > 0 else "N/A"
        )

        dataset_info = {
            "n_samples": len(df),
            "n_features": len(X.columns),
            "n_continuous": n_continuous,
            "n_categorical": n_categorical,
            "class_0_count": class_0_count,
            "class_1_count": class_1_count,
            "class_balance": class_balance,
            "data_file_path": self.config.data_file_path or "N/A",
            "feature_list": feature_names,
            "continuous_features": self.config.continuous_cols,
            "target_column": self.config.target_column,
        }

        reporter.add_overview(config_dict, dataset_info)

    def _compute_correlations(self, df, output_dir):
        """Compute and save feature correlations.

        Parameters
        ----------
        df : pandas.DataFrame
            Full dataset DataFrame.
        output_dir : pathlib.Path
            Directory to save correlation CSVs.

        Returns
        -------
        tuple
            ``(correlation_matrix, feature_target_correlation)`` DataFrames.
        """
        X = df.drop(columns=[self.config.target_column])
        y = df[self.config.target_column]

        with np.errstate(invalid="ignore", divide="ignore"):
            correlation_matrix = X.corr()

        correlation_matrix.to_csv(output_dir / "feature_correlation_matrix.csv")
        logger.info(
            f"Feature correlation matrix saved to: {output_dir / 'feature_correlation_matrix.csv'}"
        )

        with np.errstate(invalid="ignore", divide="ignore"):
            feature_target_correlation = (
                pd.DataFrame(
                    {
                        "Feature": X.columns,
                        "Correlation": [X[col].corr(y) for col in X.columns],
                    }
                )
                .set_index("Feature")
                .sort_values("Correlation", ascending=False)
            )

        feature_target_correlation.to_csv(output_dir / "feature_target_correlation.csv")
        logger.info(
            f"Feature-target correlation saved to: {output_dir / 'feature_target_correlation.csv'}"
        )

        return correlation_matrix, feature_target_correlation

    def _add_statistical_sections(
        self, reporter, caller_results, oof_predictions, output_dir
    ):
        """Add the paired pairwise statistical comparison section.

        Builds aligned pooled out-of-fold entities (ML models + variant
        callers), runs the paired McNemar / DeLong comparison, exports the full
        comparison matrix and the DeLong subset to CSV, and renders the HTML
        section.

        Parameters
        ----------
        reporter : HTMLCompareReporter
            Report builder instance.
        caller_results : dict or None
            Caller evaluation results (per-fold ``CallerResult`` lists).
        oof_predictions : dict
            Out-of-fold predictions per ML model (with operating-point
            ``y_pred`` and, where available, ``sample_indices``).
        output_dir : pathlib.Path
            Directory to save CSV exports.
        """
        from vartrustml.analysis.pairwise_comparison import (
            FAMILY_AUROC,
            build_entities,
            compare_pairwise,
            comparisons_to_dataframe,
        )

        if not oof_predictions:
            logger.warning(
                "No OOF predictions available; skipping statistical comparison"
            )
            return

        entities = build_entities(oof_predictions, caller_results)
        result = compare_pairwise(
            entities,
            primary_metric=self.config.model_comparison_metric,
            alpha=0.05,
            ci_level=self.config.bootstrap.bootstrap_ci_level,
            correction_method=self.config.correction_method,
        )
        if result is None:
            logger.warning(
                "Insufficient aligned entities for statistical comparison (need >= 2)"
            )
            return

        # Export the full comparison matrix and the DeLong subset to CSV.
        try:
            df = comparisons_to_dataframe(result)
            df.to_csv(output_dir / "pairwise_mcnemar_full.csv", index=False)
            delong_df = df[df["family"] == FAMILY_AUROC]
            if not delong_df.empty:
                delong_df.to_csv(output_dir / "auroc_delong.csv", index=False)
        except Exception as e:
            logger.warning(f"Could not export pairwise comparison CSVs: {e}")

        reporter.add_pairwise_comparison(
            result,
            output_dir=str(output_dir),
            bootstrap_n_iterations=self.config.bootstrap.bootstrap_n_iterations,
            bootstrap_ci_level=self.config.bootstrap.bootstrap_ci_level,
            bootstrap_ci_method=self.config.bootstrap.bootstrap_ci_method,
            seed=self.config.cv.seed,
        )

    def _add_threshold_section(self, reporter, cv_results, output_dir):
        """Add threshold optimization results section if enabled.

        Parameters
        ----------
        reporter : HTMLCompareReporter
            Report builder instance.
        cv_results : dict
            Cross-validation results per model.
        output_dir : pathlib.Path
            Directory containing per-model threshold files.
        """
        if not self.config.threshold.optimize_threshold:
            return

        threshold_summary = {}
        for model_name in cv_results.keys():
            model_dir = output_dir / model_name.replace(" ", "_")
            threshold_path = model_dir / "threshold.joblib"
            if threshold_path.exists():
                try:
                    threshold_summary[model_name] = joblib.load(threshold_path)
                except Exception as e:
                    logger.warning(f"Could not load threshold for {model_name}: {e}")

        if threshold_summary:
            reporter.add_threshold_results(threshold_summary)
