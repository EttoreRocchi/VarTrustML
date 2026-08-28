"""
Error analysis module for inspecting misclassified samples.

Stratifies prediction errors by confidence and aggregates them across
cross-validation folds.

Classes
-------
FoldMetrics
    Dataclass storing metrics and analysis results for a single CV fold.
ErrorAnalyzer
    Error analysis for classification models.

Notes
-----
Error analysis is performed at multiple confidence thresholds to identify
high-confidence misclassifications, which are particularly problematic in
production settings.

See Also
--------
vartrustml.core.models.ModelEvaluator : Generates FoldMetrics during evaluation.
vartrustml.analysis.bootstrap.BootstrapAnalyzer : Bootstrap CIs from fold metrics.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FoldMetrics:
    """Store metrics and analysis results for a single cross-validation fold.

    Container for all evaluation outputs from a single fold, including
    classification metrics, error analysis, SHAP values, and threshold
    optimization results.

    Attributes
    ----------
    fold_id : int
        Identifier for this fold (0-indexed).
    metrics : dict of str to float
        Dictionary of computed metrics (AUROC, MCC, etc.) at the
        optimal or default threshold.
    confusion_matrix : numpy.ndarray
        Normalized confusion matrix at the optimal or default threshold.
        Shape is (2, 2) for binary classification.
    misclassified_samples : pandas.DataFrame
        DataFrame containing all wrongly predicted samples with their
        features, true labels, and prediction probabilities.
    error_analysis : dict of float to dict
        Dictionary mapping confidence thresholds to error statistics
        at each threshold level.
    shap_values : numpy.ndarray, optional
        SHAP values for model interpretability. Shape is (n_samples, n_features).
    feature_importances : numpy.ndarray, optional
        Feature importance scores from the trained model. Indexed in the
        preprocessor's output order, which is described by
        ``transformed_feature_names`` and is generally not the input column
        order.
    transformed_feature_names : list of str, optional
        Column names of the preprocessor's output, in the order that
        ``feature_importances``, ``shap_values`` and ``X_test_transformed``
        use. Must be preferred over the input column list when labelling
        any of those three.
    best_params : dict, optional
        Best hyperparameters found during inner CV optimization.
    X_test_transformed : numpy.ndarray, optional
        Preprocessed test data after applying the sklearn pipeline.
    y_true_oof : numpy.ndarray, optional
        True labels for test fold samples.
    y_prob_oof : numpy.ndarray, optional
        Predicted probabilities for positive class on test fold.
    sample_indices : numpy.ndarray, optional
        Original DataFrame index values for test fold samples,
        preserving traceability from predictions back to input rows.
    fold_optimal_threshold : float, optional
        Optimal classification threshold found on training data
        using Youden's J statistic.
    fold_youden_j : float, optional
        Youden's J statistic value at the optimal threshold.
    fold_sensitivity_at_threshold : float, optional
        Sensitivity (TPR) at the optimal threshold.
    fold_specificity_at_threshold : float, optional
        Specificity (TNR) at the optimal threshold.
    metrics_at_default_threshold : dict of str to float, optional
        Metrics computed at threshold=0.5 for comparison with
        optimized threshold results.

    See Also
    --------
    ModelEvaluator : Produces FoldMetrics during cross-validation.
    BootstrapAnalyzer : Computes confidence intervals from fold metrics.

    Examples
    --------
    >>> # FoldMetrics are typically created by ModelEvaluator
    >>> fold_result = FoldMetrics(
    ...     fold_id=0,
    ...     metrics={"AUROC": 0.95, "MCC": 0.85},
    ...     confusion_matrix=np.array([[0.9, 0.1], [0.05, 0.95]]),
    ...     misclassified_samples=pd.DataFrame(),
    ...     error_analysis={}
    ... )
    >>> fold_result.metrics["AUROC"]
    0.95
    """

    fold_id: int
    metrics: Dict[str, float]
    confusion_matrix: np.ndarray
    misclassified_samples: pd.DataFrame
    error_analysis: Dict[float, Dict[str, Any]]
    shap_values: Optional[np.ndarray] = None
    feature_importances: Optional[np.ndarray] = None
    transformed_feature_names: Optional[List[str]] = None
    best_params: Optional[Dict[str, Any]] = None
    X_test_transformed: Optional[np.ndarray] = None
    # Test fold data (for reference)
    y_true_oof: Optional[np.ndarray] = None
    y_prob_oof: Optional[np.ndarray] = None
    sample_indices: Optional[np.ndarray] = None
    # Threshold optimization results (computed on training data)
    fold_optimal_threshold: Optional[float] = None
    fold_youden_j: Optional[float] = None
    fold_sensitivity_at_threshold: Optional[float] = None
    fold_specificity_at_threshold: Optional[float] = None
    # Metrics at default threshold for comparison
    metrics_at_default_threshold: Optional[Dict[str, float]] = None


def resolve_importance_feature_names(
    fold_results: List[FoldMetrics],
    fallback: Optional[List[str]] = None,
    expected_length: Optional[int] = None,
) -> Optional[List[str]]:
    """Names that label feature importances and SHAP values.

    Those arrays live in the preprocessor's output order, so the input column
    list would mislabel them whenever the ColumnTransformer reorders columns.
    Returns the names recorded on the first fold that has them, falling back to
    the supplied input column list when no fold recorded any, or when the
    recorded names do not match ``expected_length``.

    Parameters
    ----------
    fold_results : list of FoldMetrics
        Per-fold results for one model.
    fallback : list of str, optional
        Input column names, used only when no usable transformed names exist.
    expected_length : int, optional
        Width of the array being labelled. Candidates of a different length are
        rejected, so a checkpoint written before the names were recorded cannot
        produce a mislabelled or truncated axis.

    Returns
    -------
    list of str or None
        Names aligned with the importance/SHAP column axis.
    """
    for fold in fold_results:
        names = getattr(fold, "transformed_feature_names", None)
        if not names:
            continue
        names = list(names)
        if expected_length is not None and len(names) != expected_length:
            continue
        return names
    return fallback


class ErrorAnalyzer:
    """Error analysis for classification models.

    Analyzes misclassifications at multiple confidence thresholds to identify
    patterns in model errors and high-confidence mistakes.

    Parameters
    ----------
    confidence_thresholds : list of float or None
        List of confidence levels to analyze (e.g., [0.5, 0.7, 0.9]).
        If None, error analysis will be skipped.

    Attributes
    ----------
    confidence_thresholds : list of float or None
        Sorted list of confidence thresholds for error stratification.

    See Also
    --------
    FoldMetrics : Stores error analysis results per fold.
    ModelEvaluator : Uses ErrorAnalyzer during fold evaluation.

    Examples
    --------
    >>> analyzer = ErrorAnalyzer(confidence_thresholds=[0.5, 0.7, 0.9])
    >>> misclassified, error_stats = analyzer.analyze_misclassifications(
    ...     y_true, y_pred, y_prob, X_test
    ... )
    >>> # error_stats contains analysis at each threshold level
    """

    def __init__(self, confidence_thresholds: Optional[List[float]]):
        self.confidence_thresholds = (
            sorted(confidence_thresholds) if confidence_thresholds else None
        )

    def _safe_mean(self, values: List[float]) -> float:
        """Compute mean safely, returning NaN for empty lists."""
        if not values:
            return np.nan
        return float(np.mean(values))

    def analyze_misclassifications(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        X_test: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
        operating_threshold: float = 0.5,
    ) -> Tuple[pd.DataFrame, Dict[float, Dict[str, Any]]]:
        """Analyse the misclassified samples.

        Identifies all misclassified samples and analyzes them by confidence
        level, prediction margin, and true class.

        Confidence is measured as the distance from the operating threshold,
        normalised so that 0 sits exactly on the threshold and 1 sits at the
        extreme of the score range. The maximum class probability would not do:
        with an optimised threshold well below 0.5, every false negative has a
        class-0 probability above 0.5 by construction, so all of them would be
        counted as high-confidence errors regardless of the model.

        Parameters
        ----------
        y_true : numpy.ndarray
            True labels, shape (n_samples,).
        y_pred : numpy.ndarray
            Predicted labels, shape (n_samples,).
        y_prob : numpy.ndarray
            Predicted probabilities, shape (n_samples, n_classes).
        X_test : pandas.DataFrame
            Test features with original column names.
        feature_names : list of str, optional
            List of feature names for analysis. If None, uses
            DataFrame column names.
        operating_threshold : float, default=0.5
            Decision threshold that produced ``y_pred``.

        Returns
        -------
        misclassified_df : pandas.DataFrame
            DataFrame containing all misclassified samples with columns:
            - Original features
            - 'true_label': Actual class
            - 'predicted_label': Predicted class
            - 'predicted_proba_class_0', 'predicted_proba_class_1': Probabilities
            - 'confidence': Distance from the operating threshold on a 0 to 1 scale
            - 'prediction_margin': Distance of p(class 1) from that threshold
        error_analysis : dict of float to dict
            Dictionary mapping each confidence threshold to error statistics.
        """

        misclassified_df = X_test.copy()
        misclassified_df["true_label"] = y_true
        misclassified_df["predicted_label"] = y_pred
        misclassified_df["predicted_proba_class_0"] = y_prob[:, 0]
        misclassified_df["predicted_proba_class_1"] = y_prob[:, 1]

        p_positive = y_prob[:, 1]
        above = max(1.0 - operating_threshold, float(np.finfo(float).eps))
        below = max(operating_threshold, float(np.finfo(float).eps))
        confidence = np.where(
            p_positive >= operating_threshold,
            (p_positive - operating_threshold) / above,
            (operating_threshold - p_positive) / below,
        )
        misclassified_df["confidence"] = np.clip(confidence, 0.0, 1.0)

        misclassified_df["prediction_margin"] = np.abs(p_positive - operating_threshold)

        misclassified_df = misclassified_df[
            misclassified_df["true_label"] != misclassified_df["predicted_label"]
        ]

        error_analysis = self._analyze_by_threshold(
            misclassified_df, len(y_true), feature_names
        )

        return misclassified_df, error_analysis

    def _analyze_by_threshold(
        self,
        misclassified_df: pd.DataFrame,
        total_samples: int,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[float, Dict[str, Any]]:
        """Analyze errors at different confidence thresholds.

        Parameters
        ----------
        misclassified_df : pandas.DataFrame
            DataFrame of misclassified samples with confidence scores.
        total_samples : int
            Total number of samples in the test set.
        feature_names : list of str, optional
            Feature names for feature-level analysis.

        Returns
        -------
        dict of float to dict
            Error statistics at each confidence threshold.
        """

        error_analysis = {}
        total_misclassified = len(misclassified_df)

        for threshold in self.confidence_thresholds:
            high_conf_errors = misclassified_df[
                misclassified_df["confidence"] >= threshold
            ]

            analysis = {
                "threshold": threshold,
                "n_high_conf_errors": len(high_conf_errors),
                "pct_high_conf_errors": float(
                    len(high_conf_errors) / total_samples * 100
                ),
                "pct_of_all_errors": float(
                    len(high_conf_errors) / total_misclassified * 100
                    if total_misclassified > 0
                    else 0
                ),
                "mean_confidence": float(
                    high_conf_errors["confidence"].mean()
                    if len(high_conf_errors) > 0
                    else 0
                ),
                "std_confidence": float(
                    high_conf_errors["confidence"].std()
                    if len(high_conf_errors) > 0
                    else 0
                ),
                "mean_margin": float(
                    high_conf_errors["prediction_margin"].mean()
                    if len(high_conf_errors) > 0
                    else 0
                ),
                "error_by_class": self._analyze_by_class(high_conf_errors),
            }

            error_analysis[threshold] = analysis

        return error_analysis

    def _analyze_by_class(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Analyze errors by true class.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame of misclassified samples.

        Returns
        -------
        dict of str to dict
            Per-class error statistics including count, percentage,
            confidence distribution, and confusion patterns.
        """
        if len(df) == 0:
            return {}

        analysis = {}
        for true_class in df["true_label"].unique():
            class_df = df[df["true_label"] == true_class]

            pred_class_counts = class_df["predicted_label"].value_counts()

            analysis[f"class_{int(true_class)}"] = {
                "count": len(class_df),
                "percentage": float(len(class_df) / len(df) * 100),
                "mean_confidence": float(class_df["confidence"].mean()),
                "std_confidence": float(class_df["confidence"].std())
                if len(class_df) > 1
                else 0.0,
                "most_confused_with": int(pred_class_counts.index[0]),
                "confusion_counts": {
                    int(k): int(v) for k, v in pred_class_counts.to_dict().items()
                },
            }

        return analysis

    def generate_error_report(
        self, all_fold_analyses: List[Dict[float, Dict[str, Any]]], model_name: str
    ) -> pd.DataFrame:
        """Generate summary report aggregating error statistics across all folds.

        Parameters
        ----------
        all_fold_analyses : list of dict
            List of error analysis dictionaries, one per fold.
        model_name : str
            Name of the model for report labeling.

        Returns
        -------
        pandas.DataFrame
            Summary statistics with columns for each threshold including
            mean/std error counts, percentages, and confidence metrics.
        """

        summary_data = []
        for threshold in self.confidence_thresholds:
            threshold_data = {
                "model": model_name,
                "confidence_threshold": threshold,
                "mean_n_errors": np.mean(
                    [
                        fold[threshold]["n_high_conf_errors"]
                        for fold in all_fold_analyses
                    ]
                ),
                "std_n_errors": np.std(
                    [
                        fold[threshold]["n_high_conf_errors"]
                        for fold in all_fold_analyses
                    ]
                ),
                "mean_pct_errors": np.mean(
                    [
                        fold[threshold]["pct_high_conf_errors"]
                        for fold in all_fold_analyses
                    ]
                ),
                "std_pct_errors": np.std(
                    [
                        fold[threshold]["pct_high_conf_errors"]
                        for fold in all_fold_analyses
                    ]
                ),
                "mean_pct_of_all_errors": np.mean(
                    [fold[threshold]["pct_of_all_errors"] for fold in all_fold_analyses]
                ),
                "mean_confidence": self._safe_mean(
                    [
                        fold[threshold]["mean_confidence"]
                        for fold in all_fold_analyses
                        if fold[threshold]["mean_confidence"] > 0
                    ]
                ),
                "mean_margin": self._safe_mean(
                    [
                        fold[threshold]["mean_margin"]
                        for fold in all_fold_analyses
                        if fold[threshold]["mean_margin"] > 0
                    ]
                ),
            }

            class_errors = self._aggregate_class_errors(all_fold_analyses, threshold)
            threshold_data.update(class_errors)

            summary_data.append(threshold_data)

        return pd.DataFrame(summary_data)

    def _aggregate_class_errors(
        self, all_fold_analyses: List[Dict[float, Dict[str, Any]]], threshold: float
    ) -> Dict[str, float]:
        """Aggregate class-wise error statistics across folds.

        Parameters
        ----------
        all_fold_analyses : list of dict
            List of error analysis dictionaries, one per fold.
        threshold : float
            Confidence threshold to aggregate statistics for.

        Returns
        -------
        dict of str to float
            Mean error counts per class across all folds.
        """
        class_stats: Dict[str, float] = {}
        if not all_fold_analyses:
            return class_stats

        # A fold with no error of a given class contributes a zero, not a
        # missing entry: averaging only over the folds that happened to have
        # such errors overstates the rare error types this analysis targets
        per_fold_counts = [
            {
                class_key: class_data["count"]
                for class_key, class_data in fold[threshold]
                .get("error_by_class", {})
                .items()
            }
            for fold in all_fold_analyses
        ]

        class_keys = sorted({key for counts in per_fold_counts for key in counts})
        for class_key in class_keys:
            counts = [counts.get(class_key, 0) for counts in per_fold_counts]
            class_stats[f"mean_errors_{class_key}"] = float(np.mean(counts))

        return class_stats

    def create_detailed_error_summary(
        self, all_misclassified: pd.DataFrame, model_name: str
    ) -> Dict[str, Any]:
        """Create detailed summary of all misclassified samples.

        Parameters
        ----------
        all_misclassified : pandas.DataFrame
            Combined DataFrame of misclassified samples from all folds.
        model_name : str
            Name of the model for report labeling.

        Returns
        -------
        dict
            Detailed summary including:
            - 'model': Model name
            - 'total_misclassified': Total error count
            - 'confidence_distribution': Mean, std, quartiles
            - 'margin_distribution': Mean, std, quartiles
            - 'confusion_patterns': Count of each (true, pred) pair
        """

        summary = {
            "model": model_name,
            "total_misclassified": len(all_misclassified),
            "confidence_distribution": {
                "mean": float(all_misclassified["confidence"].mean()),
                "std": float(all_misclassified["confidence"].std()),
                "quartiles": {
                    float(k): float(v)
                    for k, v in all_misclassified["confidence"]
                    .quantile([0.25, 0.5, 0.75])
                    .to_dict()
                    .items()
                },
            },
            "margin_distribution": {
                "mean": float(all_misclassified["prediction_margin"].mean()),
                "std": float(all_misclassified["prediction_margin"].std()),
                "quartiles": {
                    float(k): float(v)
                    for k, v in all_misclassified["prediction_margin"]
                    .quantile([0.25, 0.5, 0.75])
                    .to_dict()
                    .items()
                },
            },
        }

        confusion_patterns = all_misclassified.groupby(
            ["true_label", "predicted_label"]
        ).size()
        summary["confusion_patterns"] = {
            f"{int(k[0])} to {int(k[1])}": int(v) for k, v in confusion_patterns.items()
        }

        return summary
