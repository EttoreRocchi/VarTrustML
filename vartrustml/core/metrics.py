"""
Shared classification metrics for model and caller evaluation.

The single definition of every metric, shared by ModelEvaluator and
CallerEvaluator so the two cannot drift apart.
"""

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Canonical ordered list of classification metrics tracked across the codebase.
# Probability-based metrics (AUROC, Brier, ECE, MCE) are NaN when y_prob is None.
CLASSIFICATION_METRICS: List[str] = [
    "Precision (Class 0)",
    "Precision (Class 1)",
    "Recall (Class 0)",
    "Recall (Class 1)",
    "F1 Score (Class 0)",
    "F1 Score (Class 1)",
    "F1 Score (Weighted)",
    "Matthews Corr. Coef.",
    "Balanced Accuracy",
    "AUROC",
    "Brier Score",
    "ECE",
    "MCE",
]

PROBABILITY_METRICS: List[str] = ["AUROC", "Brier Score", "ECE", "MCE"]

# Subset used for cross-dataset tracking (excludes calibration-specific metrics).
CROSS_DATASET_METRICS: List[str] = [
    "F1 Score (Weighted)",
    "AUROC",
    "Balanced Accuracy",
    "Matthews Corr. Coef.",
    "Precision (Class 0)",
    "Precision (Class 1)",
    "Recall (Class 0)",
    "Recall (Class 1)",
]


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    calibration_error_fn: Optional[Dict[str, callable]] = None,
) -> Dict[str, float]:
    """Calculate the full set of classification metrics.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred : numpy.ndarray
        Predicted binary labels.
    y_prob : numpy.ndarray of shape (n_samples, 2), optional
        Predicted probabilities for each class. When ``None``,
        probability-based metrics (AUROC, Brier, ECE, MCE) are set to NaN.
    calibration_error_fn : dict of {str: callable}, optional
        Mapping of ``{"ECE": fn, "MCE": fn}`` for calibration error
        functions. When ``None`` and ``y_prob`` is provided, ECE/MCE
        are set to NaN.

    Returns
    -------
    dict of {str: float}
        Dictionary of metric names to values.
    """
    y_true_arr = y_true.values if hasattr(y_true, "values") else np.asarray(y_true)

    precision_per_class = precision_score(
        y_true_arr, y_pred, average=None, zero_division=0
    )
    recall_per_class = recall_score(y_true_arr, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true_arr, y_pred, average=None, zero_division=0)

    metrics: Dict[str, float] = {
        "Precision (Class 0)": float(precision_per_class[0]),
        "Precision (Class 1)": float(precision_per_class[1]),
        "Recall (Class 0)": float(recall_per_class[0]),
        "Recall (Class 1)": float(recall_per_class[1]),
        "F1 Score (Class 0)": float(f1_per_class[0]),
        "F1 Score (Class 1)": float(f1_per_class[1]),
        "F1 Score (Weighted)": float(
            f1_score(y_true_arr, y_pred, average="weighted", zero_division=0)
        ),
        "Matthews Corr. Coef.": float(matthews_corrcoef(y_true_arr, y_pred)),
        "Balanced Accuracy": float(balanced_accuracy_score(y_true_arr, y_pred)),
    }

    if y_prob is not None:
        metrics["AUROC"] = float(roc_auc_score(y_true_arr, y_prob[:, 1]))
        metrics["Brier Score"] = float(brier_score_loss(y_true_arr, y_prob[:, 1]))

        if calibration_error_fn:
            metrics["ECE"] = float(
                calibration_error_fn["ECE"](y_true_arr, y_prob[:, 1])
            )
            metrics["MCE"] = float(
                calibration_error_fn["MCE"](y_true_arr, y_prob[:, 1])
            )
        else:
            metrics["ECE"] = np.nan
            metrics["MCE"] = np.nan
    else:
        metrics["AUROC"] = np.nan
        metrics["Brier Score"] = np.nan
        metrics["ECE"] = np.nan
        metrics["MCE"] = np.nan

    return metrics
