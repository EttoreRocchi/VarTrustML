"""Shared threshold optimization logic.

Provides a single source of truth for threshold optimization via inner
cross-validation, eliminating duplication across ModelEvaluator and
ModelTrainer.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from vartrustml.core.threshold import (
    ThresholdMethod,
    ThresholdOptimizer,
    ThresholdResult,
)

logger = logging.getLogger(__name__)


def optimize_threshold_from_cv(
    model: Any,
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, np.ndarray],
    cv: StratifiedKFold,
    method: str = "auto",
    auto_threshold_n_samples: int = 1000,
) -> Optional[ThresholdResult]:
    """Optimize classification threshold using inner CV on training data.

    Uses Youden's J statistic to find the optimal decision threshold
    from out-of-fold predictions.

    Parameters
    ----------
    model : estimator
        Fitted sklearn-compatible model with ``predict_proba``.
    X_train : pandas.DataFrame
        Training features.
    y_train : pandas.Series or numpy.ndarray
        Training labels.
    cv : StratifiedKFold
        Cross-validation splitter for OOF predictions.
    method : str, default="auto"
        Threshold method: ``"oof"``, ``"cv"``, or ``"auto"``.
    auto_threshold_n_samples : int, default=1000
        Sample size boundary for AUTO method selection.

    Returns
    -------
    ThresholdResult or None
        Optimization result, or None if optimization fails.
    """
    try:
        y_prob_train_oof = cross_val_predict(
            model, X_train, y_train, cv=cv, method="predict_proba"
        )[:, 1]

        threshold_method = ThresholdMethod(method)
        optimizer = ThresholdOptimizer(
            method=threshold_method,
            auto_threshold_n_samples=auto_threshold_n_samples,
        )

        if threshold_method == ThresholdMethod.AUTO:
            actual_method = optimizer.select_method(len(y_train))
        else:
            actual_method = threshold_method

        y_values = y_train.values if hasattr(y_train, "values") else np.asarray(y_train)

        if actual_method == ThresholdMethod.CV:
            fold_results: List[Tuple[np.ndarray, np.ndarray]] = []
            for _train_idx, val_idx in cv.split(X_train, y_train):
                y_val = (
                    y_train.iloc[val_idx].values
                    if hasattr(y_train, "iloc")
                    else y_train[val_idx]
                )
                fold_results.append((y_val, y_prob_train_oof[val_idx]))
            return optimizer.optimize_from_folds(fold_results)
        else:
            return optimizer.optimize_from_oof(y_values, y_prob_train_oof)
    except Exception as e:
        logger.warning(f"Threshold optimization failed: {e}. Falling back to default.")
        logger.debug("Threshold optimization traceback:", exc_info=True)
        return None


def threshold_result_to_info_dict(result: ThresholdResult) -> Dict[str, Any]:
    """Convert a ThresholdResult to the info dict used in checkpoints.

    Parameters
    ----------
    result : ThresholdResult
        Threshold optimization result.

    Returns
    -------
    dict
        Dictionary with keys: optimal_threshold, youden_j, sensitivity,
        specificity, method_used.
    """
    return {
        "optimal_threshold": float(result.optimal_threshold),
        "youden_j": float(result.youden_j),
        "sensitivity": float(result.sensitivity_at_threshold),
        "specificity": float(result.specificity_at_threshold),
        "method_used": result.method_used.value,
    }
