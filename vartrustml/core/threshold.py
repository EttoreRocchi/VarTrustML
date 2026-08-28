"""
Threshold optimization for binary classification.

Finds decision thresholds with Youden's J statistic, which maximises the sum
of sensitivity and specificity.

The Youden's J statistic is defined as:

.. math::

    J = \\text{Sensitivity} + \\text{Specificity} - 1 = \\text{TPR} - \\text{FPR}

Two optimization strategies are available:

- **OOF (Out-of-Fold)**: Uses concatenated OOF predictions to find a single
  threshold. Recommended for smaller datasets (n < 1000).
- **CV (Cross-Validation)**: Finds optimal threshold per fold and averages.
  Recommended for larger datasets (n >= 1000).

References
----------
- Youden, W.J. (1950). Index for rating diagnostic tests.
  Cancer, 3(1), 32-35.

Examples
--------
>>> from vartrustml.core.threshold import ThresholdOptimizer, ThresholdMethod
>>> optimizer = ThresholdOptimizer(method=ThresholdMethod.AUTO)
>>> result = optimizer.optimize_from_oof(y_true_oof, y_prob_oof)
>>> print(f"Optimal threshold: {result.optimal_threshold:.3f}")
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_curve

logger = logging.getLogger(__name__)


class ThresholdMethod(Enum):
    """Enumeration of threshold selection methods.

    Attributes
    ----------
    OOF : str
        Out-of-fold method: single threshold from concatenated OOF predictions.
        Recommended for n < 1000.
    CV : str
        Cross-validation method: average thresholds across folds.
        Recommended for n >= 1000.
    AUTO : str
        Auto-select method based on sample size.
    """

    OOF = "oof"
    CV = "cv"
    AUTO = "auto"


@dataclass
class ThresholdResult:
    """Container for threshold optimization results.

    Stores the optimal decision threshold found by ThresholdOptimizer
    along with associated metrics and method information.

    Attributes
    ----------
    optimal_threshold : float
        The optimal decision threshold in [0, 1].
    youden_j : float
        Youden's J statistic at the optimal threshold.
        J = Sensitivity + Specificity - 1, range [-1, 1].
    method_used : ThresholdMethod
        The method used for threshold selection (OOF, CV, or AUTO).
    sensitivity_at_threshold : float
        Sensitivity (True Positive Rate) at the optimal threshold.
    specificity_at_threshold : float
        Specificity (1 - False Positive Rate) at the optimal threshold.
    fold_thresholds : list of float, optional
        Individual fold thresholds. Only populated when method_used is CV.
    n_samples : int
        Number of samples used for threshold optimization.

    See Also
    --------
    ThresholdOptimizer : Class that produces ThresholdResult objects.
    ThresholdMethod : Enumeration of optimization methods.

    Examples
    --------
    >>> result = optimizer.optimize_from_oof(y_true, y_prob)
    >>> print(f"Threshold: {result.optimal_threshold:.3f}")
    >>> print(f"Youden's J: {result.youden_j:.3f}")
    """

    optimal_threshold: float
    youden_j: float
    method_used: ThresholdMethod
    sensitivity_at_threshold: float
    specificity_at_threshold: float
    fold_thresholds: Optional[List[float]] = None
    n_samples: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "optimal_threshold": self.optimal_threshold,
            "youden_j": self.youden_j,
            "method_used": self.method_used.value,
            "sensitivity_at_threshold": self.sensitivity_at_threshold,
            "specificity_at_threshold": self.specificity_at_threshold,
            "fold_thresholds": self.fold_thresholds,
            "n_samples": self.n_samples,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ThresholdResult":
        """Create from dictionary."""
        return cls(
            optimal_threshold=data["optimal_threshold"],
            youden_j=data["youden_j"],
            method_used=ThresholdMethod(data["method_used"]),
            sensitivity_at_threshold=data["sensitivity_at_threshold"],
            specificity_at_threshold=data["specificity_at_threshold"],
            fold_thresholds=data.get("fold_thresholds"),
            n_samples=data.get("n_samples", 0),
        )


class ThresholdOptimizer:
    """Optimize classification threshold using Youden's J statistic.

    Find the optimal decision threshold that maximizes
    :math:`J = \\text{Sensitivity} + \\text{Specificity} - 1`.

    The optimization is performed on training data only to avoid data leakage.
    Two methods are available, selected based on sample size by default.

    Parameters
    ----------
    method : ThresholdMethod, default=ThresholdMethod.AUTO
        Method for threshold selection:

        - ``OOF``: Single threshold from concatenated out-of-fold predictions.
          Recommended for n < 1000.
        - ``CV``: Average of per-fold optimal thresholds.
          Recommended for n >= 1000.
        - ``AUTO``: Automatically selects based on sample size.

    auto_threshold_n_samples : int, default=1000
        Sample size threshold for AUTO method selection.
        Samples < threshold use OOF; samples >= threshold use CV.

    Attributes
    ----------
    method : ThresholdMethod
        The threshold selection method.
    auto_threshold_n_samples : int
        Sample size threshold for AUTO method selection.

    See Also
    --------
    ThresholdResult : Container for optimization results.
    ThresholdMethod : Enumeration of available methods.

    Notes
    -----
    - OOF method pools all out-of-fold predictions and finds a single threshold.
      This works well for smaller datasets where fold-level estimates are noisy.
    - CV method finds the optimal threshold for each fold independently, then
      averages. This is more stable for larger datasets.
    - When method is AUTO, the choice is made based on total sample count.

    Examples
    --------
    Basic usage with AUTO method selection:

    >>> from vartrustml.core.threshold import ThresholdOptimizer, ThresholdMethod
    >>> optimizer = ThresholdOptimizer(method=ThresholdMethod.AUTO)
    >>> result = optimizer.optimize_from_oof(y_true_oof, y_prob_oof)
    >>> print(f"Optimal threshold: {result.optimal_threshold:.3f}")

    Using CV method for large datasets:

    >>> optimizer = ThresholdOptimizer(method=ThresholdMethod.CV)
    >>> fold_data = [(y_true_fold1, y_prob_fold1), (y_true_fold2, y_prob_fold2)]
    >>> result = optimizer.optimize_from_folds(fold_data)
    """

    DEFAULT_AUTO_THRESHOLD_N_SAMPLES = 1000

    def __init__(
        self,
        method: ThresholdMethod = ThresholdMethod.AUTO,
        auto_threshold_n_samples: int = 1000,
    ):
        self.method = method
        self.auto_threshold_n_samples = auto_threshold_n_samples

    def find_optimal_threshold(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """Find optimal threshold using Youden's J statistic.

        Computes the ROC curve and finds the threshold that maximizes
        :math:`J = TPR - FPR`.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True binary labels (0 or 1).
        y_prob : array-like of shape (n_samples,)
            Predicted probabilities for the positive class.

        Returns
        -------
        optimal_threshold : float
            The threshold that maximizes Youden's J.
        youden_j : float
            The maximum Youden's J statistic value.
        sensitivity : float
            True positive rate at the optimal threshold.
        specificity : float
            True negative rate (1 - FPR) at the optimal threshold.

        Notes
        -----
        Handles edge cases where the optimal threshold is infinity
        by selecting the next best valid threshold, or 0.5 as fallback.
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)

        # Calculate Youden's J for each threshold
        youden_j_scores = tpr - fpr

        # Find index of maximum J
        optimal_idx = np.argmax(youden_j_scores)

        optimal_threshold = thresholds[optimal_idx]
        optimal_youden_j = youden_j_scores[optimal_idx]
        sensitivity = tpr[optimal_idx]
        specificity = 1 - fpr[optimal_idx]

        # Handle edge cases where threshold might be inf
        if np.isinf(optimal_threshold):
            # Find the next best threshold that's not inf
            valid_mask = ~np.isinf(thresholds)
            if valid_mask.any():
                valid_j_scores = youden_j_scores.copy()
                valid_j_scores[~valid_mask] = -np.inf
                optimal_idx = np.argmax(valid_j_scores)
                optimal_threshold = thresholds[optimal_idx]
                optimal_youden_j = youden_j_scores[optimal_idx]
                sensitivity = tpr[optimal_idx]
                specificity = 1 - fpr[optimal_idx]
            else:
                optimal_threshold = 0.5  # Fallback

        return optimal_threshold, optimal_youden_j, sensitivity, specificity

    def optimize_from_oof(
        self, y_true_oof: np.ndarray, y_prob_oof: np.ndarray
    ) -> ThresholdResult:
        """Optimize threshold using concatenated out-of-fold predictions.

        Uses all OOF predictions pooled together to find a single optimal
        threshold. Recommended for datasets with n < 1000 samples where
        per-fold estimates would be noisy.

        Parameters
        ----------
        y_true_oof : array-like of shape (n_samples,)
            Concatenated true labels from all OOF predictions.
        y_prob_oof : array-like of shape (n_samples,)
            Concatenated predicted probabilities from all OOF predictions.

        Returns
        -------
        ThresholdResult
            Container with optimal threshold and associated metrics.
            The ``method_used`` attribute will be ``ThresholdMethod.OOF``.

        See Also
        --------
        optimize_from_folds : Alternative method for larger datasets.
        """
        threshold, youden_j, sensitivity, specificity = self.find_optimal_threshold(
            y_true_oof, y_prob_oof
        )

        logger.info(
            f"OOF threshold optimization: threshold={threshold:.4f}, "
            f"Youden's J={youden_j:.4f}, sensitivity={sensitivity:.4f}, "
            f"specificity={specificity:.4f}"
        )

        return ThresholdResult(
            optimal_threshold=threshold,
            youden_j=youden_j,
            method_used=ThresholdMethod.OOF,
            sensitivity_at_threshold=sensitivity,
            specificity_at_threshold=specificity,
            n_samples=len(y_true_oof),
        )

    @staticmethod
    def metrics_at_threshold(
        y_true: np.ndarray, y_prob: np.ndarray, threshold: float
    ) -> Tuple[float, float, float]:
        """Sensitivity, specificity and Youden's J at a given threshold.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True binary labels.
        y_prob : array-like of shape (n_samples,)
            Predicted probability of the positive class.
        threshold : float
            Decision threshold applied to ``y_prob``.

        Returns
        -------
        tuple of (float, float, float)
            Sensitivity, specificity, and Youden's J at ``threshold``.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_prob) >= threshold

        pos = y_true == 1
        neg = ~pos
        n_pos = int(pos.sum())
        n_neg = int(neg.sum())

        sensitivity = float((y_pred & pos).sum() / n_pos) if n_pos else 0.0
        specificity = float((~y_pred & neg).sum() / n_neg) if n_neg else 0.0
        return sensitivity, specificity, sensitivity + specificity - 1

    def optimize_from_folds(
        self, fold_results: List[Tuple[np.ndarray, np.ndarray]]
    ) -> ThresholdResult:
        """Optimize threshold by averaging per-fold optimal thresholds.

        Finds the optimal threshold for each fold independently, then
        averages them. Recommended for datasets with n >= 1000 samples
        where per-fold estimates are more stable.

        The reported sensitivity, specificity and Youden's J are measured on
        the pooled fold predictions **at the averaged threshold**. Averaging
        each fold's own optimum instead would report a maximum rather than the
        performance of the threshold actually returned, which is optimistically
        biased.

        Parameters
        ----------
        fold_results : list of tuple
            List of (y_true, y_prob) tuples, one per cross-validation fold.
            Each y_true and y_prob should be 1D arrays of the same length.

        Returns
        -------
        ThresholdResult
            Container with averaged optimal threshold and metrics.
            The ``method_used`` attribute will be ``ThresholdMethod.CV``.
            The ``fold_thresholds`` attribute contains individual fold values.

        See Also
        --------
        optimize_from_oof : Alternative method for smaller datasets.
        """
        fold_thresholds = []
        fold_youden_js = []
        total_samples = 0

        for y_true, y_prob in fold_results:
            threshold, youden_j, _, _ = self.find_optimal_threshold(y_true, y_prob)
            fold_thresholds.append(threshold)
            fold_youden_js.append(youden_j)
            total_samples += len(y_true)

        avg_threshold = float(np.mean(fold_thresholds))

        # Measure at the threshold actually returned, on the pooled folds
        y_true_all = np.concatenate([np.asarray(yt) for yt, _ in fold_results])
        y_prob_all = np.concatenate([np.asarray(yp) for _, yp in fold_results])
        sensitivity, specificity, youden_j = self.metrics_at_threshold(
            y_true_all, y_prob_all, avg_threshold
        )

        logger.info(
            f"CV threshold optimization: avg_threshold={avg_threshold:.4f}, "
            f"Youden's J at that threshold={youden_j:.4f} "
            f"(mean of per-fold optima was {np.mean(fold_youden_js):.4f}), "
            f"fold_thresholds={[f'{t:.4f}' for t in fold_thresholds]}"
        )

        return ThresholdResult(
            optimal_threshold=avg_threshold,
            youden_j=youden_j,
            method_used=ThresholdMethod.CV,
            sensitivity_at_threshold=sensitivity,
            specificity_at_threshold=specificity,
            fold_thresholds=fold_thresholds,
            n_samples=total_samples,
        )

    def select_method(self, n_samples: int) -> ThresholdMethod:
        """Auto-select optimization method based on sample size.

        Chooses between OOF and CV methods based on the
        ``auto_threshold_n_samples`` attribute.

        Parameters
        ----------
        n_samples : int
            Total number of samples in the dataset.

        Returns
        -------
        ThresholdMethod
            ``OOF`` if n_samples < auto_threshold_n_samples,
            ``CV`` otherwise.
        """
        if n_samples < self.auto_threshold_n_samples:
            logger.info(
                f"Auto-selected OOF method (n={n_samples} < {self.auto_threshold_n_samples})"
            )
            return ThresholdMethod.OOF
        else:
            logger.info(
                f"Auto-selected CV method (n={n_samples} >= {self.auto_threshold_n_samples})"
            )
            return ThresholdMethod.CV
