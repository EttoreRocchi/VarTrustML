"""
Calibration metrics for probabilistic classifiers.

Expected Calibration Error (ECE) and Maximum Calibration Error (MCE), which
measure how far predicted probabilities drift from observed frequencies.
"""

from typing import List, Tuple

import numpy as np


def _compute_calibration_bins(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> List[Tuple[int, float, float]]:
    """Compute per-bin calibration statistics.

    Parameters
    ----------
    y_true : numpy.ndarray
        True binary labels.
    y_prob : numpy.ndarray
        Predicted probabilities for the positive class.
    n_bins : int, default=10
        Number of bins to use for grouping predictions.

    Returns
    -------
    list of (n_in_bin, avg_confidence, avg_accuracy)
        Statistics for each non-empty bin.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []

    for i in range(n_bins):
        if i == 0:
            in_bin = (y_prob >= bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        else:
            in_bin = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])

        n_in_bin = int(np.sum(in_bin))
        if n_in_bin > 0:
            avg_confidence = float(np.mean(y_prob[in_bin]))
            avg_accuracy = float(np.mean(y_true[in_bin]))
            bins.append((n_in_bin, avg_confidence, avg_accuracy))

    return bins


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error (ECE).

    ECE measures how well the predicted probabilities match the true frequencies.
    A perfectly calibrated model has ECE = 0.

    Parameters
    ----------
    y_true : numpy.ndarray
        True binary labels.
    y_prob : numpy.ndarray
        Predicted probabilities for the positive class.
    n_bins : int, default=10
        Number of bins to use for grouping predictions.

    Returns
    -------
    float
        Expected Calibration Error in range [0, 1].

    References
    ----------
    - Naeini, M. P., Cooper, G., & Hauskrecht, M. (2015). Obtaining well
      calibrated probabilities using Bayesian binning. AAAI Conference
      on Artificial Intelligence.
    """
    total_samples = len(np.asarray(y_true))
    ece = 0.0
    for n_in_bin, avg_conf, avg_acc in _compute_calibration_bins(
        y_true, y_prob, n_bins
    ):
        ece += (n_in_bin / total_samples) * abs(avg_acc - avg_conf)
    return float(ece)


def maximum_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Compute Maximum Calibration Error (MCE).

    MCE is the maximum absolute difference between predicted probability
    and actual accuracy across all bins. It captures the worst-case
    calibration error.

    Parameters
    ----------
    y_true : numpy.ndarray
        True binary labels.
    y_prob : numpy.ndarray
        Predicted probabilities for the positive class.
    n_bins : int, default=10
        Number of bins to use for grouping predictions.

    Returns
    -------
    float
        Maximum Calibration Error in range [0, 1].

    References
    ----------
    - Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On
      calibration of modern neural networks. ICML.
    """
    mce = 0.0
    for _n, avg_conf, avg_acc in _compute_calibration_bins(y_true, y_prob, n_bins):
        mce = max(mce, abs(avg_acc - avg_conf))
    return float(mce)
