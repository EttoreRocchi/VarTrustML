"""
Bootstrap confidence intervals for model evaluation metrics.

:class:`BootstrapAnalyzer` computes bootstrap confidence intervals by
resampling individual predictions.

Bootstrap CIs offer several advantages over parametric methods:

- No normality assumption required
- Less sensitive to skewed distributions
- Can handle any metric (including MCC, AUROC)
- Uses all individual predictions rather than fold-level summaries

By default the bias-corrected and accelerated (BCa) interval is used. BCa
adjusts the percentile bounds for bias and skewness of the bootstrap
distribution, giving more accurate (second-order correct) coverage than the
plain percentile method, especially for bounded or skewed metrics (MCC, AUROC
near 1, ECE).

References
----------
- Efron, B. & Tibshirani, R. (1993). An Introduction to the Bootstrap.
  Chapman & Hall/CRC.
- Efron, B. (1987). Better bootstrap confidence intervals. Journal of the
  American Statistical Association, 82(397), 171-185.

See Also
--------
compare_pairwise : Paired McNemar/DeLong comparison between classifiers.

Examples
--------
Compute CIs from concatenated predictions:

>>> from vartrustml.analysis.bootstrap import BootstrapAnalyzer
>>> from sklearn.metrics import matthews_corrcoef
>>> analyzer = BootstrapAnalyzer(n_iterations=1000, ci_level=0.95)
>>> result = analyzer.compute_ci_from_predictions(
...     y_true, y_pred, y_prob=None,
...     metric_func=matthews_corrcoef, metric_name="MCC"
... )
>>> print(f"{result.point_estimate:.3f} [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
"""

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
from scipy.stats import norm
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

# Maximum number of leave-one-block-out groups used to estimate the BCa
# acceleration. For ``n <= _BCA_MAX_JACKKNIFE_BLOCKS`` this is the ordinary
# delete-1 jackknife; for larger samples a grouped (delete-d) jackknife with
# this many roughly equal blocks is used so the acceleration is estimated at
# bounded cost (~_BCA_MAX_JACKKNIFE_BLOCKS metric evaluations) instead of O(n).
_BCA_MAX_JACKKNIFE_BLOCKS = 200


@dataclass
class BootstrapCIResult:
    """Container for bootstrap confidence interval results.

    Stores the point estimate, confidence interval bounds, and metadata
    from a bootstrap confidence interval computation.

    Attributes
    ----------
    metric_name : str
        Name of the metric (e.g., "Matthews Corr. Coef.", "AUROC").
    point_estimate : float
        Point estimate (mean across folds or from full data).
    ci_lower : float
        Lower bound of the confidence interval.
    ci_upper : float
        Upper bound of the confidence interval.
    ci_level : float
        Confidence level (e.g., 0.95 for 95% CI).
    n_iterations : int
        Number of bootstrap resamples used.
    std : float
        Standard deviation of the bootstrap distribution.
    ci_method : str
        Interval method actually used: ``"bca"`` or ``"percentile"``. May differ
        from the requested method when BCa is not applicable for a given metric
        (e.g. the point estimate lies at the edge of the bootstrap distribution),
        in which case the analyzer falls back to the percentile interval.
    is_valid : bool
        False when no interval could be computed because more than half of the
        resamples failed. The bounds then collapse onto ``point_estimate`` and
        must not be read as a genuinely narrow interval.
    reason : str or None
        Why the interval is not valid, when ``is_valid`` is False.

    See Also
    --------
    BootstrapAnalyzer : Class that produces BootstrapCIResult objects.
    format_ci : Format result as human-readable string.

    Examples
    --------
    >>> ci_results = analyzer.compute_all_cis_from_predictions(y_true, y_pred, y_prob)
    >>> result = ci_results["Balanced Accuracy"]
    >>> print(f"95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
    """

    metric_name: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    n_iterations: int
    std: float
    ci_method: str = "bca"
    is_valid: bool = True
    reason: Optional[str] = None


class BootstrapAnalyzer:
    """Compute bootstrap confidence intervals for model evaluation metrics.

    Resamples at the prediction level by
    resampling individual predictions rather than fold-level metrics.

    Parameters
    ----------
    n_iterations : int, default=1000
        Number of bootstrap resamples. Higher values give more stable CIs
        but increase computation time. Minimum 100.
    ci_level : float, default=0.95
        Confidence level for the interval (e.g., 0.95 for 95% CI).
        Must be between 0 and 1 (exclusive).
    seed : int, default=42
        Random seed for reproducibility.
    ci_method : {"bca", "percentile"}, default="bca"
        Confidence interval method. ``"bca"`` (bias-corrected and accelerated)
        corrects the percentile bounds for bias and skewness and is the
        recommended default; ``"percentile"`` uses the raw bootstrap percentiles.
        BCa falls back to the percentile interval per-metric when it is not
        applicable (see :attr:`BootstrapCIResult.ci_method`).

    Attributes
    ----------
    n_iterations : int
        Number of bootstrap resamples.
    ci_level : float
        Confidence level.
    seed : int
        Random seed.
    ci_method : str
        Confidence interval method (``"bca"`` or ``"percentile"``).
    rng : numpy.random.Generator
        NumPy random number generator instance.

    Raises
    ------
    ValueError
        If n_iterations < 100, ci_level not in (0, 1), or ci_method is invalid.

    See Also
    --------
    BootstrapCIResult : Container for CI results.
    compare_pairwise : Paired McNemar/DeLong comparison between classifiers.

    Examples
    --------
    >>> analyzer = BootstrapAnalyzer(n_iterations=2000, ci_level=0.95)
    >>> ci_results = analyzer.compute_all_cis_from_predictions(y_true, y_pred, y_prob)
    >>> mcc_result = ci_results["Matthews Corr. Coef."]
    >>> print(f"{mcc_result.point_estimate:.3f} [{mcc_result.ci_lower:.3f}, {mcc_result.ci_upper:.3f}]")
    """

    def __init__(
        self,
        n_iterations: int = 1000,
        ci_level: float = 0.95,
        seed: int = 42,
        ci_method: str = "bca",
    ):
        if n_iterations < 100:
            raise ValueError("n_iterations should be at least 100 for reliable CIs")
        if not 0 < ci_level < 1:
            raise ValueError("ci_level must be between 0 and 1")
        if ci_method not in ("bca", "percentile"):
            raise ValueError(
                f"ci_method must be 'bca' or 'percentile', got {ci_method!r}"
            )

        self.n_iterations = n_iterations
        self.ci_level = ci_level
        self.seed = seed
        self.ci_method = ci_method
        self.rng = np.random.default_rng(seed)

    def compute_ci_from_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray],
        metric_func: Callable,
        metric_name: str = "metric",
        requires_prob: bool = False,
    ) -> BootstrapCIResult:
        """Compute bootstrap CI by resampling predictions directly.

        More powerful than fold-level resampling as it uses all individual
        predictions. Provide concatenated predictions from all test folds.

        Parameters
        ----------
        y_true : numpy.ndarray
            Concatenated true labels from all test folds.
        y_pred : numpy.ndarray
            Concatenated predictions from all test folds.
        y_prob : numpy.ndarray or None
            Concatenated probabilities from all test folds. Required if
            ``requires_prob=True``.
        metric_func : callable
            Function to compute metric. Signature: ``metric_func(y_true, y_pred)``
            or ``metric_func(y_true, y_prob)`` if ``requires_prob=True``.
        metric_name : str, default="metric"
            Name of the metric for labeling the result.
        requires_prob : bool, default=False
            Whether metric_func requires probabilities instead of predictions.

        Returns
        -------
        BootstrapCIResult
            Container with point estimate and confidence interval.

        See Also
        --------
        compute_all_cis_from_predictions : Compute CIs for all standard metrics.

        Notes
        -----
        The interval is computed with the method set on the analyzer
        (``ci_method``): BCa by default, otherwise the raw percentile interval.
        BCa estimates the acceleration with a leave-one-block-out jackknife and
        falls back to the percentile interval per-metric when it is not
        applicable; the method actually used is recorded on
        :attr:`BootstrapCIResult.ci_method`.

        Bootstrap samples that contain only one class are skipped. If more
        than 50% of bootstrap samples fail, falls back to point estimate.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_prob is not None:
            y_prob = np.asarray(y_prob)

        n_samples = len(y_true)

        # Compute point estimate
        if requires_prob and y_prob is not None:
            point_estimate = metric_func(y_true, y_prob)
        else:
            point_estimate = metric_func(y_true, y_pred)

        # Bootstrap resampling
        # Pre-generate indices when memory allows (<100 MB), otherwise generate
        # per-iteration to avoid excessive memory usage on large datasets.
        bootstrap_metrics = np.empty(self.n_iterations)
        index_array_bytes = self.n_iterations * n_samples * 8  # int64
        use_pregenerated = index_array_bytes < 100 * 1024 * 1024  # 100 MB

        if use_pregenerated:
            all_indices = self.rng.choice(
                n_samples, size=(self.n_iterations, n_samples), replace=True
            )
        else:
            all_indices = None

        single_class_count = 0
        exception_count = 0

        for i in range(self.n_iterations):
            if all_indices is not None:
                indices = all_indices[i]
            else:
                indices = self.rng.choice(n_samples, size=n_samples, replace=True)

            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]

            # Skip if only one class in bootstrap sample
            if len(np.unique(y_true_boot)) < 2:
                bootstrap_metrics[i] = np.nan
                single_class_count += 1
                continue

            try:
                if requires_prob and y_prob is not None:
                    y_prob_boot = y_prob[indices]
                    bootstrap_metrics[i] = metric_func(y_true_boot, y_prob_boot)
                else:
                    bootstrap_metrics[i] = metric_func(y_true_boot, y_pred_boot)
            except (ValueError, ZeroDivisionError, IndexError) as e:
                logger.debug(
                    f"Bootstrap iteration {i} failed for metric '{metric_name}': "
                    f"{type(e).__name__}: {e}"
                )
                bootstrap_metrics[i] = np.nan
                exception_count += 1
            except Exception as e:
                # Catch unexpected exceptions but still log them
                logger.debug(
                    f"Bootstrap iteration {i} failed with unexpected error for "
                    f"metric '{metric_name}': {type(e).__name__}: {e}"
                )
                bootstrap_metrics[i] = np.nan
                exception_count += 1

        # Log warning if failure rate exceeds 10%
        failure_rate = (single_class_count + exception_count) / self.n_iterations
        if failure_rate > 0.1:
            logger.warning(
                f"Bootstrap CI for '{metric_name}': {failure_rate:.1%} of iterations failed "
                f"({single_class_count} single-class samples, {exception_count} exceptions). "
                f"Results may be less reliable."
            )

        # Remove NaN values
        bootstrap_metrics = bootstrap_metrics[~np.isnan(bootstrap_metrics)]

        alpha = 1 - self.ci_level
        n_ok = len(bootstrap_metrics)
        is_valid = True
        reason = None
        if n_ok < self.n_iterations * 0.5:
            # Too many failures: no interval is computable. The bounds collapse
            # onto the point estimate, so is_valid marks them as unusable
            # rather than letting them read as a zero-width interval.
            ci_lower = float(point_estimate)
            ci_upper = float(point_estimate)
            std = 0.0
            method_used = self.ci_method
            is_valid = False
            reason = (
                f"Only {n_ok}/{self.n_iterations} bootstrap resamples produced a "
                f"value for '{metric_name}'; no interval could be computed"
            )
        else:
            std = float(np.std(bootstrap_metrics, ddof=1))

            bca_bounds = None
            if self.ci_method == "bca":
                jackknife_estimates = self._jackknife_estimates(
                    y_true, y_pred, y_prob, metric_func, requires_prob
                )
                bca_bounds = self._bca_interval(
                    bootstrap_metrics, float(point_estimate), jackknife_estimates
                )
                if bca_bounds is None:
                    logger.debug(
                        f"BCa not applicable for metric '{metric_name}'; "
                        f"falling back to percentile interval."
                    )

            if bca_bounds is not None:
                ci_lower, ci_upper = bca_bounds
                method_used = "bca"
            else:
                # Percentile interval: requested directly, or BCa fallback.
                ci_lower = float(np.percentile(bootstrap_metrics, alpha / 2 * 100))
                ci_upper = float(
                    np.percentile(bootstrap_metrics, (1 - alpha / 2) * 100)
                )
                method_used = "percentile"

        return BootstrapCIResult(
            metric_name=metric_name,
            point_estimate=float(point_estimate),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            ci_level=self.ci_level,
            n_iterations=self.n_iterations,
            std=std,
            ci_method=method_used,
            is_valid=is_valid,
            reason=reason,
        )

    def _jackknife_estimates(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray],
        metric_func: Callable,
        requires_prob: bool,
    ) -> np.ndarray:
        """Leave-one-block-out jackknife estimates for the BCa acceleration.

        For ``n <= _BCA_MAX_JACKKNIFE_BLOCKS`` this is the ordinary delete-1
        jackknife (one estimate per observation). For larger samples a grouped
        (delete-d) jackknife with ``_BCA_MAX_JACKKNIFE_BLOCKS`` roughly equal,
        randomly assigned blocks is used, so the acceleration is estimated at
        bounded cost rather than recomputing the metric ``n`` times. The
        acceleration is the standardized skewness of these values and is
        scale-invariant, so the grouped jackknife yields a faithful estimate.

        Blocks that become single-class (or for which the metric raises) are
        skipped. Returns an array of jackknife estimates (possibly shorter than
        the number of blocks); the caller treats fewer than 3 values as
        insufficient and sets the acceleration to zero.
        """
        n = len(y_true)
        jrng = np.random.default_rng(self.seed + 1)
        order = jrng.permutation(n)
        n_blocks = n if n <= _BCA_MAX_JACKKNIFE_BLOCKS else _BCA_MAX_JACKKNIFE_BLOCKS
        blocks = np.array_split(order, n_blocks)

        estimates = []
        keep = np.ones(n, dtype=bool)
        for block in blocks:
            keep[block] = False
            y_true_jack = y_true[keep]
            if len(np.unique(y_true_jack)) >= 2:
                try:
                    if requires_prob and y_prob is not None:
                        est = metric_func(y_true_jack, y_prob[keep])
                    else:
                        est = metric_func(y_true_jack, y_pred[keep])
                    if np.isfinite(est):
                        estimates.append(float(est))
                except (ValueError, ZeroDivisionError, IndexError):
                    pass
            keep[block] = True  # restore for the next leave-one-block-out

        return np.asarray(estimates, dtype=float)

    def _bca_interval(
        self,
        bootstrap_metrics: np.ndarray,
        point_estimate: float,
        jackknife_estimates: np.ndarray,
    ) -> Optional[tuple]:
        """Compute bias-corrected and accelerated (BCa) interval bounds.

        Returns ``(ci_lower, ci_upper)`` or ``None`` when BCa is not applicable
        (point estimate at the edge of the bootstrap distribution, or degenerate
        adjusted percentiles), in which case the caller uses the percentile
        interval.

        References
        ----------
        - Efron, B. (1987). Better bootstrap confidence intervals. JASA,
          82(397), 171-185.
        """
        alpha = 1 - self.ci_level

        # Bias-correction z0: inverse-normal of the fraction of bootstrap
        # replicates below the point estimate.
        prop_less = float(np.mean(bootstrap_metrics < point_estimate))
        if prop_less <= 0.0 or prop_less >= 1.0:
            return None
        z0 = norm.ppf(prop_less)

        # Acceleration a: standardized skewness of the jackknife estimates.
        if len(jackknife_estimates) >= 3:
            diff = jackknife_estimates.mean() - jackknife_estimates
            denom = 6.0 * (float(np.sum(diff**2)) ** 1.5)
            a = float(np.sum(diff**3)) / denom if denom > 0.0 else 0.0
        else:
            a = 0.0  # not enough jackknife points -> reduces to bias-corrected

        z_lo = norm.ppf(alpha / 2)
        z_hi = norm.ppf(1 - alpha / 2)

        def _adjust(z):
            return norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))

        a_lo = _adjust(z_lo)
        a_hi = _adjust(z_hi)
        if not (np.isfinite(a_lo) and np.isfinite(a_hi)) or a_lo >= a_hi:
            return None

        ci_lower = float(np.percentile(bootstrap_metrics, 100 * a_lo))
        ci_upper = float(np.percentile(bootstrap_metrics, 100 * a_hi))
        return ci_lower, ci_upper

    def compute_all_cis_from_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> Dict[str, BootstrapCIResult]:
        """Compute bootstrap CIs for all standard classification metrics.

        Computes CIs for precision, recall, F1 (per-class and weighted),
        MCC, balanced accuracy, and AUROC (if probabilities provided).

        Parameters
        ----------
        y_true : numpy.ndarray
            Concatenated true labels from all test folds.
        y_pred : numpy.ndarray
            Concatenated predictions from all test folds.
        y_prob : numpy.ndarray, optional
            Concatenated probabilities (positive class) from all test folds.
            Required for AUROC computation.

        Returns
        -------
        dict of {str: BootstrapCIResult}
            Dictionary mapping metric names to BootstrapCIResult objects.
        """
        results = {}

        # Precision per class
        def precision_class_0(y_t, y_p):
            return precision_score(y_t, y_p, average=None, zero_division=0)[0]

        def precision_class_1(y_t, y_p):
            return precision_score(y_t, y_p, average=None, zero_division=0)[1]

        # Recall per class
        def recall_class_0(y_t, y_p):
            return recall_score(y_t, y_p, average=None, zero_division=0)[0]

        def recall_class_1(y_t, y_p):
            return recall_score(y_t, y_p, average=None, zero_division=0)[1]

        # F1 per class
        def f1_class_0(y_t, y_p):
            return f1_score(y_t, y_p, average=None, zero_division=0)[0]

        def f1_class_1(y_t, y_p):
            return f1_score(y_t, y_p, average=None, zero_division=0)[1]

        def f1_weighted(y_t, y_p):
            return f1_score(y_t, y_p, average="weighted", zero_division=0)

        # Compute CIs for each metric
        metrics = [
            ("Precision (Class 0)", precision_class_0, False),
            ("Precision (Class 1)", precision_class_1, False),
            ("Recall (Class 0)", recall_class_0, False),
            ("Recall (Class 1)", recall_class_1, False),
            ("F1 Score (Class 0)", f1_class_0, False),
            ("F1 Score (Class 1)", f1_class_1, False),
            ("F1 Score (Weighted)", f1_weighted, False),
            ("Matthews Corr. Coef.", matthews_corrcoef, False),
            ("Balanced Accuracy", balanced_accuracy_score, False),
        ]

        for metric_name, metric_func, requires_prob in metrics:
            results[metric_name] = self.compute_ci_from_predictions(
                y_true, y_pred, y_prob, metric_func, metric_name, requires_prob
            )

        # Probability-based metrics require probabilities
        if y_prob is not None:
            results["AUROC"] = self.compute_ci_from_predictions(
                y_true, y_pred, y_prob, roc_auc_score, "AUROC", requires_prob=True
            )
            results["Brier Score"] = self.compute_ci_from_predictions(
                y_true,
                y_pred,
                y_prob,
                brier_score_loss,
                "Brier Score",
                requires_prob=True,
            )

            # ECE and MCE (import lazily to avoid circular imports)
            try:
                from vartrustml.core.calibration import (
                    expected_calibration_error,
                    maximum_calibration_error,
                )

                results["ECE"] = self.compute_ci_from_predictions(
                    y_true,
                    y_pred,
                    y_prob,
                    expected_calibration_error,
                    "ECE",
                    requires_prob=True,
                )
                results["MCE"] = self.compute_ci_from_predictions(
                    y_true,
                    y_pred,
                    y_prob,
                    maximum_calibration_error,
                    "MCE",
                    requires_prob=True,
                )
            except ImportError:
                logger.debug("Calibration module not available for bootstrap CI")

        return results


def format_ci(result: BootstrapCIResult, precision: int = 3) -> str:
    """Format bootstrap CI result as human-readable string.

    Parameters
    ----------
    result : BootstrapCIResult
        Bootstrap CI result object to format.
    precision : int, default=3
        Number of decimal places.

    Returns
    -------
    str
        Formatted string like "0.850 [0.823, 0.877]".

    Examples
    --------
    >>> ci_results = analyzer.compute_all_cis_from_predictions(y_true, y_pred, y_prob)
    >>> print(format_ci(ci_results["Matthews Corr. Coef."]))
    0.847 [0.820, 0.870]
    """
    return (
        f"{result.point_estimate:.{precision}f} "
        f"[{result.ci_lower:.{precision}f}, {result.ci_upper:.{precision}f}]"
    )
