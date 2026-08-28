"""
Cliff's delta effect size computation with bootstrap confidence intervals.

Provides non-parametric effect size measurement using Cliff's delta,
including confidence interval estimation via bootstrap resampling.

Effect Size Interpretation (Cliff's Delta)
------------------------------------------
- |delta| < 0.147: Negligible
- 0.147 <= |delta| < 0.33: Small
- 0.33 <= |delta| < 0.474: Medium
- |delta| >= 0.474: Large

References
----------
- Cliff, N. (1993). Dominance statistics: Ordinal analyses to answer
  ordinal questions. Psychological Bulletin, 114(3), 494-509.
- Romano, J., et al. (2006). Exploring methods for evaluating group
  differences on the NSSE and other surveys: Are the t-test and
  Cohen's d indices the most appropriate choices?
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cliff's delta effect size (non-parametric).

    Measures the probability that a randomly selected value from group x
    is greater than a randomly selected value from group y, minus the
    reverse probability.

    Parameters
    ----------
    x : numpy.ndarray
        First sample array.
    y : numpy.ndarray
        Second sample array.

    Returns
    -------
    float
        Cliff's delta in range [-1, 1]:

        - +1: All values in x are greater than all values in y
        - -1: All values in x are less than all values in y
        - 0: No difference between groups

    References
    ----------
    - Cliff, N. (1993). Dominance statistics: Ordinal analyses to answer
      ordinal questions. Psychological Bulletin, 114(3), 494-509.
    """
    # NaNs must be removed before sorting: np.sort places them last, so
    # searchsorted would count every NaN as dominating the whole comparison
    # group and push delta toward the corresponding extreme
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    x_arr = x_arr[~np.isnan(x_arr)]
    y_arr = y_arr[~np.isnan(y_arr)]

    n1, n2 = len(x_arr), len(y_arr)
    if n1 == 0 or n2 == 0:
        return 0.0

    # O(n log n + m log m) sorted-merge algorithm - avoids O(n*m) memory
    x_arr = np.sort(x_arr)
    y_arr = np.sort(y_arr)

    # For each x_i, count how many y_j it dominates vs. is dominated by
    # searchsorted('left') gives count of y values strictly less than x_i
    # n2 - searchsorted('right') gives count of y values strictly greater
    j_less = np.searchsorted(y_arr, x_arr, side="left")
    j_greater = n2 - np.searchsorted(y_arr, x_arr, side="right")
    dominance = int(j_less.sum()) - int(j_greater.sum())
    return float(dominance) / (n1 * n2)


@dataclass
class CliffsDeltaResult:
    """Container for Cliff's delta with confidence interval.

    Attributes
    ----------
    delta : float
        Cliff's delta effect size in range [-1, 1].
    ci_lower : float
        Lower bound of confidence interval.
    ci_upper : float
        Upper bound of confidence interval.
    ci_level : float
        Confidence level (e.g., 0.95 for 95% CI).
    n_iterations : int
        Number of bootstrap iterations.
    interpretation : str
        Effect size interpretation (negligible, small, medium, large).

    References
    ----------
    - Cliff, N. (1993). Dominance statistics: Ordinal analyses to answer
      ordinal questions. Psychological Bulletin, 114(3), 494-509.
    """

    delta: float
    ci_lower: float
    ci_upper: float
    ci_level: float = 0.95
    n_iterations: int = 1000
    interpretation: str = ""

    def __post_init__(self):
        """Set interpretation if not provided."""
        if not self.interpretation:
            self.interpretation = interpret_cliffs_delta(self.delta)


def interpret_cliffs_delta(delta: float) -> str:
    """Interpret Cliff's delta effect size magnitude.

    Uses standard thresholds from Romano et al. (2006).

    Parameters
    ----------
    delta : float
        Cliff's delta value in range [-1, 1].

    Returns
    -------
    str
        Interpretation: 'negligible', 'small', 'medium', or 'large'.

    References
    ----------
    - Romano, J., et al. (2006). Exploring methods for evaluating group
      differences on the NSSE and other surveys: Are the t-test and
      Cohen's d indices the most appropriate choices?
    """
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        return "negligible"
    elif abs_delta < 0.33:
        return "small"
    elif abs_delta < 0.474:
        return "medium"
    else:
        return "large"


def cliffs_delta_with_ci(
    x: np.ndarray,
    y: np.ndarray,
    ci_level: float = 0.95,
    n_iterations: int = 1000,
    seed: Optional[int] = None,
) -> CliffsDeltaResult:
    """Compute Cliff's delta with bootstrap confidence interval.

    Parameters
    ----------
    x : numpy.ndarray
        First sample array.
    y : numpy.ndarray
        Second sample array.
    ci_level : float, default=0.95
        Confidence level for the interval.
    n_iterations : int, default=1000
        Number of bootstrap resamples.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    CliffsDeltaResult
        Cliff's delta with confidence interval and interpretation.

    Examples
    --------
    >>> result = cliffs_delta_with_ci(group_a_scores, group_b_scores)
    >>> print(f"delta={result.delta:.3f} [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
    >>> print(f"Effect size: {result.interpretation}")
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Point estimate
    delta = cliffs_delta(x, y)

    if len(x) == 0 or len(y) == 0:
        return CliffsDeltaResult(
            delta=delta,
            ci_lower=delta,
            ci_upper=delta,
            ci_level=ci_level,
            n_iterations=0,
        )

    # Bootstrap CI
    rng = np.random.default_rng(seed)
    bootstrap_deltas = np.empty(n_iterations)

    for i in range(n_iterations):
        x_boot = rng.choice(x, size=len(x), replace=True)
        y_boot = rng.choice(y, size=len(y), replace=True)
        bootstrap_deltas[i] = cliffs_delta(x_boot, y_boot)

    alpha = 1 - ci_level
    ci_lower = float(np.percentile(bootstrap_deltas, alpha / 2 * 100))
    ci_upper = float(np.percentile(bootstrap_deltas, (1 - alpha / 2) * 100))

    return CliffsDeltaResult(
        delta=float(delta),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci_level,
        n_iterations=n_iterations,
    )
