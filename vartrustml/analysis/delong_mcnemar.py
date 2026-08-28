"""
DeLong and McNemar statistical tests for classifier comparison.

Provides the DeLong test for comparing two correlated AUROC values and
McNemar's test for comparing classifier prediction disagreement patterns.

Also includes:
- Benjamini-Hochberg FDR correction for multiple testing
- Power analysis for sample size estimation

References
----------
- DeLong, E.R., DeLong, D.M., Clarke-Pearson, D.L. (1988). Comparing
  the areas under two or more correlated receiver operating
  characteristic curves: a nonparametric approach. Biometrics, 44(3),
  837-845.
- McNemar, Q. (1947). Note on the sampling error of the difference
  between correlated proportions or percentages. Psychometrika, 12(2),
  153-157.
- Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery
  rate: a practical and powerful approach to multiple testing.
  Journal of the Royal Statistical Society B, 57(1), 289-300.
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from scipy import stats
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def holm_bonferroni_correction(
    p_values: np.ndarray, alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Holm-Bonferroni step-down correction (controls the FWER)."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([]), np.array([], dtype=bool)
    order = np.argsort(p)
    factors = n - np.arange(n)
    adj_sorted = np.minimum(np.maximum.accumulate(factors * p[order]), 1.0)
    adjusted = np.empty(n)
    adjusted[order] = adj_sorted
    return adjusted, adjusted <= alpha


def correct_pvalues(
    p_values: np.ndarray, method: str = "holm", alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """Dispatch multiple-comparison correction ('holm' FWER or 'bh' FDR)."""
    if method == "holm":
        return holm_bonferroni_correction(p_values, alpha)
    return benjamini_hochberg_correction(p_values, alpha)


def benjamini_hochberg_correction(
    p_values: np.ndarray, alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Benjamini-Hochberg FDR correction for multiple testing.

    Controls the False Discovery Rate (FDR) rather than the Family-Wise
    Error Rate (FWER) like Bonferroni/Holm corrections.

    Parameters
    ----------
    p_values : numpy.ndarray
        Array of raw p-values.
    alpha : float, default=0.05
        Target FDR level.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
        - adjusted_pvalues: FDR-adjusted p-values (q-values)
        - is_significant: Boolean array indicating significant tests

    References
    ----------
    - Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery
      rate: a practical and powerful approach to multiple testing.
      Journal of the Royal Statistical Society B, 57(1), 289-300.

    Examples
    --------
    >>> p_values = np.array([0.001, 0.03, 0.04, 0.05, 0.5])
    >>> adjusted, significant = benjamini_hochberg_correction(p_values, alpha=0.05)
    """
    p_values = np.asarray(p_values)
    n = len(p_values)

    if n == 0:
        return np.array([]), np.array([], dtype=bool)

    # Sort p-values and get sort indices
    sorted_indices = np.argsort(p_values)
    sorted_pvalues = p_values[sorted_indices]

    # Compute adjusted p-values
    adjusted = np.empty(n)
    for i in range(n):
        rank = i + 1
        adjusted[sorted_indices[i]] = sorted_pvalues[i] * n / rank

    # Ensure monotonicity (cumulative minimum from the end)
    adjusted_sorted = adjusted[sorted_indices]
    for i in range(n - 2, -1, -1):
        if adjusted_sorted[i] > adjusted_sorted[i + 1]:
            adjusted_sorted[i] = adjusted_sorted[i + 1]
    adjusted[sorted_indices] = adjusted_sorted

    # Cap at 1.0
    adjusted = np.minimum(adjusted, 1.0)

    # Determine significance
    is_significant = adjusted <= alpha

    return adjusted, is_significant


def power_analysis_sample_size(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80,
    test_type: str = "wilcoxon",
) -> int:
    """Estimate required sample size for desired statistical power.

    Provides sample size recommendations for planning experiments with
    adequate power to detect effects of a given size.

    Parameters
    ----------
    effect_size : float
        Expected effect size as Cohen's d (standardized mean difference)
        for both test types. Cohen's d is defined as the difference in
        means divided by the pooled standard deviation. Typical values:
        0.2 (small), 0.5 (medium), 0.8 (large).

        Note: this parameter expects Cohen's d, not Cliff's delta. If you
        have a Cliff's delta value, convert it to Cohen's d first (e.g.,
        using the relationship d ≈ delta * sqrt(2) / sqrt(1 - delta^2)
        under normality assumptions).
    alpha : float, default=0.05
        Significance level.
    power : float, default=0.80
        Desired statistical power (1 - beta).
    test_type : str, default="wilcoxon"
        Type of test: "wilcoxon" or "ttest".

    Returns
    -------
    int
        Required sample size per group.

    Notes
    -----
    The base formula ``n = 2 * ((z_alpha + z_beta) / d)^2`` is derived for
    the two-sample t-test with Cohen's d as the effect size measure.

    For Wilcoxon test, the t-test sample size is divided by the asymptotic
    relative efficiency (ARE) of 0.955 compared to the t-test under normal
    distributions, yielding a conservative (slightly larger) estimate.
    For non-normal distributions, the Wilcoxon test may actually require
    fewer samples than the t-test.

    Examples
    --------
    >>> n = power_analysis_sample_size(effect_size=0.5, power=0.80)
    >>> print(f"Required samples per group: {n}")

    References
    ----------
    - Cohen, J. (1988). Statistical Power Analysis for the Behavioral
      Sciences (2nd ed.). Lawrence Erlbaum Associates.
    """
    if effect_size <= 0:
        raise ValueError(f"Effect size must be positive, got {effect_size}")

    if effect_size > 1:
        logger.warning(f"Effect size {effect_size} outside typical range (0, 1]")

    if not 0 < power < 1:
        raise ValueError(f"Power must be in (0, 1), got {power}")

    # Get z-scores for alpha and power
    z_alpha = norm.ppf(1 - alpha / 2)  # Two-tailed
    z_beta = norm.ppf(power)

    # Basic formula for t-test
    # n = 2 * ((z_alpha + z_beta) / effect_size)^2
    n_ttest = 2 * ((z_alpha + z_beta) / effect_size) ** 2

    if test_type == "ttest":
        return int(np.ceil(n_ttest))
    elif test_type == "wilcoxon":
        # Wilcoxon has ARE ~ 0.955 for normal distributions
        # For other distributions it can be more efficient
        # Conservative estimate: divide by ARE
        are = 0.955
        n_wilcoxon = n_ttest / are
        return int(np.ceil(n_wilcoxon))
    else:
        raise ValueError(f"Unknown test_type: {test_type}. Use 'wilcoxon' or 'ttest'.")


@dataclass
class DeLongTestResult:
    """Container for DeLong test results.

    The DeLong test compares two correlated AUROC values from the same
    sample, making it more powerful than comparing AUROCs across CV folds.

    Attributes
    ----------
    z_statistic : float
        The z-statistic from the DeLong test.
    p_value : float
        Two-sided p-value.
    auroc_a : float
        AUROC of model A.
    auroc_b : float
        AUROC of model B.
    auroc_diff : float
        Difference in AUROCs (A - B).
    is_significant : bool
        Whether the test is significant at the given alpha level.
    alpha : float
        Significance level used (default: 0.05).
    is_valid : bool
        Whether the test result is statistically valid. False when the test
        could not be performed properly (e.g., non-positive variance due to
        nearly identical predictions).
    reason : str or None
        Explanation when is_valid is False, describing why the test failed.

    References
    ----------
    - DeLong, E.R., DeLong, D.M., Clarke-Pearson, D.L. (1988). Comparing
      the areas under two or more correlated receiver operating
      characteristic curves: a nonparametric approach. Biometrics, 44(3),
      837-845.
    """

    z_statistic: float
    p_value: float
    auroc_a: float
    auroc_b: float
    auroc_diff: float
    is_significant: bool
    alpha: float = 0.05
    is_valid: bool = True
    reason: Optional[str] = None


@dataclass
class McNemarTestResult:
    """Container for McNemar's test results.

    McNemar's test compares the predictions of two classifiers by examining
    their disagreement pattern on the same test set.

    Attributes
    ----------
    statistic : float
        Edwards-corrected chi-squared statistic when the chi-squared branch was
        used, or the discordant count ``b`` when the exact binomial was used.
        Read ``exact_used`` to know which one this is.
    p_value : float
        P-value of the test.
    n_both_correct : int
        Number of samples both classifiers predicted correctly.
    n_both_wrong : int
        Number of samples both classifiers predicted incorrectly.
    n_a_correct_b_wrong : int
        Number of samples where A was correct and B was wrong.
    n_a_wrong_b_correct : int
        Number of samples where A was wrong and B was correct.
    is_significant : bool
        Whether the test is significant at the given alpha level.
    alpha : float
        Significance level used (default: 0.05).
    exact_used : bool
        Whether the exact binomial branch produced ``p_value`` and
        ``statistic``. False means the chi-squared approximation was used.

    References
    ----------
    - McNemar, Q. (1947). Note on the sampling error of the difference
      between correlated proportions or percentages. Psychometrika, 12(2),
      153-157.
    """

    statistic: float
    p_value: float
    n_both_correct: int
    n_both_wrong: int
    n_a_correct_b_wrong: int
    n_a_wrong_b_correct: int
    is_significant: bool
    alpha: float = 0.05
    exact_used: bool = True


def delong_test(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
    alpha: float = 0.05,
) -> DeLongTestResult:
    """Perform DeLong test to compare two AUROCs from the same sample.

    The DeLong test is the standard method for comparing AUROC values
    of two classifiers on the same dataset. It accounts for the correlation
    between the predictions.

    Parameters
    ----------
    y_true : numpy.ndarray
        True binary labels.
    y_prob_a : numpy.ndarray
        Predicted probabilities from classifier A (positive class).
    y_prob_b : numpy.ndarray
        Predicted probabilities from classifier B (positive class).
    alpha : float, default=0.05
        Significance level for the test.

    Returns
    -------
    DeLongTestResult
        Container with test results including z-statistic, p-value,
        and AUROC values for both classifiers.

    References
    ----------
    - DeLong, E.R., DeLong, D.M., Clarke-Pearson, D.L. (1988). Comparing
      the areas under two or more correlated receiver operating
      characteristic curves: a nonparametric approach. Biometrics, 44(3),
      837-845.
    - Sun, X. & Xu, W. (2014). Fast implementation of DeLong's algorithm
      for comparing the areas under correlated receiver operating
      characteristic curves. IEEE Signal Processing Letters, 21(11),
      1389-1393.
    """
    y_true = np.asarray(y_true)
    y_prob_a = np.asarray(y_prob_a)
    y_prob_b = np.asarray(y_prob_b)

    # Compute AUROCs
    auroc_a = roc_auc_score(y_true, y_prob_a)
    auroc_b = roc_auc_score(y_true, y_prob_b)
    auroc_diff = auroc_a - auroc_b

    # Separate positive and negative samples
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    n_pos = np.sum(pos_mask)
    n_neg = np.sum(neg_mask)

    if n_pos == 0 or n_neg == 0:
        logger.warning("DeLong test requires both positive and negative samples")
        return DeLongTestResult(
            z_statistic=0.0,
            p_value=1.0,
            auroc_a=auroc_a,
            auroc_b=auroc_b,
            auroc_diff=auroc_diff,
            is_significant=False,
            alpha=alpha,
            is_valid=False,
            reason="Test requires both positive and negative samples",
        )

    # Extract predictions for positive and negative classes
    pos_prob_a = y_prob_a[pos_mask]
    neg_prob_a = y_prob_a[neg_mask]
    pos_prob_b = y_prob_b[pos_mask]
    neg_prob_b = y_prob_b[neg_mask]

    # Compute structural components (placement values) - vectorized
    def compute_placements(pos_probs, neg_probs):
        """Compute placement values for positive samples (vectorized)."""
        # pos_probs[:, None] vs neg_probs[None, :] -> broadcasting
        return np.mean(
            (pos_probs[:, None] > neg_probs[None, :])
            + 0.5 * (pos_probs[:, None] == neg_probs[None, :]),
            axis=1,
        )

    # Placement values for positive samples
    v10_a = compute_placements(pos_prob_a, neg_prob_a)
    v10_b = compute_placements(pos_prob_b, neg_prob_b)

    # Placement values for negative samples
    def compute_placements_neg(neg_probs, pos_probs):
        """Compute placement values for negative samples (vectorized)."""
        return np.mean(
            (pos_probs[None, :] > neg_probs[:, None])
            + 0.5 * (pos_probs[None, :] == neg_probs[:, None]),
            axis=1,
        )

    v01_a = compute_placements_neg(neg_prob_a, pos_prob_a)
    v01_b = compute_placements_neg(neg_prob_b, pos_prob_b)

    # Compute variances and covariance
    s10_a = np.var(v10_a, ddof=1) if len(v10_a) > 1 else 0.0
    s10_b = np.var(v10_b, ddof=1) if len(v10_b) > 1 else 0.0
    s01_a = np.var(v01_a, ddof=1) if len(v01_a) > 1 else 0.0
    s01_b = np.var(v01_b, ddof=1) if len(v01_b) > 1 else 0.0

    # Covariances
    s10_ab = np.cov(v10_a, v10_b, ddof=1)[0, 1] if len(v10_a) > 1 else 0.0
    s01_ab = np.cov(v01_a, v01_b, ddof=1)[0, 1] if len(v01_a) > 1 else 0.0

    # Variance of the difference
    var_a = s10_a / n_pos + s01_a / n_neg
    var_b = s10_b / n_pos + s01_b / n_neg
    cov_ab = s10_ab / n_pos + s01_ab / n_neg

    var_diff = var_a + var_b - 2 * cov_ab

    if var_diff <= 0:
        logger.warning(
            "DeLong test variance is non-positive (predictions may be nearly identical). "
            "Test is not applicable for this comparison."
        )
        return DeLongTestResult(
            z_statistic=0.0,
            p_value=1.0,
            auroc_a=float(auroc_a),
            auroc_b=float(auroc_b),
            auroc_diff=float(auroc_diff),
            is_significant=False,
            alpha=alpha,
            is_valid=False,
            reason="Non-positive variance - predictions may be nearly identical or sample size too small",
        )

    # Compute z-statistic
    z = auroc_diff / np.sqrt(var_diff)

    # Two-sided p-value; norm.sf avoids the catastrophic cancellation of
    # 1 - cdf, which underflows to exactly 0.0 for |z| above ~8.3
    p_value = 2 * stats.norm.sf(np.abs(z))

    return DeLongTestResult(
        z_statistic=float(z),
        p_value=float(p_value),
        auroc_a=float(auroc_a),
        auroc_b=float(auroc_b),
        auroc_diff=float(auroc_diff),
        is_significant=p_value < alpha,
        alpha=alpha,
        is_valid=True,
        reason=None,
    )


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    alpha: float = 0.05,
    exact: Optional[bool] = None,
) -> McNemarTestResult:
    """Perform McNemar's test to compare two classifier predictions.

    McNemar's test compares the disagreement patterns between two classifiers.
    It tests whether the number of samples where A is correct and B is wrong
    differs significantly from the number where B is correct and A is wrong.

    Parameters
    ----------
    y_true : numpy.ndarray
        True binary labels.
    y_pred_a : numpy.ndarray
        Predicted labels from classifier A.
    y_pred_b : numpy.ndarray
        Predicted labels from classifier B.
    alpha : float, default=0.05
        Significance level for the test.
    exact : bool or None, default=None
        If None, selects automatically: exact binomial when the discordant
        count b + c is below 25, chi-squared approximation otherwise. Pass
        True or False to force one branch. ``McNemarTestResult.exact_used``
        reports which branch produced the result.

    Returns
    -------
    McNemarTestResult
        Container with test results including statistic, p-value,
        and disagreement counts.

    References
    ----------
    - McNemar, Q. (1947). Note on the sampling error of the difference
      between correlated proportions or percentages. Psychometrika, 12(2),
      153-157.
    - Edwards, A. L. (1948). Note on the correction for continuity in
      testing the significance of the difference between correlated
      proportions. Psychometrika, 13(3), 185-187.
    """
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)

    # Compute correctness
    correct_a = y_pred_a == y_true
    correct_b = y_pred_b == y_true

    # Count disagreement cells
    n_both_correct = np.sum(correct_a & correct_b)
    n_both_wrong = np.sum(~correct_a & ~correct_b)
    n_a_correct_b_wrong = np.sum(correct_a & ~correct_b)  # b
    n_a_wrong_b_correct = np.sum(~correct_a & correct_b)  # c

    # The discordant cells (b and c) are what matter for McNemar's test
    b = n_a_correct_b_wrong
    c = n_a_wrong_b_correct

    if b + c == 0:
        # No disagreements, classifiers are identical
        return McNemarTestResult(
            statistic=0.0,
            p_value=1.0,
            n_both_correct=int(n_both_correct),
            n_both_wrong=int(n_both_wrong),
            n_a_correct_b_wrong=int(b),
            n_a_wrong_b_correct=int(c),
            is_significant=False,
            alpha=alpha,
            exact_used=True,
        )

    use_exact = (b + c) < 25 if exact is None else exact
    if use_exact:
        # Exact binomial test (more accurate for small samples)
        # Under null hypothesis, P(b successes in b+c trials) with p=0.5
        # Two-sided test: sum probabilities in both tails
        p_value = stats.binomtest(b, b + c, 0.5, alternative="two-sided").pvalue
        statistic = float(b)  # Report the count as "statistic" for exact test
    else:
        # Chi-squared approximation with continuity correction (Edwards correction)
        statistic = (np.abs(b - c) - 1) ** 2 / (b + c)
        p_value = stats.chi2.sf(statistic, df=1)

    return McNemarTestResult(
        statistic=float(statistic),
        p_value=float(p_value),
        n_both_correct=int(n_both_correct),
        n_both_wrong=int(n_both_wrong),
        n_a_correct_b_wrong=int(b),
        n_a_wrong_b_correct=int(c),
        is_significant=p_value < alpha,
        alpha=alpha,
        exact_used=bool(use_exact),
    )


def format_delong_result(result: DeLongTestResult) -> str:
    """Format DeLong test result as human-readable string.

    Parameters
    ----------
    result : DeLongTestResult
        DeLong test result object to format.

    Returns
    -------
    str
        Formatted string with test results.
    """
    significance = "SIGNIFICANT" if result.is_significant else "NOT SIGNIFICANT"
    return (
        f"DeLong Test: z={result.z_statistic:.4f}, p={result.p_value:.4f} "
        f"({significance} at alpha={result.alpha})\n"
        f"  AUROC A: {result.auroc_a:.4f}, AUROC B: {result.auroc_b:.4f}, "
        f"Diff: {result.auroc_diff:+.4f}"
    )


def format_mcnemar_result(result: McNemarTestResult) -> str:
    """Format McNemar's test result as human-readable string.

    Parameters
    ----------
    result : McNemarTestResult
        McNemar's test result object to format.

    Returns
    -------
    str
        Formatted string with test results.
    """
    significance = "SIGNIFICANT" if result.is_significant else "NOT SIGNIFICANT"
    return (
        f"McNemar's Test: statistic={result.statistic:.4f}, p={result.p_value:.4f} "
        f"({significance} at alpha={result.alpha})\n"
        f"  Both correct: {result.n_both_correct}, Both wrong: {result.n_both_wrong}\n"
        f"  A correct/B wrong: {result.n_a_correct_b_wrong}, "
        f"A wrong/B correct: {result.n_a_wrong_b_correct}"
    )
