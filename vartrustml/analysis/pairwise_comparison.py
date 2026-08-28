"""
Paired pairwise comparison of classifiers on pooled out-of-fold predictions.

This module replaces the earlier family-level (median-aggregated Friedman /
Wilcoxon) analysis with paired tests computed on *pooled out-of-fold (OOF)*
predictions, where every variant is predicted exactly once (at the optimized
threshold of its hold-out fold). Because each instance contributes a single
paired observation, there is no cross-fold dependence to correct for, and the
appropriate tests are:

- **McNemar's test** for two classifiers at a fixed operating point. This is the
  textbook-correct paired test for comparing two classifiers on the same test
  set (Dietterich, 1998) and is the only inferential comparison available for
  variant callers, which emit hard binary calls (no scores).
- **DeLong's test** for the AUROC of two ML models, which requires probability
  scores and is therefore restricted to ML-vs-ML comparisons.

Every comparison reports paired effect sizes alongside the p-value:

- the discordant counts ``b`` (A right / B wrong) and ``c`` (A wrong / B right);
- the paired accuracy difference ``(b - c) / n`` with a 95% confidence interval
  (positive favours A);
- the discordant odds ratio ``b / c`` with a 95% confidence interval.

P-values within a comparison *family* (all operating-point McNemar tests; all
AUROC DeLong tests) are corrected for multiple comparisons (Holm-Bonferroni by
default, Benjamini-Hochberg optional).

References
----------
- Dietterich, T. G. (1998). Approximate statistical tests for comparing
  supervised classification learning algorithms. Neural Computation,
  10(7), 1895-1923.
- McNemar, Q. (1947). Note on the sampling error of the difference between
  correlated proportions or percentages. Psychometrika, 12(2), 153-157.
- DeLong, E.R., DeLong, D.M., Clarke-Pearson, D.L. (1988). Biometrics,
  44(3), 837-845.
- Benjamini, Y. & Hochberg, Y. (1995). JRSS B, 57(1), 289-300.

See Also
--------
vartrustml.analysis.delong_mcnemar : Underlying McNemar / DeLong implementations.
vartrustml.analysis.bootstrap : Bias-corrected bootstrap confidence intervals.
"""

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import norm
from sklearn.metrics import matthews_corrcoef, roc_auc_score

from vartrustml.analysis.delong_mcnemar import (
    correct_pvalues,
    delong_test,
    mcnemar_test,
)

logger = logging.getLogger(__name__)

# Entity / comparison type labels.
TYPE_ML = "ML Model"
TYPE_SINGLE_CALLER = "Single Caller"
TYPE_COMBINATION = "Caller Combination"

# Correction-family labels.
FAMILY_OPERATING_POINT = "operating_point"  # McNemar tests
FAMILY_AUROC = "auroc"  # DeLong tests

# Floor for displaying p-values.
P_DISPLAY_FLOOR = 1e-10


def format_pvalue(p: Optional[float], floor: float = P_DISPLAY_FLOOR) -> str:
    """Format a p/q-value for display, flooring negligibly small values.

    Values below ``floor`` (including 0 from numerical underflow) render as
    ``"< 1e-10"`` instead of e.g. ``3.4e-235`` or ``0.0e+00``. Larger values use
    one-significant-digit scientific notation below 1e-3 and three decimals
    otherwise. Raw values in the CSV exports are never modified.
    """
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    if p < floor:
        return f"< {floor:.0e}"
    if p < 1e-3:
        return f"{p:.1e}"
    return f"{p:.3f}"


@dataclass
class Entity:
    """Aligned pooled out-of-fold predictions for one classifier.

    Attributes
    ----------
    name : str
        Identifier (e.g. ``"RandomForest"``, ``"MANTA"``, ``"MANTA AND DELLY"``).
    entity_type : str
        One of :data:`TYPE_ML`, :data:`TYPE_SINGLE_CALLER`,
        :data:`TYPE_COMBINATION`.
    y_true : numpy.ndarray
        Ground-truth labels aligned to this entity's predictions. Carried per
        entity so alignment can verify agreement across classifiers.
    y_pred : numpy.ndarray
        Hard predictions at the operating point (per-fold optimized threshold
        for ML models; native binary calls for callers).
    y_prob : numpy.ndarray or None
        Probability scores for the positive class. ``None`` for callers.
    sample_indices : numpy.ndarray or None
        Original row indices used to align entities across models/callers.
    """

    name: str
    entity_type: str
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: Optional[np.ndarray] = None
    sample_indices: Optional[np.ndarray] = None


@dataclass
class PairwiseComparison:
    """Result of a single paired comparison between two classifiers.

    Attributes
    ----------
    name_a, name_b : str
        Names of the two classifiers (A is the reference, typically the ML
        model).
    comparison_type : str
        Human-readable pairing, e.g. ``"ML Model vs Single Caller"``.
    test_name : str
        ``"McNemar (exact)"``, ``"McNemar (chi2)"`` or ``"DeLong (AUROC)"``.
    family : str
        Correction family the p-value belongs to
        (:data:`FAMILY_OPERATING_POINT` or :data:`FAMILY_AUROC`).
    statistic : float
        Test statistic (discordant count for exact McNemar, chi-square for the
        approximation, z for DeLong).
    p_value : float
        Raw two-sided p-value.
    p_value_corrected : float
        Multiple-comparison adjusted p-value (q-value) within ``family``.
    is_significant : bool
        Whether ``p_value_corrected`` <= alpha.
    better : str
        Name of the favoured classifier, or ``"tie"``.
    n_total : int
        Number of paired instances.
    n_discordant_b, n_discordant_c : int or None
        McNemar discordant counts (A right/B wrong, A wrong/B right). ``None``
        for DeLong.
    acc_diff, acc_diff_ci_lower, acc_diff_ci_upper : float or None
        Paired accuracy difference (A - B) with CI. ``None`` for DeLong.
    odds_ratio, odds_ratio_ci_lower, odds_ratio_ci_upper : float or None
        Discordant odds ratio b/c with CI. ``None`` for DeLong.
    auroc_a, auroc_b, auroc_diff : float or None
        AUROC values and difference (A - B). Only set for DeLong.
    """

    name_a: str
    name_b: str
    comparison_type: str
    test_name: str
    family: str
    statistic: float
    p_value: float
    p_value_corrected: float
    is_significant: bool
    better: str
    n_total: int
    n_discordant_b: Optional[int] = None
    n_discordant_c: Optional[int] = None
    acc_diff: Optional[float] = None
    acc_diff_ci_lower: Optional[float] = None
    acc_diff_ci_upper: Optional[float] = None
    odds_ratio: Optional[float] = None
    odds_ratio_ci_lower: Optional[float] = None
    odds_ratio_ci_upper: Optional[float] = None
    auroc_a: Optional[float] = None
    auroc_b: Optional[float] = None
    auroc_diff: Optional[float] = None


@dataclass
class PairwiseComparisonResult:
    """Container for a full set of pairwise comparisons on one dataset.

    Attributes
    ----------
    y_true : numpy.ndarray
        Common (aligned) ground-truth labels shared by all entities.
    entities : dict of {str: Entity}
        The aligned entities that were compared.
    comparisons : list of PairwiseComparison
        Every pairwise comparison computed (the full matrix).
    best_ml : str or None
        Name of the best ML model by the primary metric (used for the main
        ML-vs-callers table).
    primary_metric : str
        Metric used to rank ML models / determine ``best_ml``.
    alpha : float
        Significance level used for correction.
    metric_scores : dict of {str: float}
        Primary-metric point estimate per entity on the pooled OOF data
        (descriptive; used for the forest plot ranking).
    """

    y_true: np.ndarray
    entities: Dict[str, Entity]
    comparisons: List[PairwiseComparison]
    best_ml: Optional[str]
    primary_metric: str
    alpha: float
    metric_scores: Dict[str, float] = field(default_factory=dict)
    correction_method: str = "holm"

    def main_comparisons(self) -> List[PairwiseComparison]:
        """Best-ML-vs-caller/combination operating-point comparisons (main table)."""
        if self.best_ml is None:
            return []
        return [
            c
            for c in self.comparisons
            if c.family == FAMILY_OPERATING_POINT
            and c.name_a == self.best_ml
            and self.entities[c.name_b].entity_type != TYPE_ML
        ]


def build_entities(
    oof_predictions: Dict[str, Dict[str, np.ndarray]],
    caller_results: Optional[Dict[str, List]] = None,
) -> Dict[str, Entity]:
    """Assemble :class:`Entity` objects from ML OOF predictions and callers.

    Parameters
    ----------
    oof_predictions : dict
        Mapping model name -> dict with keys ``y_true``, ``y_pred`` (at the
        operating point), ``y_prob`` and optionally ``sample_indices``.
    caller_results : dict of {str: list}, optional
        Mapping caller/combination name -> list of per-fold ``CallerResult``
        objects (with ``y_true``, ``y_pred`` and optionally ``sample_indices``).
        Names containing ``" AND "`` or ``" OR "`` are treated as combinations.

    Returns
    -------
    dict of {str: Entity}
    """
    entities: Dict[str, Entity] = {}

    for name, entry in oof_predictions.items():
        if "y_pred" not in entry:
            logger.warning(
                f"Model '{name}' OOF entry lacks operating-point predictions; "
                f"skipping it from the paired comparison."
            )
            continue
        entities[name] = Entity(
            name=name,
            entity_type=TYPE_ML,
            y_true=np.asarray(entry["y_true"]),
            y_pred=np.asarray(entry["y_pred"]),
            y_prob=(
                np.asarray(entry["y_prob"]) if entry.get("y_prob") is not None else None
            ),
            sample_indices=(
                np.asarray(entry["sample_indices"])
                if entry.get("sample_indices") is not None
                else None
            ),
        )

    if caller_results:
        for name, fold_results in caller_results.items():
            if not fold_results:
                continue
            y_true = np.concatenate([np.asarray(r.y_true) for r in fold_results])
            y_pred = np.concatenate([np.asarray(r.y_pred) for r in fold_results])
            have_idx = all(
                getattr(r, "sample_indices", None) is not None for r in fold_results
            )
            sample_indices = (
                np.concatenate([np.asarray(r.sample_indices) for r in fold_results])
                if have_idx
                else None
            )
            is_combination = " AND " in name or " OR " in name
            entities[name] = Entity(
                name=name,
                entity_type=TYPE_COMBINATION if is_combination else TYPE_SINGLE_CALLER,
                y_true=y_true,
                y_pred=y_pred,
                y_prob=None,
                sample_indices=sample_indices,
            )

    return entities


def _paired_accuracy_diff_ci(b: int, c: int, n: int, ci_level: float = 0.95) -> tuple:
    """Paired accuracy difference (A - B) and its Wald CI from discordant counts.

    The difference in accuracy between two classifiers evaluated on the same
    instances is ``(b - c) / n`` (concordant cells cancel). The standard error
    of this difference of correlated proportions is
    ``sqrt(b + c - (b - c)^2 / n) / n``.
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    diff = (b - c) / n
    var = (b + c - (b - c) ** 2 / n) / (n**2)
    se = float(np.sqrt(var)) if var > 0 else 0.0
    z = norm.ppf(1 - (1 - ci_level) / 2)
    return diff, diff - z * se, diff + z * se


def _discordant_odds_ratio_ci(b: int, c: int, ci_level: float = 0.95) -> tuple:
    """Discordant odds ratio b/c with a log-OR Wald CI.

    A Haldane-Anscombe 0.5 correction is applied when either discordant cell is
    zero so the (log) odds ratio and its variance stay finite.
    """
    if b == 0 and c == 0:
        return float("nan"), float("nan"), float("nan")
    b_adj, c_adj = b, c
    if b == 0 or c == 0:
        b_adj, c_adj = b + 0.5, c + 0.5
    odds = b_adj / c_adj
    se = float(np.sqrt(1.0 / b_adj + 1.0 / c_adj))
    z = norm.ppf(1 - (1 - ci_level) / 2)
    log_or = np.log(odds)
    return (
        float(odds),
        float(np.exp(log_or - z * se)),
        float(np.exp(log_or + z * se)),
    )


def align_entities(entities: Dict[str, Entity]) -> tuple:
    """Align all entities onto a common instance ordering.

    When every entity carries ``sample_indices``, entities are reindexed onto
    the sorted intersection of their indices so that position *i* refers to the
    same variant for all of them. Otherwise a positional alignment is used,
    requiring equal length. In both cases ground-truth agreement is verified;
    an entity whose labels disagree with the reference is dropped with a
    warning (a misaligned entity would silently invalidate every paired test).

    Returns
    -------
    tuple of (dict of {str: Entity}, numpy.ndarray)
        The aligned entities and the common ``y_true``. Returns ``({}, empty)``
        when fewer than two entities can be aligned.
    """
    items = [(name, e) for name, e in entities.items() if e.y_pred is not None]
    if len(items) < 2:
        logger.warning("Need at least 2 entities with predictions to compare")
        return {}, np.array([])

    have_indices = all(e.sample_indices is not None for _, e in items)

    if have_indices:
        common = None
        for _, e in items:
            idx = set(int(i) for i in e.sample_indices)
            common = idx if common is None else (common & idx)
        common_sorted = np.array(sorted(common))
        if len(common_sorted) == 0:
            logger.warning("Entities share no common sample indices")
            return {}, np.array([])

        aligned: Dict[str, Entity] = {}
        ref_y_true = None
        for name, e in items:
            pos = {int(i): p for p, i in enumerate(e.sample_indices)}
            take = np.array([pos[i] for i in common_sorted])
            y_true_e = np.asarray(e.y_true)[take]
            if ref_y_true is None:
                ref_y_true = y_true_e
            elif not np.array_equal(y_true_e, ref_y_true):
                logger.warning(
                    f"Entity '{name}' ground truth disagrees after alignment; "
                    f"dropping it from the comparison."
                )
                continue
            aligned[name] = Entity(
                name=name,
                entity_type=e.entity_type,
                y_true=y_true_e,
                y_pred=np.asarray(e.y_pred)[take],
                y_prob=(np.asarray(e.y_prob)[take] if e.y_prob is not None else None),
                sample_indices=common_sorted,
            )
        return (aligned if len(aligned) >= 2 else {}), (
            ref_y_true if ref_y_true is not None else np.array([])
        )

    # Positional fallback: require equal length and identical ground truth.
    lengths = {len(e.y_pred) for _, e in items}
    if len(lengths) > 1:
        logger.warning(
            "Entities lack sample indices and have differing lengths; cannot "
            "align for paired comparison."
        )
        return {}, np.array([])

    ref_y_true = np.asarray(items[0][1].y_true)
    aligned = {}
    for name, e in items:
        if not np.array_equal(np.asarray(e.y_true), ref_y_true):
            logger.warning(
                f"Entity '{name}' ground truth disagrees (positional alignment); "
                f"dropping it from the comparison."
            )
            continue
        aligned[name] = e
    return (aligned if len(aligned) >= 2 else {}), ref_y_true


def compare_pairwise(
    entities: Dict[str, Entity],
    primary_metric: str = "Matthews Corr. Coef.",
    alpha: float = 0.05,
    ci_level: float = 0.95,
    correction_method: str = "holm",
) -> Optional[PairwiseComparisonResult]:
    """Run the full paired pairwise comparison on pooled OOF predictions.

    Parameters
    ----------
    entities : dict of {str: Entity}
        Classifiers to compare. Each :class:`Entity` carries its own
        ``y_true`` so alignment can verify ground-truth agreement.
    primary_metric : str, default="Matthews Corr. Coef."
        Metric used to rank ML models and select ``best_ml``. Supported:
        ``"Matthews Corr. Coef."`` and ``"AUROC"``; otherwise MCC is used.
    alpha : float, default=0.05
        Significance level for the multiple-comparison correction.
    ci_level : float, default=0.95
        Confidence level for effect-size intervals.

    Returns
    -------
    PairwiseComparisonResult or None
        ``None`` when fewer than two entities can be aligned.
    """
    aligned, y_true = align_entities(entities)
    if len(aligned) < 2 or len(y_true) == 0:
        logger.warning("Pairwise comparison skipped: insufficient aligned entities")
        return None

    names = list(aligned.keys())
    ml_names = [n for n in names if aligned[n].entity_type == TYPE_ML]

    # Descriptive primary-metric score per entity (for forest-plot ranking and
    # best-ML selection).
    metric_scores: Dict[str, float] = {}
    for name, e in aligned.items():
        try:
            if primary_metric == "AUROC" and e.y_prob is not None:
                metric_scores[name] = float(roc_auc_score(y_true, e.y_prob))
            else:
                metric_scores[name] = float(matthews_corrcoef(y_true, e.y_pred))
        except ValueError:
            metric_scores[name] = float("nan")

    best_ml = None
    if ml_names:
        best_ml = max(
            ml_names,
            key=lambda n: (
                metric_scores[n] if not np.isnan(metric_scores[n]) else -np.inf
            ),
        )

    comparisons: List[PairwiseComparison] = []

    # --- Operating-point McNemar: every ML vs every caller/combination, and
    #     every ML vs ML pair. ML is always side A for callers; for ML vs ML the
    #     better-scoring model is placed on side A for readable "better".
    mcnemar_specs = []
    for ml in ml_names:
        for other in names:
            if aligned[other].entity_type != TYPE_ML:
                mcnemar_specs.append((ml, other))
    for a, b in combinations(ml_names, 2):
        # order so A is the higher-MCC model (purely cosmetic for `better`)
        if metric_scores.get(b, -np.inf) > metric_scores.get(a, -np.inf):
            a, b = b, a
        mcnemar_specs.append((a, b))

    mcnemar_raw = []
    for name_a, name_b in mcnemar_specs:
        res = mcnemar_test(
            y_true, aligned[name_a].y_pred, aligned[name_b].y_pred, alpha=alpha
        )
        mcnemar_raw.append((name_a, name_b, res))

    op_pvals = [r.p_value for _, _, r in mcnemar_raw]
    op_qvals, _ = correct_pvalues(np.array(op_pvals), correction_method, alpha)

    for i, (name_a, name_b, res) in enumerate(mcnemar_raw):
        b, c = res.n_a_correct_b_wrong, res.n_a_wrong_b_correct
        n = res.n_both_correct + res.n_both_wrong + b + c
        diff, lo, hi = _paired_accuracy_diff_ci(b, c, n, ci_level)
        odds, or_lo, or_hi = _discordant_odds_ratio_ci(b, c, ci_level)
        if b > c:
            better = name_a
        elif c > b:
            better = name_b
        else:
            better = "tie"
        type_b = aligned[name_b].entity_type
        test_name = "McNemar (exact)" if res.exact_used else "McNemar (chi2)"
        comparisons.append(
            PairwiseComparison(
                name_a=name_a,
                name_b=name_b,
                comparison_type=f"{aligned[name_a].entity_type} vs {type_b}",
                test_name=test_name,
                family=FAMILY_OPERATING_POINT,
                statistic=float(res.statistic),
                p_value=float(res.p_value),
                p_value_corrected=float(op_qvals[i]),
                is_significant=bool(op_qvals[i] <= alpha),
                better=better,
                n_total=int(n),
                n_discordant_b=int(b),
                n_discordant_c=int(c),
                acc_diff=float(diff),
                acc_diff_ci_lower=float(lo),
                acc_diff_ci_upper=float(hi),
                odds_ratio=odds,
                odds_ratio_ci_lower=or_lo,
                odds_ratio_ci_upper=or_hi,
            )
        )

    # --- DeLong AUROC: every ML vs ML pair with probability scores.
    delong_specs = [
        (a, b)
        for a, b in combinations(ml_names, 2)
        if aligned[a].y_prob is not None and aligned[b].y_prob is not None
    ]
    delong_raw = []
    for name_a, name_b in delong_specs:
        res = delong_test(
            y_true, aligned[name_a].y_prob, aligned[name_b].y_prob, alpha=alpha
        )
        delong_raw.append((name_a, name_b, res))

    if delong_raw:
        au_pvals = [r.p_value for _, _, r in delong_raw]
        au_qvals, _ = correct_pvalues(np.array(au_pvals), correction_method, alpha)
        for i, (name_a, name_b, res) in enumerate(delong_raw):
            if res.auroc_diff > 0:
                better = name_a
            elif res.auroc_diff < 0:
                better = name_b
            else:
                better = "tie"
            comparisons.append(
                PairwiseComparison(
                    name_a=name_a,
                    name_b=name_b,
                    comparison_type="ML Model vs ML Model (AUROC)",
                    test_name="DeLong (AUROC)",
                    family=FAMILY_AUROC,
                    statistic=float(res.z_statistic),
                    p_value=float(res.p_value),
                    p_value_corrected=float(au_qvals[i]),
                    is_significant=bool(au_qvals[i] <= alpha and res.is_valid),
                    better=better if res.is_valid else "tie",
                    n_total=int(len(y_true)),
                    auroc_a=float(res.auroc_a),
                    auroc_b=float(res.auroc_b),
                    auroc_diff=float(res.auroc_diff),
                )
            )

    return PairwiseComparisonResult(
        y_true=y_true,
        entities=aligned,
        comparisons=comparisons,
        best_ml=best_ml,
        primary_metric=primary_metric,
        alpha=alpha,
        metric_scores=metric_scores,
        correction_method=correction_method,
    )


def comparisons_to_dataframe(result: PairwiseComparisonResult):
    """Flatten all pairwise comparisons into a tidy DataFrame for CSV export.

    Produces one row per comparison with the test, p-values (raw and corrected), the
    favoured classifier and the paired effect sizes (discordant counts, paired
    accuracy difference with CI, discordant odds ratio with CI) plus the AUROC
    fields for DeLong rows.
    """
    import pandas as pd

    rows = []
    for c in result.comparisons:
        rows.append(
            {
                "name_a": c.name_a,
                "name_b": c.name_b,
                "comparison_type": c.comparison_type,
                "test": c.test_name,
                "family": c.family,
                "better": c.better,
                "statistic": c.statistic,
                "p_value": c.p_value,
                "p_value_corrected": c.p_value_corrected,
                "significant": c.is_significant,
                "n_total": c.n_total,
                "discordant_b": c.n_discordant_b,
                "discordant_c": c.n_discordant_c,
                "acc_diff_a_minus_b": c.acc_diff,
                "acc_diff_ci_lower": c.acc_diff_ci_lower,
                "acc_diff_ci_upper": c.acc_diff_ci_upper,
                "odds_ratio_b_over_c": c.odds_ratio,
                "odds_ratio_ci_lower": c.odds_ratio_ci_lower,
                "odds_ratio_ci_upper": c.odds_ratio_ci_upper,
                "auroc_a": c.auroc_a,
                "auroc_b": c.auroc_b,
                "auroc_diff": c.auroc_diff,
            }
        )
    return pd.DataFrame(rows)
