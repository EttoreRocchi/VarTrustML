"""
Unit tests for the paired pairwise comparison module.

Covers entity construction/alignment, McNemar (operating-point) and DeLong
(AUROC) comparisons on pooled out-of-fold predictions, paired effect sizes,
Benjamini-Hochberg correction, best-ML selection, and the report plots.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pytest

from vartrustml.analysis import pairwise_plots as pp
from vartrustml.analysis.pairwise_comparison import (
    FAMILY_AUROC,
    FAMILY_OPERATING_POINT,
    TYPE_COMBINATION,
    TYPE_ML,
    TYPE_SINGLE_CALLER,
    Entity,
    _discordant_odds_ratio_ci,
    _paired_accuracy_diff_ci,
    align_entities,
    build_entities,
    comparisons_to_dataframe,
    compare_pairwise,
)


@dataclass
class MockCallerResult:
    """Minimal stand-in for core.caller_evaluator.CallerResult."""

    y_true: np.ndarray
    y_pred: np.ndarray
    sample_indices: Optional[np.ndarray] = None


def _make_ml(y, signal, rng, idx, threshold=0.5):
    prob = np.clip(
        signal * y + (1 - signal) * 0.5 + 0.15 * rng.standard_normal(len(y)), 0, 1
    )
    return {
        "y_true": y,
        "y_pred": (prob >= threshold).astype(int),
        "y_prob": prob,
        "sample_indices": idx,
    }


def _make_caller(y, err, rng, n_folds=3):
    """Per-fold caller results with sample indices."""
    idx = np.arange(len(y))
    res = []
    for f in np.array_split(idx, n_folds):
        yt = y[f]
        yp = yt.copy()
        flip = rng.random(len(f)) < err
        yp[flip] = 1 - yp[flip]
        res.append(MockCallerResult(yt, yp, f))
    return res


@pytest.fixture
def scenario():
    rng = np.random.default_rng(0)
    n = 2000
    idx = np.arange(n)
    y = rng.integers(0, 2, n)
    oof = {
        "RF": _make_ml(y, 0.8, rng, idx),
        "LR": _make_ml(y, 0.6, rng, idx),
    }
    callers = {
        "MANTA": _make_caller(y, 0.2, rng),
        "DELLY": _make_caller(y, 0.35, rng),
        "MANTA AND DELLY": _make_caller(y, 0.18, rng),
    }
    return oof, callers, y


class TestBuildEntities:
    def test_classifies_entity_types(self, scenario):
        oof, callers, _ = scenario
        ents = build_entities(oof, callers)
        assert ents["RF"].entity_type == TYPE_ML
        assert ents["MANTA"].entity_type == TYPE_SINGLE_CALLER
        assert ents["MANTA AND DELLY"].entity_type == TYPE_COMBINATION

    def test_or_combination_detected(self):
        y = np.array([0, 1, 0, 1])
        ents = build_entities({}, {"A OR B": [MockCallerResult(y, y, np.arange(4))]})
        assert ents["A OR B"].entity_type == TYPE_COMBINATION

    def test_ml_without_y_pred_skipped(self):
        y = np.array([0, 1, 0, 1])
        ents = build_entities({"M": {"y_true": y, "y_prob": y.astype(float)}}, None)
        assert "M" not in ents


class TestAlignment:
    def test_aligns_by_sample_indices(self):
        # two entities with shuffled, overlapping indices
        y_full = np.array([0, 1, 0, 1, 1, 0])
        e1 = Entity(
            "a",
            TYPE_ML,
            y_full[[0, 1, 2, 3]],
            y_full[[0, 1, 2, 3]],
            sample_indices=np.array([0, 1, 2, 3]),
        )
        order = np.array([3, 2, 1, 0])
        e2 = Entity(
            "b", TYPE_SINGLE_CALLER, y_full[order], y_full[order], sample_indices=order
        )
        aligned, y_true = align_entities({"a": e1, "b": e2})
        assert set(aligned) == {"a", "b"}
        assert np.array_equal(aligned["a"].y_true, aligned["b"].y_true)
        assert np.array_equal(y_true, np.array([0, 1, 0, 1]))

    def test_drops_entity_with_disagreeing_ground_truth(self):
        idx = np.arange(4)
        good = Entity(
            "good",
            TYPE_ML,
            np.array([0, 1, 0, 1]),
            np.array([0, 1, 0, 1]),
            sample_indices=idx,
        )
        bad = Entity(
            "bad",
            TYPE_SINGLE_CALLER,
            np.array([1, 1, 1, 1]),
            np.array([0, 0, 0, 0]),
            sample_indices=idx,
        )
        third = Entity(
            "third",
            TYPE_ML,
            np.array([0, 1, 0, 1]),
            np.array([0, 0, 0, 1]),
            sample_indices=idx,
        )
        aligned, _ = align_entities({"good": good, "bad": bad, "third": third})
        assert "bad" not in aligned
        assert {"good", "third"} <= set(aligned)

    def test_insufficient_entities_returns_empty(self):
        e = Entity("a", TYPE_ML, np.array([0, 1]), np.array([0, 1]))
        aligned, y_true = align_entities({"a": e})
        assert aligned == {}
        assert len(y_true) == 0


class TestComparePairwise:
    def test_full_matrix_and_best_ml(self, scenario):
        oof, callers, _ = scenario
        res = compare_pairwise(build_entities(oof, callers))
        assert res is not None
        assert res.best_ml == "RF"  # stronger signal
        # 2 ML x 3 callers = 6, plus 1 ML-ML McNemar, plus 1 ML-ML DeLong
        op = [c for c in res.comparisons if c.family == FAMILY_OPERATING_POINT]
        au = [c for c in res.comparisons if c.family == FAMILY_AUROC]
        assert len(op) == 7
        assert len(au) == 1

    def test_main_comparisons_are_best_ml_vs_callers(self, scenario):
        oof, callers, _ = scenario
        res = compare_pairwise(build_entities(oof, callers))
        main = res.main_comparisons()
        assert len(main) == 3
        assert all(c.name_a == "RF" for c in main)
        assert all(res.entities[c.name_b].entity_type != TYPE_ML for c in main)

    def test_better_direction_matches_acc_diff_sign(self, scenario):
        oof, callers, _ = scenario
        res = compare_pairwise(build_entities(oof, callers))
        for c in res.main_comparisons():
            # ML is much stronger than callers -> ML wins, positive acc diff
            assert c.better == "RF"
            assert c.acc_diff > 0

    def test_bh_correction_not_below_raw(self, scenario):
        oof, callers, _ = scenario
        res = compare_pairwise(build_entities(oof, callers))
        for c in res.comparisons:
            assert c.p_value_corrected >= c.p_value - 1e-12

    def test_delong_present_for_ml_pair(self, scenario):
        oof, callers, _ = scenario
        res = compare_pairwise(build_entities(oof, callers))
        delong = [c for c in res.comparisons if c.family == FAMILY_AUROC]
        assert delong[0].test_name == "DeLong (AUROC)"
        assert delong[0].auroc_a is not None

    def test_returns_none_with_single_entity(self):
        y = np.array([0, 1, 0, 1])
        ents = build_entities(
            {"M": _make_ml(y, 0.8, np.random.default_rng(0), np.arange(4))}, None
        )
        assert compare_pairwise(ents) is None

    def test_dataframe_export_has_expected_columns(self, scenario):
        oof, callers, _ = scenario
        res = compare_pairwise(build_entities(oof, callers))
        df = comparisons_to_dataframe(res)
        for col in [
            "name_a",
            "name_b",
            "test",
            "family",
            "better",
            "p_value",
            "p_value_corrected",
            "discordant_b",
            "discordant_c",
            "acc_diff_a_minus_b",
            "odds_ratio_b_over_c",
            "auroc_diff",
        ]:
            assert col in df.columns
        assert len(df) == len(res.comparisons)


class TestEffectSizeHelpers:
    def test_paired_accuracy_diff_sign(self):
        diff, lo, hi = _paired_accuracy_diff_ci(b=80, c=20, n=1000)
        assert diff == pytest.approx(0.06)
        assert lo < diff < hi

    def test_paired_accuracy_diff_zero_when_symmetric(self):
        diff, lo, hi = _paired_accuracy_diff_ci(b=50, c=50, n=1000)
        assert diff == pytest.approx(0.0)
        assert lo < 0 < hi

    def test_odds_ratio_haldane_when_zero_cell(self):
        odds, lo, hi = _discordant_odds_ratio_ci(b=30, c=0)
        assert np.isfinite(odds) and odds > 1
        assert np.isfinite(lo) and np.isfinite(hi)

    def test_odds_ratio_nan_when_no_discordants(self):
        odds, lo, hi = _discordant_odds_ratio_ci(b=0, c=0)
        assert np.isnan(odds)


class TestMcNemarTestSelection:
    def test_exact_for_few_discordants(self):
        # construct two predictors with very few disagreements
        y = np.array([0, 1] * 100)
        a = y.copy()
        b = y.copy()
        b[0] = 1 - b[0]  # one discordant
        ents = {
            "A": Entity("A", TYPE_ML, y, a, sample_indices=np.arange(len(y))),
            "B": Entity("B", TYPE_ML, y, b, sample_indices=np.arange(len(y))),
        }
        res = compare_pairwise(ents)
        op = [c for c in res.comparisons if c.family == FAMILY_OPERATING_POINT]
        assert op[0].test_name == "McNemar (exact)"


class TestPlots:
    def test_forest_plot_returns_base64(self, scenario):
        oof, callers, _ = scenario
        res = compare_pairwise(build_entities(oof, callers))
        b64 = pp.plot_metric_forest(res, bootstrap_n_iterations=200, return_base64=True)
        assert isinstance(b64, str) and len(b64) > 100

    def test_dominance_plot_returns_base64(self, scenario):
        oof, callers, _ = scenario
        res = compare_pairwise(build_entities(oof, callers))
        b64 = pp.plot_roc_pr_dominance(res, return_base64=True)
        assert isinstance(b64, str) and len(b64) > 100

    def test_heatmap_returns_base64(self, scenario):
        oof, callers, _ = scenario
        res = compare_pairwise(build_entities(oof, callers))
        b64 = pp.plot_pvalue_heatmap(res, return_base64=True)
        assert isinstance(b64, str) and len(b64) > 100

    def test_heatmap_none_with_single_ml(self):
        y = np.array([0, 1] * 50)
        idx = np.arange(len(y))
        ents = build_entities(
            {"M": _make_ml(y, 0.8, np.random.default_rng(0), idx)},
            {"MANTA": [MockCallerResult(y, y, idx)]},
        )
        res = compare_pairwise(ents)
        assert pp.plot_pvalue_heatmap(res) is None
