"""
Integration tests for caller comparison functionality.

Tests the full flow from caller evaluation to bootstrap CI computation.
"""

import numpy as np
import pandas as pd
import pytest

from vartrustml.analysis.bootstrap import BootstrapAnalyzer
from vartrustml.config.caller import CallerConfig
from vartrustml.core.caller_evaluator import CallerEvaluator, validate_caller_columns


class TestCallerIntegration:
    """Integration tests for caller comparison workflow."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a realistic sample dataset with callers and target."""
        np.random.seed(42)
        n_samples = 500

        state = np.random.randint(0, 2, n_samples)

        manta = state.copy()
        flip_idx = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
        manta[flip_idx] = 1 - manta[flip_idx]

        delly = state.copy()
        flip_idx = np.random.choice(
            n_samples, size=int(n_samples * 0.15), replace=False
        )
        delly[flip_idx] = 1 - delly[flip_idx]

        smoove = state.copy()
        flip_idx = np.random.choice(n_samples, size=int(n_samples * 0.2), replace=False)
        smoove[flip_idx] = 1 - smoove[flip_idx]

        return pd.DataFrame(
            {
                "MANTA": manta,
                "DELLY": delly,
                "SMOOVE": smoove,
                "state": state,
            }
        )

    @pytest.fixture
    def cv_fold_indices(self, sample_dataset):
        """Create CV fold indices (simulating 5-fold CV)."""
        n_samples = len(sample_dataset)
        indices = np.arange(n_samples)
        np.random.seed(42)
        np.random.shuffle(indices)

        n_folds = 5
        fold_size = n_samples // n_folds
        folds = []

        for i in range(n_folds):
            test_start = i * fold_size
            test_end = test_start + fold_size if i < n_folds - 1 else n_samples
            test_idx = indices[test_start:test_end]
            train_idx = np.concatenate([indices[:test_start], indices[test_end:]])
            folds.append((train_idx, test_idx))

        return folds

    def test_full_caller_evaluation_workflow(self, sample_dataset, cv_fold_indices):
        """Test complete caller evaluation workflow."""
        caller_columns = ["MANTA", "DELLY", "SMOOVE"]
        target_column = "state"

        validate_caller_columns(sample_dataset, caller_columns, target_column)

        evaluator = CallerEvaluator(caller_columns)

        manta_results = []
        for fold_id, (train_idx, test_idx) in enumerate(cv_fold_indices):
            y_test = sample_dataset[target_column].iloc[test_idx].values
            caller_pred = sample_dataset["MANTA"].iloc[test_idx].values

            result = evaluator.evaluate_single_caller(
                "MANTA", y_test, caller_pred, fold_id
            )
            manta_results.append(result)

        assert len(manta_results) == 5
        for i, result in enumerate(manta_results):
            assert result.fold_id == i
            assert result.name == "MANTA"

    def test_caller_combination_evaluation(self, sample_dataset, cv_fold_indices):
        """Test caller combination evaluation across folds."""
        caller_columns = ["MANTA", "DELLY", "SMOOVE"]
        evaluator = CallerEvaluator(caller_columns)

        and_results = []
        or_results = []

        for fold_id, (train_idx, test_idx) in enumerate(cv_fold_indices):
            y_test = sample_dataset["state"].iloc[test_idx].values
            test_data = sample_dataset[caller_columns].iloc[test_idx]

            and_result = evaluator.evaluate_combination(
                ["MANTA", "DELLY"], "AND", y_test, test_data, fold_id
            )
            and_results.append(and_result)

            or_result = evaluator.evaluate_combination(
                ["MANTA", "DELLY"], "OR", y_test, test_data, fold_id
            )
            or_results.append(or_result)

        and_recalls = [r.metrics["Recall (Class 1)"] for r in and_results]
        or_recalls = [r.metrics["Recall (Class 1)"] for r in or_results]

        assert np.mean(or_recalls) >= np.mean(and_recalls)

    def test_bootstrap_ci_with_caller_results(self, sample_dataset, cv_fold_indices):
        """Test computing bootstrap CIs for caller metrics using prediction-level bootstrap."""
        bootstrap = BootstrapAnalyzer(n_iterations=500, ci_level=0.95, seed=42)

        # Collect predictions from all folds
        y_true_all = []
        y_pred_all = []

        for fold_id, (train_idx, test_idx) in enumerate(cv_fold_indices):
            y_test = sample_dataset["state"].iloc[test_idx].values
            caller_pred = sample_dataset["MANTA"].iloc[test_idx].values

            y_true_all.extend(y_test)
            y_pred_all.extend(caller_pred)

        # Compute CIs using prediction-level bootstrap
        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)

        ci_results = bootstrap.compute_all_cis_from_predictions(
            y_true_all, y_pred_all, y_prob=None
        )

        mcc_result = ci_results["Matthews Corr. Coef."]
        assert mcc_result.ci_lower <= mcc_result.point_estimate <= mcc_result.ci_upper
        assert mcc_result.n_iterations == 500
        assert mcc_result.ci_level == 0.95

    def test_caller_config_integration(self, sample_dataset, cv_fold_indices):
        """Test CallerConfig generates expected combinations."""
        caller_columns = ["MANTA", "DELLY", "SMOOVE"]

        config = CallerConfig.from_experiment_config(
            caller_columns=caller_columns,
            caller_combinations=["MANTA AND DELLY"],
            include_default_combinations=True,
        )

        assert "MANTA AND DELLY" in config.combinations
        assert "DELLY AND SMOOVE" in config.combinations
        assert "MANTA OR SMOOVE" in config.combinations
        assert "MANTA AND DELLY AND SMOOVE" in config.combinations
        assert "MANTA OR DELLY OR SMOOVE" in config.combinations

        evaluator = CallerEvaluator(caller_columns)

        for combo_expr in config.combinations:
            fold_id = 0
            train_idx, test_idx = cv_fold_indices[fold_id]
            y_test = sample_dataset["state"].iloc[test_idx].values
            test_data = sample_dataset[caller_columns].iloc[test_idx]

            result = evaluator.evaluate_from_expression(
                combo_expr, y_test, test_data, fold_id
            )

            assert result is not None
            assert result.name == combo_expr
            assert "Matthews Corr. Coef." in result.metrics

    def test_metrics_same_as_ml_format(self, sample_dataset, cv_fold_indices):
        """Test that caller metrics match ML model metric format."""
        evaluator = CallerEvaluator(["MANTA", "DELLY", "SMOOVE"])

        fold_id = 0
        train_idx, test_idx = cv_fold_indices[fold_id]
        y_test = sample_dataset["state"].iloc[test_idx].values
        caller_pred = sample_dataset["MANTA"].iloc[test_idx].values

        result = evaluator.evaluate_single_caller("MANTA", y_test, caller_pred, fold_id)

        expected_metrics = [
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
        ]

        for metric in expected_metrics:
            assert metric in result.metrics, f"Missing metric: {metric}"

        assert np.isnan(result.metrics["AUROC"])

    def test_fair_comparison_same_folds(self, sample_dataset, cv_fold_indices):
        """Test that using same folds enables fair comparison."""
        evaluator = CallerEvaluator(["MANTA", "DELLY", "SMOOVE"])

        manta_recalls = []
        delly_recalls = []

        for fold_id, (train_idx, test_idx) in enumerate(cv_fold_indices):
            y_test = sample_dataset["state"].iloc[test_idx].values

            manta_pred = sample_dataset["MANTA"].iloc[test_idx].values
            delly_pred = sample_dataset["DELLY"].iloc[test_idx].values

            manta_result = evaluator.evaluate_single_caller(
                "MANTA", y_test, manta_pred, fold_id
            )
            delly_result = evaluator.evaluate_single_caller(
                "DELLY", y_test, delly_pred, fold_id
            )

            manta_recalls.append(manta_result.metrics["Recall (Class 1)"])
            delly_recalls.append(delly_result.metrics["Recall (Class 1)"])

        assert np.mean(manta_recalls) > np.mean(delly_recalls)
