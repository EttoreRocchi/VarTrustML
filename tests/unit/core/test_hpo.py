"""
Unit tests for HyperparameterOptimizer and Optuna search space conversion.
"""

import numpy as np
import pandas as pd
import pytest
from optuna.distributions import CategoricalDistribution, FloatDistribution
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from vartrustml.core.hpo import HyperparameterOptimizer, to_optuna_distributions


@pytest.fixture
def binary_data():
    """Small separable binary classification problem."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "a": np.concatenate([rng.normal(0, 1, 40), rng.normal(3, 1, 40)]),
            "b": np.concatenate([rng.normal(0, 1, 40), rng.normal(2, 1, 40)]),
        }
    )
    y = pd.Series([0] * 40 + [1] * 40)
    return X, y


@pytest.fixture
def pipeline():
    return Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]
    )


class TestToOptunaDistributions:
    """Tests for scikit-learn grid to Optuna distribution conversion."""

    def test_lists_become_categorical_over_same_values(self):
        distributions = to_optuna_distributions({"clf__C": [0.01, 0.1, 1, 10]})

        assert isinstance(distributions["clf__C"], CategoricalDistribution)
        assert distributions["clf__C"].choices == (0.01, 0.1, 1, 10)

    def test_string_options_are_preserved(self):
        distributions = to_optuna_distributions({"clf__penalty": ["l1", "l2"]})

        assert distributions["clf__penalty"].choices == ("l1", "l2")

    def test_nested_lists_become_hashable_tuples(self):
        """JSON configs give layer sizes as lists; choices must be hashable."""
        distributions = to_optuna_distributions(
            {"clf__hidden_layer_sizes": [[50], [100, 50]]}
        )

        assert distributions["clf__hidden_layer_sizes"].choices == ((50,), (100, 50))

    def test_tuples_are_kept_as_is(self):
        distributions = to_optuna_distributions(
            {"clf__hidden_layer_sizes": [(50,), (100, 50)]}
        )

        assert distributions["clf__hidden_layer_sizes"].choices == ((50,), (100, 50))

    def test_existing_distributions_pass_through(self):
        given = FloatDistribution(0.01, 10.0, log=True)

        distributions = to_optuna_distributions({"clf__C": given})

        assert distributions["clf__C"] is given

    def test_empty_grid_gives_empty_mapping(self):
        assert to_optuna_distributions({}) == {}


class TestOptimizeSearch:
    """Tests for search execution."""

    def test_optuna_search_runs_on_a_sklearn_style_grid(self, binary_data, pipeline):
        """Regression test: grids must be converted, not passed through raw."""
        X, y = binary_data
        optimizer = HyperparameterOptimizer(
            hpo_method="optuna", seed=42, optuna_n_trials=3
        )

        best_model, search = optimizer.run_search(
            pipeline,
            {"clf__C": [0.1, 1.0, 10.0]},
            "Logistic Regression",
            X,
            y,
            StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
            "roc_auc",
        )

        assert best_model is not None
        assert search.best_params_["clf__C"] in (0.1, 1.0, 10.0)

    def test_optuna_falls_back_to_grid_search_without_a_grid(
        self, binary_data, pipeline
    ):
        """An empty grid has nothing to sample, so a single fit is enough."""
        X, y = binary_data
        optimizer = HyperparameterOptimizer(hpo_method="optuna", seed=42)

        best_model, search = optimizer.run_search(
            pipeline,
            {},
            "Logistic Regression",
            X,
            y,
            StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
            "roc_auc",
        )

        assert isinstance(search, GridSearchCV)
        assert best_model is not None

    def test_grid_search_is_used_by_default(self, binary_data, pipeline):
        X, y = binary_data
        optimizer = HyperparameterOptimizer(hpo_method="grid", seed=42)

        _, search = optimizer.run_search(
            pipeline,
            {"clf__C": [0.1, 1.0]},
            "Logistic Regression",
            X,
            y,
            StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
            "roc_auc",
        )

        assert isinstance(search, GridSearchCV)

    def test_calibration_rewrites_parameter_prefixes(self, binary_data):
        """Calibrated pipelines nest the estimator one level deeper."""
        X, y = binary_data
        calibrated_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    CalibratedClassifierCV(LogisticRegression(max_iter=1000), cv=2),
                ),
            ]
        )
        optimizer = HyperparameterOptimizer(
            hpo_method="optuna", calibrate_models=True, seed=42, optuna_n_trials=2
        )

        _, search = optimizer.run_search(
            calibrated_pipeline,
            {"clf__C": [0.1, 1.0]},
            "Logistic Regression",
            X,
            y,
            StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
            "roc_auc",
        )

        assert "clf__estimator__C" in search.best_params_
