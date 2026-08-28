"""Hyperparameter optimization strategies."""

import logging
import warnings
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from optuna.distributions import BaseDistribution, CategoricalDistribution
from optuna_integration.sklearn import OptunaSearchCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def to_optuna_distributions(
    param_grid: Dict[str, Any],
) -> Dict[str, BaseDistribution]:
    """Convert a scikit-learn parameter grid into Optuna distributions.

    Model specifications declare search spaces as lists of candidate values, the
    format ``GridSearchCV`` expects. ``OptunaSearchCV`` instead requires Optuna
    distribution objects, so each list is wrapped in a
    :class:`~optuna.distributions.CategoricalDistribution` over exactly those
    values. The searched space is therefore identical to the declared grid, with
    Optuna's sampler choosing which combinations to evaluate.

    Parameters
    ----------
    param_grid : dict
        Mapping of parameter name to a list of candidate values. Values that are
        already Optuna distributions are passed through unchanged.

    Returns
    -------
    dict
        Mapping of parameter name to Optuna distribution.
    """
    distributions: Dict[str, BaseDistribution] = {}
    for name, values in param_grid.items():
        if isinstance(values, BaseDistribution):
            distributions[name] = values
            continue
        # Categorical choices must be hashable: lists from JSON configs (for
        # example MLP layer sizes) become tuples, which estimators accept.
        choices = tuple(tuple(v) if isinstance(v, list) else v for v in values)
        distributions[name] = CategoricalDistribution(choices)
    return distributions


class HyperparameterOptimizer:
    """Handles hyperparameter grid initialization and HPO search execution.

    Parameters
    ----------
    hpo_method : str
        HPO strategy: "grid" or "optuna".
    calibrate_models : bool
        Whether models use calibration (affects param grid key prefixes).
    n_jobs : int
        Number of parallel jobs for search.
    seed : int
        Random seed for Optuna.
    verbose : int
        Verbosity level.
    optuna_n_trials : int
        Number of Optuna trials.
    optuna_timeout : int or None
        Optuna timeout in seconds.
    """

    def __init__(
        self,
        hpo_method: str = "grid",
        calibrate_models: bool = False,
        n_jobs: int = 1,
        seed: int = 42,
        verbose: int = 0,
        optuna_n_trials: int = 50,
        optuna_timeout: Optional[int] = None,
    ):
        self.hpo_method = hpo_method
        self.calibrate_models = calibrate_models
        self.n_jobs = n_jobs
        self.seed = seed
        self.verbose = verbose
        self.optuna_n_trials = optuna_n_trials
        self.optuna_timeout = optuna_timeout

    def run_search(
        self,
        pipeline: Pipeline,
        param_grid: Dict[str, Any],
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        inner_cv: StratifiedKFold,
        scoring: str,
    ) -> Tuple[Any, Any]:
        """Run hyperparameter optimization and return (best_model, search).

        Parameters
        ----------
        pipeline : Pipeline
            Sklearn pipeline with preprocessor and classifier.
        param_grid : dict
            Hyperparameter search space.
        model_name : str
            Name of the model (for logging).
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training labels.
        inner_cv : StratifiedKFold
            Cross-validation splitter.
        scoring : str
            Scoring metric.

        Returns
        -------
        Tuple[Any, Any]
            (best_estimator, search_object)
        """
        # Adjust parameter names if using calibration
        if self.calibrate_models and param_grid:
            calibrated_param_grid = {}
            for key, value in param_grid.items():
                new_key = key.replace("clf__", "clf__estimator__")
                calibrated_param_grid[new_key] = value
            param_grid = calibrated_param_grid

        if self.hpo_method == "optuna" and not param_grid:
            logger.warning(
                f"{model_name} declares no hyperparameters to tune; "
                "falling back to a single fit."
            )

        if self.hpo_method == "optuna" and param_grid:
            search = OptunaSearchCV(
                estimator=pipeline,
                param_distributions=to_optuna_distributions(param_grid),
                cv=inner_cv,
                n_trials=self.optuna_n_trials,
                timeout=self.optuna_timeout,
                n_jobs=self.n_jobs,
                scoring=scoring,
                random_state=self.seed,
                verbose=1 if self.verbose >= 1 else 0,
            )
            with warnings.catch_warnings():
                # OptunaSearchCV imports the deprecated optuna.terminator module
                # internally and the notice is re-emitted on every trial.
                warnings.filterwarnings(
                    "ignore",
                    message=r".*optuna\.terminator.*",
                    category=FutureWarning,
                )
                search.fit(X_train, y_train)
            return search.best_estimator_, search

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            cv=inner_cv,
            n_jobs=self.n_jobs,
            verbose=0,
        )
        search.fit(X_train, y_train)
        return search.best_estimator_, search
