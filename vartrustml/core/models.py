"""
Model definitions and evaluation functionality.

:class:`ModelEvaluator` trains and evaluates models with hyperparameter
optimization, optional probability calibration, and SHAP attributions.

See Also
--------
ExperimentConfig : Configuration for model training.
ModelConfig : Hyperparameter search space configuration.
FoldMetrics : Container for per-fold evaluation results.
model_registry : Registry of available model specifications.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from vartrustml.analysis.error_analysis import ErrorAnalyzer, FoldMetrics
from vartrustml.config import ExperimentConfig, ModelConfig
from vartrustml.core.calibration import (
    expected_calibration_error,
    maximum_calibration_error,
)
from vartrustml.core.hpo import HyperparameterOptimizer
from vartrustml.core.metrics import calculate_classification_metrics
from vartrustml.core.missing import IMPUTE_STRATEGIES, make_imputer
from vartrustml.core.model_registry import (
    create_models,
    create_param_grids,
    extract_feature_importances,
)
from vartrustml.utils.reproducibility import set_all_seeds

if TYPE_CHECKING:
    from vartrustml.core.checkpoint_manager import CheckpointManager
    from vartrustml.core.shap_explainer import SHAPExplainer

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Model training and evaluation with hyperparameter optimization.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration controlling training behavior.
    model_config : ModelConfig, optional
        Model-specific configuration for hyperparameter search spaces.
    error_analyzer : ErrorAnalyzer, optional
        Handler for misclassification analysis. Created from config if None.
    shap_explainer : SHAPExplainer, optional
        Helper for SHAP value computation. Created from config if None.
    hpo : HyperparameterOptimizer, optional
        Helper for hyperparameter optimization. Created from config if None.
    checkpoint_mgr : CheckpointManager, optional
        Helper for model checkpoint save/load. Created from config if None.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        model_config: Optional[ModelConfig] = None,
        error_analyzer: Optional[ErrorAnalyzer] = None,
        shap_explainer: Optional[SHAPExplainer] = None,
        hpo: Optional[HyperparameterOptimizer] = None,
        checkpoint_mgr: Optional[CheckpointManager] = None,
    ):
        self.config = config
        self.model_config = model_config or ModelConfig()
        self.error_analyzer = error_analyzer or ErrorAnalyzer(
            config.confidence_thresholds
        )
        if shap_explainer is not None:
            self.shap = shap_explainer
        else:
            from vartrustml.core.shap_explainer import SHAPExplainer

            self.shap = SHAPExplainer(
                output_dir=config.output_dir,
                shap_cache_dir=config.shap_cache_dir,
                shap_cache_enabled=config.shap_cache_enabled,
                seed=config.cv.seed,
            )
        self.hpo = hpo or HyperparameterOptimizer(
            hpo_method=config.hpo_method,
            calibrate_models=config.calibration.calibrate_models,
            n_jobs=config.n_jobs,
            seed=config.cv.seed,
            verbose=config.verbose,
            optuna_n_trials=config.optuna_n_trials,
            optuna_timeout=config.optuna_timeout,
        )
        if checkpoint_mgr is not None:
            self.checkpoint_mgr = checkpoint_mgr
        else:
            from vartrustml.core.checkpoint_manager import CheckpointManager

            self.checkpoint_mgr = CheckpointManager()
        self.models = create_models(
            self.config.models_to_use, self.config.cv.seed, self.model_config
        )
        self.param_grids = create_param_grids(
            self.config.models_to_use, self.model_config
        )

    def resolve_column_roles(
        self, columns: Optional[Iterable[str]] = None
    ) -> Tuple[List[str], List[str]]:
        """Split the input columns into continuous and categorical roles.

        When only ``categorical_cols`` is configured, the continuous columns are
        inferred by exclusion, which is the behaviour the CLI documents. The
        resolved list is written back onto the config so downstream consumers
        and the saved config describe the same split.

        Parameters
        ----------
        columns : iterable of str, optional
            Input column names. Required for the inference by exclusion.

        Returns
        -------
        tuple of (list of str, list of str)
            Continuous columns and categorical columns.
        """
        categorical = list(self.config.categorical_cols or [])
        continuous = list(self.config.continuous_cols or [])

        if not continuous and categorical and columns is not None:
            excluded = set(categorical)
            continuous = [c for c in columns if c not in excluded]
            self.config.continuous_cols = continuous
            logger.info(
                f"Inferred {len(continuous)} continuous columns by exclusion from "
                f"{len(categorical)} categorical columns"
            )

        return continuous, categorical

    def create_pipeline(
        self, model: Any, columns: Optional[Iterable[str]] = None
    ) -> Pipeline:
        """Create sklearn pipeline with preprocessing."""
        continuous_cols, categorical_cols = self.resolve_column_roles(columns)
        preprocessors = []

        if continuous_cols:
            imputer = make_imputer(self.config.nan_strategy)
            if imputer is not None:
                continuous_transform = Pipeline(
                    [("imputer", imputer), ("scaler", StandardScaler())]
                )
            else:
                continuous_transform = StandardScaler()
            scaler = ("scaler", continuous_transform, continuous_cols)
            preprocessors.append(scaler)

        if categorical_cols:
            # Categorical columns are not scaled, but they still need the
            # configured missing-value handling; a most_frequent imputer is the
            # meaningful counterpart of the continuous strategy
            if self.config.nan_strategy in IMPUTE_STRATEGIES:
                categorical_transform: Any = make_imputer("most_frequent")
            else:
                categorical_transform = "passthrough"
            preprocessors.append(
                ("categorical", categorical_transform, categorical_cols)
            )

        if preprocessors:
            preprocessor = ColumnTransformer(
                preprocessors, remainder="passthrough", verbose_feature_names_out=False
            ).set_output(transform="pandas")
        else:
            preprocessor = ColumnTransformer(
                [("passthrough", "passthrough", slice(None))],
                verbose_feature_names_out=False,
            ).set_output(transform="pandas")

        if self.config.calibration.calibrate_models:
            calibration_cv = StratifiedKFold(
                n_splits=self.config.calibration.calibration_cv,
                shuffle=True,
                random_state=self.config.cv.seed,
            )
            calibrated_model = CalibratedClassifierCV(
                estimator=model,
                method=self.config.calibration.calibration_method,
                cv=calibration_cv,
                n_jobs=1,
            )
            return Pipeline([("preprocessor", preprocessor), ("clf", calibrated_model)])
        else:
            return Pipeline([("preprocessor", preprocessor), ("clf", model)])

    def train_single_fold(
        self,
        model_name: str,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        fold_id: int,
        fold_seed: int,
        inner_cv: StratifiedKFold,
        feature_names: Optional[List[str]] = None,
        scoring: str = "roc_auc",
        background_sample_size: int = 100,
        save_checkpoint: bool = True,
        checkpoint_dir: Optional[Path] = None,
    ) -> FoldMetrics:
        """Train model on a single CV fold with hyperparameter optimization.

        Parameters
        ----------
        model_name : str
            Name of the model (e.g., "Random Forest", "XGBoost").
        model : estimator
            Scikit-learn compatible model instance.
        X_train, y_train : DataFrame, Series
            Training data for this fold.
        X_test, y_test : DataFrame, Series
            Test data for this fold.
        fold_id : int
            Identifier for the current fold (0-indexed).
        fold_seed : int
            Random seed for this specific fold.
        inner_cv : StratifiedKFold
            Cross-validation splitter for hyperparameter optimization.
        feature_names : list of str, optional
            Feature column names for error analysis.
        scoring : str, default="roc_auc"
            Scoring metric for hyperparameter optimization.
        background_sample_size : int, default=100
            Number of samples for SHAP background dataset.
        save_checkpoint : bool, default=True
            Whether to save model checkpoint after training.
        checkpoint_dir : Path, optional
            Directory for saving checkpoints.

        Returns
        -------
        FoldMetrics
            Container with all evaluation results for this fold.
        """
        if save_checkpoint and checkpoint_dir is not None:
            result = self._load_checkpoint_fold(
                model_name,
                fold_id,
                checkpoint_dir,
                X_train,
                X_test,
                y_test,
                feature_names,
                background_sample_size,
            )
            if result is not None:
                return result

        logger.info(f"Training {model_name} on fold {fold_id} from scratch")
        set_all_seeds(fold_seed)

        best_model, search = self._run_hpo_search(
            model, model_name, X_train, y_train, inner_cv, scoring
        )

        optimal_threshold, threshold_info = self._optimize_fold_threshold(
            best_model, fold_id, X_train, y_train, inner_cv
        )

        fold_metrics = self._evaluate_fold_predictions(
            best_model,
            model_name,
            fold_id,
            X_train,
            X_test,
            y_test,
            y_prob=None,
            optimal_threshold=optimal_threshold,
            threshold_info=threshold_info,
            feature_names=feature_names,
            background_sample_size=background_sample_size,
            best_params=search.best_params_,
        )

        if save_checkpoint and checkpoint_dir is not None:
            self._save_fold_checkpoint(
                best_model,
                checkpoint_dir,
                fold_id,
                threshold_info,
                optimal_threshold,
                search.best_params_,
            )

        return fold_metrics

    def _load_checkpoint_fold(
        self,
        model_name: str,
        fold_id: int,
        checkpoint_dir: Path,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_names: Optional[List[str]],
        background_sample_size: int,
    ) -> Optional[FoldMetrics]:
        """Load and evaluate a model from a checkpoint file."""
        checkpoint_data = self.checkpoint_mgr.load_fold_checkpoint(
            model_name, fold_id, checkpoint_dir
        )
        if checkpoint_data is None:
            return None

        best_model = checkpoint_data["model"]
        threshold_info = checkpoint_data.get("threshold_info")
        optimal_threshold = checkpoint_data.get("optimal_threshold", 0.5)
        best_params = checkpoint_data.get("best_params", {})

        return self._evaluate_fold_predictions(
            best_model,
            model_name,
            fold_id,
            X_train,
            X_test,
            y_test,
            y_prob=None,
            optimal_threshold=optimal_threshold,
            threshold_info=threshold_info,
            feature_names=feature_names,
            background_sample_size=background_sample_size,
            best_params=best_params,
        )

    def _run_hpo_search(
        self,
        model: Any,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        inner_cv: StratifiedKFold,
        scoring: str,
    ) -> Tuple[Any, Any]:
        """Run hyperparameter optimization and return (best_model, search)."""
        pipeline = self.create_pipeline(model, getattr(X_train, "columns", None))
        param_grid = self.param_grids.get(model_name, {})
        return self.hpo.run_search(
            pipeline, param_grid, model_name, X_train, y_train, inner_cv, scoring
        )

    def _optimize_fold_threshold(
        self,
        best_model: Any,
        fold_id: int,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        inner_cv: StratifiedKFold,
    ) -> Tuple[float, Optional[Dict[str, Any]]]:
        """Optimize the classification threshold using inner CV on training data."""
        if not self.config.threshold.optimize_threshold:
            return 0.5, None

        from vartrustml.core.threshold_helper import (
            optimize_threshold_from_cv,
            threshold_result_to_info_dict,
        )

        logger.info(f"Fold {fold_id}: Optimizing threshold on training data...")
        result = optimize_threshold_from_cv(
            best_model,
            X_train,
            y_train,
            inner_cv,
            method=self.config.threshold.threshold_method,
            auto_threshold_n_samples=self.config.threshold_auto_n_samples,
        )
        if result is None:
            return 0.5, None

        threshold_info = threshold_result_to_info_dict(result)
        logger.info(
            f"Fold {fold_id}: Optimal threshold = {result.optimal_threshold:.4f} "
            f"(J = {result.youden_j:.4f}, method = {result.method_used.value})"
        )
        return float(result.optimal_threshold), threshold_info

    def _evaluate_fold_predictions(
        self,
        best_model: Any,
        model_name: str,
        fold_id: int,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_prob: Optional[np.ndarray],
        optimal_threshold: float,
        threshold_info: Optional[Dict[str, Any]],
        feature_names: Optional[List[str]],
        background_sample_size: int,
        best_params: Dict[str, Any],
    ) -> FoldMetrics:
        """Evaluate model predictions on a test fold and build FoldMetrics."""
        if y_prob is None:
            y_prob = best_model.predict_proba(X_test)

        y_pred = (y_prob[:, 1] >= optimal_threshold).astype(int)
        metrics = self._calculate_metrics(y_test, y_pred, y_prob)
        conf_matrix = confusion_matrix(y_test, y_pred, normalize="true")

        metrics_at_default = None
        if threshold_info and optimal_threshold != 0.5:
            y_pred_default = (y_prob[:, 1] >= 0.5).astype(int)
            metrics_at_default = self._calculate_metrics(y_test, y_pred_default, y_prob)

        if self.error_analyzer.confidence_thresholds:
            misclassified_df, error_analysis = (
                self.error_analyzer.analyze_misclassifications(
                    y_test.values,
                    y_pred,
                    y_prob,
                    X_test,
                    feature_names,
                    operating_threshold=optimal_threshold,
                )
            )
        else:
            misclassified_df = pd.DataFrame()
            error_analysis = {}

        feature_importances = self._extract_feature_importances(best_model, model_name)

        # Importances, SHAP values and the transformed matrix all live in the
        # preprocessor's output order, which the ColumnTransformer reorders
        # (continuous columns first, passthrough remainder after). Record that
        # order so consumers never label them with the input column list.
        transformed_feature_names = self._transformed_feature_names(best_model)

        X_test_transformed = None
        shap_values = None
        try:
            preprocessor = best_model.named_steps["preprocessor"]
            X_test_transformed = preprocessor.transform(X_test)
            if hasattr(X_test_transformed, "values"):
                X_test_transformed = X_test_transformed.values

            shap_values = self._calculate_shap_values(
                best_model,
                model_name,
                X_train,
                X_test,
                fold_id,
                background_sample_size,
            )
        except Exception as e:
            logger.debug(
                f"SHAP preprocessing/calculation failed for "
                f"{model_name} fold {fold_id}: {e}"
            )

        return FoldMetrics(
            fold_id=fold_id,
            metrics=metrics,
            confusion_matrix=conf_matrix,
            misclassified_samples=misclassified_df,
            error_analysis=error_analysis,
            shap_values=shap_values,
            feature_importances=feature_importances,
            transformed_feature_names=transformed_feature_names,
            best_params=best_params,
            X_test_transformed=X_test_transformed,
            y_true_oof=y_test.values,
            y_prob_oof=y_prob[:, 1],
            sample_indices=X_test.index.values,
            fold_optimal_threshold=(
                threshold_info["optimal_threshold"] if threshold_info else None
            ),
            fold_youden_j=(threshold_info["youden_j"] if threshold_info else None),
            fold_sensitivity_at_threshold=(
                threshold_info["sensitivity"] if threshold_info else None
            ),
            fold_specificity_at_threshold=(
                threshold_info["specificity"] if threshold_info else None
            ),
            metrics_at_default_threshold=metrics_at_default,
        )

    def _save_fold_checkpoint(
        self,
        best_model: Any,
        checkpoint_dir: Path,
        fold_id: int,
        threshold_info: Optional[Dict[str, Any]],
        optimal_threshold: float,
        best_params: Dict[str, Any],
    ) -> None:
        """Save model checkpoint to disk."""
        self.checkpoint_mgr.save_fold_checkpoint(
            best_model,
            checkpoint_dir,
            fold_id,
            threshold_info,
            optimal_threshold,
            best_params,
        )

    def _calculate_metrics(
        self, y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray
    ) -> Dict[str, float]:
        """Calculate the full set of classification metrics."""
        return calculate_classification_metrics(
            y_true,
            y_pred,
            y_prob=y_prob,
            calibration_error_fn={
                "ECE": expected_calibration_error,
                "MCE": maximum_calibration_error,
            },
        )

    @staticmethod
    def _transformed_feature_names(model: Any) -> Optional[List[str]]:
        """Column names produced by the fitted preprocessor, in output order."""
        try:
            preprocessor = model.named_steps["preprocessor"]
            return [str(name) for name in preprocessor.get_feature_names_out()]
        except Exception as e:
            logger.warning(f"Could not resolve transformed feature names: {e}")
            return None

    def _extract_feature_importances(
        self, model: Any, model_name: str
    ) -> Optional[np.ndarray]:
        """Extract feature importances from fitted model via model registry."""
        try:
            clf = model.named_steps["clf"]

            if hasattr(clf, "calibrated_classifiers_"):
                base_clf = clf.calibrated_classifiers_[0].estimator
            else:
                base_clf = clf

            return extract_feature_importances(model_name, base_clf)
        except Exception as e:
            logger.warning(
                f"Could not extract feature importances for {model_name}: {e}"
            )
            logger.debug("Feature importance extraction traceback:", exc_info=True)
            return None

    def clear_shap_cache(self) -> int:
        """Clear all cached SHAP values."""
        return self.shap.clear_shap_cache()

    def _calculate_shap_values(
        self,
        model: Any,
        model_name: str,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        fold_id: int = 0,
        background_sample_size: int = 100,
    ) -> Optional[np.ndarray]:
        """Calculate SHAP values for interpretability."""
        return self.shap.compute_shap_values(
            model, model_name, X_train, X_test, fold_id, background_sample_size
        )

    def evaluate_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        scoring: str = "roc_auc",
        n_splits: int = 5,
        save_checkpoint: bool = False,
        checkpoint_path: Optional[Path] = None,
        fold_idx: int = 0,
    ) -> Tuple[Dict[str, float], np.ndarray, Optional[float]]:
        """Evaluate model for cross-dataset generalization testing.

        Trains on one dataset and evaluates on another (no CV on test set).

        Parameters
        ----------
        model_name : str
            Name of the model to evaluate.
        X_train, y_train : DataFrame, Series
            Training data (from source dataset).
        X_test, y_test : DataFrame, Series
            Test data (from target dataset).
        scoring : str, default="roc_auc"
            Scoring metric for hyperparameter optimization.
        n_splits : int, default=5
            Number of CV splits for hyperparameter tuning on training data.
        save_checkpoint : bool, default=False
            Whether to save trained model checkpoint.
        checkpoint_path : Path, optional
            Path for saving/loading checkpoint.
        fold_idx : int, default=0
            Fold index used.

        Returns
        -------
        metrics : dict of {str: float}
        y_prob : numpy.ndarray of shape (n_samples, 2)
        optimal_threshold : float or None
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        if save_checkpoint and checkpoint_path is not None and checkpoint_path.exists():
            import joblib

            logger.info(f"Loading existing checkpoint: {checkpoint_path}")
            checkpoint_data = joblib.load(checkpoint_path)

            best_model = checkpoint_data["model"]
            optimal_threshold = checkpoint_data.get("optimal_threshold", 0.5)

            y_prob = best_model.predict_proba(X_test)
            y_pred = (y_prob[:, 1] >= optimal_threshold).astype(int)
            metrics = self._calculate_metrics(y_test, y_pred, y_prob)

            return (
                metrics,
                y_prob,
                optimal_threshold if optimal_threshold != 0.5 else None,
            )

        logger.info(f"Training {model_name} from scratch (fold {fold_idx})")
        set_all_seeds(self.config.cv.seed + fold_idx)

        model = self.models[model_name]
        pipeline = self.create_pipeline(model, getattr(X_train, "columns", None))
        param_grid = self.param_grids.get(model_name, {})

        cv = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.config.cv.seed
        )

        best_model, search = self.hpo.run_search(
            pipeline, param_grid, model_name, X_train, y_train, cv, scoring
        )

        optimal_threshold: Optional[float] = None
        threshold_info = None

        if self.config.threshold.optimize_threshold:
            from vartrustml.core.threshold_helper import (
                optimize_threshold_from_cv,
                threshold_result_to_info_dict,
            )

            logger.debug(f"Optimizing threshold for {model_name} on training data...")
            result = optimize_threshold_from_cv(
                best_model,
                X_train,
                y_train,
                cv,
                method=self.config.threshold.threshold_method,
                auto_threshold_n_samples=self.config.threshold_auto_n_samples,
            )
            if result is not None:
                optimal_threshold = float(result.optimal_threshold)
                threshold_info = threshold_result_to_info_dict(result)
                logger.debug(
                    f"{model_name}: Optimal threshold = {optimal_threshold:.4f} "
                    f"(J = {result.youden_j:.4f}, "
                    f"method = {result.method_used.value})"
                )
            else:
                optimal_threshold = 0.5

        if save_checkpoint and checkpoint_path is not None:
            import joblib

            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_data = {
                "model": best_model,
                "threshold_info": threshold_info,
                "optimal_threshold": (
                    optimal_threshold if optimal_threshold is not None else 0.5
                ),
                "best_params": search.best_params_,
            }
            joblib.dump(checkpoint_data, checkpoint_path, compress=9)
            logger.info(f"Checkpoint saved: {checkpoint_path}")

        y_prob = best_model.predict_proba(X_test)

        threshold_to_use = optimal_threshold if optimal_threshold is not None else 0.5
        y_pred = (y_prob[:, 1] >= threshold_to_use).astype(int)

        metrics = self._calculate_metrics(y_test, y_pred, y_prob)

        return metrics, y_prob, optimal_threshold
