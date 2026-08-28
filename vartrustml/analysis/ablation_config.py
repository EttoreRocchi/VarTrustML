"""
Config-driven ablation study framework.

Provides ConfigAblationAnalyzer which extends AblationAnalyzer with methods
that create fresh model instances from configuration, supporting the CLI
ablation workflow.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from vartrustml.analysis.ablation import AblationAnalyzer, AblationStudyResult

logger = logging.getLogger(__name__)

# Supported model names for ablation_from_config
SUPPORTED_MODELS = [
    "XGBoost",
    "CatBoost",
    "Random Forest",
    "Logistic Regression",
    "MLP",
    "KNN",
]


class ConfigAblationAnalyzer(AblationAnalyzer):
    """Ablation analyzer with config-driven fresh model training.

    Extends :class:`AblationAnalyzer` with methods that create fresh model
    instances from configuration, supporting the CLI ablation workflow.

    Parameters
    ----------
    n_splits : int, default=3
        Number of cross-validation splits.
    seed : int, default=42
        Random seed for reproducibility.
    alpha : float, default=0.05
        Significance level for statistical tests.
    n_jobs : int, default=-1
        Number of parallel jobs for cross-validation.
    """

    def __init__(
        self,
        n_splits: int = 3,
        seed: int = 42,
        alpha: float = 0.05,
        n_jobs: int = -1,
    ):
        """Initialize the config-driven ablation analyzer."""
        super().__init__(n_splits=n_splits, seed=seed, alpha=alpha, n_jobs=n_jobs)

    def _create_model_instance(
        self,
        model_name: str,
        best_params: Optional[Dict[str, Any]] = None,
    ) -> "BaseEstimator":
        """Create a fresh model instance by name.

        Parameters
        ----------
        model_name : str
            Name of the model (e.g., 'XGBoost', 'CatBoost', 'Random Forest').
        best_params : dict, optional
            Hyperparameters to set on the model. These should be the raw
            model parameters (without 'clf__' prefix).

        Returns
        -------
        BaseEstimator
            Fresh, unfitted model instance.

        Raises
        ------
        ValueError
            If model_name is not supported.
        """
        # Import here to avoid circular imports
        from catboost import CatBoostClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.neural_network import MLPClassifier
        from xgboost import XGBClassifier

        model_map = {
            "XGBoost": lambda: XGBClassifier(
                eval_metric="logloss",
                random_state=self.seed,
                device="cpu",
                enable_categorical=False,
            ),
            "CatBoost": lambda: CatBoostClassifier(
                auto_class_weights="Balanced",
                verbose=0,
                allow_writing_files=False,
                random_state=self.seed,
            ),
            "Random Forest": lambda: RandomForestClassifier(
                class_weight="balanced",
                random_state=self.seed,
            ),
            "Logistic Regression": lambda: LogisticRegression(
                class_weight="balanced",
                solver="liblinear",
                max_iter=1000,
                random_state=self.seed,
            ),
            "MLP": lambda: MLPClassifier(
                max_iter=500,
                random_state=self.seed,
                early_stopping=True,
            ),
            "KNN": lambda: KNeighborsClassifier(metric="euclidean"),
        }

        if model_name not in model_map:
            raise ValueError(
                f"Unsupported model: {model_name}. "
                f"Supported models: {list(model_map.keys())}"
            )

        model = model_map[model_name]()

        # Apply best_params if provided
        if best_params:
            # Strip 'clf__' or 'clf__estimator__' prefix if present
            clean_params = {}
            for key, value in best_params.items():
                clean_key = key.replace("clf__estimator__", "").replace("clf__", "")
                clean_params[clean_key] = value

            # Only set params that the model accepts
            valid_params = model.get_params()
            params_to_set = {k: v for k, v in clean_params.items() if k in valid_params}
            if params_to_set:
                model.set_params(**params_to_set)

        return model

    def _create_fresh_pipeline(
        self,
        model_name: str,
        X: pd.DataFrame,
        continuous_cols: Optional[List[str]] = None,
        best_params: Optional[Dict[str, Any]] = None,
        calibrate: bool = False,
        calibration_method: str = "isotonic",
        calibration_cv: int = 3,
    ) -> Pipeline:
        """Create a fresh pipeline for the given feature set.

        This method creates a new pipeline with a ColumnTransformer that only
        references columns present in X, avoiding the issue with pre-trained
        pipelines that have fixed column names.

        Parameters
        ----------
        model_name : str
            Name of the model to use.
        X : pd.DataFrame
            Feature matrix (used to determine available columns).
        continuous_cols : list of str, optional
            Columns to apply StandardScaler to. Only columns present in X
            will be used.
        best_params : dict, optional
            Hyperparameters to set on the model.
        calibrate : bool
            Whether to wrap the model in CalibratedClassifierCV.
        calibration_method : str
            Calibration method ('isotonic' or 'sigmoid').
        calibration_cv : int
            Number of CV folds for calibration.

        Returns
        -------
        Pipeline
            Fresh sklearn Pipeline ready for fitting.
        """
        # Create model instance
        model = self._create_model_instance(model_name, best_params)

        # Determine which continuous columns are actually in X
        if continuous_cols:
            available_continuous = [c for c in continuous_cols if c in X.columns]
        else:
            available_continuous = []

        # Build preprocessor
        if available_continuous:
            preprocessor = ColumnTransformer(
                [("scaler", StandardScaler(), available_continuous)],
                remainder="passthrough",
                verbose_feature_names_out=False,
            ).set_output(transform="pandas")
        else:
            # Passthrough all columns
            preprocessor = ColumnTransformer(
                [("passthrough", "passthrough", slice(None))],
                verbose_feature_names_out=False,
            ).set_output(transform="pandas")

        # Wrap in calibration if requested
        if calibrate:
            calibration_cv_split = StratifiedKFold(
                n_splits=calibration_cv,
                shuffle=True,
                random_state=self.seed,
            )
            calibrated_model = CalibratedClassifierCV(
                estimator=model,
                method=calibration_method,
                cv=calibration_cv_split,
                n_jobs=1,
            )
            return Pipeline([("preprocessor", preprocessor), ("clf", calibrated_model)])
        else:
            return Pipeline([("preprocessor", preprocessor), ("clf", model)])

    def _compute_cv_scores_fresh(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        model_name: str,
        metric_func: Callable,
        continuous_cols: Optional[List[str]] = None,
        best_params: Optional[Dict[str, Any]] = None,
        calibrate: bool = False,
        calibration_method: str = "isotonic",
        calibration_cv: int = 3,
        optimize_threshold: bool = False,
        threshold_method: str = "auto",
    ) -> Tuple[np.ndarray, Optional[float]]:
        """Compute CV scores by creating fresh pipelines for each fold.

        Unlike _compute_cv_scores which clones an existing model, this method
        creates a fresh pipeline for each fold with the correct column
        configuration for the given X.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : ndarray
            Target labels.
        model_name : str
            Name of the model to use.
        metric_func : callable
            Scoring function (y_true, y_pred) -> float.
        continuous_cols : list of str, optional
            Columns to scale.
        best_params : dict, optional
            Hyperparameters for the model.
        calibrate : bool
            Whether to apply calibration.
        calibration_method : str
            Calibration method.
        calibration_cv : int
            Calibration CV folds.
        optimize_threshold : bool
            Whether to optimize classification threshold.
        threshold_method : str
            Threshold method ('oof', 'cv', 'auto').

        Returns
        -------
        scores : ndarray
            Array of per-fold scores.
        optimal_threshold : float or None
            Optimal threshold if optimize_threshold=True, else None.
        """
        scores = []
        thresholds = []
        oof_probs = np.zeros(len(y))
        oof_true = np.zeros(len(y))

        for fold_idx, (train_idx, test_idx) in enumerate(self._cv.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Create fresh pipeline for this fold
            pipeline = self._create_fresh_pipeline(
                model_name=model_name,
                X=X_train,
                continuous_cols=continuous_cols,
                best_params=best_params,
                calibrate=calibrate,
                calibration_method=calibration_method,
                calibration_cv=calibration_cv,
            )

            # Fit and predict
            pipeline.fit(X_train, y_train)

            if optimize_threshold and hasattr(pipeline, "predict_proba"):
                # Collect OOF probabilities for threshold computation
                y_prob = pipeline.predict_proba(X_test)[:, 1]
                oof_probs[test_idx] = y_prob
                oof_true[test_idx] = y_test

                # Per-fold threshold (used only for the returned optimal_threshold)
                from sklearn.metrics import roc_curve

                fpr, tpr, thresh = roc_curve(y_test, y_prob)
                youden_j = tpr - fpr
                best_idx = np.argmax(youden_j)
                fold_threshold = thresh[best_idx]
                thresholds.append(fold_threshold)

            # Always score with default threshold to avoid data leakage
            y_pred = pipeline.predict(X_test)

            score = metric_func(y_test, y_pred)
            scores.append(score)

        # Compute optimal threshold
        optimal_threshold = None
        if optimize_threshold and thresholds:
            if threshold_method == "cv" or (
                threshold_method == "auto" and len(y) >= 1000
            ):
                # Average of per-fold thresholds
                optimal_threshold = float(np.mean(thresholds))
            else:
                # OOF method: find threshold on pooled predictions
                from sklearn.metrics import roc_curve

                fpr, tpr, thresh = roc_curve(oof_true, oof_probs)
                youden_j = tpr - fpr
                best_idx = np.argmax(youden_j)
                optimal_threshold = float(thresh[best_idx])

        return np.array(scores), optimal_threshold

    def _run_ablation_study(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        model_name: str,
        items: List[Tuple[str, List[str]]],
        metric_func: Callable,
        metric_name: str,
        study_type: str,
        continuous_cols: Optional[List[str]] = None,
        best_params: Optional[Dict[str, Any]] = None,
        calibrate: bool = False,
        calibration_method: str = "isotonic",
        calibration_cv: int = 3,
        optimize_threshold: bool = False,
        threshold_method: str = "auto",
    ) -> AblationStudyResult:
        """Shared implementation for feature and group ablation with fresh models.

        Contains the common logic for ``ablation_from_config`` and
        ``group_ablation_from_config``: baseline computation, per-item loop
        (drop columns, adjust continuous_cols, score), Holm correction, and
        result construction.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with named columns.
        y : ndarray
            Target labels.
        model_name : str
            Name of the model to use.
        items : list of (str, list of str)
            Each tuple is ``(name, columns_to_drop)`` where *name* becomes
            the ``ablation_name`` in the result and *columns_to_drop* are
            the columns removed from *X* for that ablation.
        metric_func : callable
            Scoring function ``(y_true, y_pred) -> float``.
        metric_name : str
            Name of the metric for reporting.
        study_type : str
            Study type label for the returned ``AblationStudyResult``
            (e.g., ``"leave_one_out"`` or ``"group"``).
        continuous_cols : list of str, optional
            Columns to apply StandardScaler to.
        best_params : dict, optional
            Hyperparameters for the model.
        calibrate : bool, default=False
            Whether to apply probability calibration.
        calibration_method : str, default="isotonic"
            Calibration method (``'isotonic'`` or ``'sigmoid'``).
        calibration_cv : int, default=3
            Number of CV folds for calibration.
        optimize_threshold : bool, default=False
            Whether to optimize classification threshold.
        threshold_method : str, default="auto"
            Threshold method (``'oof'``, ``'cv'``, or ``'auto'``).

        Returns
        -------
        AblationStudyResult
            Complete ablation study results.
        """
        logger.info("Computing baseline scores with all features")
        baseline_scores, baseline_threshold = self._compute_cv_scores_fresh(
            X=X,
            y=y,
            model_name=model_name,
            metric_func=metric_func,
            continuous_cols=continuous_cols,
            best_params=best_params,
            calibrate=calibrate,
            calibration_method=calibration_method,
            calibration_cv=calibration_cv,
            optimize_threshold=optimize_threshold,
            threshold_method=threshold_method,
        )

        if baseline_threshold is not None:
            logger.info(f"Baseline optimal threshold: {baseline_threshold:.4f}")

        results = []
        for i, (name, cols_to_drop) in enumerate(items):
            logger.debug(
                f"Ablating item {i + 1}/{len(items)}: {name} "
                f"(dropping {len(cols_to_drop)} column(s))"
            )

            # Create ablated dataset
            X_ablated = X.drop(columns=cols_to_drop)

            # Adjust continuous_cols for ablated data
            ablated_continuous = None
            if continuous_cols:
                drop_set = set(cols_to_drop)
                ablated_continuous = [c for c in continuous_cols if c not in drop_set]

            ablated_scores, _ = self._compute_cv_scores_fresh(
                X=X_ablated,
                y=y,
                model_name=model_name,
                metric_func=metric_func,
                continuous_cols=ablated_continuous,
                best_params=best_params,
                calibrate=calibrate,
                calibration_method=calibration_method,
                calibration_cv=calibration_cv,
                optimize_threshold=optimize_threshold,
                threshold_method=threshold_method,
            )

            result = self._compute_ablation_result(
                ablation_name=name,
                baseline_scores=baseline_scores,
                ablated_scores=ablated_scores,
                metric_name=metric_name,
            )
            results.append(result)

        logger.info(
            f"Ablation study ({study_type}) complete. {len(results)} item(s) analyzed."
        )

        # Apply Holm-Bonferroni correction for multiple comparisons
        self._apply_holm_correction(results)

        return AblationStudyResult(
            study_type=study_type,
            results=results,
            baseline_score=float(np.mean(baseline_scores)),
            metric_name=metric_name,
            n_splits=self.n_splits,
            seed=self.seed,
        )

    def ablation_from_config(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        model_name: str,
        metric_func: Callable,
        metric_name: str = "metric",
        features_to_ablate: Optional[List[str]] = None,
        continuous_cols: Optional[List[str]] = None,
        best_params: Optional[Dict[str, Any]] = None,
        calibrate: bool = False,
        calibration_method: str = "isotonic",
        calibration_cv: int = 3,
        optimize_threshold: bool = False,
        threshold_method: str = "auto",
    ) -> AblationStudyResult:
        """Perform feature ablation by training fresh models from configuration.

        This method creates fresh pipelines for the baseline and each ablation,
        avoiding issues with pre-trained models that have fixed column
        transformers. Recommended for CLI use and when working with models
        that include preprocessing steps.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with named columns.
        y : ndarray
            Target labels.
        model_name : str
            Name of the model to use (e.g., 'XGBoost', 'CatBoost').
        metric_func : callable
            Scoring function (y_true, y_pred) -> float.
        metric_name : str, default="metric"
            Name of the metric for reporting.
        features_to_ablate : list of str, optional
            Specific features to ablate. If None, ablates all features.
        continuous_cols : list of str, optional
            Columns to apply StandardScaler to.
        best_params : dict, optional
            Hyperparameters for the model (from previous training).
        calibrate : bool, default=False
            Whether to apply probability calibration.
        calibration_method : str, default="isotonic"
            Calibration method ('isotonic' or 'sigmoid').
        calibration_cv : int, default=3
            Number of CV folds for calibration.
        optimize_threshold : bool, default=False
            Whether to optimize classification threshold.
        threshold_method : str, default="auto"
            Threshold method ('oof', 'cv', or 'auto').

        Returns
        -------
        AblationStudyResult
            Complete ablation study results.

        Examples
        --------
        >>> analyzer = ConfigAblationAnalyzer(n_splits=3, seed=42)
        >>> results = analyzer.ablation_from_config(
        ...     X, y, model_name='XGBoost',
        ...     metric_func=balanced_accuracy_score,
        ...     continuous_cols=['feature1', 'feature2'],
        ...     calibrate=True,
        ... )

        Notes
        -----
        Unlike `feature_ablation()` which requires a pre-built model object,
        this method creates fresh models for each ablation. This is necessary
        when the model is a Pipeline with a ColumnTransformer that stores
        column names, as cloning such a model would preserve the original
        column configuration.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame for ablation_from_config")

        if model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model: {model_name}. Supported: {SUPPORTED_MODELS}"
            )

        features = features_to_ablate or list(X.columns)
        logger.info(
            f"Starting ablation_from_config with {len(features)} features "
            f"using {model_name}"
        )

        items = [(feat, [feat]) for feat in features]

        return self._run_ablation_study(
            X=X,
            y=y,
            model_name=model_name,
            items=items,
            metric_func=metric_func,
            metric_name=metric_name,
            study_type="feature",
            continuous_cols=continuous_cols,
            best_params=best_params,
            calibrate=calibrate,
            calibration_method=calibration_method,
            calibration_cv=calibration_cv,
            optimize_threshold=optimize_threshold,
            threshold_method=threshold_method,
        )

    def group_ablation_from_config(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        model_name: str,
        feature_groups: Dict[str, List[str]],
        metric_func: Callable,
        metric_name: str = "metric",
        continuous_cols: Optional[List[str]] = None,
        best_params: Optional[Dict[str, Any]] = None,
        calibrate: bool = False,
        calibration_method: str = "isotonic",
        calibration_cv: int = 3,
        optimize_threshold: bool = False,
        threshold_method: str = "auto",
    ) -> AblationStudyResult:
        """Perform feature group ablation by training fresh models.

        Similar to `ablation_from_config()` but ablates entire feature groups
        at once rather than individual features.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with named columns.
        y : ndarray
            Target labels.
        model_name : str
            Name of the model to use.
        feature_groups : dict
            Dictionary mapping group names to lists of feature names.
        metric_func : callable
            Scoring function (y_true, y_pred) -> float.
        metric_name : str, default="metric"
            Name of the metric for reporting.
        continuous_cols : list of str, optional
            Columns to apply StandardScaler to.
        best_params : dict, optional
            Hyperparameters for the model.
        calibrate : bool, default=False
            Whether to apply probability calibration.
        calibration_method : str, default="isotonic"
            Calibration method.
        calibration_cv : int, default=3
            Calibration CV folds.
        optimize_threshold : bool, default=False
            Whether to optimize classification threshold.
        threshold_method : str, default="auto"
            Threshold method.

        Returns
        -------
        AblationStudyResult
            Complete ablation study results.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "X must be a pandas DataFrame for group_ablation_from_config"
            )

        logger.info(
            f"Starting group_ablation_from_config with {len(feature_groups)} groups "
            f"using {model_name}"
        )

        # Build items list, validating features and filtering missing ones
        items: List[Tuple[str, List[str]]] = []
        for group_name, features in feature_groups.items():
            missing = [f for f in features if f not in X.columns]
            if missing:
                logger.warning(
                    f"Group '{group_name}' contains missing features: {missing}"
                )
                features = [f for f in features if f in X.columns]
                if not features:
                    continue
            items.append((group_name, features))

        return self._run_ablation_study(
            X=X,
            y=y,
            model_name=model_name,
            items=items,
            metric_func=metric_func,
            metric_name=metric_name,
            study_type="feature_group",
            continuous_cols=continuous_cols,
            best_params=best_params,
            calibrate=calibrate,
            calibration_method=calibration_method,
            calibration_cv=calibration_cv,
            optimize_threshold=optimize_threshold,
            threshold_method=threshold_method,
        )

    def pipeline_step_ablation(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        model_name: str,
        metric_func: Callable,
        metric_name: str = "metric",
        continuous_cols: Optional[List[str]] = None,
        best_params: Optional[Dict[str, Any]] = None,
        ablate_calibration: bool = True,
        ablate_threshold: bool = True,
        ablate_scaling: bool = True,
    ) -> AblationStudyResult:
        """Ablate pipeline steps to measure their contribution.

        This method compares the full pipeline (with all components) against
        versions with specific components removed, helping understand the
        contribution of calibration, threshold optimization, and scaling.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : ndarray
            Target labels.
        model_name : str
            Name of the model to use.
        metric_func : callable
            Scoring function (y_true, y_pred) -> float.
        metric_name : str, default="metric"
            Name of the metric for reporting.
        continuous_cols : list of str, optional
            Columns to scale (required if ablate_scaling=True).
        best_params : dict, optional
            Hyperparameters for the model.
        ablate_calibration : bool, default=True
            Whether to test removing calibration.
        ablate_threshold : bool, default=True
            Whether to test removing threshold optimization.
        ablate_scaling : bool, default=True
            Whether to test removing feature scaling.

        Returns
        -------
        AblationStudyResult
            Complete ablation study results showing the impact of each
            pipeline step.

        Examples
        --------
        >>> results = analyzer.pipeline_step_ablation(
        ...     X, y, model_name='XGBoost',
        ...     metric_func=balanced_accuracy_score,
        ...     continuous_cols=['feature1', 'feature2'],
        ... )
        >>> for r in results.results:
        ...     print(f"{r.ablation_name}: delta={r.delta:.4f}")

        Notes
        -----
        The baseline is the full pipeline with all enabled components
        (scaling, calibration, threshold optimization). Each ablation
        removes one component to measure its contribution.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame for pipeline_step_ablation")

        logger.info(f"Starting pipeline_step_ablation using {model_name}")

        # Baseline: full pipeline with all components
        logger.info("Computing baseline (full pipeline)")
        baseline_scores, _ = self._compute_cv_scores_fresh(
            X=X,
            y=y,
            model_name=model_name,
            metric_func=metric_func,
            continuous_cols=continuous_cols,
            best_params=best_params,
            calibrate=True,
            calibration_method="isotonic",
            calibration_cv=3,
            optimize_threshold=True,
            threshold_method="auto",
        )

        results = []

        # Ablate calibration
        if ablate_calibration:
            logger.debug("Testing: no calibration")
            scores, _ = self._compute_cv_scores_fresh(
                X=X,
                y=y,
                model_name=model_name,
                metric_func=metric_func,
                continuous_cols=continuous_cols,
                best_params=best_params,
                calibrate=False,  # No calibration
                optimize_threshold=True,
                threshold_method="auto",
            )
            result = self._compute_ablation_result(
                ablation_name="no_calibration",
                baseline_scores=baseline_scores,
                ablated_scores=scores,
                metric_name=metric_name,
            )
            results.append(result)

        # Ablate threshold optimization
        if ablate_threshold:
            logger.debug("Testing: no threshold optimization")
            scores, _ = self._compute_cv_scores_fresh(
                X=X,
                y=y,
                model_name=model_name,
                metric_func=metric_func,
                continuous_cols=continuous_cols,
                best_params=best_params,
                calibrate=True,
                calibration_method="isotonic",
                calibration_cv=3,
                optimize_threshold=False,  # No threshold optimization
            )
            result = self._compute_ablation_result(
                ablation_name="no_threshold_optimization",
                baseline_scores=baseline_scores,
                ablated_scores=scores,
                metric_name=metric_name,
            )
            results.append(result)

        # Ablate scaling
        if ablate_scaling and continuous_cols:
            logger.debug("Testing: no scaling")
            scores, _ = self._compute_cv_scores_fresh(
                X=X,
                y=y,
                model_name=model_name,
                metric_func=metric_func,
                continuous_cols=None,  # No scaling
                best_params=best_params,
                calibrate=True,
                calibration_method="isotonic",
                calibration_cv=3,
                optimize_threshold=True,
                threshold_method="auto",
            )
            result = self._compute_ablation_result(
                ablation_name="no_scaling",
                baseline_scores=baseline_scores,
                ablated_scores=scores,
                metric_name=metric_name,
            )
            results.append(result)

        logger.info(f"pipeline_step_ablation complete. {len(results)} steps analyzed.")

        return AblationStudyResult(
            study_type="pipeline_step",
            results=results,
            baseline_score=float(np.mean(baseline_scores)),
            metric_name=metric_name,
            n_splits=self.n_splits,
            seed=self.seed,
        )
