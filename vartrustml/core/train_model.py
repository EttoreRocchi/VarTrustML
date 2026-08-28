"""
Standalone model fitting with hyperparameter tuning.

:class:`ModelTrainer` fits a single model with hyperparameter optimization,
outside the cross-validation pipeline. This is the path for producing a final
deployable model.

See Also
--------
CrossValidationPipeline : Full nested CV pipeline for model comparison.
ModelConfig : Hyperparameter search space configuration.

Examples
--------
>>> from vartrustml import ModelTrainer, TrainConfig
>>> config = TrainConfig(model_name="XGBoost", calibrate_model=True)
>>> trainer = ModelTrainer(config)
>>> results = trainer.fit(X_train, y_train, X_test, y_test)
"""

import logging
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from vartrustml.config import (
    CalibrationConfig,
    CVConfig,
    ExperimentConfig,
    ModelConfig,
    ThresholdConfig,
)
from vartrustml.core.models import ModelEvaluator
from vartrustml.utils.reproducibility import get_library_versions
from vartrustml.utils.validation import validate_target_for_cv

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Configuration for standalone model fitting.

    Simplified configuration for training a single model with hyperparameter
    optimization, separate from the full cross-validation pipeline.

    Attributes
    ----------
    model_name : str
        Name of the model to fit (default: "XGBoost").
    seed : int
        Random seed for reproducibility (default: 42).
    n_cv_folds : int
        Number of CV folds for hyperparameter tuning (default: 5).
    scoring : str
        Scoring metric for optimization (default: "roc_auc").
    n_jobs : int
        Number of parallel jobs (default: -1).
    calibrate_model : bool
        Whether to calibrate model probabilities (default: False).
    calibration_method : str
        Calibration method ('isotonic' or 'sigmoid') (default: "isotonic").
    calibration_cv : int
        Number of CV folds for calibration (default: 3).
    target_column : str or None
        Name of target column (default: None).
    continuous_cols : list of str
        List of continuous feature columns.
    categorical_cols : list of str
        List of categorical feature columns.
    output_dir : str
        Directory for saving outputs (default: "results/model_fitting").
    save_model : bool
        Whether to save fitted model (default: True).
    save_predictions : bool
        Whether to save predictions (default: True).
    save_report : bool
        Whether to save training report (default: True).
    optimize_threshold : bool
        Whether to optimize classification threshold (default: False).
    threshold_method : str
        Threshold optimization method ('oof', 'cv', or 'auto') (default: "auto").
    threshold_auto_n_samples : int
        Sample count threshold for automatic method selection (default: 1000).

    See Also
    --------
    ModelTrainer : Uses this configuration for training.
    ExperimentConfig : Full experiment configuration for CV pipeline.
    """

    # Model settings
    model_name: str = "XGBoost"
    seed: int = 42
    n_cv_folds: int = 5
    scoring: str = "roc_auc"
    n_jobs: int = -1

    # Calibration settings
    calibrate_model: bool = False
    calibration_method: str = "isotonic"
    calibration_cv: int = 3

    # Data settings
    target_column: Optional[str] = None  # Required from CLI
    continuous_cols: list = field(
        default_factory=list
    )  # Required from CLI if scaling needed
    categorical_cols: list = field(
        default_factory=list
    )  # Optional: if provided, continuous inferred by exclusion

    # Output settings
    output_dir: str = "results/model_fitting"
    save_model: bool = True
    save_predictions: bool = True
    save_report: bool = True

    # Threshold optimization settings
    optimize_threshold: bool = False
    threshold_method: str = "auto"
    threshold_auto_n_samples: int = 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainConfig":
        """Create from dictionary"""
        return cls(**config_dict)


class ModelTrainer:
    """Standalone model fitting with hyperparameter tuning.

    Provides a simpler interface for training a single model with hyperparameter
    optimization, separate from the full cross-validation pipeline. Ideal for
    training final production models after model selection.

    Parameters
    ----------
    train_config : TrainConfig
        Configuration for model fitting.
    model_config : ModelConfig, optional
        Model-specific hyperparameter configuration.
        If None, uses default hyperparameter grids.

    Attributes
    ----------
    train_config : TrainConfig
        Training configuration.
    model_config : ModelConfig
        Model hyperparameter configuration.
    evaluator : ModelEvaluator
        Model evaluator instance.
    model : estimator
        The sklearn-compatible model instance.
    param_grid : dict
        Hyperparameter grid for the selected model.
    pipeline : Pipeline or None
        Fitted sklearn Pipeline (after fit()).
    grid_search : GridSearchCV or None
        Fitted GridSearchCV instance (after fit()).
    results : dict
        Training results and metrics.
    optimal_threshold : float
        Classification threshold (default: 0.5).
    threshold_result : ThresholdResult or None
        Threshold optimization result (if enabled).

    See Also
    --------
    TrainConfig : Configuration for this trainer.
    CrossValidationPipeline : Full CV pipeline for model comparison.

    Examples
    --------
    >>> config = TrainConfig(model_name="XGBoost", calibrate_model=True)
    >>> trainer = ModelTrainer(config)
    >>> results = trainer.fit(X_train, y_train, X_test, y_test)
    >>> print(f"Best score: {results['best_score']:.4f}")
    """

    def __init__(
        self, train_config: TrainConfig, model_config: Optional[ModelConfig] = None
    ):
        self.train_config = train_config
        self.model_config = model_config or ModelConfig()

        self._create_minimal_config()
        self.evaluator = ModelEvaluator(self.minimal_config, self.model_config)

        if train_config.model_name not in self.evaluator.models:
            raise ValueError(
                f"Model '{train_config.model_name}' not available. "
                f"Choose from: {list(self.evaluator.models.keys())}"
            )

        self.model = self.evaluator.models[train_config.model_name]
        self.param_grid = self.evaluator.param_grids.get(train_config.model_name, {})
        self.pipeline = None
        self.grid_search = None
        self.results = {}
        self.optimal_threshold = 0.5
        self.threshold_result = None

    def _create_minimal_config(self):
        """Create ExperimentConfig for ModelEvaluator from TrainConfig."""
        tc = self.train_config
        self.minimal_config = ExperimentConfig(
            cv=CVConfig(
                seed=tc.seed,
                n_inner_splits=tc.n_cv_folds,
            ),
            calibration=CalibrationConfig(
                calibrate_models=tc.calibrate_model,
                calibration_method=tc.calibration_method,
                calibration_cv=tc.calibration_cv,
            ),
            threshold=ThresholdConfig(
                optimize_threshold=tc.optimize_threshold,
                threshold_method=tc.threshold_method,
            ),
            continuous_cols=tc.continuous_cols,
            categorical_cols=tc.categorical_cols,
            target_column=tc.target_column,
            n_jobs=tc.n_jobs,
            models_to_use=[tc.model_name],
            confidence_thresholds=None,
            verbose=1,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """Fit model with hyperparameter tuning.

        Parameters
        ----------
        X : pandas.DataFrame
            Training features.
        y : pandas.Series or numpy.ndarray
            Training labels.
        X_test : pandas.DataFrame, optional
            Test features for evaluation.
        y_test : pandas.Series or numpy.ndarray, optional
            Test labels for evaluation.

        Returns
        -------
        dict
            Results dictionary containing best parameters, CV scores,
            and test metrics (if test data provided).

        Raises
        ------
        ValueError
            If target variable fails validation for CV.
        """
        logger.info(f"Fitting {self.train_config.model_name} on {len(X)} samples")

        # Convert to Series if needed for validation
        if isinstance(y, np.ndarray):
            y_series = pd.Series(y)
        else:
            y_series = y

        # Validate target variable
        is_valid, error_msg = validate_target_for_cv(
            y_series,
            self.train_config.n_cv_folds,
            1,  # No inner CV in ModelTrainer, just use 1
            "training data",
        )

        if not is_valid:
            logger.error(error_msg)
            raise ValueError(f"Cannot fit model: {error_msg}")

        self.pipeline = self.evaluator.create_pipeline(
            self.model, getattr(X, "columns", None)
        )

        if self.train_config.calibrate_model:
            logger.info(
                f"Using CV-based calibration: method={self.train_config.calibration_method}, cv={self.train_config.calibration_cv}"
            )

        param_grid = self.param_grid

        # Adjust parameter names if using calibration (params go through estimator attribute)
        if self.train_config.calibrate_model and param_grid:
            calibrated_param_grid = {}
            for key, value in param_grid.items():
                new_key = key.replace("clf__", "clf__estimator__")
                calibrated_param_grid[new_key] = value
            param_grid = calibrated_param_grid

        cv = StratifiedKFold(
            n_splits=self.train_config.n_cv_folds,
            shuffle=True,
            random_state=self.train_config.seed,
        )

        logger.info(
            f"Performing grid search with {self.train_config.n_cv_folds}-fold CV"
        )
        logger.info(f"Parameter grid: {param_grid}")

        self.grid_search = GridSearchCV(
            estimator=self.pipeline,
            param_grid=param_grid,
            scoring=self.train_config.scoring,
            cv=cv,
            n_jobs=self.train_config.n_jobs,
            verbose=1,
            return_train_score=True,
            refit=True,  # Explicitly ensure model is refitted on full training data
        )

        # Perform HPO on full training set (calibration happens inside pipeline if enabled)
        self.grid_search.fit(X, y)

        # Perform threshold optimization if enabled
        if self.train_config.optimize_threshold:
            self._optimize_threshold(X, y)

        self.results = {
            "model_name": self.train_config.model_name,
            "calibrated": self.train_config.calibrate_model,
            "best_params": self.grid_search.best_params_,
            "best_score": self.grid_search.best_score_,
            "cv_results": pd.DataFrame(self.grid_search.cv_results_),
            "n_samples_train": len(X),
            "n_features": X.shape[1],
            "feature_names": list(X.columns),
            "optimal_threshold": self.optimal_threshold,
        }

        if self.threshold_result is not None:
            self.results["threshold_result"] = self.threshold_result  # Object for CLI
            self.results["threshold_optimization"] = (
                self.threshold_result.to_dict()
            )  # Dict for serialization

        logger.info(f"Best parameters: {self.grid_search.best_params_}")
        logger.info(
            f"Best CV score ({self.train_config.scoring}): {self.grid_search.best_score_:.4f}"
        )

        if self.train_config.optimize_threshold:
            logger.info(f"Optimal threshold: {self.optimal_threshold:.4f}")

        if X_test is not None and y_test is not None:
            self._evaluate_test_set(X_test, y_test)

        if self.train_config.save_model or self.train_config.save_report:
            self._save_outputs(X, y)

        return self.results

    def _optimize_threshold(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]):
        """Optimize classification threshold using Youden's J on internal CV."""
        assert self.grid_search is not None, (
            "fit() must be called before _optimize_threshold()"
        )
        from vartrustml.core.threshold_helper import optimize_threshold_from_cv

        logger.info("Optimizing classification threshold using Youden's J statistic...")

        cv = StratifiedKFold(
            n_splits=self.train_config.n_cv_folds,
            shuffle=True,
            random_state=self.train_config.seed,
        )

        result = optimize_threshold_from_cv(
            self.grid_search.best_estimator_,
            X,
            y,
            cv,
            method=self.train_config.threshold_method,
            auto_threshold_n_samples=self.train_config.threshold_auto_n_samples,
        )

        if result is not None:
            self.threshold_result = result
            self.optimal_threshold = result.optimal_threshold
            logger.info(
                f"Threshold optimization complete: threshold={self.optimal_threshold:.4f}, "
                f"Youden's J={result.youden_j:.4f}, "
                f"method={result.method_used.value}"
            )
        else:
            logger.warning("Threshold optimization failed, using default 0.5")
            self.optimal_threshold = 0.5

    def _evaluate_test_set(
        self, X_test: pd.DataFrame, y_test: Union[pd.Series, np.ndarray]
    ):
        """Evaluate fitted model on test set"""
        assert self.grid_search is not None, (
            "fit() must be called before _evaluate_test_set()"
        )
        logger.info(f"Evaluating on test set ({len(X_test)} samples)")

        y_pred = self.grid_search.predict(X_test)
        y_prob = self.grid_search.predict_proba(X_test)

        # Also compute predictions using optimized threshold
        if self.train_config.optimize_threshold:
            y_pred_threshold = (y_prob[:, 1] >= self.optimal_threshold).astype(int)
        else:
            y_pred_threshold = y_pred

        from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef

        test_results = {
            "n_samples_test": len(X_test),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "matthews corr. coef.": matthews_corrcoef(y_test, y_pred),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
            "auroc": roc_auc_score(y_test, y_prob[:, 1])
            if y_prob.shape[1] == 2
            else None,
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True
            ),
        }

        # Add threshold-optimized metrics if applicable
        if self.train_config.optimize_threshold:
            test_results["balanced_accuracy_threshold"] = balanced_accuracy_score(
                y_test, y_pred_threshold
            )
            test_results["matthews_coef_threshold"] = matthews_corrcoef(
                y_test, y_pred_threshold
            )
            test_results["f1_weighted_threshold"] = f1_score(
                y_test, y_pred_threshold, average="weighted"
            )

        self.results["test_results"] = test_results

        logger.info(
            f"Test matthews corr. coef.: {test_results['matthews corr. coef.']:.4f}"
        )
        logger.info(f"Test balanced accuracy: {test_results['balanced_accuracy']:.4f}")
        if test_results["auroc"]:
            logger.info(f"Test AUROC: {test_results['auroc']:.4f}")

    def _save_outputs(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]):
        """Save model and results"""
        assert self.grid_search is not None, (
            "fit() must be called before _save_outputs()"
        )
        output_dir = Path(self.train_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_suffix = "_calibrated" if self.train_config.calibrate_model else ""
        base_name = f"{self.train_config.model_name.replace(' ', '_')}{model_suffix}"

        if self.train_config.save_model:
            model_path = output_dir / f"{base_name}_model.joblib"

            # Save model with threshold metadata and library versions
            model_data = {
                "model": self.grid_search.best_estimator_,
                "optimal_threshold": self.optimal_threshold,
                "threshold_metadata": self.threshold_result.to_dict()
                if self.threshold_result
                else None,
                "config": self.train_config.to_dict(),
                "library_versions": get_library_versions(),
            }

            joblib.dump(model_data, model_path, compress=9)
            logger.info(f"Model saved to {model_path} with compression level 9")

        if self.train_config.save_report:
            report_path = output_dir / f"{base_name}_report.joblib"

            report = {
                "config": self.train_config.to_dict(),
                "model_name": self.results["model_name"],
                "calibrated": self.results["calibrated"],
                "best_params": self.results["best_params"],
                "best_cv_score": float(self.results["best_score"]),
                "n_samples_train": int(self.results["n_samples_train"]),
                "n_features": int(self.results["n_features"]),
                "feature_names": self.results["feature_names"],
                "optimal_threshold": self.optimal_threshold,
            }

            if self.threshold_result:
                report["threshold_optimization"] = self.threshold_result.to_dict()

            cv_df = self.results["cv_results"]
            report["cv_summary"] = {
                "mean_fit_time": float(cv_df["mean_fit_time"].mean()),
                "mean_score_time": float(cv_df["mean_score_time"].mean()),
                "params_evaluated": len(cv_df),
                "score_summary": {
                    "mean": float(cv_df["mean_test_score"].mean()),
                    "std": float(cv_df["std_test_score"].mean()),
                    "min": float(cv_df["mean_test_score"].min()),
                    "max": float(cv_df["mean_test_score"].max()),
                },
            }

            joblib.dump(report, report_path)

            logger.info(f"Report saved to {report_path}")

            cv_path = output_dir / f"{base_name}_cv_results.csv"
            cv_df.to_csv(cv_path, index=False)
            logger.info(f"CV results saved to {cv_path}")

    def predict(self, X: pd.DataFrame, use_threshold: bool = True) -> np.ndarray:
        """Make predictions with fitted model.

        Parameters
        ----------
        X : pandas.DataFrame
            Features to predict.
        use_threshold : bool, default=True
            If True and threshold optimization was enabled,
            use the optimized threshold instead of default 0.5.

        Returns
        -------
        numpy.ndarray
            Predicted binary labels.

        Raises
        ------
        ValueError
            If model has not been fitted yet.
        """
        if self.grid_search is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        if use_threshold and self.train_config.optimize_threshold:
            y_prob = self.grid_search.predict_proba(X)[:, 1]
            return (y_prob >= self.optimal_threshold).astype(int)
        else:
            return self.grid_search.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities.

        Parameters
        ----------
        X : pandas.DataFrame
            Features to predict.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_classes)
            Predicted probabilities for each class.

        Raises
        ------
        ValueError
            If model has not been fitted yet.
        """
        if self.grid_search is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        return self.grid_search.predict_proba(X)

    @classmethod
    def load_model(cls, model_path: Union[str, Path]) -> Dict[str, Any]:
        """Load a saved model with threshold metadata.

        Parameters
        ----------
        model_path : str or Path
            Path to saved model file (.joblib).

        Returns
        -------
        dict
            Dictionary containing:
            - 'model': Fitted estimator
            - 'optimal_threshold': Classification threshold
            - 'threshold_metadata': ThresholdResult dict (if available)
            - 'config': TrainConfig dict (if available)
            - 'library_versions': Dict of library versions used during training

        Warns
        -----
        UserWarning
            If library versions differ between training and current environment.
        """
        model_data = joblib.load(model_path)

        if not isinstance(model_data, dict) or "model" not in model_data:
            raise ValueError(
                f"Invalid model format in {model_path}. "
                "Expected dict with 'model' key. Old format models are not supported."
            )

        # Check for library version mismatches
        if "library_versions" in model_data:
            saved_versions = model_data["library_versions"]
            current_versions = get_library_versions()
            mismatches = []
            for lib, saved_ver in saved_versions.items():
                current_ver = current_versions.get(lib)
                if current_ver and current_ver != saved_ver:
                    mismatches.append(f"  {lib}: {saved_ver} → {current_ver}")
            if mismatches:
                warnings.warn(
                    "Model was trained with different library versions:\n"
                    + "\n".join(mismatches)
                    + "\nThis may cause compatibility issues.",
                    UserWarning,
                    stacklevel=2,
                )

        return model_data
