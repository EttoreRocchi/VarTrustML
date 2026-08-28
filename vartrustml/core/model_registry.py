"""
Model registry for VarTrustML.

Provides a registry pattern for model specifications, enabling new models
to be added without modifying the ModelEvaluator class (Open/Closed Principle).

Each model is defined by a :class:`ModelSpec` that knows how to create its
estimator, define its hyperparameter grid, and extract feature importances.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from vartrustml.config.model import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelSpec(ABC):
    """Specification for a machine learning model.

    Attributes
    ----------
    name : str
        Display name of the model.
    shap_explainer_type : str
        SHAP explainer type: ``"tree"``, ``"linear"``, or ``"kernel"``.
    """

    name: str
    shap_explainer_type: str

    @abstractmethod
    def create_estimator(self, seed: int, model_config: ModelConfig) -> Any:
        """Create a fresh estimator instance.

        Parameters
        ----------
        seed : int
            Random seed for reproducibility.
        model_config : ModelConfig
            Model hyperparameter configuration.

        Returns
        -------
        Any
            Unfitted scikit-learn compatible estimator.
        """
        ...

    @abstractmethod
    def get_param_grid(self, model_config: ModelConfig) -> Dict[str, List]:
        """Return the hyperparameter search grid.

        Parameters
        ----------
        model_config : ModelConfig
            Model hyperparameter configuration.

        Returns
        -------
        dict
            Mapping of parameter names to candidate values.
        """
        ...

    def extract_feature_importances(self, fitted_clf: Any) -> Optional[np.ndarray]:
        """Extract feature importances from a fitted classifier.

        Parameters
        ----------
        fitted_clf : Any
            Fitted scikit-learn compatible classifier.

        Returns
        -------
        numpy.ndarray or None
            Feature importance array, or None if not available.
        """
        return None


_MODEL_REGISTRY: Dict[str, ModelSpec] = {}


def register_model(spec: ModelSpec) -> ModelSpec:
    """Register a model spec in the global registry."""
    _MODEL_REGISTRY[spec.name] = spec
    return spec


def get_model_spec(name: str) -> ModelSpec:
    """Get a registered model spec by name."""
    return _MODEL_REGISTRY[name]


def get_registered_model_names() -> List[str]:
    """Get names of all registered models."""
    return list(_MODEL_REGISTRY.keys())


def create_models(
    model_names: List[str], seed: int, model_config: ModelConfig
) -> Dict[str, Any]:
    """Create model instances for the given names."""
    models = {}
    for name in model_names:
        if name in _MODEL_REGISTRY:
            models[name] = _MODEL_REGISTRY[name].create_estimator(seed, model_config)
    return models


def create_param_grids(
    model_names: List[str], model_config: ModelConfig
) -> Dict[str, Dict[str, List]]:
    """Create parameter grids for the given model names."""
    grids = {}
    for name in model_names:
        if name in _MODEL_REGISTRY:
            grids[name] = _MODEL_REGISTRY[name].get_param_grid(model_config)
    return grids


def extract_feature_importances(
    model_name: str, fitted_clf: Any
) -> Optional[np.ndarray]:
    """Extract feature importances using the model's spec."""
    if model_name in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[model_name].extract_feature_importances(fitted_clf)
    return None


# ---------------------------------------------------------------------------
# Built-in model specifications
# ---------------------------------------------------------------------------


class RandomForestSpec(ModelSpec):
    """Specification for scikit-learn RandomForestClassifier."""

    def __init__(self):
        super().__init__(name="Random Forest", shap_explainer_type="tree")

    def create_estimator(self, seed: int, model_config: ModelConfig) -> Any:
        return RandomForestClassifier(class_weight="balanced", random_state=seed)

    def get_param_grid(self, model_config: ModelConfig) -> Dict[str, List]:
        return {
            "clf__n_estimators": model_config.rf_n_estimators_options,
            "clf__max_depth": model_config.rf_max_depth_options,
        }

    def extract_feature_importances(self, fitted_clf: Any) -> Optional[np.ndarray]:
        return fitted_clf.feature_importances_


class XGBoostSpec(ModelSpec):
    """Specification for XGBoost XGBClassifier."""

    def __init__(self):
        super().__init__(name="XGBoost", shap_explainer_type="tree")

    def create_estimator(self, seed: int, model_config: ModelConfig) -> Any:
        return XGBClassifier(
            eval_metric="logloss",
            random_state=seed,
            device="cpu",
            enable_categorical=False,
            n_jobs=1,  # parallelism handled by grid-search
        )

    def get_param_grid(self, model_config: ModelConfig) -> Dict[str, List]:
        return {
            "clf__n_estimators": model_config.xgb_n_estimators_options,
            "clf__max_depth": model_config.xgb_max_depth_options,
            "clf__learning_rate": model_config.xgb_learning_rate_options,
        }

    def extract_feature_importances(self, fitted_clf: Any) -> Optional[np.ndarray]:
        return fitted_clf.feature_importances_


class CatBoostSpec(ModelSpec):
    """Specification for CatBoost CatBoostClassifier."""

    def __init__(self):
        super().__init__(name="CatBoost", shap_explainer_type="tree")

    def create_estimator(self, seed: int, model_config: ModelConfig) -> Any:
        return CatBoostClassifier(
            auto_class_weights="Balanced",
            verbose=0,
            allow_writing_files=False,
            random_state=seed,
            thread_count=1,  # parallelism handled by grid-search
        )

    def get_param_grid(self, model_config: ModelConfig) -> Dict[str, List]:
        return {
            "clf__iterations": model_config.catboost_iterations_options,
            "clf__depth": model_config.catboost_depth_options,
            "clf__learning_rate": model_config.catboost_learning_rate_options,
        }

    def extract_feature_importances(self, fitted_clf: Any) -> Optional[np.ndarray]:
        return fitted_clf.feature_importances_


class LogisticRegressionSpec(ModelSpec):
    """Specification for scikit-learn LogisticRegression."""

    def __init__(self):
        super().__init__(name="Logistic Regression", shap_explainer_type="linear")

    def create_estimator(self, seed: int, model_config: ModelConfig) -> Any:
        return LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            max_iter=1000,
            random_state=seed,
        )

    def get_param_grid(self, model_config: ModelConfig) -> Dict[str, List]:
        return {
            "clf__C": model_config.lr_C_options,
            "clf__penalty": model_config.lr_penalty_options,
        }

    def extract_feature_importances(self, fitted_clf: Any) -> Optional[np.ndarray]:
        return np.abs(fitted_clf.coef_[0])


class MLPSpec(ModelSpec):
    """Specification for scikit-learn MLPClassifier."""

    def __init__(self):
        super().__init__(name="MLP", shap_explainer_type="kernel")

    def create_estimator(self, seed: int, model_config: ModelConfig) -> Any:
        return MLPClassifier(
            max_iter=model_config.mlp_max_iter,
            random_state=seed,
            early_stopping=True,
        )

    def get_param_grid(self, model_config: ModelConfig) -> Dict[str, List]:
        return {
            "clf__hidden_layer_sizes": model_config.mlp_hidden_layer_options,
            "clf__activation": model_config.mlp_activation_options,
            "clf__alpha": model_config.mlp_alpha_options,
        }


class KNNSpec(ModelSpec):
    """Specification for scikit-learn KNeighborsClassifier."""

    def __init__(self):
        super().__init__(name="KNN", shap_explainer_type="kernel")

    def create_estimator(self, seed: int, model_config: ModelConfig) -> Any:
        return KNeighborsClassifier(metric="euclidean")

    def get_param_grid(self, model_config: ModelConfig) -> Dict[str, List]:
        return {
            "clf__n_neighbors": model_config.knn_n_neighbors_options,
            "clf__weights": model_config.knn_weights_options,
            "clf__leaf_size": model_config.knn_leaf_size_options,
        }


# Register all built-in models
register_model(RandomForestSpec())
register_model(XGBoostSpec())
register_model(CatBoostSpec())
register_model(LogisticRegressionSpec())
register_model(MLPSpec())
register_model(KNNSpec())
