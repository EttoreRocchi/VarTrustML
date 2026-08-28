"""
Model hyperparameter configuration for VarTrustML.

:class:`ModelConfig` defines the hyperparameter search space of a single
model.

See Also
--------
ExperimentConfig : Main experiment configuration.
ModelEvaluator : Uses ModelConfig for hyperparameter search.

Examples
--------
>>> from vartrustml import ModelConfig
>>> model_config = ModelConfig(
...     rf_n_estimators_options=[100, 200, 500],
...     rf_max_depth_options=[3, 5, 7]
... )
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ModelConfig:
    """Configuration for model hyperparameter search spaces.

    Defines the search spaces for hyperparameter optimization for each
    supported model type. Used by ModelEvaluator during grid search or
    Optuna optimization.

    Attributes
    ----------
    rf_n_estimators_options : list of int
        Random Forest n_estimators search space.
    rf_max_depth_options : list of int
        Random Forest max_depth search space.
    lr_C_options : list of float
        Logistic Regression C (regularization) search space.
    lr_penalty_options : list of str
        Logistic Regression penalty type search space.
    xgb_n_estimators_options : list of int
        XGBoost n_estimators search space.
    xgb_max_depth_options : list of int
        XGBoost max_depth search space.
    xgb_learning_rate_options : list of float
        XGBoost learning_rate search space.
    mlp_hidden_layer_options : list of tuple
        MLP hidden layer sizes search space.
    mlp_activation_options : list of str
        MLP activation function search space.
    mlp_alpha_options : list of float
        MLP alpha (L2 penalty) search space.
    mlp_max_iter : int
        MLP maximum iterations (default: 1000).
    catboost_iterations_options : list of int
        CatBoost iterations search space.
    catboost_depth_options : list of int
        CatBoost depth search space.
    catboost_learning_rate_options : list of float
        CatBoost learning_rate search space.
    knn_n_neighbors_options : list of int
        KNN n_neighbors search space.
    knn_weights_options : list of str
        KNN weights search space ('uniform' or 'distance').
    knn_leaf_size_options : list of int
        KNN leaf_size search space.

    See Also
    --------
    ExperimentConfig : Main experiment configuration.
    ModelEvaluator : Uses this config for hyperparameter search.

    Examples
    --------
    >>> config = ModelConfig(
    ...     rf_n_estimators_options=[100, 200, 500],
    ...     xgb_learning_rate_options=[0.01, 0.05, 0.1]
    ... )
    """

    # Random Forest settings
    rf_n_estimators_options: List[int] = field(default_factory=lambda: [50, 100, 200])
    rf_max_depth_options: List[int] = field(default_factory=lambda: [3, 5])

    # Logistic Regression settings
    lr_C_options: List[float] = field(default_factory=lambda: [0.01, 0.1, 1, 10])
    lr_penalty_options: List[str] = field(default_factory=lambda: ["l1", "l2"])

    # XGBoost settings
    xgb_n_estimators_options: List[int] = field(default_factory=lambda: [50, 100, 200])
    xgb_max_depth_options: List[int] = field(default_factory=lambda: [3, 5])
    xgb_learning_rate_options: List[float] = field(default_factory=lambda: [0.01, 0.1])

    # MLP settings
    mlp_hidden_layer_options: List[Tuple[int, ...]] = field(
        default_factory=lambda: [(50,), (100,), (50, 50)]
    )
    mlp_activation_options: List[str] = field(default_factory=lambda: ["relu", "tanh"])
    mlp_alpha_options: List[float] = field(default_factory=lambda: [0.0001, 0.001])
    mlp_max_iter: int = 1000

    # CatBoost settings
    catboost_iterations_options: List[int] = field(
        default_factory=lambda: [100, 200, 500]
    )
    catboost_depth_options: List[int] = field(default_factory=lambda: [4, 6, 8])
    catboost_learning_rate_options: List[float] = field(
        default_factory=lambda: [0.01, 0.05, 0.1]
    )

    # KNN settings
    knn_n_neighbors_options: List[int] = field(default_factory=lambda: [3, 5, 7, 9])
    knn_weights_options: List[str] = field(
        default_factory=lambda: ["uniform", "distance"]
    )
    knn_leaf_size_options: List[int] = field(default_factory=lambda: [20, 30, 40])
