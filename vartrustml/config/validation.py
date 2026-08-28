"""
Configuration validation utilities.

Checks configuration parameters before an experiment starts, so errors
surface before any model is fitted.
"""

import logging
from typing import List, Optional, Tuple

import pandas as pd

from vartrustml.config.experiment import SUPPORTED_MODELS, ExperimentConfig
from vartrustml.config.model import ModelConfig

logger = logging.getLogger(__name__)


def validate_experiment_config(config: ExperimentConfig) -> Tuple[bool, List[str]]:
    """Validate ExperimentConfig parameters.

    Parameters
    ----------
    config : ExperimentConfig
        ExperimentConfig instance to validate.

    Returns
    -------
    tuple of (bool, list of str)
        ``(is_valid, list_of_errors)`` where ``is_valid`` is True when the
        configuration passes all checks.
    """
    errors = []

    # Validate seed
    if config.cv.seed < 0:
        errors.append(f"seed must be non-negative, got {config.cv.seed}")

    # Validate CV splits
    if config.cv.n_outer_splits < 2:
        errors.append(f"n_outer_splits must be >= 2, got {config.cv.n_outer_splits}")

    if config.cv.n_inner_splits < 2:
        errors.append(f"n_inner_splits must be >= 2, got {config.cv.n_inner_splits}")

    # Validate calibration settings
    if config.calibration.calibrate_models:
        if config.calibration.calibration_cv < 2:
            errors.append(
                f"calibration_cv must be >= 2, got {config.calibration.calibration_cv}"
            )

        if config.calibration.calibration_method not in ["sigmoid", "isotonic"]:
            errors.append(
                f"calibration_method must be 'sigmoid' or 'isotonic', got '{config.calibration.calibration_method}'"
            )

    # Validate models
    if not config.models_to_use:
        errors.append("models_to_use cannot be empty")
    else:
        for model in config.models_to_use:
            if model not in SUPPORTED_MODELS:
                errors.append(
                    f"Unknown model '{model}'. Valid models: {', '.join(SUPPORTED_MODELS)}"
                )

    # Validate NaN handling strategy
    valid_nan_strategies = ("median", "mean", "most_frequent", "drop")
    if config.nan_strategy not in valid_nan_strategies:
        errors.append(
            f"nan_strategy must be one of {valid_nan_strategies}, "
            f"got {config.nan_strategy!r}"
        )

    # Validate confidence thresholds
    if config.confidence_thresholds:
        for threshold in config.confidence_thresholds:
            if not (0 < threshold < 1):
                errors.append(
                    f"Confidence thresholds must be between 0 and 1, got {threshold}"
                )

    # Validate visualization settings
    if config.visualization.plot_top_n_features < 1:
        errors.append(
            f"plot_top_n_features must be >= 1, got {config.visualization.plot_top_n_features}"
        )

    if config.visualization.figure_dpi < 50:
        errors.append(
            f"figure_dpi must be >= 50, got {config.visualization.figure_dpi}"
        )

    # Validate parallel settings
    if config.n_jobs == 0:
        errors.append("n_jobs cannot be 0 (use -1 for all cores or positive integer)")

    # Validate verbosity
    if config.verbose < 0 or config.verbose > 3:
        errors.append(f"verbose must be 0, 1, 2, or 3, got {config.verbose}")

    # Validate threshold optimization settings
    if config.threshold.optimize_threshold:
        valid_threshold_methods = ["oof", "cv", "auto"]
        if config.threshold.threshold_method not in valid_threshold_methods:
            errors.append(
                f"threshold_method must be one of {valid_threshold_methods}, "
                f"got '{config.threshold.threshold_method}'"
            )

        if config.threshold_auto_n_samples < 100:
            errors.append(
                f"threshold_auto_n_samples must be >= 100, got {config.threshold_auto_n_samples}"
            )

    # ============================================
    # Inter-field validation (cross-field checks)
    # ============================================

    # Validate caller comparison requires caller columns
    if (
        config.caller_comparison.compare_callers
        and not config.caller_comparison.caller_columns
    ):
        errors.append(
            "compare_callers=True requires caller_columns to be specified. "
            "Set caller_columns to list of binary caller feature names."
        )

    # Validate threshold_method='cv' requires sufficient inner splits
    if (
        config.threshold.optimize_threshold
        and config.threshold.threshold_method == "cv"
    ):
        if config.cv.n_inner_splits < 3:
            errors.append(
                f"threshold_method='cv' requires n_inner_splits >= 3, "
                f"got {config.cv.n_inner_splits}. Use threshold_method='oof' or "
                f"increase n_inner_splits."
            )

    # Validate bootstrap settings consistency
    if config.bootstrap.bootstrap_n_iterations < 100:
        errors.append(
            f"bootstrap_n_iterations should be >= 100 for reliable confidence intervals, "
            f"got {config.bootstrap.bootstrap_n_iterations}"
        )

    if config.bootstrap.bootstrap_ci_method not in ("bca", "percentile"):
        errors.append(
            f"bootstrap_ci_method must be 'bca' or 'percentile', "
            f"got {config.bootstrap.bootstrap_ci_method!r}"
        )

    if not 0 < config.bootstrap.bootstrap_ci_level < 1:
        errors.append(
            f"bootstrap_ci_level must be a proportion in (0, 1), e.g. 0.95 for a "
            f"95% interval, got {config.bootstrap.bootstrap_ci_level!r}"
        )

    if config.hpo_method not in ("grid", "optuna"):
        errors.append(
            f"hpo_method must be 'grid' or 'optuna', got {config.hpo_method!r}"
        )

    if config.correction_method not in ("holm", "bh"):
        errors.append(
            f"correction_method must be 'holm' or 'bh', got {config.correction_method!r}"
        )

    is_valid = len(errors) == 0

    if not is_valid:
        logger.error("Configuration validation failed:")
        for error in errors:
            logger.error(f"  - {error}")

    return is_valid, errors


def validate_model_config(config: ModelConfig) -> Tuple[bool, List[str]]:
    """Validate ModelConfig parameters.

    Parameters
    ----------
    config : ModelConfig
        ModelConfig instance to validate.

    Returns
    -------
    tuple of (bool, list of str)
        ``(is_valid, list_of_errors)`` where ``is_valid`` is True when the
        configuration passes all checks.
    """
    errors = []

    # Validate all hyperparameter options are non-empty
    if not config.rf_n_estimators_options:
        errors.append("rf_n_estimators_options cannot be empty")

    if not config.rf_max_depth_options:
        errors.append("rf_max_depth_options cannot be empty")

    if not config.lr_C_options:
        errors.append("lr_C_options cannot be empty")

    if not config.xgb_n_estimators_options:
        errors.append("xgb_n_estimators_options cannot be empty")

    # Validate ranges
    for est in config.rf_n_estimators_options:
        if est < 1:
            errors.append(f"rf_n_estimators must be >= 1, got {est}")

    for depth in config.rf_max_depth_options:
        if depth < 1:
            errors.append(f"rf_max_depth must be >= 1, got {depth}")

    for C in config.lr_C_options:
        if C <= 0:
            errors.append(f"lr_C must be > 0, got {C}")

    is_valid = len(errors) == 0

    if not is_valid:
        logger.error("Model configuration validation failed:")
        for error in errors:
            logger.error(f"  - {error}")

    return is_valid, errors


def validate_dataset_for_cv(
    df: pd.DataFrame,
    target_column: str,
    n_outer_splits: int,
    n_inner_splits: int,
    dataset_name: Optional[str] = None,
) -> Tuple[bool, List[str]]:
    """Validate dataset is suitable for cross-validation.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to validate.
    target_column : str
        Name of target column.
    n_outer_splits : int
        Number of outer CV folds.
    n_inner_splits : int
        Number of inner CV folds.
    dataset_name : str, optional
        Name for error messages.

    Returns
    -------
    tuple of (bool, list of str)
        ``(is_valid, list_of_errors)`` where ``is_valid`` is True when the
        dataset passes all checks.
    """
    errors = []
    name = dataset_name or "Dataset"

    # Check dataset is not empty
    if len(df) == 0:
        errors.append(f"{name}: Dataset is empty")
        return False, errors

    # Check target column exists
    if target_column not in df.columns:
        errors.append(f"{name}: Target column '{target_column}' not found in dataset")
        return False, errors

    # Check target has at least 2 classes
    target = df[target_column]
    n_classes = target.nunique()

    if n_classes < 2:
        errors.append(f"{name}: Target has only {n_classes} class(es), need at least 2")

    # Check sufficient samples per class for CV
    class_counts = target.value_counts()
    min_samples_per_class = class_counts.min()

    min_required = max(n_outer_splits, n_inner_splits)

    if min_samples_per_class < min_required:
        errors.append(
            f"{name}: Minimum class has only {min_samples_per_class} samples, "
            f"but need at least {min_required} for {max(n_outer_splits, n_inner_splits)}-fold CV"
        )

    # Check for missing values in target
    if target.isna().any():
        n_missing = target.isna().sum()
        errors.append(f"{name}: Target column has {n_missing} missing values")

    # Check dataset has features
    n_features = len([col for col in df.columns if col != target_column])
    if n_features == 0:
        errors.append(f"{name}: Dataset has no feature columns")

    is_valid = len(errors) == 0

    return is_valid, errors
