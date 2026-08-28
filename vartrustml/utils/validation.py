"""
Data validation utilities for cross-validation pipelines.
"""

import math
from typing import Tuple

import pandas as pd


def calculate_minimum_samples_for_cv(n_outer_splits: int, n_inner_splits: int) -> int:
    """
    Calculate minimum required samples in minority class for nested CV.

    Formula: ``K >= max(M, ceil((M * N) / (M - 1)))``

    Where K = minimum samples in minority class, M = n_outer_splits (outer folds),
    N = n_inner_splits (inner folds).

    Parameters
    ----------
    n_outer_splits : int
        Number of outer CV folds
    n_inner_splits : int
        Number of inner CV folds

    Returns
    -------
    int
        Minimum required number of samples in minority class
    """
    M = n_outer_splits
    N = n_inner_splits

    if M <= 1:
        raise ValueError("n_outer_splits must be > 1 for cross-validation")

    term1 = M
    term2 = math.ceil((M * N) / (M - 1))

    return max(term1, term2)


def validate_target_for_cv(
    y: pd.Series,
    n_outer_splits: int,
    n_inner_splits: int,
    dataset_name: str = "dataset",
) -> Tuple[bool, str]:
    """
    Validate that target variable is suitable for nested cross-validation.

    Checks:
    1. Target has more than one unique class
    2. Minority class has sufficient samples for nested CV

    Parameters
    ----------
    y : pd.Series
        Target variable
    n_outer_splits : int
        Number of outer CV folds
    n_inner_splits : int
        Number of inner CV folds
    dataset_name : str
        Name of dataset for logging

    Returns
    -------
    Tuple[bool, str]
        ``(True, "")`` when the target is usable, otherwise ``(False, reason)``.
    """
    # Check for single class
    unique_classes = y.nunique()
    if unique_classes < 2:
        error_msg = (
            f"{dataset_name}: Target has only {unique_classes} unique value(s). "
            f"Classification requires at least 2 classes. Skipping dataset."
        )
        return False, error_msg

    # Check minority class sample size
    class_counts = y.value_counts()
    min_class_samples = class_counts.min()

    required_samples = calculate_minimum_samples_for_cv(n_outer_splits, n_inner_splits)

    if min_class_samples < required_samples:
        minority_class = class_counts.idxmin()
        error_msg = (
            f"{dataset_name}: Insufficient samples in minority class. "
            f"Class '{minority_class}' has {min_class_samples} samples, "
            f"but {required_samples} required for {n_outer_splits}-fold outer "
            f"and {n_inner_splits}-fold inner CV. "
            f"Formula: K >= max({n_outer_splits}, ceil(({n_outer_splits} * {n_inner_splits}) / ({n_outer_splits} - 1))) = {required_samples}. "
            f"Skipping dataset."
        )
        return False, error_msg

    return True, ""
