"""
Caller Evaluator Module

Evaluates individual variant callers, and their AND/OR combinations, as
baseline classifiers.

The key feature is that callers are evaluated on the same CV test folds
as ML models, enabling fair comparison.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from vartrustml.core.metrics import calculate_classification_metrics


@dataclass
class CallerResult:
    """Store evaluation results for a single caller or combination.

    Attributes:
        name (str): Identifier for the caller or combination (e.g., "MANTA", "MANTA AND DELLY").
        fold_id (int): CV fold identifier.
        metrics (Dict[str, float]): Dictionary of computed metrics (same as ML models).
        confusion_matrix (np.ndarray): Normalized confusion matrix.
        y_true (np.ndarray): True labels for this fold.
        y_pred (np.ndarray): Caller predictions for this fold.
    """

    name: str
    fold_id: int
    metrics: Dict[str, float]
    confusion_matrix: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    sample_indices: Optional[np.ndarray] = None


class CallerEvaluator:
    """
    Evaluate variant callers and their logical combinations.

    This class provides methods to:
    - Evaluate individual callers (MANTA, DELLY, SMOOVE, etc.)
    - Evaluate logical combinations (AND/OR)
    - Generate default combinations (all pairwise AND/OR + all-callers AND/OR)

    All evaluations use the same metrics as ML models for fair comparison.

    Attributes:
        caller_columns (List[str]): List of column names for variant callers.
    """

    def __init__(self, caller_columns: List[str]):
        """
        Initialize CallerEvaluator with explicit caller columns.

        Parameters
        ----------
        caller_columns : List[str]
            List of column names for variant callers
            (e.g., ["MANTA", "DELLY", "SMOOVE"])

        Raises
        ------
        ValueError
            If caller_columns is empty
        """
        if not caller_columns:
            raise ValueError("caller_columns cannot be empty")

        self.caller_columns = caller_columns

    def evaluate_single_caller(
        self,
        caller_name: str,
        y_true: np.ndarray,
        caller_predictions: np.ndarray,
        fold_id: int,
    ) -> CallerResult:
        """
        Evaluate a single variant caller as a binary classifier.

        Parameters
        ----------
        caller_name : str
            Name of the caller (e.g., "MANTA")
        y_true : np.ndarray
            True labels
        caller_predictions : np.ndarray
            Binary predictions from the caller (0/1)
        fold_id : int
            CV fold identifier

        Returns
        -------
        CallerResult
            Metrics, confusion matrix and predictions for this caller.

        Raises
        ------
        ValueError
            If caller_name not in caller_columns
        """
        if caller_name not in self.caller_columns:
            raise ValueError(
                f"Caller '{caller_name}' not in caller_columns: {self.caller_columns}"
            )

        metrics = self._calculate_metrics(y_true, caller_predictions)
        cm = self._compute_confusion_matrix(y_true, caller_predictions)

        return CallerResult(
            name=caller_name,
            fold_id=fold_id,
            metrics=metrics,
            confusion_matrix=cm,
            y_true=y_true,
            y_pred=caller_predictions,
        )

    def evaluate_combination(
        self,
        caller_names: List[str],
        operation: str,
        y_true: np.ndarray,
        caller_data: pd.DataFrame,
        fold_id: int,
    ) -> CallerResult:
        """
        Evaluate a logical combination of callers.

        Parameters
        ----------
        caller_names : List[str]
            List of caller names to combine
        operation : str
            Logical operation ("AND" or "OR")
        y_true : np.ndarray
            True labels
        caller_data : pd.DataFrame
            DataFrame with caller columns
        fold_id : int
            CV fold identifier

        Returns
        -------
        CallerResult
            Metrics and predictions for the combined call.

        Raises
        ------
        ValueError
            If operation is not "AND" or "OR"
        ValueError
            If any caller_name not in caller_columns
        """
        operation = operation.upper()
        if operation not in ("AND", "OR"):
            raise ValueError(f"Operation must be 'AND' or 'OR', got: {operation}")

        for caller in caller_names:
            if caller not in self.caller_columns:
                raise ValueError(
                    f"Caller '{caller}' not in caller_columns: {self.caller_columns}"
                )

        # Get predictions from each caller
        caller_preds = [caller_data[c].values for c in caller_names]

        # Apply logical operation
        if operation == "AND":
            combined_pred = np.all(np.column_stack(caller_preds), axis=1).astype(int)
        else:  # OR
            combined_pred = np.any(np.column_stack(caller_preds), axis=1).astype(int)

        # Create combination name
        combination_name = f" {operation} ".join(caller_names)

        metrics = self._calculate_metrics(y_true, combined_pred)
        cm = self._compute_confusion_matrix(y_true, combined_pred)

        return CallerResult(
            name=combination_name,
            fold_id=fold_id,
            metrics=metrics,
            confusion_matrix=cm,
            y_true=y_true,
            y_pred=combined_pred,
        )

    def parse_combination_expression(self, expression: str) -> Tuple[List[str], str]:
        """
        Parse a combination expression like "MANTA AND DELLY".

        Parameters
        ----------
        expression : str
            Expression string (e.g., "MANTA AND DELLY", "MANTA OR DELLY OR SMOOVE")

        Returns
        -------
        Tuple[List[str], str]
            The caller names, and the operation joining them.

        Raises
        ------
        ValueError
            If expression is malformed or uses mixed operations
        """
        expression = expression.strip()

        # Check for AND operation
        if " AND " in expression:
            if " OR " in expression:
                raise ValueError(
                    f"Mixed AND/OR not supported in expression: {expression}"
                )
            callers = [c.strip() for c in expression.split(" AND ")]
            return callers, "AND"

        # Check for OR operation
        if " OR " in expression:
            callers = [c.strip() for c in expression.split(" OR ")]
            return callers, "OR"

        # Single caller (no operation)
        raise ValueError(f"Expression must contain 'AND' or 'OR': {expression}")

    def get_default_combinations(self) -> List[str]:
        """
        Generate default logical combinations of callers.

        Generates:
        - All pairwise AND combinations
        - All pairwise OR combinations
        - All-callers AND (unanimous consensus)
        - All-callers OR (any caller)

        Returns
        -------
        List[str]
            One expression per generated combination.
        """
        combinations = []
        callers = self.caller_columns

        # Pairwise combinations
        for i, c1 in enumerate(callers):
            for c2 in callers[i + 1 :]:
                combinations.append(f"{c1} AND {c2}")
                combinations.append(f"{c1} OR {c2}")

        # All-callers combinations (only if >2 callers)
        if len(callers) > 2:
            combinations.append(" AND ".join(callers))
            combinations.append(" OR ".join(callers))

        return combinations

    def evaluate_from_expression(
        self,
        expression: str,
        y_true: np.ndarray,
        caller_data: pd.DataFrame,
        fold_id: int,
    ) -> CallerResult:
        """
        Evaluate a combination from an expression string.

        Parameters
        ----------
        expression : str
            Expression string (e.g., "MANTA AND DELLY")
        y_true : np.ndarray
            True labels
        caller_data : pd.DataFrame
            DataFrame with caller columns
        fold_id : int
            CV fold identifier

        Returns
        -------
        CallerResult
            Metrics and predictions for the combined call.
        """
        callers, operation = self.parse_combination_expression(expression)
        return self.evaluate_combination(
            callers, operation, y_true, caller_data, fold_id
        )

    def _calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate metrics for caller evaluation.

        Uses the same metrics as ML model evaluation for fair comparison.
        Note: AUROC is not computed since callers provide binary outputs,
        not probability scores.

        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels (binary)

        Returns
        -------
        Dict[str, float]
            Metric name to value. Probability-based entries are NaN.
        """
        return calculate_classification_metrics(y_true, y_pred, y_prob=None)

    def _compute_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Compute normalized confusion matrix.

        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels

        Returns
        -------
        np.ndarray
            Normalized confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        # Normalize (guard against zero row sums from single-class folds)
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm_normalized = cm.astype("float") / row_sums
        return cm_normalized


def validate_caller_columns(
    df: pd.DataFrame, caller_columns: List[str], target_column: str
) -> None:
    """
    Validate that caller columns exist and are binary.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    caller_columns : List[str]
        List of caller column names
    target_column : str
        Name of the target column (to exclude from validation)

    Raises
    ------
    ValueError
        If any caller column doesn't exist or isn't binary
    """
    for col in caller_columns:
        if col not in df.columns:
            raise ValueError(f"Caller column '{col}' not found in dataset")

        if col == target_column:
            raise ValueError(
                f"Caller column '{col}' cannot be the same as target column"
            )

        unique_vals = df[col].dropna().unique()
        if not set(unique_vals).issubset({0, 1}):
            raise ValueError(
                f"Caller column '{col}' must be binary (0/1), "
                f"found values: {sorted(unique_vals)}"
            )


def caller_baseline_table(
    datasets: List[Tuple[pd.DataFrame, str]],
    caller_columns: List[str],
    target_column: str,
    include_combinations: bool = True,
    metric: str = "Matthews Corr. Coef.",
) -> pd.DataFrame:
    """Operating-point metric of each caller (+ default combinations) per dataset.

    Callers are train-invariant, so each value is the caller's metric on that
    sample. Rows = caller / combination, columns = dataset name.
    """
    evaluator = CallerEvaluator(caller_columns)
    combos = evaluator.get_default_combinations() if include_combinations else []
    table = {}
    for df, name in datasets:
        y = df[target_column].to_numpy()
        col = {}
        for caller in caller_columns:
            res = evaluator.evaluate_single_caller(caller, y, df[caller].to_numpy(), 0)
            col[caller] = res.metrics.get(metric)
        for expr in combos:
            res = evaluator.evaluate_from_expression(expr, y, df[caller_columns], 0)
            col[expr] = res.metrics.get(metric)
        table[name] = col
    return pd.DataFrame(table)
