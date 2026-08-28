"""
Protocol definitions for dependency inversion in the core pipeline.

These protocols define the contracts that pipeline components must satisfy,
enabling loose coupling between high-level orchestrators and low-level
implementations.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

import numpy as np

if TYPE_CHECKING:
    from vartrustml.analysis.error_analysis import FoldMetrics


@runtime_checkable
class ModelEvaluatorProtocol(Protocol):
    """Protocol for model evaluation components.

    Any class satisfying this protocol can be used by
    ``CrossValidationPipeline`` and ``CrossDatasetEvaluator``
    for model training and evaluation.
    """

    models: dict

    def train_single_fold(
        self,
        model_name: str,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        fold_id: int,
        fold_seed: int,
        inner_cv: Any,
        **kwargs: Any,
    ) -> "FoldMetrics": ...

    def evaluate_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        **kwargs: Any,
    ) -> Tuple[Dict[str, float], np.ndarray, Optional[float]]: ...


@runtime_checkable
class MetricAggregatorProtocol(Protocol):
    """Protocol for metric aggregation components."""

    def aggregate_scores(
        self, all_results: Dict[str, List[Any]]
    ) -> Dict[str, Dict[str, Dict[str, Any]]]: ...

    def concatenate_oof_predictions(
        self,
        all_results: Dict[str, List[Any]],
        config: Any,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]: ...

    def aggregate_threshold_results(
        self,
        all_results: Dict[str, List[Any]],
        dataset_name: str,
        output_dir: Any,
    ) -> None: ...


@runtime_checkable
class ReportGeneratorProtocol(Protocol):
    """Protocol for report generation components."""

    def generate_reports(
        self,
        results: Dict[str, List[Any]],
        dataset_name: str,
        feature_names: List[str],
        df: Any,
        caller_results: Optional[Dict[str, List[Any]]] = None,
    ) -> None: ...
