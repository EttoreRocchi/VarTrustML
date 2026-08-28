"""
Main pipeline orchestration for cross-validation experiments.

:class:`CrossValidationPipeline` runs nested cross-validation over several
models, with optional probability calibration, threshold optimization, and
variant caller comparison.

The pipeline implements a complete ML evaluation workflow:

1. Outer CV for model evaluation (unbiased performance estimates)
2. Inner CV for hyperparameter optimization (within each outer fold)
3. Optional probability calibration and threshold optimization
4. Bootstrap confidence intervals for all metrics
5. Report generation (CSV, HTML, visualizations)

See Also
--------
ExperimentConfig : Configuration dataclass for pipeline settings.
ModelEvaluator : Individual model training and evaluation.
BootstrapAnalyzer : Confidence interval computation.

Examples
--------
>>> from vartrustml import ExperimentConfig
>>> from vartrustml.config.experiment import CVConfig, CalibrationConfig
>>> from vartrustml.core.pipeline import CrossValidationPipeline
>>> config = ExperimentConfig(
...     cv=CVConfig(n_outer_splits=10),
...     calibration=CalibrationConfig(calibrate_models=True),
... )
>>> pipeline = CrossValidationPipeline(config)
>>> results, caller_results = pipeline.run_cross_validation(df, "my_dataset")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from vartrustml.config import ExperimentConfig, ModelConfig
from vartrustml.core.metric_aggregator import MetricAggregator
from vartrustml.core.pipeline_checkpoint import PipelineCheckpointManager
from vartrustml.core.protocols import (
    MetricAggregatorProtocol,
    ModelEvaluatorProtocol,
    ReportGeneratorProtocol,
)
from vartrustml.core.report_generator import ReportGenerator
from vartrustml.core.caller_evaluator import (
    CallerEvaluator,
    CallerResult,
    validate_caller_columns,
)
from vartrustml.core.models import ModelEvaluator
from vartrustml.utils.reproducibility import log_reproducibility_info, set_all_seeds
from vartrustml.utils.validation import validate_target_for_cv

if TYPE_CHECKING:
    from vartrustml.analysis.error_analysis import FoldMetrics
    from vartrustml.core.manifest import ManifestGenerator
    from vartrustml.visualization.plots import Visualizer

logger = logging.getLogger(__name__)


class CrossValidationPipeline:
    """Main pipeline for nested cross-validation experiments.

    Orchestrates the complete ML evaluation workflow including data splitting,
    model training with hyperparameter optimization, evaluation, and report
    generation. Supports checkpoint/resume for long-running experiments.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration controlling all pipeline behavior.
    model_config : ModelConfig, optional
        Model-specific configuration for hyperparameter search spaces.
        If None, uses default hyperparameter grids.
    evaluator : ModelEvaluator, optional
        Model training and evaluation handler. Created from config if None.
    visualizer : Visualizer, optional
        Visualization generation handler. Created from config if None.
    manifest_generator : ManifestGenerator, optional
        Output manifest generator. Created from config if None.
    metric_aggregator : MetricAggregator, optional
        Metric aggregation handler. Created from config if None.
    pipeline_checkpoint : PipelineCheckpointManager, optional
        Checkpoint manager. Created from config if None.
    report_generator : ReportGenerator, optional
        Report generation handler. Created from config if None.

    Attributes
    ----------
    config : ExperimentConfig
        The experiment configuration.
    model_config : ModelConfig or None
        Model-specific configuration.
    evaluator : ModelEvaluator
        Model training and evaluation handler.
    visualizer : Visualizer
        Visualization generation handler.

    See Also
    --------
    ExperimentConfig : Configuration for pipeline settings.
    ModelConfig : Configuration for model hyperparameters.
    ModelEvaluator : Individual model training logic.
    ReportGenerator : Report and visualization generation.

    Notes
    -----
    The pipeline implements nested cross-validation:

    - **Outer CV** (n_outer_splits): Provides unbiased performance estimates.
      Each outer fold uses a held-out test set never seen during training.
    - **Inner CV** (n_inner_splits): Optimizes hyperparameters within each
      outer fold's training set, preventing data leakage.

    Checkpointing saves results after each fold, allowing interrupted
    experiments to resume without recomputing completed folds.

    Examples
    --------
    Basic usage:

    >>> from vartrustml import ExperimentConfig
    >>> from vartrustml.config.experiment import CVConfig
    >>> from vartrustml.core.pipeline import CrossValidationPipeline
    >>> config = ExperimentConfig(cv=CVConfig(seed=42, n_outer_splits=10))
    >>> pipeline = CrossValidationPipeline(config)
    >>> results, _ = pipeline.run_cross_validation(df, "experiment_1")

    With model configuration and caller comparison:

    >>> from vartrustml.config import ModelConfig
    >>> from vartrustml.config.experiment import CallerComparisonConfig
    >>> config = ExperimentConfig(
    ...     caller_comparison=CallerComparisonConfig(
    ...         compare_callers=True,
    ...         caller_columns=["MANTA", "DELLY"],
    ...     ),
    ... )
    >>> model_config = ModelConfig()
    >>> pipeline = CrossValidationPipeline(config, model_config)
    >>> results, caller_results = pipeline.run_cross_validation(df, "with_callers")
    """

    def __init__(
        self,
        config: ExperimentConfig,
        model_config: Optional[ModelConfig] = None,
        evaluator: Optional[ModelEvaluatorProtocol] = None,
        visualizer: Optional["Visualizer"] = None,
        manifest_generator: Optional["ManifestGenerator"] = None,
        metric_aggregator: Optional[MetricAggregatorProtocol] = None,
        pipeline_checkpoint: Optional[PipelineCheckpointManager] = None,
        report_generator: Optional[ReportGeneratorProtocol] = None,
    ):
        self.config = config
        self.model_config = model_config
        self.evaluator = evaluator or ModelEvaluator(config, model_config)

        if visualizer is not None:
            self.visualizer = visualizer
        else:
            from vartrustml.visualization.plots import Visualizer

            self.visualizer = Visualizer(config)

        if manifest_generator is not None:
            self.manifest = manifest_generator
        else:
            from vartrustml.core.manifest import ManifestGenerator

            self.manifest = ManifestGenerator(config)
        self.metrics = metric_aggregator or MetricAggregator(
            bootstrap_n_iterations=config.bootstrap.bootstrap_n_iterations,
            bootstrap_ci_level=config.bootstrap.bootstrap_ci_level,
            seed=config.cv.seed,
            optimize_threshold=config.threshold.optimize_threshold,
            bootstrap_ci_method=config.bootstrap.bootstrap_ci_method,
        )
        self.checkpoint = pipeline_checkpoint or PipelineCheckpointManager(
            output_dir=config.output_dir,
            checkpoint_dir=config.checkpoint_dir,
            save_checkpoints=config.save_checkpoints,
        )
        self.report_generator = report_generator or ReportGenerator(
            config=config,
            visualizer=self.visualizer,
            metric_aggregator=self.metrics,
            error_analyzer=self.evaluator.error_analyzer,
        )

    def run_cross_validation(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        feature_names: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, List[FoldMetrics]], Optional[Dict[str, List[CallerResult]]]]:
        """Run complete nested cross-validation pipeline.

        Executes the full evaluation workflow: data validation, stratified
        k-fold splitting, model training with hyperparameter optimization,
        performance evaluation, and report generation.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataframe containing features and target column.
            The target column name is specified in ``config.target_column``.
        dataset_name : str
            Name of the dataset, used for organizing output directories
            and naming output files.
        feature_names : list of str, optional
            List of feature column names. If None, inferred from dataframe
            columns (excluding target column).

        Returns
        -------
        results : dict of {str: list of FoldMetrics}
            Dictionary mapping model names to lists of FoldMetrics objects,
            one per outer CV fold. Empty dict if validation fails.
        caller_results : dict of {str: list of CallerResult} or None
            Dictionary mapping caller/combination names to lists of
            CallerResult objects. None if ``compare_callers=False`` or
            validation fails.

        See Also
        --------
        FoldMetrics : Container for per-fold evaluation results.
        CallerResult : Container for variant caller evaluation results.

        Notes
        -----
        The method validates the target variable before proceeding:

        - Checks for binary labels (0/1)
        - Ensures sufficient samples per class for stratified splitting
        - Returns empty results if validation fails (with logged error)

        Output files are organized under ``{output_dir}/{dataset_name}/``.
        """
        logger.info(f"Starting cross-validation for dataset: {dataset_name}")
        logger.info(f"Dataset shape: {df.shape}")

        # Set all random seeds for reproducibility at the start of the pipeline
        set_all_seeds(self.config.cv.seed)

        # Log reproducibility info to output directory
        output_dir = Path(self.config.output_dir) / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        log_reproducibility_info(
            self.config.cv.seed, output_path=output_dir / "reproducibility_info.json"
        )

        X = df.drop(columns=[self.config.target_column])
        y = df[self.config.target_column]

        # Validate target variable
        is_valid, error_msg = validate_target_for_cv(
            y,
            self.config.cv.n_outer_splits,
            self.config.cv.n_inner_splits,
            dataset_name,
        )

        if not is_valid:
            logger.error(error_msg)
            logger.warning(f"Skipping cross-validation for {dataset_name}")
            return {}, None

        if feature_names is None:
            feature_names = X.columns.tolist()

        # Isolate this run's checkpoints: a different seed, split count, feature
        # set or dataset content must not resume another run's folds
        self.checkpoint.set_run_key(self.config, X, y)

        class_dist = y.value_counts()
        logger.info(f"Class distribution:\n{class_dist}")

        # Run model training across folds
        all_results = self._train_all_folds(X, y, dataset_name, feature_names)

        # Aggregate threshold optimization results if enabled
        if self.config.threshold.optimize_threshold:
            output_dir = Path(self.config.output_dir) / dataset_name
            self.metrics.aggregate_threshold_results(
                all_results, dataset_name, output_dir
            )

        # Evaluate callers on same folds if enabled
        caller_results = None
        if (
            self.config.caller_comparison.compare_callers
            and self.config.caller_comparison.caller_columns
        ):
            caller_results = self._evaluate_callers(df, dataset_name, X, y)

        # Generate reports and visualizations
        self.report_generator.generate_reports(
            all_results, dataset_name, feature_names, df, caller_results
        )

        output_dir = Path(self.config.output_dir) / dataset_name
        self.visualizer.plot_metrics_comparison(all_results, output_dir)

        # Generate manifest.json for output tracking
        self.manifest.generate(output_dir, dataset_name)

        return all_results, caller_results

    def _train_all_folds(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dataset_name: str,
        feature_names: List[str],
    ) -> Dict[str, List[FoldMetrics]]:
        """Train all models across all outer CV folds."""
        outer_cv = StratifiedKFold(
            n_splits=self.config.cv.n_outer_splits,
            shuffle=True,
            random_state=self.config.cv.seed,
        )
        inner_cv = StratifiedKFold(
            n_splits=self.config.cv.n_inner_splits,
            shuffle=True,
            random_state=self.config.cv.seed,
        )

        all_results = {model_name: [] for model_name in self.evaluator.models.keys()}

        # Generate independent fold seeds
        fold_seed_rng = np.random.default_rng(self.config.cv.seed)
        fold_seeds = fold_seed_rng.integers(
            0, 2**31, size=self.config.cv.n_outer_splits
        ).tolist()

        for model_name, model in self.evaluator.models.items():
            logger.info(f"\nTraining model: {model_name}")

            fold_iterator = list(enumerate(outer_cv.split(X, y)))

            if self.config.verbose > 0:
                fold_iterator = tqdm(
                    fold_iterator,
                    total=self.config.cv.n_outer_splits,
                    desc=f"{model_name}",
                )

            for fold_id, (train_idx, test_idx) in fold_iterator:
                # Check if checkpoint exists
                if (
                    self.checkpoint.save_checkpoints
                    and self.checkpoint.checkpoint_exists(
                        dataset_name, model_name, fold_id
                    )
                ):
                    fold_result = self.checkpoint.load_checkpoint(
                        dataset_name, model_name, fold_id
                    )
                    if fold_result is not None:
                        all_results[model_name].append(fold_result)
                        logger.info(f"Skipping fold {fold_id} - using checkpoint")
                        continue

                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # Prepare checkpoint directory for model files
                model_checkpoint_dir = None
                if self.checkpoint.save_checkpoints:
                    model_checkpoint_dir = self.checkpoint.get_fold_dir(
                        dataset_name, model_name, fold_id
                    )

                fold_result = self.evaluator.train_single_fold(
                    model_name=model_name,
                    model=model,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    fold_id=fold_id,
                    fold_seed=fold_seeds[fold_id],
                    inner_cv=inner_cv,
                    feature_names=feature_names,
                    save_checkpoint=self.checkpoint.save_checkpoints,
                    checkpoint_dir=model_checkpoint_dir,
                )

                all_results[model_name].append(fold_result)

                # Save fold result checkpoint
                if self.checkpoint.save_checkpoints:
                    self.checkpoint.save_checkpoint(
                        dataset_name, model_name, fold_id, fold_result
                    )

        return all_results

    def _evaluate_callers(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict[str, List[CallerResult]]:
        """Evaluate variant callers on identical CV folds as ML models.

        Ensures fair comparison by using the same train/test splits
        used for ML model evaluation.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataframe containing caller prediction columns.
        dataset_name : str
            Name of dataset for logging and output organization.
        X : pandas.DataFrame
            Feature matrix (used to regenerate fold indices).
        y : pandas.Series
            Target variable (used to regenerate fold indices).

        Returns
        -------
        dict of {str: list of CallerResult}
            Dictionary mapping caller/combination names to lists of
            CallerResult objects, one per fold.
        """
        logger.info("\nEvaluating variant callers...")

        # Validate caller columns
        caller_columns = self.config.caller_comparison.caller_columns
        assert caller_columns is not None, (
            "caller_columns must be set when evaluating callers"
        )
        assert self.config.target_column is not None, (
            "target_column must be set when evaluating callers"
        )
        validate_caller_columns(df, caller_columns, self.config.target_column)

        # Create CallerConfig with combinations
        from vartrustml.config.caller import CallerConfig

        caller_config = CallerConfig.from_experiment_config(
            caller_columns=caller_columns,
            caller_combinations=self.config.caller_comparison.caller_combinations,
            include_default_combinations=self.config.caller_comparison.include_default_combinations,
        )

        # Regenerate fold indices (same seed ensures same splits)
        outer_cv = StratifiedKFold(
            n_splits=self.config.cv.n_outer_splits,
            shuffle=True,
            random_state=self.config.cv.seed,
        )
        fold_indices = list(outer_cv.split(X, y))

        # Initialize evaluator
        evaluator = CallerEvaluator(caller_columns)
        caller_data = df[caller_columns]

        all_caller_results: Dict[str, List[CallerResult]] = {}

        # Evaluate individual callers
        for caller in caller_columns:
            logger.info(f"Evaluating caller: {caller}")
            caller_results = []

            for fold_id, (train_idx, test_idx) in enumerate(fold_indices):
                y_test = y.iloc[test_idx].values
                caller_pred = caller_data[caller].iloc[test_idx].values

                result = evaluator.evaluate_single_caller(
                    caller, y_test, caller_pred, fold_id
                )
                result.sample_indices = df.index[test_idx].values
                caller_results.append(result)

            all_caller_results[caller] = caller_results

        # Evaluate combinations
        for combo_expr in caller_config.combinations:
            logger.info(f"Evaluating combination: {combo_expr}")
            combo_results = []

            for fold_id, (train_idx, test_idx) in enumerate(fold_indices):
                y_test = y.iloc[test_idx].values
                test_caller_data = caller_data.iloc[test_idx]

                result = evaluator.evaluate_from_expression(
                    combo_expr, y_test, test_caller_data, fold_id
                )
                result.sample_indices = df.index[test_idx].values
                combo_results.append(result)

            all_caller_results[combo_expr] = combo_results

        # Save caller results
        self.report_generator.save_caller_results(all_caller_results, dataset_name)

        return all_caller_results

    def _concatenate_oof_predictions(
        self,
        fold_results: List[FoldMetrics],
        use_optimized_threshold: bool = False,
    ) -> Tuple[
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """Delegate to MetricAggregator for OOF prediction concatenation."""
        return self.metrics.concatenate_oof_predictions(
            fold_results, use_optimized_threshold
        )

    def _aggregate_metrics(self, fold_results: List[FoldMetrics]) -> pd.DataFrame:
        """Delegate to MetricAggregator for metric aggregation."""
        return self.metrics.aggregate_metrics(fold_results)
