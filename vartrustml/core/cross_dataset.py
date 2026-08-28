"""
Cross-dataset evaluation module for VarTrustML.

Trains on one dataset and tests on the others, over aligned cross-validation
splits, to estimate how far performance carries between datasets.

Classes
-------
CrossDatasetEvaluator
    Evaluator for cross-dataset generalizability experiments.

See Also
--------
vartrustml.core.pipeline.CrossValidationPipeline : Single-dataset evaluation.
vartrustml.io.data_loader.DataLoader : Dataset loading utilities.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from vartrustml.config import ExperimentConfig, ModelConfig
from vartrustml.core.cross_dataset_reporting import generate_cross_dataset_summary
from vartrustml.core.cross_dataset_visualization import (
    create_comparison_plots,
    plot_heatmap_with_uncertainty,
)
from vartrustml.core.metrics import CROSS_DATASET_METRICS
from vartrustml.core.models import ModelEvaluator
from vartrustml.core.pipeline_checkpoint import compute_run_key, data_fingerprint
from vartrustml.core.protocols import ModelEvaluatorProtocol
from vartrustml.utils.reproducibility import set_all_seeds
from vartrustml.utils.validation import validate_target_for_cv

logger = logging.getLogger(__name__)


class CrossDatasetEvaluator:
    """Evaluator for cross-dataset generalizability experiments.

    Trains models on each dataset and tests on all others using
    aligned outer cross-validation splits to assess how well models
    generalize across different data distributions.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration containing CV settings, models to use,
        and output paths.
    model_config : ModelConfig, optional
        Model-specific hyperparameter configuration.

    Attributes
    ----------
    config : ExperimentConfig
        Experiment configuration.
    model_config : ModelConfig or None
        Model-specific hyperparameter configuration.
    evaluator : ModelEvaluator
        Model evaluator instance for training and evaluation.
    METRICS_TO_TRACK : list of str
        Class attribute listing metrics tracked during evaluation.

    See Also
    --------
    CrossValidationPipeline : Single-dataset nested CV evaluation.
    DataLoader.validate_datasets_compatibility : Validate datasets before evaluation.
    HTMLCrossDatasetReporter : Generate interactive cross-dataset reports.

    Examples
    --------
    >>> from vartrustml import ExperimentConfig, CrossDatasetEvaluator, DataLoader
    >>> config = ExperimentConfig(n_outer_splits=5, models_to_use=["XGBoost"])
    >>> evaluator = CrossDatasetEvaluator(config)
    >>> loader = DataLoader("data/")
    >>> datasets = [
    ...     (loader.load_dataset("dataset1.csv"), "Dataset1"),
    ...     (loader.load_dataset("dataset2.csv"), "Dataset2"),
    ... ]
    >>> results = evaluator.evaluate_cross_dataset(datasets)
    >>> # Results contain performance matrices per model and metric
    """

    # Metrics to track in cross-dataset evaluation
    METRICS_TO_TRACK = CROSS_DATASET_METRICS

    def __init__(
        self,
        config: ExperimentConfig,
        model_config: Optional[ModelConfig] = None,
        evaluator: Optional[ModelEvaluatorProtocol] = None,
    ):
        self.config = config
        self.model_config = model_config
        self.evaluator = evaluator or ModelEvaluator(config, model_config)
        self.run_key = "unkeyed"

    def evaluate_cross_dataset(
        self,
        datasets: List[Tuple[pd.DataFrame, str]],
        cv_scheme: str = "pairwise",
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Run cross-dataset evaluation with aligned outer CV splits.

        For each source dataset, trains models on K-fold splits and
        evaluates on the corresponding test folds of all target datasets.
        This estimates how far performance carries to an unseen dataset.

        Parameters
        ----------
        datasets : list of tuple
            List of (dataframe, name) tuples. Each dataframe must
            contain the target column specified in config.

        Returns
        -------
        dict of str to dict of str to pandas.DataFrame
            Nested dictionary: model_name -> metric_name -> DataFrame.
            Each DataFrame has shape (n_datasets, n_datasets) with
            rows=training datasets and columns=test datasets.

        Raises
        ------
        ValueError
            If fewer than 2 valid datasets remain after validation.
        """
        logger.info("Starting cross-dataset evaluation with aligned outer CV")

        # Reset RNG for reproducibility (matches compare-models behavior)
        set_all_seeds(self.config.cv.seed)

        logger.info(f"Number of datasets: {len(datasets)}")
        logger.info(
            f"CV settings: {self.config.cv.n_outer_splits} outer folds, "
            f"{self.config.cv.n_inner_splits} inner folds"
        )

        # Validate all datasets
        valid_datasets = self._validate_datasets(datasets)

        if len(valid_datasets) < 2:
            logger.error("Cross-dataset evaluation requires at least 2 valid datasets")
            raise ValueError("Insufficient valid datasets for cross-dataset evaluation")

        if len(valid_datasets) < len(datasets):
            logger.warning(
                f"Proceeding with {len(valid_datasets)}/{len(datasets)} valid datasets"
            )

        datasets = valid_datasets
        dataset_names = [name for _, name in datasets]

        # Isolate this run's checkpoints from runs with different settings or
        # different dataset content
        self.run_key = compute_run_key(
            self.config,
            extra={
                "cv_scheme": cv_scheme,
                "datasets": {name: data_fingerprint(df) for df, name in datasets},
            },
        )
        logger.info(f"Checkpoint run key: {self.run_key}")

        # Initialize result structures
        results, results_std = self._initialize_result_matrices(dataset_names)

        # Prepare aligned CV splits for all datasets
        outer_splits = self._prepare_cv_splits(datasets)

        # Run cross-dataset evaluation
        results, results_std, threshold_results = self._run_evaluation_loop(
            datasets, dataset_names, outer_splits, results, results_std
        )

        # Save all results and generate reports
        self._save_results(results, results_std, dataset_names, threshold_results)

        out_dir = Path(self.config.output_dir)
        self.generalization_gap = self._compute_generalization_gap(dataset_names)
        if not self.generalization_gap.empty:
            out_dir.mkdir(parents=True, exist_ok=True)
            self.generalization_gap.to_csv(
                out_dir / "generalization_gap.csv", index=False
            )

        self.lodo_results = None
        if cv_scheme in ("lodo", "both") and len(datasets) >= 3:
            self.lodo_results = self._run_lodo(datasets, dataset_names, outer_splits)
            if self.lodo_results is not None and not self.lodo_results.empty:
                out_dir.mkdir(parents=True, exist_ok=True)
                self.lodo_results.to_csv(out_dir / "lodo_results.csv", index=False)

        return results

    @staticmethod
    def _bootstrap_mean_ci(values, n_iterations, ci_level, seed):
        """Percentile bootstrap CI of the mean of a 1D array (resampling elements)."""
        v = np.asarray(values, dtype=float)
        v = v[~np.isnan(v)]
        if v.size == 0:
            return float("nan"), float("nan"), float("nan")
        point = float(v.mean())
        if v.size == 1:
            return point, point, point
        rng = np.random.default_rng(seed)
        means = v[rng.integers(0, v.size, size=(n_iterations, v.size))].mean(axis=1)
        lo = float(np.percentile(means, 100 * (1 - ci_level) / 2))
        hi = float(np.percentile(means, 100 * (1 + ci_level) / 2))
        return point, lo, hi

    def _compute_generalization_gap(self, dataset_names):
        """Per-source gap (in-sample minus cross-sample) with bootstrap CI over folds."""
        metric = self.config.model_comparison_metric
        fv = getattr(self, "_fold_values", {})
        n_boot = self.config.bootstrap.bootstrap_n_iterations
        ci = self.config.bootstrap.bootstrap_ci_level
        seed = self.config.cv.seed
        rows = []
        for model, by_metric in fv.items():
            mm = by_metric.get(metric)
            if not mm:
                continue
            for src in dataset_names:
                if src not in mm or src not in mm[src]:
                    continue
                in_by_fold = mm[src][src]
                targets = [t for t in dataset_names if t != src and t in mm[src]]
                if not in_by_fold or not targets:
                    continue

                # Pair by fold id, never by list position: a fold that failed
                # for one cell is simply absent there, and pairing positionally
                # would subtract different folds from each other
                shared = set(in_by_fold)
                for tgt in targets:
                    shared &= set(mm[src][tgt])
                fold_ids = sorted(shared)
                if not fold_ids:
                    continue

                in_folds = np.array([in_by_fold[f] for f in fold_ids], dtype=float)
                cross = (
                    np.vstack([[mm[src][tgt][f] for f in fold_ids] for tgt in targets])
                    .astype(float)
                    .mean(axis=0)
                )
                gap_folds = in_folds - cross
                point, lo, hi = self._bootstrap_mean_ci(gap_folds, n_boot, ci, seed)
                rows.append(
                    {
                        "model": model,
                        "source": src,
                        "metric": metric,
                        "n_folds": len(fold_ids),
                        "in_sample": float(np.nanmean(in_folds)),
                        "cross_sample": float(np.nanmean(cross)),
                        "gap": point,
                        "gap_ci_lower": lo,
                        "gap_ci_upper": hi,
                    }
                )
        return pd.DataFrame(rows)

    def _run_lodo(self, datasets, dataset_names, outer_splits):
        """Leave-one-dataset-out: train on the N-1 pooled samples, test on held-out.

        Reuses the aligned per-fold machinery (pooled source vs held-out target) and
        reports the primary metric with a bootstrap CI over folds, plus the pairwise
        cross-sample mean for the same held-out sample.
        """
        metric = self.config.model_comparison_metric
        target = self.config.target_column
        n_boot = self.config.bootstrap.bootstrap_n_iterations
        ci = self.config.bootstrap.bootstrap_ci_level
        seed = self.config.cv.seed
        fv = getattr(self, "_fold_values", {})
        rows = []
        for h_idx, (h_df, h_name) in enumerate(datasets):
            others = [datasets[k][0] for k in range(len(datasets)) if k != h_idx]
            if not others:
                continue
            pooled_df = pd.concat(others, ignore_index=True)
            pooled_splits = self._prepare_cv_splits([(pooled_df, "__pooled__")])
            splits = dict(outer_splits)
            splits["__pooled__"] = pooled_splits["__pooled__"]
            X_pooled = pooled_df.drop(columns=[target])
            y_pooled = pooled_df[target]
            common = self._get_common_features(X_pooled, h_df)
            cv_results, _ = self._evaluate_dataset_pair(
                "__pooled__",
                h_name,
                X_pooled[common],
                y_pooled,
                h_df[common],
                h_df[target],
                splits,
            )
            for model_name, metric_vals in cv_results.items():
                by_fold = metric_vals.get(metric, {})
                folds = np.asarray([by_fold[f] for f in sorted(by_fold)], dtype=float)
                folds = folds[~np.isnan(folds)]
                if folds.size == 0:
                    continue
                point, lo, hi = self._bootstrap_mean_ci(folds, n_boot, ci, seed)
                mm = fv.get(model_name, {}).get(metric, {})
                cross = [
                    float(np.nanmean(list(mm[s][h_name].values())))
                    for s in dataset_names
                    if s != h_name and s in mm and h_name in mm.get(s, {})
                ]
                pairwise_cross = float(np.mean(cross)) if cross else float("nan")
                rows.append(
                    {
                        "model": model_name,
                        "held_out": h_name,
                        "lodo": point,
                        "lodo_ci_lower": lo,
                        "lodo_ci_upper": hi,
                        "pairwise_cross": pairwise_cross,
                        "delta": (point - pairwise_cross) if cross else float("nan"),
                    }
                )
        return pd.DataFrame(rows)

    def _validate_datasets(
        self, datasets: List[Tuple[pd.DataFrame, str]]
    ) -> List[Tuple[pd.DataFrame, str]]:
        """Validate datasets for cross-validation compatibility.

        Parameters
        ----------
        datasets : list of tuple
            List of (dataframe, name) tuples.

        Returns
        -------
        list of tuple
            List of valid (dataframe, name) tuples.
        """
        valid_datasets = []

        for df, name in datasets:
            y = df[self.config.target_column]

            is_valid, error_msg = validate_target_for_cv(
                y, self.config.cv.n_outer_splits, self.config.cv.n_inner_splits, name
            )

            if not is_valid:
                logger.error(error_msg)
                logger.warning(f"Skipping dataset {name} from cross-dataset evaluation")
            else:
                valid_datasets.append((df, name))
                logger.info(
                    f"Dataset {name} validated: {len(df)} samples, "
                    f"class distribution: {y.value_counts().to_dict()}"
                )

        return valid_datasets

    def _initialize_result_matrices(
        self, dataset_names: List[str]
    ) -> Tuple[Dict[str, Dict[str, pd.DataFrame]], Dict[str, Dict[str, pd.DataFrame]]]:
        """Initialize empty result matrices for all models and metrics.

        Parameters
        ----------
        dataset_names : list of str
            List of dataset names.

        Returns
        -------
        results : dict
            Empty matrices for mean values.
        results_std : dict
            Empty matrices for standard deviation values.
        """
        results: Dict[str, Dict[str, pd.DataFrame]] = {
            model_name: {
                metric: pd.DataFrame(
                    index=dataset_names, columns=dataset_names, dtype=float
                )
                for metric in self.METRICS_TO_TRACK
            }
            for model_name in self.evaluator.models.keys()
        }

        results_std: Dict[str, Dict[str, pd.DataFrame]] = {
            model_name: {
                metric: pd.DataFrame(
                    index=dataset_names, columns=dataset_names, dtype=float
                )
                for metric in self.METRICS_TO_TRACK
            }
            for model_name in self.evaluator.models.keys()
        }

        return results, results_std

    def _prepare_cv_splits(
        self, datasets: List[Tuple[pd.DataFrame, str]]
    ) -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
        """Prepare aligned CV splits for all datasets.

        Uses the same random seed to ensure aligned splits across datasets.

        Parameters
        ----------
        datasets : list of tuple
            List of (dataframe, name) tuples.

        Returns
        -------
        dict of str to list of tuple
            Maps dataset names to lists of (train_idx, test_idx) tuples.
        """
        outer_cv = StratifiedKFold(
            n_splits=self.config.cv.n_outer_splits,
            shuffle=True,
            random_state=self.config.cv.seed,
        )

        outer_splits: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}

        for df, name in datasets:
            y = df[self.config.target_column]
            X = df.drop(columns=[self.config.target_column])
            outer_splits[name] = list(outer_cv.split(X, y))
            logger.debug(f"Prepared {len(outer_splits[name])} folds for {name}")

        return outer_splits

    def _run_evaluation_loop(
        self,
        datasets: List[Tuple[pd.DataFrame, str]],
        dataset_names: List[str],
        outer_splits: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
        results: Dict[str, Dict[str, pd.DataFrame]],
        results_std: Dict[str, Dict[str, pd.DataFrame]],
    ) -> Tuple[
        Dict[str, Dict[str, pd.DataFrame]],
        Dict[str, Dict[str, pd.DataFrame]],
        Dict[str, Dict[str, List[float]]],
    ]:
        """Run the main evaluation loop across all dataset pairs.

        Uses per-fold threshold optimization (like compare-models) so that the
        diagonal values match compare-models exactly.

        Parameters
        ----------
        datasets : list of tuple
            List of (dataframe, name) tuples.
        dataset_names : list of str
            List of dataset names.
        outer_splits : dict
            Pre-computed CV splits.
        results : dict
            Results matrices to populate.
        results_std : dict
            Standard deviation matrices to populate.

        Returns
        -------
        tuple
            Updated (results, results_std, threshold_results) tuple.
            threshold_results maps source_name -> model_name -> list of per-fold thresholds.
        """
        # Initialize threshold results storage (per-fold thresholds)
        threshold_results: Dict[str, Dict[str, List[float]]] = {}
        # Per-fold metric values keyed model -> metric -> source -> target
        fold_values: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = {}

        for i, (src_df, src_name) in enumerate(
            tqdm(datasets, desc="Training datasets")
        ):
            logger.info(f"\n{'=' * 60}\nSource dataset: {src_name}")

            X_src_full = src_df.drop(columns=[self.config.target_column])
            y_src_full = src_df[self.config.target_column]

            # Initialize per-fold threshold storage for this source
            if self.config.threshold.optimize_threshold:
                threshold_results[src_name] = {
                    model_name: [] for model_name in self.evaluator.models.keys()
                }

            for j, (tgt_df, tgt_name) in enumerate(
                tqdm(datasets, desc="Test datasets", leave=False)
            ):
                # Align features between source and target datasets
                common_features = self._get_common_features(X_src_full, tgt_df)

                X_src_aligned = X_src_full[common_features]
                X_tgt_aligned_full = tgt_df[common_features]
                y_tgt_full = tgt_df[self.config.target_column]

                # Collect CV results for this dataset pair
                # Per-fold threshold optimization happens inside _evaluate_dataset_pair
                cv_results, fold_thresholds = self._evaluate_dataset_pair(
                    src_name,
                    tgt_name,
                    X_src_aligned,
                    y_src_full,
                    X_tgt_aligned_full,
                    y_tgt_full,
                    outer_splits,
                )

                # Store per-fold thresholds (only once per source, using diagonal)
                if (
                    self.config.threshold.optimize_threshold
                    and src_name == tgt_name
                    and fold_thresholds
                ):
                    for model_name, thresholds in fold_thresholds.items():
                        threshold_results[src_name][model_name] = thresholds

                # Aggregate and store results
                self._aggregate_and_store_results(
                    cv_results, results, results_std, src_name, tgt_name
                )

                for model_name, metric_vals in cv_results.items():
                    for metric_name, vals in metric_vals.items():
                        (
                            fold_values.setdefault(model_name, {})
                            .setdefault(metric_name, {})
                            .setdefault(src_name, {})[tgt_name]
                        ) = dict(vals)

        self._fold_values = fold_values
        return results, results_std, threshold_results

    def _get_common_features(
        self, X_src: pd.DataFrame, tgt_df: pd.DataFrame
    ) -> List[str]:
        """Get common features between source and target datasets.

        Parameters
        ----------
        X_src : pandas.DataFrame
            Source features DataFrame.
        tgt_df : pandas.DataFrame
            Target DataFrame (including target column).

        Returns
        -------
        list of str
            Common feature names in the original source column order.
        """
        tgt_feature_set = set(tgt_df.columns)
        common_features = [col for col in X_src.columns if col in tgt_feature_set]

        if len(common_features) < len(X_src.columns):
            logger.warning(
                f"Feature mismatch: source has {len(X_src.columns)} features, "
                f"target has {len(tgt_df.columns)} - intersection: {len(common_features)}"
            )

        return common_features

    def _evaluate_dataset_pair(
        self,
        src_name: str,
        tgt_name: str,
        X_src: pd.DataFrame,
        y_src: pd.Series,
        X_tgt: pd.DataFrame,
        y_tgt: pd.Series,
        outer_splits: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
    ) -> Tuple[Dict[str, Dict[str, List[float]]], Dict[str, List[float]]]:
        """Evaluate all models on a source-target dataset pair.

        Uses per-fold threshold optimization (like compare-models) when enabled.

        Parameters
        ----------
        src_name : str
            Name of source (training) dataset.
        tgt_name : str
            Name of target (test) dataset.
        X_src : pandas.DataFrame
            Source features.
        y_src : pandas.Series
            Source labels.
        X_tgt : pandas.DataFrame
            Target features.
        y_tgt : pandas.Series
            Target labels.
        outer_splits : dict
            Pre-computed CV splits.

        Returns
        -------
        cv_results : dict of str to dict of str to dict of int to float
            Maps model names to metric names to ``{fold_id: value}``. Keying by
            fold id keeps the values alignable across cells even when a fold
            fails for one model and is therefore absent.
        fold_thresholds : dict of str to list of float
            Maps model names to lists of per-fold thresholds (if threshold
            optimization is enabled, otherwise empty dict).
        """
        cv_results: Dict[str, Dict[str, Dict[int, float]]] = {
            model_name: {metric: {} for metric in self.METRICS_TO_TRACK}
            for model_name in self.evaluator.models.keys()
        }

        # Track per-fold thresholds for each model
        fold_thresholds: Dict[str, List[float]] = {
            model_name: [] for model_name in self.evaluator.models.keys()
        }

        for fold_idx in tqdm(
            range(self.config.cv.n_outer_splits), desc="CV folds", leave=False
        ):
            src_train_idx, _src_test_idx = outer_splits[src_name][fold_idx]
            _tgt_train_idx, tgt_test_idx = outer_splits[tgt_name][fold_idx]

            X_train = X_src.iloc[src_train_idx]
            y_train = y_src.iloc[src_train_idx]
            X_test = X_tgt.iloc[tgt_test_idx]
            y_test = y_tgt.iloc[tgt_test_idx]

            for model_name in tqdm(
                self.evaluator.models.keys(), desc="Models", leave=False
            ):
                try:
                    metrics, threshold = self._train_and_evaluate_fold(
                        model_name,
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        src_name,
                        tgt_name,
                        fold_idx,
                    )

                    for metric_name in self.METRICS_TO_TRACK:
                        if metric_name in metrics:
                            cv_results[model_name][metric_name][fold_idx] = metrics[
                                metric_name
                            ]

                    # Store the threshold used for this fold
                    if threshold is not None:
                        fold_thresholds[model_name].append(threshold)

                except Exception as e:
                    logger.error(
                        f"Error in fold {fold_idx} (train: {src_name}, test: {tgt_name}) "
                        f"for model {model_name}: {e}"
                    )

        return cv_results, fold_thresholds

    def _train_and_evaluate_fold(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        src_name: str,
        tgt_name: str,
        fold_idx: int,
    ) -> Tuple[Dict[str, float], Optional[float]]:
        """Train a model on one fold and evaluate on test data.

        Uses per-fold threshold optimization (like compare-models) when enabled.

        Parameters
        ----------
        model_name : str
            Name of the model to train.
        X_train : pandas.DataFrame
            Training features.
        y_train : pandas.Series
            Training labels.
        X_test : pandas.DataFrame
            Test features.
        y_test : pandas.Series
            Test labels.
        src_name : str
            Source dataset name (for checkpointing).
        tgt_name : str
            Target dataset name (for checkpointing).
        fold_idx : int
            Fold index (for checkpointing).

        Returns
        -------
        metrics : dict of str to float
            Dictionary of metric values.
        threshold : float or None
            The threshold used for this fold (if threshold optimization is
            enabled), otherwise None.
        """
        # Setup checkpoint path if saving is enabled
        checkpoint_path = None
        if self.config.save_checkpoints:
            checkpoint_dir = (
                Path(self.config.output_dir)
                / Path(self.config.checkpoint_dir)
                / self.run_key
                / f"{src_name}_to_{tgt_name}"
                / model_name.replace(" ", "_")
                / f"fold_{fold_idx}"
            )
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"fold_{fold_idx}_model.joblib"

        metrics, y_prob, threshold_used = self.evaluator.evaluate_model(
            model_name,
            X_train,
            y_train,
            X_test,
            y_test,
            n_splits=self.config.cv.n_inner_splits,
            save_checkpoint=self.config.save_checkpoints,
            checkpoint_path=checkpoint_path,
            fold_idx=fold_idx,
        )

        return metrics, threshold_used

    def _aggregate_and_store_results(
        self,
        cv_results: Dict[str, Dict[str, Dict[int, float]]],
        results: Dict[str, Dict[str, pd.DataFrame]],
        results_std: Dict[str, Dict[str, pd.DataFrame]],
        src_name: str,
        tgt_name: str,
    ):
        """Aggregate CV fold results and store in result matrices.

        Parameters
        ----------
        cv_results : dict
            Raw CV results for each model/metric.
        results : dict
            Results matrices to update with mean values.
        results_std : dict
            Std matrices to update with std values.
        src_name : str
            Source dataset name.
        tgt_name : str
            Target dataset name.
        """
        expected_folds = self.config.cv.n_outer_splits
        for model_name in self.evaluator.models.keys():
            for metric_name in self.METRICS_TO_TRACK:
                by_fold = cv_results[model_name][metric_name]
                if not by_fold:
                    continue

                values = [by_fold[fold_id] for fold_id in sorted(by_fold)]
                if len(values) < expected_folds:
                    logger.warning(
                        f"{model_name} - {metric_name} ({src_name} -> {tgt_name}): "
                        f"averaging {len(values)}/{expected_folds} folds; the "
                        f"missing folds failed and were skipped"
                    )

                mean_val = float(np.mean(values))
                std_val = float(np.std(values))

                results[model_name][metric_name].loc[src_name, tgt_name] = mean_val
                results_std[model_name][metric_name].loc[src_name, tgt_name] = std_val

                logger.debug(
                    f"  {model_name} - {metric_name}: {mean_val:.3f} (+/-{std_val:.3f})"
                )

    def _save_results(
        self,
        results: Dict[str, Dict[str, pd.DataFrame]],
        results_std: Dict[str, Dict[str, pd.DataFrame]],
        dataset_names: List[str],
        threshold_results: Optional[Dict[str, Dict[str, List[float]]]] = None,
    ):
        """Save all cross-dataset results, plots, and reports.

        Parameters
        ----------
        results : dict
            Mean results matrices.
        results_std : dict
            Std results matrices.
        dataset_names : list of str
            List of dataset names.
        threshold_results : dict, optional
            Maps source_name -> model_name -> list of per-fold thresholds.
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save threshold optimization results if present
        if threshold_results:
            # Per-fold thresholds are already simple lists, save directly
            threshold_path = output_dir / "threshold_optimization.joblib"
            joblib.dump(threshold_results, threshold_path)
            logger.info(f"Per-fold threshold results saved to {threshold_path}")

        # Save CSV matrices per model
        for model_name, model_results in results.items():
            model_dir = output_dir / model_name.replace(" ", "_")
            model_dir.mkdir(parents=True, exist_ok=True)

            for metric_name, matrix in model_results.items():
                safe_metric = metric_name.replace(" ", "_").lower()
                matrix.to_csv(model_dir / f"{safe_metric}_mean.csv")

                std_matrix = results_std[model_name][metric_name]
                std_matrix.to_csv(model_dir / f"{safe_metric}_std.csv")

                # Generate heatmap
                self._plot_heatmap_with_uncertainty(
                    matrix,
                    std_matrix,
                    model_name,
                    metric_name,
                    dataset_names,
                    model_dir,
                )

        # Generate comparison plots
        self._create_comparison_plots(results, results_std, dataset_names, output_dir)

        # Generate summary report
        self._generate_summary_report(
            results, results_std, dataset_names, output_dir, threshold_results
        )

    def _plot_heatmap_with_uncertainty(
        self,
        mean_matrix: pd.DataFrame,
        std_matrix: pd.DataFrame,
        model_name: str,
        metric_name: str,
        dataset_names: List[str],
        output_dir: Path,
    ):
        """Create a heatmap showing mean +/- std for a model/metric.

        Delegates to :func:`cross_dataset_visualization.plot_heatmap_with_uncertainty`.
        """
        plot_heatmap_with_uncertainty(
            mean_matrix=mean_matrix,
            std_matrix=std_matrix,
            model_name=model_name,
            metric_name=metric_name,
            dataset_names=dataset_names,
            output_dir=output_dir,
            n_outer_splits=self.config.cv.n_outer_splits,
            figure_dpi=self.config.visualization.figure_dpi,
        )

    def _create_comparison_plots(
        self,
        results: Dict[str, Dict[str, pd.DataFrame]],
        results_std: Dict[str, Dict[str, pd.DataFrame]],
        dataset_names: List[str],
        output_dir: Path,
    ):
        """Create plots comparing same-dataset vs cross-dataset performance.

        Delegates to :func:`cross_dataset_visualization.create_comparison_plots`.
        """
        create_comparison_plots(
            results=results,
            results_std=results_std,
            dataset_names=dataset_names,
            output_dir=output_dir,
            figure_dpi=self.config.visualization.figure_dpi,
        )

    def _generate_summary_report(
        self,
        results: Dict[str, Dict[str, pd.DataFrame]],
        results_std: Dict[str, Dict[str, pd.DataFrame]],
        dataset_names: List[str],
        output_dir: Path,
        threshold_results: Optional[Dict[str, Dict[str, List[float]]]] = None,
    ):
        """Generate the text and JSON summary reports.

        Delegates to :func:`cross_dataset_reporting.generate_cross_dataset_summary`.
        """
        generate_cross_dataset_summary(
            results=results,
            results_std=results_std,
            dataset_names=dataset_names,
            output_dir=output_dir,
            metrics_to_track=self.METRICS_TO_TRACK,
            n_outer_splits=self.config.cv.n_outer_splits,
            n_inner_splits=self.config.cv.n_inner_splits,
            optimize_threshold=self.config.threshold.optimize_threshold,
            threshold_method=self.config.threshold.threshold_method,
            threshold_auto_n_samples=self.config.threshold_auto_n_samples,
            threshold_results=threshold_results,
            bootstrap_n_iterations=self.config.bootstrap.bootstrap_n_iterations,
            bootstrap_ci_level=self.config.bootstrap.bootstrap_ci_level,
            seed=self.config.cv.seed,
        )
