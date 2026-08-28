"""Pipeline-level checkpoint management for fold results.

Handles saving and loading of per-fold FoldMetrics objects to enable
resuming interrupted cross-validation experiments.

Checkpoints are stored under a run key: a digest of every setting that
determines what a fold contains (seed, split counts, feature roles, missing
value strategy, model list, calibration, threshold and HPO settings) together
with a fingerprint of the data itself. A run whose settings or data differ
therefore looks at a different directory and recomputes, instead of silently
reusing folds that belong to a different experiment.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd

from vartrustml.analysis.error_analysis import FoldMetrics

logger = logging.getLogger(__name__)

#: Length of the hexadecimal run key used as a directory name.
RUN_KEY_LENGTH = 12


def data_fingerprint(X: pd.DataFrame, y: Optional[pd.Series] = None) -> str:
    """Content hash of a feature matrix and, optionally, its labels."""
    digest = hashlib.sha256()
    digest.update(str(X.shape).encode())
    digest.update(",".join(str(c) for c in X.columns).encode())
    digest.update(pd.util.hash_pandas_object(X, index=True).values.tobytes())
    if y is not None:
        digest.update(
            pd.util.hash_pandas_object(pd.Series(y), index=True).values.tobytes()
        )
    return digest.hexdigest()


def compute_run_key(
    config: Any,
    X: Optional[pd.DataFrame] = None,
    y: Optional[pd.Series] = None,
    extra: Optional[dict] = None,
) -> str:
    """Digest of the settings and data that determine every fold's content.

    Parameters
    ----------
    config : ExperimentConfig
        Configuration of the run.
    X : pandas.DataFrame, optional
        Feature matrix, included as a content fingerprint when given.
    y : pandas.Series, optional
        Labels, included alongside ``X``.
    extra : dict, optional
        Further entries to fold into the digest, for callers whose run is
        determined by more than one dataset.

    Returns
    -------
    str
        Hexadecimal key of length :data:`RUN_KEY_LENGTH`.
    """
    payload = {
        "seed": config.cv.seed,
        "n_outer_splits": config.cv.n_outer_splits,
        "n_inner_splits": config.cv.n_inner_splits,
        "target_column": config.target_column,
        "continuous_cols": list(config.continuous_cols or []),
        "categorical_cols": list(config.categorical_cols or []),
        "nan_strategy": config.nan_strategy,
        "models_to_use": sorted(config.models_to_use or []),
        "calibrate_models": config.calibration.calibrate_models,
        "calibration_method": config.calibration.calibration_method,
        "calibration_cv": config.calibration.calibration_cv,
        "optimize_threshold": config.threshold.optimize_threshold,
        "threshold_method": config.threshold.threshold_method,
        "threshold_auto_n_samples": config.threshold_auto_n_samples,
        "hpo_method": config.hpo_method,
        "optuna_n_trials": config.optuna_n_trials,
    }
    if X is not None:
        payload["data"] = data_fingerprint(X, y)
    if extra:
        payload["extra"] = extra

    encoded = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:RUN_KEY_LENGTH]


class PipelineCheckpointManager:
    """Manage fold-result checkpoints for cross-validation pipelines.

    Parameters
    ----------
    output_dir : str
        Base output directory for experiment results.
    checkpoint_dir : str
        Subdirectory name for checkpoint storage.
    save_checkpoints : bool
        Whether checkpoint saving is enabled.
    run_key : str, optional
        Digest isolating this run's checkpoints from runs with different
        settings or data. Set it with :meth:`set_run_key` once the data is
        known; until then checkpoints live under ``"unkeyed"``.
    """

    def __init__(
        self,
        output_dir: str,
        checkpoint_dir: str,
        save_checkpoints: bool,
        run_key: Optional[str] = None,
    ):
        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir
        self.save_checkpoints = save_checkpoints
        self.run_key = run_key or "unkeyed"

    def set_run_key(
        self,
        config: Any,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
    ) -> str:
        """Derive and store the run key from a config and its data."""
        self.run_key = compute_run_key(config, X, y)
        logger.info(f"Checkpoint run key: {self.run_key}")
        return self.run_key

    def get_fold_dir(self, dataset_name: str, model_name: str, fold_id: int) -> Path:
        """Directory holding every checkpoint file of one model-fold."""
        fold_dir = (
            Path(self.output_dir)
            / Path(self.checkpoint_dir)
            / self.run_key
            / dataset_name
            / model_name.replace(" ", "_")
            / f"fold_{fold_id}"
        )
        fold_dir.mkdir(parents=True, exist_ok=True)
        return fold_dir

    def get_checkpoint_path(
        self, dataset_name: str, model_name: str, fold_id: int
    ) -> Path:
        """Get checkpoint path for a specific model and fold."""
        fold_dir = self.get_fold_dir(dataset_name, model_name, fold_id)
        return fold_dir / f"fold_{fold_id}_results.joblib"

    def checkpoint_exists(
        self, dataset_name: str, model_name: str, fold_id: int
    ) -> bool:
        """Check if a checkpoint exists for a specific fold."""
        checkpoint_path = self.get_checkpoint_path(dataset_name, model_name, fold_id)
        return checkpoint_path.exists()

    def load_checkpoint(
        self, dataset_name: str, model_name: str, fold_id: int
    ) -> Optional[FoldMetrics]:
        """Load a checkpoint if it exists."""
        checkpoint_path = self.get_checkpoint_path(dataset_name, model_name, fold_id)
        if checkpoint_path.exists():
            try:
                checkpoint_data = joblib.load(checkpoint_path)
                logger.info(f"Loaded checkpoint for {model_name} fold {fold_id}")
                return checkpoint_data
            except Exception as e:
                logger.warning(f"Failed to load checkpoint {checkpoint_path}: {e}")
                return None
        return None

    def save_checkpoint(
        self, dataset_name: str, model_name: str, fold_id: int, fold_result: FoldMetrics
    ) -> None:
        """Save a checkpoint for a specific fold."""
        if not self.save_checkpoints:
            return

        checkpoint_path = self.get_checkpoint_path(dataset_name, model_name, fold_id)
        try:
            joblib.dump(fold_result, checkpoint_path, compress=9)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
        except Exception as e:
            logger.error(
                f"Failed to save checkpoint {checkpoint_path}: {e}. "
                f"Progress for this fold may be lost if interrupted."
            )
