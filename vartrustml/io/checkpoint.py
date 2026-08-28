"""
Checkpoint utilities for saving and loading model results.

Saves fold results, manages checkpoints, and reloads saved models.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _sanitize_checkpoint_path(path: str, base_dir: Optional[str] = None) -> Path:
    """Sanitize and validate checkpoint path to prevent directory traversal.

    Parameters
    ----------
    path : str
        Path to sanitize.
    base_dir : str, optional
        Base directory that path must be within. If None, only resolves path.

    Returns
    -------
    pathlib.Path
        Resolved and validated path.

    Raises
    ------
    ValueError
        If path attempts to escape base_dir (path traversal attack).
    """
    resolved_path = Path(path).resolve()

    if base_dir is not None:
        base_resolved = Path(base_dir).resolve()
        try:
            resolved_path.relative_to(base_resolved)
        except ValueError:
            raise ValueError(
                f"Path traversal detected: '{path}' resolves outside base directory. "
                f"Path must be within '{base_resolved}'."
            )

    return resolved_path


def save_fold_results(fold_results: List, output_dir: Path) -> None:
    """Save detailed results for each fold.

    Parameters
    ----------
    fold_results : list
        List of FoldMetrics objects.
    output_dir : pathlib.Path
        Directory to save results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for fold in fold_results:
        fold_dir = output_dir / f"fold_{fold.fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        pd.Series(fold.metrics).to_csv(fold_dir / "metrics.csv")

        np.save(fold_dir / "confusion_matrix.npy", fold.confusion_matrix)

        conf_df = pd.DataFrame(
            fold.confusion_matrix,
            index=["True_0", "True_1"],
            columns=["Pred_0", "Pred_1"],
        )
        conf_df.to_csv(fold_dir / "confusion_matrix.csv")

        if len(fold.misclassified_samples) > 0:
            fold.misclassified_samples.to_csv(
                fold_dir / "misclassified.csv", index=True
            )

        if fold.sample_indices is not None:
            np.save(fold_dir / "sample_indices.npy", fold.sample_indices)

        joblib.dump(fold.error_analysis, fold_dir / "error_analysis.joblib")

        if fold.best_params:
            joblib.dump(fold.best_params, fold_dir / "best_params.joblib")

        if fold.feature_importances is not None:
            np.save(fold_dir / "feature_importances.npy", fold.feature_importances)

        if fold.shap_values is not None:
            np.save(fold_dir / "shap_values.npy", fold.shap_values)

        if fold.X_test_transformed is not None:
            np.save(fold_dir / "X_test_transformed.npy", fold.X_test_transformed)

    logger.info(f"Saved fold results to {output_dir}")


def list_checkpoints(checkpoint_dir: str) -> Dict[str, Dict[str, List[int]]]:
    """List all available checkpoints.

    Parameters
    ----------
    checkpoint_dir : str
        Directory containing checkpoints.

    Returns
    -------
    dict
        Dictionary mapping dataset -> model -> list of fold IDs.
    """
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        logger.warning(f"Checkpoint directory {checkpoint_dir} does not exist")
        return {}

    checkpoints = {}

    for dataset_dir in checkpoint_path.iterdir():
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name
        checkpoints[dataset_name] = {}

        for model_dir in dataset_dir.iterdir():
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name
            fold_ids = []

            # Look for fold_N subdirectories containing fold_N_results.joblib
            for fold_dir in model_dir.glob("fold_*"):
                if not fold_dir.is_dir():
                    continue
                try:
                    fold_id = int(fold_dir.name.split("_")[1])
                    results_file = fold_dir / f"fold_{fold_id}_results.joblib"
                    if results_file.exists():
                        fold_ids.append(fold_id)
                except (ValueError, IndexError):
                    continue

            if fold_ids:
                checkpoints[dataset_name][model_name] = sorted(fold_ids)

    return checkpoints


def cleanup_checkpoints(
    checkpoint_dir: str,
    dataset_name: Optional[str] = None,
    model_name: Optional[str] = None,
    dry_run: bool = False,
):
    """Clean up checkpoint files.

    Parameters
    ----------
    checkpoint_dir : str
        Directory containing checkpoints.
    dataset_name : str, optional
        Specific dataset to clean.
    model_name : str, optional
        Specific model to clean.
    dry_run : bool
        If True, log what would be deleted without actually deleting.
    """
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        logger.warning(f"Checkpoint directory {checkpoint_dir} does not exist")
        return

    if dataset_name and model_name:
        target = checkpoint_path / dataset_name / model_name.replace(" ", "_")
        label = f"{dataset_name}/{model_name}"
    elif dataset_name:
        target = checkpoint_path / dataset_name
        label = dataset_name
    else:
        target = checkpoint_path
        label = "all"

    if not target.exists():
        return

    files_to_delete = [f for f in target.glob("**/*") if f.is_file()]

    if dry_run:
        for file in files_to_delete:
            logger.info(f"Would delete: {file}")
        logger.info(
            f"Dry run: {len(files_to_delete)} file(s) would be deleted ({label})"
        )
    else:
        for file in files_to_delete:
            file.unlink()
            logger.info(f"Deleted: {file}")
        logger.info(f"Cleaned checkpoints for {label} ({len(files_to_delete)} files)")


def load_checkpoint_model(checkpoint_path: str) -> Optional[Any]:
    """
    Load a trained model from a checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to the checkpoint file.

    Returns
    -------
    Optional[Any]
        Loaded model pipeline or model data dict, or None if loading fails.

    Raises
    ------
    ValueError
        If path traversal is detected in checkpoint_path.
    """
    # Sanitize path (basic resolution, no base_dir constraint for flexibility)
    safe_path = _sanitize_checkpoint_path(checkpoint_path)

    try:
        model_data = joblib.load(safe_path)
        logger.info(f"Loaded model from {checkpoint_path}")

        if not isinstance(model_data, dict) or "model" not in model_data:
            raise ValueError(
                f"Invalid checkpoint format in {checkpoint_path}. "
                "Expected dict with 'model' key. Old format checkpoints are not supported."
            )
        return model_data
    except Exception as e:
        logger.error(f"Failed to load model from {checkpoint_path}: {e}")
        return None


def get_checkpoint_summary(checkpoint_dir: str) -> pd.DataFrame:
    """Generate a summary DataFrame of all checkpoints.

    Parameters
    ----------
    checkpoint_dir : str
        Directory containing checkpoints.

    Returns
    -------
    pandas.DataFrame
        DataFrame with checkpoint information (columns: Dataset, Model,
        Completed Folds, Fold IDs).
    """
    checkpoints = list_checkpoints(checkpoint_dir)

    summary_data = []
    for dataset_name, models in checkpoints.items():
        for model_name, fold_ids in models.items():
            summary_data.append(
                {
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "Completed Folds": len(fold_ids),
                    "Fold IDs": ", ".join(map(str, fold_ids)),
                }
            )

    return pd.DataFrame(summary_data)
