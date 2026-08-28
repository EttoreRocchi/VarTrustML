"""Model checkpoint management for fold-level persistence."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Handles saving and loading of model checkpoints."""

    def save_fold_checkpoint(
        self,
        best_model: Any,
        checkpoint_dir: Path,
        fold_id: int,
        threshold_info: Optional[Dict[str, Any]],
        optimal_threshold: float,
        best_params: Dict[str, Any],
    ) -> None:
        """Save model checkpoint to disk.

        Parameters
        ----------
        best_model : Any
            Fitted model or pipeline to save.
        checkpoint_dir : pathlib.Path
            Directory to store checkpoint files.
        fold_id : int
            Cross-validation fold identifier.
        threshold_info : dict or None
            Threshold optimization information.
        optimal_threshold : float
            Optimal decision threshold.
        best_params : dict
            Best hyperparameters found during search.
        """
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"fold_{fold_id}_model.joblib"
        checkpoint_data = {
            "model": best_model,
            "threshold_info": threshold_info,
            "optimal_threshold": optimal_threshold,
            "best_params": best_params,
        }
        joblib.dump(checkpoint_data, checkpoint_path, compress=9)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_fold_checkpoint(
        self,
        model_name: str,
        fold_id: int,
        checkpoint_dir: Path,
    ) -> Optional[Dict[str, Any]]:
        """Load a model checkpoint from disk.

        Returns
        -------
        dict or None
            Checkpoint data dict with keys 'model', 'threshold_info',
            'optimal_threshold', 'best_params', or None if no checkpoint exists.
        """
        checkpoint_path = checkpoint_dir / f"fold_{fold_id}_model.joblib"
        if not checkpoint_path.exists():
            return None

        logger.info(
            f"Loading existing checkpoint for {model_name} fold {fold_id}: {checkpoint_path}"
        )
        return joblib.load(checkpoint_path)
