"""SHAP-based model interpretability.

:class:`SHAPExplainer` computes, caches, and manages SHAP values across the
supported model types.
"""

import hashlib
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import shap

from vartrustml.core.model_registry import get_model_spec

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """Handles SHAP value computation, caching, and cache management.

    Parameters
    ----------
    output_dir : str or Path
        Base output directory for cache storage.
    shap_cache_dir : str
        Subdirectory name for SHAP cache files.
    shap_cache_enabled : bool
        Whether to enable SHAP value caching.
    seed : int
        Random seed for reproducibility (used by KernelExplainer).
    """

    def __init__(
        self,
        output_dir: str = "results",
        shap_cache_dir: str = ".shap_cache",
        shap_cache_enabled: bool = True,
        seed: int = 42,
    ):
        self.output_dir = output_dir
        self.shap_cache_dir = shap_cache_dir
        self.shap_cache_enabled = shap_cache_enabled
        self.seed = seed

    def _get_shap_cache_path(
        self, model_name: str, fold_id: int, X_test: pd.DataFrame
    ) -> Path:
        """Generate cache file path for SHAP values."""
        # Create hash of X_test data for cache key
        data_hash = hashlib.sha256(
            pd.util.hash_pandas_object(X_test).values.tobytes()
        ).hexdigest()[:12]

        cache_dir = Path(self.output_dir) / self.shap_cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        safe_model_name = model_name.replace(" ", "_")
        return cache_dir / f"{safe_model_name}_fold{fold_id}_{data_hash}.npy"

    def _load_shap_cache(
        self, model_name: str, fold_id: int, X_test: pd.DataFrame
    ) -> Optional[np.ndarray]:
        """Load cached SHAP values if available."""
        if not self.shap_cache_enabled:
            return None

        cache_path = self._get_shap_cache_path(model_name, fold_id, X_test)
        if cache_path.exists():
            try:
                shap_values = np.load(cache_path)
                logger.info(f"Loaded SHAP values from cache: {cache_path}")
                return shap_values
            except Exception as e:
                logger.warning(f"Failed to load SHAP cache {cache_path}: {e}")
                return None
        return None

    def _save_shap_cache(
        self,
        model_name: str,
        fold_id: int,
        X_test: pd.DataFrame,
        shap_values: np.ndarray,
    ) -> None:
        """Save SHAP values to cache."""
        if not self.shap_cache_enabled:
            return

        cache_path = self._get_shap_cache_path(model_name, fold_id, X_test)
        try:
            np.save(cache_path, shap_values)
            logger.debug(f"Saved SHAP values to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save SHAP cache {cache_path}: {e}")

    def clear_shap_cache(self) -> int:
        """Clear all cached SHAP values.

        Returns
        -------
        int
            Number of cache files deleted.
        """
        cache_dir = Path(self.output_dir) / self.shap_cache_dir
        if not cache_dir.exists():
            return 0

        deleted = 0
        for cache_file in cache_dir.glob("*.npy"):
            try:
                cache_file.unlink()
                deleted += 1
            except Exception as e:
                logger.warning(f"Failed to delete cache file {cache_file}: {e}")

        logger.info(f"Cleared {deleted} SHAP cache files from {cache_dir}")
        return deleted

    def compute_shap_values(
        self,
        model: Any,
        model_name: str,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        fold_id: int = 0,
        background_sample_size: int = 100,
    ) -> Optional[np.ndarray]:
        """Calculate SHAP values for interpretability.

        Parameters
        ----------
        model : sklearn.pipeline.Pipeline
            Trained model pipeline.
        model_name : str
            Name of the model.
        X_train : pd.DataFrame
            Training data (for background sampling).
        X_test : pd.DataFrame
            Test data to explain.
        fold_id : int, default=0
            Fold identifier for caching.
        background_sample_size : int, default=100
            Number of background samples for KernelExplainer.

        Returns
        -------
        np.ndarray or None
            SHAP values array or None if computation fails.
        """
        # Check cache first
        cached_values = self._load_shap_cache(model_name, fold_id, X_test)
        if cached_values is not None:
            return cached_values

        # Suppress SHAP's verbose output and warnings
        import warnings

        shap_logger = logging.getLogger("shap")
        original_level = shap_logger.level
        shap_logger.setLevel(logging.WARNING)

        try:
            # Also suppress warnings during SHAP calculation
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                # Get preprocessor and classifier from pipeline
                preprocessor = model.named_steps["preprocessor"]
                clf = model.named_steps["clf"]

                # Handle calibration wrapper (CalibratedClassifierCV)
                if hasattr(clf, "calibrated_classifiers_"):
                    # CalibratedClassifierCV wraps the model
                    # Access the base estimator from the first calibrated classifier
                    base_clf = clf.calibrated_classifiers_[0].estimator
                else:
                    # Direct access when no calibration
                    base_clf = clf

                X_test_transformed = preprocessor.transform(X_test)
                if hasattr(X_test_transformed, "values"):
                    X_test_transformed = X_test_transformed.values

                spec = get_model_spec(model_name)
                explainer_type = spec.shap_explainer_type

                if explainer_type == "tree":
                    explainer = shap.TreeExplainer(base_clf)
                    shap_values = explainer.shap_values(X_test_transformed)

                    # Handle multi-output for binary classification
                    if isinstance(shap_values, list) and len(shap_values) == 2:
                        shap_values = shap_values[1]
                    elif len(shap_values.shape) == 3:
                        shap_values = shap_values[:, :, 1]

                elif explainer_type == "linear":
                    background = preprocessor.transform(X_train)
                    if hasattr(background, "values"):
                        background = background.values
                    explainer = shap.LinearExplainer(base_clf, background)
                    shap_values = explainer.shap_values(X_test_transformed)

                elif explainer_type == "kernel":
                    background = preprocessor.transform(X_train)
                    if hasattr(background, "values"):
                        background = background.values
                    sample_size = min(background_sample_size, background.shape[0])
                    background_sample = background[:sample_size]

                    rng = np.random.default_rng(self.seed)
                    np.random.seed(rng.integers(2**31))

                    explainer = shap.KernelExplainer(
                        base_clf.predict_proba, background_sample
                    )
                    shap_values = explainer.shap_values(
                        X_test_transformed,
                        nsamples=background_sample_size,
                        silent=True,
                    )

                    if isinstance(shap_values, list) and len(shap_values) == 2:
                        shap_values = shap_values[1]

                else:
                    logger.warning(
                        f"Unknown SHAP explainer type '{explainer_type}' "
                        f"for model '{model_name}'"
                    )
                    return None

                # Save to cache
                self._save_shap_cache(model_name, fold_id, X_test, shap_values)

                return shap_values

        except MemoryError as e:
            logger.warning(
                f"SHAP calculation for {model_name} failed due to insufficient memory: {e}\n"
                "Consider reducing the background_sample_size or disabling SHAP for large datasets."
            )
            return None
        except ValueError as e:
            logger.warning(
                f"SHAP calculation for {model_name} failed due to data issue: {e}\n"
                "This may be caused by NaN/infinite values in features or incompatible data types."
            )
            logger.debug("SHAP calculation traceback:", exc_info=True)
            return None
        except Exception as e:
            logger.warning(
                f"SHAP calculation for {model_name} failed: {type(e).__name__}: {e}\n"
                "Feature importance will not be available for this fold. "
                "If this persists, try disabling SHAP via shap_cache_enabled=False."
            )
            logger.debug("SHAP calculation traceback:", exc_info=True)
            return None
        finally:
            # Restore original SHAP logging level
            shap_logger.setLevel(original_level)
