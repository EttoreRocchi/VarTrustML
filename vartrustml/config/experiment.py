"""
Experiment configuration for VarTrustML.

:class:`ExperimentConfig` configures cross-validation experiments, model
training, and evaluation pipelines.

Examples
--------
Basic configuration:

>>> from vartrustml.config.experiment import ExperimentConfig, CVConfig
>>> config = ExperimentConfig(
...     cv=CVConfig(seed=42, n_outer_splits=10),
...     calibration=CalibrationConfig(calibrate_models=True, calibration_cv=5),
... )

Configuration with threshold optimization and caller comparison:

>>> config = ExperimentConfig(
...     threshold=ThresholdConfig(optimize_threshold=True, threshold_method="auto"),
...     caller_comparison=CallerComparisonConfig(
...         compare_callers=True,
...         caller_columns=["MANTA", "DELLY", "LUMPY"],
...     ),
... )

See Also
--------
CrossValidationPipeline : Main pipeline that uses this configuration.
ModelConfig : Configuration for individual model hyperparameters.
"""

import json
from dataclasses import asdict, dataclass, field, fields
from typing import List, Optional

SUPPORTED_MODELS: List[str] = [
    "MLP",
    "Random Forest",
    "XGBoost",
    "CatBoost",
    "Logistic Regression",
    "KNN",
]


# ---------------------------------------------------------------------------
# Nested sub-config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CVConfig:
    """Cross-validation settings.

    Parameters
    ----------
    n_outer_splits : int, default=10
        Number of folds for model evaluation (outer cross-validation).
    n_inner_splits : int, default=5
        Number of folds for hyperparameter tuning (inner cross-validation).
    seed : int, default=42
        Random seed for reproducibility across all random operations.
    """

    n_outer_splits: int = 10
    n_inner_splits: int = 5
    seed: int = 42


@dataclass
class VisualizationConfig:
    """Visualization settings embedded in ExperimentConfig.

    Parameters
    ----------
    plot_top_n_features : int, default=20
        Number of top features to display in importance plots.
    figure_dpi : int, default=300
        Resolution (DPI) for saved figure files.
    error_analysis_features : list of str
        Feature names to analyze in error distribution plots.
    """

    plot_top_n_features: int = 20
    figure_dpi: int = 300
    error_analysis_features: List[str] = field(default_factory=list)


@dataclass
class CalibrationConfig:
    """Probability calibration settings.

    Parameters
    ----------
    calibrate_models : bool, default=False
        Whether to calibrate model probabilities using cross-validation.
    calibration_method : str, default="isotonic"
        Calibration method: ``"isotonic"`` or ``"sigmoid"``.
    calibration_cv : int, default=3
        Number of cross-validation folds for probability calibration.
    """

    calibrate_models: bool = False
    calibration_method: str = "isotonic"
    calibration_cv: int = 3


@dataclass
class CallerComparisonConfig:
    """Variant caller comparison settings.

    Parameters
    ----------
    compare_callers : bool, default=False
        Whether to compare ML models against external variant callers.
    caller_columns : list of str or None, default=None
        Column names for variant caller predictions.
    caller_combinations : list of str
        Custom logical combinations to evaluate.
    include_default_combinations : bool, default=True
        Whether to auto-generate default AND/OR combinations from callers.
    """

    compare_callers: bool = False
    caller_columns: Optional[List[str]] = None
    caller_combinations: List[str] = field(default_factory=list)
    include_default_combinations: bool = True


@dataclass
class ThresholdConfig:
    """Threshold optimization settings.

    Parameters
    ----------
    optimize_threshold : bool, default=False
        Whether to optimize classification threshold using Youden's J statistic.
    threshold_method : str, default="auto"
        Threshold optimization method: ``"oof"``, ``"cv"``, or ``"auto"``.
    """

    optimize_threshold: bool = False
    threshold_method: str = "auto"


@dataclass
class BootstrapConfig:
    """Bootstrap confidence interval settings.

    Parameters
    ----------
    bootstrap_n_iterations : int, default=1000
        Number of bootstrap resamples for confidence interval estimation.
    bootstrap_ci_level : float, default=0.95
        Confidence level for bootstrap CIs.
    bootstrap_ci_method : str, default="bca"
        Confidence interval method: ``"bca"`` (bias-corrected and accelerated,
        recommended) or ``"percentile"`` (raw bootstrap percentiles).
    """

    bootstrap_n_iterations: int = 1000
    bootstrap_ci_level: float = 0.95
    bootstrap_ci_method: str = "bca"


@dataclass
class ExperimentConfig:
    """Configuration for machine learning experiments.

    Central configuration dataclass that controls all aspects of the
    VarTrustML cross-validation pipeline including data processing,
    model training, evaluation, and report generation.

    Settings are organized into nested sub-configs for clarity:

    - ``cv``: Cross-validation settings (seed, folds)
    - ``visualization``: Plot settings (DPI, feature count)
    - ``calibration``: Probability calibration settings
    - ``caller_comparison``: Variant caller comparison settings
    - ``threshold``: Threshold optimization settings
    - ``bootstrap``: Bootstrap CI settings

    See Also
    --------
    CrossValidationPipeline : Main pipeline that uses this configuration.
    ThresholdOptimizer : Threshold optimization using Youden's J.
    BootstrapAnalyzer : Bootstrap confidence interval computation.

    Examples
    --------
    Minimal configuration for quick experiments:

    >>> config = ExperimentConfig(
    ...     cv=CVConfig(seed=42, n_outer_splits=5),
    ...     models_to_use=["Random Forest", "XGBoost"],
    ... )

    Full configuration with calibration and threshold optimization:

    >>> config = ExperimentConfig(
    ...     cv=CVConfig(seed=42, n_outer_splits=10),
    ...     calibration=CalibrationConfig(
    ...         calibrate_models=True, calibration_method="isotonic"
    ...     ),
    ...     threshold=ThresholdConfig(optimize_threshold=True, threshold_method="auto"),
    ...     bootstrap=BootstrapConfig(bootstrap_n_iterations=2000),
    ... )

    Save and load configuration:

    >>> config.save("experiment_config.json")
    >>> loaded_config = ExperimentConfig.load("experiment_config.json")
    """

    # --- Nested sub-configs ---
    cv: CVConfig = field(default_factory=CVConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    caller_comparison: CallerComparisonConfig = field(
        default_factory=CallerComparisonConfig
    )
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)

    # --- Fields that remain directly on ExperimentConfig ---
    output_dir: str = "results"

    # Data settings
    target_column: Optional[str] = None
    continuous_cols: List[str] = field(default_factory=list)
    categorical_cols: List[str] = field(default_factory=list)

    # Missing-value (NaN) handling strategy. Impute strategies ("median",
    # "mean", "most_frequent") add a SimpleImputer before scaling inside the
    # per-fold pipeline; "drop" removes rows with NaN before cross-validation.
    nan_strategy: str = "median"

    # Error analysis settings
    confidence_thresholds: List[float] = field(
        default_factory=lambda: [0.6, 0.7, 0.8, 0.9, 0.95]
    )

    # Model settings
    models_to_use: List[str] = field(default_factory=lambda: list(SUPPORTED_MODELS))

    # Advanced settings
    n_jobs: int = -1
    verbose: int = 1

    # Checkpoint settings
    save_checkpoints: bool = True
    checkpoint_dir: str = "checkpoints"

    # Hyperparameter optimization settings
    hpo_method: str = "grid"
    optuna_n_trials: int = 50
    optuna_timeout: int = 3600

    # Report generation settings
    generate_html_report: bool = True
    html_report_path: str = "report.html"

    # Threshold auto settings (not in ThresholdConfig as it's a standalone knob)
    threshold_auto_n_samples: int = 1000

    # Model comparison settings
    model_comparison_metric: str = "Matthews Corr. Coef."

    # Multiple-comparison correction for the pairwise tests: "holm" (FWER) or "bh" (FDR)
    correction_method: str = "holm"

    # SHAP caching settings
    shap_cache_enabled: bool = False
    shap_cache_dir: str = ".shap_cache"

    # Metadata tracking
    data_file_path: Optional[str] = None

    # ------------------------------------------------------------------
    # Serialization (nested JSON format)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Convert configuration to a nested dictionary.

        Returns
        -------
        dict
            Configuration as a nested dictionary with sub-configs
            under their respective keys.

        See Also
        --------
        save : Save configuration to JSON file.
        """
        return asdict(self)

    def save(self, filepath: str) -> None:
        """Save configuration to JSON file.

        Parameters
        ----------
        filepath : str
            Path to save the configuration file.

        See Also
        --------
        load : Load configuration from JSON file.
        to_dict : Convert configuration to dictionary.
        """
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "ExperimentConfig":
        """Load configuration from JSON file.

        Parameters
        ----------
        filepath : str
            Path to the configuration file.

        Returns
        -------
        ExperimentConfig
            Loaded configuration instance.

        See Also
        --------
        save : Save configuration to JSON file.
        """
        with open(filepath) as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ExperimentConfig":
        """Create an ExperimentConfig from a dictionary.

        Accepts nested format (sub-configs as dicts under their key).

        Parameters
        ----------
        config_dict : dict
            Configuration dictionary in nested format.

        Returns
        -------
        ExperimentConfig
            Configuration instance.
        """
        import logging

        _SUB_CONFIG_CLASSES = {
            "cv": CVConfig,
            "visualization": VisualizationConfig,
            "calibration": CalibrationConfig,
            "caller_comparison": CallerComparisonConfig,
            "threshold": ThresholdConfig,
            "bootstrap": BootstrapConfig,
        }

        direct_field_names = {
            f.name for f in fields(cls) if f.name not in _SUB_CONFIG_CLASSES
        }
        all_known = set(_SUB_CONFIG_CLASSES.keys()) | direct_field_names

        unknown_keys = set(config_dict) - all_known
        if unknown_keys:
            logging.getLogger(__name__).warning(
                "Ignoring unknown config keys: %s", sorted(unknown_keys)
            )

        kwargs = {}
        for key, value in config_dict.items():
            if key in _SUB_CONFIG_CLASSES and isinstance(value, dict):
                sub_cls = _SUB_CONFIG_CLASSES[key]
                valid_keys = {f.name for f in fields(sub_cls)}
                unknown_sub = set(value) - valid_keys
                if unknown_sub:
                    logging.getLogger(__name__).warning(
                        "Ignoring unknown keys in '%s' config: %s",
                        key,
                        sorted(unknown_sub),
                    )
                filtered = {k: v for k, v in value.items() if k in valid_keys}
                kwargs[key] = sub_cls(**filtered)
            elif key in direct_field_names:
                kwargs[key] = value

        return cls(**kwargs)
