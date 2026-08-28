"""
Configuration module for VarTrustML.

Configuration classes for machine learning experiments:
- ExperimentConfig: Main configuration for cross-validation experiments
- ModelConfig: Hyperparameter search spaces for individual models
- VisualizationConfig: Settings for plot generation and styling
- CallerConfig: Configuration for variant caller comparison
"""

from vartrustml.config.caller import CallerConfig
from vartrustml.config.experiment import (
    BootstrapConfig,
    CalibrationConfig,
    CallerComparisonConfig,
    CVConfig,
    ExperimentConfig,
    ThresholdConfig,
)
from vartrustml.config.model import ModelConfig
from vartrustml.config.validation import (
    validate_dataset_for_cv,
    validate_experiment_config,
    validate_model_config,
)
from vartrustml.config.visualization import VisualizationConfig

__all__ = [
    "BootstrapConfig",
    "CalibrationConfig",
    "CallerComparisonConfig",
    "CallerConfig",
    "CVConfig",
    "ExperimentConfig",
    "ModelConfig",
    "ThresholdConfig",
    "VisualizationConfig",
    "validate_dataset_for_cv",
    "validate_experiment_config",
    "validate_model_config",
]
