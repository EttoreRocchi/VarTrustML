"""
Core module for VarTrustML.

The main ML functionality:
- CrossValidationPipeline: Main orchestration for CV experiments
- ModelEvaluator: Model training and evaluation
- ModelTrainer: Standalone model fitting
- ThresholdOptimizer: Optional threshold optimization
"""

from vartrustml.core.cross_dataset import CrossDatasetEvaluator
from vartrustml.core.models import ModelEvaluator
from vartrustml.core.pipeline import CrossValidationPipeline
from vartrustml.core.train_model import ModelTrainer, TrainConfig

__all__ = [
    "CrossValidationPipeline",
    "ModelEvaluator",
    "ModelTrainer",
    "TrainConfig",
    "CrossDatasetEvaluator",
]
