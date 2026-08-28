"""VarTrustML - Reliable Variant Calling Through Machine Learning."""

# Version info - safe to import at build time (no heavy dependencies)
from vartrustml._version import __author__, __email__, __version__

__all__ = [
    # Version
    "__version__",
    "__author__",
    "__email__",
    # Core
    "CrossValidationPipeline",
    "CrossDatasetEvaluator",
    "ModelEvaluator",
    "ModelTrainer",
    "TrainConfig",
    "ThresholdOptimizer",
    "ThresholdMethod",
    "ThresholdResult",
    "ReportGenerator",
    "PipelineCheckpointManager",
    # Config
    "ExperimentConfig",
    "ModelConfig",
    # Analysis - Error Analysis
    "ErrorAnalyzer",
    "FoldMetrics",
    # Analysis - Ablation
    "AblationAnalyzer",
    "AblationResult",
    "AblationStudyResult",
    # Analysis - Statistical
    "compare_pairwise",
    "PairwiseComparisonResult",
    "cliffs_delta",
    "cliffs_delta_with_ci",
    "CliffsDeltaResult",
    "benjamini_hochberg_correction",
    "power_analysis_sample_size",
    # Analysis - Bootstrap
    "BootstrapAnalyzer",
    "BootstrapCIResult",
    # I/O
    "DataLoader",
    "save_fold_results",
    "load_checkpoint_model",
    "list_checkpoints",
    "cleanup_checkpoints",
    "get_checkpoint_summary",
    # Visualization
    "Visualizer",
    "HTMLCompareReporter",
    "HTMLTrainReporter",
    "HTMLCrossDatasetReporter",
    # Utils
    "create_summary_report",
    "create_feature_importance_report",
    "calculate_minimum_samples_for_cv",
    "validate_target_for_cv",
]

# Lazy imports
_lazy_imports = {
    # Core
    "CrossValidationPipeline": ("vartrustml.core.pipeline", "CrossValidationPipeline"),
    "CrossDatasetEvaluator": ("vartrustml.core.cross_dataset", "CrossDatasetEvaluator"),
    "ModelEvaluator": ("vartrustml.core.models", "ModelEvaluator"),
    "ModelTrainer": ("vartrustml.core.train_model", "ModelTrainer"),
    "TrainConfig": ("vartrustml.core.train_model", "TrainConfig"),
    "ThresholdOptimizer": ("vartrustml.core.threshold", "ThresholdOptimizer"),
    "ThresholdMethod": ("vartrustml.core.threshold", "ThresholdMethod"),
    "ThresholdResult": ("vartrustml.core.threshold", "ThresholdResult"),
    "ReportGenerator": ("vartrustml.core.report_generator", "ReportGenerator"),
    "PipelineCheckpointManager": (
        "vartrustml.core.pipeline_checkpoint",
        "PipelineCheckpointManager",
    ),
    # Config
    "ExperimentConfig": ("vartrustml.config.experiment", "ExperimentConfig"),
    "ModelConfig": ("vartrustml.config.model", "ModelConfig"),
    # Analysis - Error Analysis
    "ErrorAnalyzer": ("vartrustml.analysis.error_analysis", "ErrorAnalyzer"),
    "FoldMetrics": ("vartrustml.analysis.error_analysis", "FoldMetrics"),
    # Analysis - Ablation
    "AblationAnalyzer": ("vartrustml.analysis.ablation", "AblationAnalyzer"),
    "AblationResult": ("vartrustml.analysis.ablation", "AblationResult"),
    "AblationStudyResult": ("vartrustml.analysis.ablation", "AblationStudyResult"),
    # Analysis - Statistical
    "compare_pairwise": (
        "vartrustml.analysis.pairwise_comparison",
        "compare_pairwise",
    ),
    "PairwiseComparisonResult": (
        "vartrustml.analysis.pairwise_comparison",
        "PairwiseComparisonResult",
    ),
    "cliffs_delta": ("vartrustml.analysis.cliffs_delta", "cliffs_delta"),
    "cliffs_delta_with_ci": (
        "vartrustml.analysis.cliffs_delta",
        "cliffs_delta_with_ci",
    ),
    "CliffsDeltaResult": ("vartrustml.analysis.cliffs_delta", "CliffsDeltaResult"),
    "benjamini_hochberg_correction": (
        "vartrustml.analysis.delong_mcnemar",
        "benjamini_hochberg_correction",
    ),
    "power_analysis_sample_size": (
        "vartrustml.analysis.delong_mcnemar",
        "power_analysis_sample_size",
    ),
    # Analysis - Bootstrap
    "BootstrapAnalyzer": ("vartrustml.analysis.bootstrap", "BootstrapAnalyzer"),
    "BootstrapCIResult": ("vartrustml.analysis.bootstrap", "BootstrapCIResult"),
    # I/O
    "DataLoader": ("vartrustml.io.data_loader", "DataLoader"),
    "save_fold_results": ("vartrustml.io.checkpoint", "save_fold_results"),
    "load_checkpoint_model": ("vartrustml.io.checkpoint", "load_checkpoint_model"),
    "list_checkpoints": ("vartrustml.io.checkpoint", "list_checkpoints"),
    "cleanup_checkpoints": ("vartrustml.io.checkpoint", "cleanup_checkpoints"),
    "get_checkpoint_summary": ("vartrustml.io.checkpoint", "get_checkpoint_summary"),
    # Visualization
    "Visualizer": ("vartrustml.visualization.plots", "Visualizer"),
    "HTMLCompareReporter": (
        "vartrustml.visualization.html_compare_reporter",
        "HTMLCompareReporter",
    ),
    "HTMLTrainReporter": (
        "vartrustml.visualization.html_train_reporter",
        "HTMLTrainReporter",
    ),
    "HTMLCrossDatasetReporter": (
        "vartrustml.visualization.html_cross_dataset_reporter",
        "HTMLCrossDatasetReporter",
    ),
    # Utils
    "create_summary_report": ("vartrustml.utils.reporting", "create_summary_report"),
    "create_feature_importance_report": (
        "vartrustml.utils.reporting",
        "create_feature_importance_report",
    ),
    "calculate_minimum_samples_for_cv": (
        "vartrustml.utils.validation",
        "calculate_minimum_samples_for_cv",
    ),
    "validate_target_for_cv": (
        "vartrustml.utils.validation",
        "validate_target_for_cv",
    ),
}


def __getattr__(name: str):
    """Lazy import for heavy modules to avoid import errors during build."""
    if name in _lazy_imports:
        module_path, attr_name = _lazy_imports[name]
        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
