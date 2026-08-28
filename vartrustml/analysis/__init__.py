"""
Analysis module for VarTrustML.

Error analysis, bootstrap confidence intervals, statistical testing
utilities, and the ablation study framework.
"""

from vartrustml.analysis.ablation import (
    AblationAnalyzer,
    AblationResult,
    AblationStudyResult,
)
from vartrustml.analysis.ablation_config import (
    ConfigAblationAnalyzer,
    SUPPORTED_MODELS,
)
from vartrustml.analysis.ablation_formatters import (
    format_ablation_result,
    format_ablation_study,
)
from vartrustml.analysis.bootstrap import (
    BootstrapAnalyzer,
    BootstrapCIResult,
    format_ci,
)
from vartrustml.analysis.cliffs_delta import (
    CliffsDeltaResult,
    cliffs_delta,
    cliffs_delta_with_ci,
    interpret_cliffs_delta,
)
from vartrustml.analysis.delong_mcnemar import (
    DeLongTestResult,
    McNemarTestResult,
    benjamini_hochberg_correction,
    correct_pvalues,
    delong_test,
    holm_bonferroni_correction,
    mcnemar_test,
    power_analysis_sample_size,
)
from vartrustml.analysis.error_analysis import ErrorAnalyzer, FoldMetrics
from vartrustml.analysis.pairwise_comparison import (
    FAMILY_AUROC,
    FAMILY_OPERATING_POINT,
    TYPE_COMBINATION,
    TYPE_ML,
    TYPE_SINGLE_CALLER,
    Entity,
    PairwiseComparison,
    PairwiseComparisonResult,
    build_entities,
    compare_pairwise,
    comparisons_to_dataframe,
)

__all__ = [
    # Ablation studies
    "AblationAnalyzer",
    "ConfigAblationAnalyzer",
    "AblationResult",
    "AblationStudyResult",
    "SUPPORTED_MODELS",
    "format_ablation_result",
    "format_ablation_study",
    # Error analysis
    "ErrorAnalyzer",
    "FoldMetrics",
    # Bootstrap
    "BootstrapAnalyzer",
    "BootstrapCIResult",
    "format_ci",
    # Paired pairwise comparison (pooled out-of-fold)
    "Entity",
    "PairwiseComparison",
    "PairwiseComparisonResult",
    "compare_pairwise",
    "build_entities",
    "comparisons_to_dataframe",
    "TYPE_ML",
    "TYPE_SINGLE_CALLER",
    "TYPE_COMBINATION",
    "FAMILY_OPERATING_POINT",
    "FAMILY_AUROC",
    # Paired tests (McNemar / DeLong)
    "mcnemar_test",
    "delong_test",
    "McNemarTestResult",
    "DeLongTestResult",
    # Effect size and CI
    "cliffs_delta",
    "cliffs_delta_with_ci",
    "CliffsDeltaResult",
    "interpret_cliffs_delta",
    # Multiple testing correction
    "benjamini_hochberg_correction",
    "holm_bonferroni_correction",
    "correct_pvalues",
    # Power analysis
    "power_analysis_sample_size",
]
