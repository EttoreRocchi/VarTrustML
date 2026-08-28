"""Formatting utilities for ablation study results.

Human-readable formatting for the :class:`AblationResult` and
:class:`AblationStudyResult` objects the ablation framework produces.
"""

from typing import Optional

from vartrustml.analysis.ablation import AblationResult, AblationStudyResult


def format_ablation_result(result: AblationResult, include_scores: bool = False) -> str:
    """Format ablation result as human-readable string.

    Parameters
    ----------
    result : AblationResult
        Ablation result to format.
    include_scores : bool, default=False
        Whether to include per-fold scores.

    Returns
    -------
    str
        Formatted string representation.
    """
    sig_marker = "*" if result.is_significant else ""
    lines = [
        f"Ablation: {result.ablation_name}",
        f"  Baseline: {result.baseline_score:.4f} +/- {result.baseline_std:.4f}",
        f"  Ablated:  {result.ablated_score:.4f} +/- {result.ablated_std:.4f}",
        f"  Delta:    {result.delta:+.4f} ({result.delta_pct:+.1f}%){sig_marker}",
        f"  p-value:  {result.p_value:.4f}"
        + (
            f" (corrected: {result.p_value_corrected:.4f})"
            if result.p_value_corrected is not None
            else ""
        ),
        f"  Effect size (Cohen's d_av): {result.effect_size:.3f}",
    ]

    if include_scores:
        lines.append(f"  Baseline scores: {result.baseline_scores}")
        lines.append(f"  Ablated scores:  {result.ablated_scores}")

    return "\n".join(lines)


def format_ablation_study(
    study: AblationStudyResult,
    top_k: Optional[int] = None,
    significant_only: bool = False,
) -> str:
    """Format complete ablation study as human-readable string.

    Parameters
    ----------
    study : AblationStudyResult
        Complete ablation study results.
    top_k : int, optional
        Show only top-k most impactful ablations.
    significant_only : bool, default=False
        Show only statistically significant results.

    Returns
    -------
    str
        Formatted string representation.
    """
    lines = [
        f"Ablation Study Results ({study.study_type})",
        f"{'=' * 50}",
        f"Metric: {study.metric_name}",
        f"Baseline score: {study.baseline_score:.4f}",
        f"CV splits: {study.n_splits}, Seed: {study.seed}",
        "",
    ]

    results = study.results
    if significant_only:
        results = study.get_significant_ablations()
        lines.append(f"Showing {len(results)} significant results only\n")

    if top_k is not None:
        results = study.get_top_k_features(top_k)
        lines.append(f"Showing top {top_k} most impactful ablations\n")

    for result in results:
        lines.append(format_ablation_result(result))
        lines.append("")

    return "\n".join(lines)
