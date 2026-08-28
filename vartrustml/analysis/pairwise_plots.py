"""
Plots for the paired pairwise comparison report.

Three figures replace the retired family/model boxplots:

- :func:`plot_metric_forest`:per-entity forest plot of the primary metric with
  bias-corrected bootstrap (BCa) confidence intervals, coloured by entity type
  (ML model / single caller / caller combination). This is the *descriptive*
  backbone: it shows where every classifier sits with honest per-entity CIs,
  while inference is carried by the paired McNemar / DeLong tests.
- :func:`plot_roc_pr_dominance`:ROC and precision-recall curves for the ML
  models with each caller's single operating point overlaid. A caller lying
  below/inside every ML curve is dominated regardless of threshold, a
  threshold-free argument that complements the operating-point McNemar tests.
- :func:`plot_pvalue_heatmap`:heatmap of multiple-comparison adjusted McNemar
  p-values among ML models (model selection).

All functions accept ``save_path`` to write a PNG and ``return_base64`` to get a
base64-encoded PNG for HTML embedding.

See Also
--------
vartrustml.analysis.pairwise_comparison : Produces the PairwiseComparisonResult.
"""

import base64
import io
import logging
from typing import Optional

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from vartrustml.analysis.pairwise_comparison import (
    FAMILY_OPERATING_POINT,
    P_DISPLAY_FLOOR,
    TYPE_COMBINATION,
    TYPE_ML,
    TYPE_SINGLE_CALLER,
    PairwiseComparisonResult,
    format_pvalue,
)
from vartrustml.visualization.colors import model_color

logger = logging.getLogger(__name__)

_TYPE_COLORS = {
    TYPE_ML: "#2ecc71",  # green
    TYPE_SINGLE_CALLER: "#3498db",  # blue
    TYPE_COMBINATION: "#9b59b6",  # purple
}


def _fig_to_base64(fig) -> str:
    """Render a Matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return encoded


def _finalize(fig, save_path: Optional[str], return_base64: bool):
    """Save and/or encode a figure, then close it."""
    import matplotlib.pyplot as plt

    encoded = _fig_to_base64(fig) if return_base64 else None
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return encoded


def plot_metric_forest(
    result: PairwiseComparisonResult,
    metric_label: str = "Matthews Corr. Coef.",
    bootstrap_n_iterations: int = 1000,
    ci_level: float = 0.95,
    ci_method: str = "bca",
    seed: int = 42,
    save_path: Optional[str] = None,
    return_base64: bool = False,
) -> Optional[str]:
    """Forest plot of the primary metric per entity with bootstrap CIs.

    Confidence intervals are computed on each entity's pooled out-of-fold
    predictions with the same bootstrap method used elsewhere in the report
    (BCa by default). The primary metric is Matthews' correlation coefficient
    (works for both scored ML models and binary callers).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from sklearn.metrics import matthews_corrcoef

    from vartrustml.analysis.bootstrap import BootstrapAnalyzer

    bootstrap = BootstrapAnalyzer(
        n_iterations=bootstrap_n_iterations,
        ci_level=ci_level,
        seed=seed,
        ci_method=ci_method,
    )

    rows = []
    for name, e in result.entities.items():
        ci = bootstrap.compute_ci_from_predictions(
            e.y_true, e.y_pred, None, matthews_corrcoef, metric_label
        )
        if not ci.is_valid:
            # Bounds collapsed onto the point estimate: plotting them would
            # show a spuriously tight interval
            logger.warning(f"Omitting '{name}' from the forest plot: {ci.reason}")
            continue
        rows.append((name, e.entity_type, ci.point_estimate, ci.ci_lower, ci.ci_upper))

    if not rows:
        return None

    # Sort ascending so the best entity is at the top of the chart.
    rows.sort(key=lambda r: r[2])
    names = [r[0] for r in rows]
    point = np.array([r[2] for r in rows])
    lower = np.array([r[3] for r in rows])
    upper = np.array([r[4] for r in rows])
    colors = [_TYPE_COLORS.get(r[1], "#7f8c8d") for r in rows]

    fig, ax = plt.subplots(figsize=(9, max(3, 0.45 * len(rows) + 1.5)))
    y = np.arange(len(rows))
    err_lower = np.clip(point - lower, 0, None)
    err_upper = np.clip(upper - point, 0, None)
    ax.errorbar(
        point,
        y,
        xerr=[err_lower, err_upper],
        fmt="none",
        ecolor="#555555",
        elinewidth=1.2,
        capsize=3,
        zorder=1,
    )
    ax.scatter(point, y, c=colors, s=60, zorder=2, edgecolors="white", linewidths=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel(f"{metric_label}  (point estimate, {int(ci_level * 100)}% BCa CI)")
    ax.set_title("Per-classifier performance (pooled out-of-fold)")
    ax.axvline(0.0, color="#bbbbbb", linestyle="--", linewidth=0.8, zorder=0)
    ax.grid(axis="x", alpha=0.3)

    handles = [
        mpatches.Patch(color=c, label=t)
        for t, c in _TYPE_COLORS.items()
        if any(r[1] == t for r in rows)
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()

    return _finalize(fig, save_path, return_base64)


def plot_roc_pr_dominance(
    result: PairwiseComparisonResult,
    save_path: Optional[str] = None,
    return_base64: bool = False,
) -> Optional[str]:
    """ROC and PR curves for ML models with caller operating points overlaid.

    ML models contribute full curves (they have probability scores); callers and
    combinations contribute a single (FPR, TPR) / (recall, precision) marker each
    at their native binary operating point.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_true = result.y_true
    ml = [(n, e) for n, e in result.entities.items() if e.entity_type == TYPE_ML]
    callers = [(n, e) for n, e in result.entities.items() if e.entity_type != TYPE_ML]
    if not ml:
        return None

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5.5))

    pos_rate = float(np.mean(y_true)) if len(y_true) else 0.0

    for name, e in ml:
        if e.y_prob is None:
            continue
        col = model_color(name)
        fpr, tpr, _ = roc_curve(y_true, e.y_prob)
        ax_roc.plot(fpr, tpr, linewidth=1.8, label=name, color=col)
        prec, rec, _ = precision_recall_curve(y_true, e.y_prob)
        ax_pr.plot(rec, prec, linewidth=1.8, label=name, color=col)

    # Caller operating points.
    caller_markers = ["s", "^", "D", "v", "P", "X", "*"]
    for i, (name, e) in enumerate(callers):
        tp = int(np.sum((e.y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((e.y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((e.y_pred == 0) & (y_true == 1)))
        tn = int(np.sum((e.y_pred == 0) & (y_true == 0)))
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        color = _TYPE_COLORS.get(e.entity_type, "#7f8c8d")
        marker = caller_markers[i % len(caller_markers)]
        ax_roc.scatter(
            fpr,
            tpr,
            marker=marker,
            s=90,
            color=color,
            edgecolors="black",
            linewidths=0.7,
            zorder=5,
            label=name,
        )
        ax_pr.scatter(
            tpr,
            precision,
            marker=marker,
            s=90,
            color=color,
            edgecolors="black",
            linewidths=0.7,
            zorder=5,
            label=name,
        )

    ax_roc.plot([0, 1], [0, 1], "--", color="#bbbbbb", linewidth=0.8)
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title("ROC: ML curves vs caller operating points")
    ax_roc.set_xlim(-0.02, 1.02)
    ax_roc.set_ylim(-0.02, 1.02)
    ax_roc.grid(alpha=0.3)

    ax_pr.axhline(pos_rate, color="#bbbbbb", linestyle="--", linewidth=0.8)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall: ML curves vs caller operating points")
    ax_pr.set_xlim(-0.02, 1.02)
    ax_pr.set_ylim(-0.02, 1.02)
    ax_pr.grid(alpha=0.3)

    # Single shared legend outside the panels (both panels show the same
    # entities), instead of a duplicated legend inside each subplot.
    handles, labels = ax_roc.get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=8,
        framealpha=0.9,
        title="Models / callers",
    )
    return _finalize(fig, save_path, return_base64)


def plot_pvalue_heatmap(
    result: PairwiseComparisonResult,
    save_path: Optional[str] = None,
    return_base64: bool = False,
) -> Optional[str]:
    """Heatmap of multiple-comparison adjusted McNemar p-values among ML models.

    Each off-diagonal cell shows the adjusted operating-point McNemar q-value for
    that ML-vs-ML pair (Holm-Bonferroni or Benjamini-Hochberg, per the configured
    correction); the cell is annotated with the q-value and coloured on a
    -log10(q) scale (darker = stronger evidence of a difference).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ml_names = [n for n, e in result.entities.items() if e.entity_type == TYPE_ML]
    if len(ml_names) < 2:
        return None

    # Alphabetical order for a stable, unambiguous layout.
    ml_names.sort()
    idx = {n: i for i, n in enumerate(ml_names)}
    k = len(ml_names)
    q = np.full((k, k), np.nan)

    for c in result.comparisons:
        if c.family != FAMILY_OPERATING_POINT:
            continue
        if c.name_a in idx and c.name_b in idx:
            i, j = idx[c.name_a], idx[c.name_b]
            q[i, j] = c.p_value_corrected
            q[j, i] = c.p_value_corrected

    with np.errstate(divide="ignore"):
        neglog = -np.log10(np.clip(q, P_DISPLAY_FLOOR, 1.0))

    fig, ax = plt.subplots(figsize=(1.1 * k + 3, 1.1 * k + 2))
    im = ax.imshow(neglog, cmap="viridis", aspect="auto")
    # The global seaborn "darkgrid" theme would draw gridlines through each cell
    # centre, making every column look duplicated. Disable the grid here.
    ax.grid(False)
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(ml_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ml_names, fontsize=9)
    adj = (
        "Holm-adjusted"
        if getattr(result, "correction_method", "bh") == "holm"
        else "BH-adjusted"
    )
    ax.set_title(f"ML vs ML: McNemar q-values ({adj})")

    for i in range(k):
        for j in range(k):
            if i == j:
                ax.text(j, i, "·", ha="center", va="center", color="#888888")
            elif not np.isnan(q[i, j]):
                val = q[i, j]
                txt = format_pvalue(val)
                color = "black" if neglog[i, j] > np.nanmax(neglog) / 2 else "white"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("-log10(q), floored at 1e-10")
    fig.tight_layout()
    return _finalize(fig, save_path, return_base64)
