"""
Reusable HTML section mixins for VarTrustML report generators.

These mixins provide modular HTML section builders that can be composed
into different reporter classes via multiple inheritance.
"""

import logging
from datetime import datetime
from html import escape as h
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from vartrustml._version import __version__ as VARTRUSTML_VERSION

logger = logging.getLogger(__name__)


def render_confusion_matrix_html(
    cm: np.ndarray,
    normalize: bool = True,
    div_id: str = "confusion_matrix",
    title: Optional[str] = None,
) -> str:
    """Render a single confusion matrix as a Plotly HTML snippet.

    Parameters
    ----------
    cm : numpy.ndarray
        Confusion matrix array.
    normalize : bool
        Whether the matrix values are normalized.
    div_id : str, default="confusion_matrix"
        HTML div ID for the Plotly figure.
    title : str, optional
        Custom title. Defaults to 'Confusion Matrix (Normalized)' or 'Confusion Matrix'.

    Returns
    -------
    str
        HTML string for the confusion matrix section.
    """
    if title is None:
        title = "Confusion Matrix (Normalized)" if normalize else "Confusion Matrix"

    cm_flipped = cm[::-1]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm_flipped,
            x=["Predicted 0", "Predicted 1"],
            y=["Actual 1", "Actual 0"],
            colorscale="Viridis",
            text=np.round(cm_flipped, 3) if normalize else cm_flipped,
            texttemplate="%{text}",
            textfont={"size": 18},
            colorbar=dict(
                title=dict(text="Value" if normalize else "Count", font=dict(size=14)),
                tickfont=dict(size=12),
            ),
        )
    )

    fig.update_layout(
        title=title,
        title_font_size=18,
        height=600,
        width=600,
        font=dict(size=14),
    )
    fig.update_xaxes(tickfont=dict(size=14))
    fig.update_yaxes(tickfont=dict(size=14))

    return f"""
    <div class="section">
        <h2>{title}</h2>
        {fig.to_html(include_plotlyjs="cdn", div_id=div_id)}
    </div>
    """


class _OverviewMixin:
    """Mixin for overview and best-models-table sections."""

    def add_overview(self, config: Dict[str, Any], dataset_info: Dict[str, Any]):
        """Add experiment overview section with enhanced metadata.

        Parameters
        ----------
        config : dict
            Experiment configuration dictionary.
        dataset_info : dict
            Dataset information including shape, class distribution, etc.
        """
        # Format calibration info
        calibration_enabled = config.get("calibrate_models", False)
        if calibration_enabled:
            calibration_method = config.get("calibration_method", "N/A")
            calibration_cv = config.get("calibration_cv", "N/A")
            calibration_info = (
                f"Enabled ({calibration_method}, {calibration_cv}-fold CV)"
            )
        else:
            calibration_info = "Disabled"

        # Format bootstrap CI info
        bootstrap_n = config.get("bootstrap_n_iterations", 1000)
        bootstrap_ci = config.get("bootstrap_ci_level", 0.95)
        bootstrap_method = config.get("bootstrap_ci_method", "bca")
        bootstrap_info = (
            f"{int(bootstrap_ci * 100)}% {bootstrap_method.upper()} CI "
            f"({bootstrap_n} iterations)"
        )

        # Format threshold optimization info
        optimize_threshold = config.get("optimize_threshold", False)
        if optimize_threshold:
            threshold_method = config.get("threshold_method", "youden")
            threshold_info = f"Enabled ({threshold_method})"
        else:
            threshold_info = "Disabled"

        # Format caller comparison info
        compare_callers = config.get("compare_callers", False)
        caller_columns = config.get("caller_columns", [])
        caller_info = ", ".join(caller_columns) if caller_columns else "Disabled"

        # Get timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Format feature list for display
        continuous_features = dataset_info.get("continuous_features", [])
        n_continuous = (
            len(continuous_features)
            if continuous_features
            else dataset_info.get("n_continuous", 0)
        )
        n_categorical = dataset_info.get("n_categorical", 0)

        # Build content using templates
        table_tpl = self.env.get_template("components/metadata_table.html.j2")

        metadata_table = table_tpl.render(
            title="Metadata",
            items=[
                {"key": "VarTrustML Version", "value": VARTRUSTML_VERSION},
                {"key": "Report Generated", "value": timestamp},
                {
                    "key": "Data File",
                    "value": dataset_info.get("data_file_path", "N/A"),
                },
                {
                    "key": "Target Column",
                    "value": dataset_info.get("target_column", "N/A"),
                },
            ],
        )

        config_table = table_tpl.render(
            title="Experiment Configuration",
            items=[
                {"key": "Random Seed", "value": config.get("seed", "N/A")},
                {
                    "key": "Outer CV Splits",
                    "value": config.get("n_outer_splits", "N/A"),
                },
                {
                    "key": "Inner CV Splits",
                    "value": config.get("n_inner_splits", "N/A"),
                },
                {"key": "HPO Method", "value": config.get("hpo_method", "grid")},
                {"key": "Calibration", "value": calibration_info},
                {"key": "Threshold Optimization", "value": threshold_info},
                {"key": "Bootstrap CI", "value": bootstrap_info},
                {"key": "Models", "value": ", ".join(config.get("models_to_use", []))},
                {
                    "key": "Caller Comparison",
                    "value": caller_info if compare_callers else "Disabled",
                },
            ],
        )

        dataset_table = table_tpl.render(
            title="Dataset Information",
            items=[
                {"key": "Total Samples", "value": dataset_info.get("n_samples", "N/A")},
                {
                    "key": "Total Features",
                    "value": dataset_info.get("n_features", "N/A"),
                },
                {"key": "Continuous Features", "value": n_continuous},
                {"key": "Categorical Features", "value": n_categorical},
                {
                    "key": "Class 0 Count",
                    "value": dataset_info.get("class_0_count", "N/A"),
                },
                {
                    "key": "Class 1 Count",
                    "value": dataset_info.get("class_1_count", "N/A"),
                },
                {
                    "key": "Class Balance",
                    "value": dataset_info.get("class_balance", "N/A"),
                },
            ],
        )

        section_tpl = self.env.get_template("components/section_wrapper.html.j2")
        html = section_tpl.render(
            title="Experiment Overview",
            content=metadata_table + config_table + dataset_table,
        )
        self.sections.append(html)

    def add_best_models_table(self, results_df: pd.DataFrame):
        """
        Add table showing best model for each metric.

        Parameters
        ----------
        results_df : pd.DataFrame
            DataFrame with model results (rows=models, cols=metrics)
        """
        # Filter to mean columns only (exclude std columns)
        metric_cols = [col for col in results_df.columns if not col.endswith("_std")]

        if len(metric_cols) == 0:
            logger.warning("No metrics found for best models table")
            return

        # Metrics where lower is better
        lower_is_better = {"Brier Score", "ECE", "MCE"}

        rows = []
        for metric in metric_cols:
            # Use idxmin for lower-is-better metrics, idxmax otherwise
            if metric in lower_is_better:
                best_idx = results_df[metric].idxmin()
            else:
                best_idx = results_df[metric].idxmax()
            best_model = best_idx
            best_score = results_df.loc[best_idx, metric]
            rows.append(
                [
                    h(metric),
                    f"<strong>{h(str(best_model))}</strong>",
                    f"{best_score:.4f}",
                ]
            )

        table_tpl = self.env.get_template("components/info_table.html.j2")
        table_html = table_tpl.render(
            headers=["Metric", "Best Model", "Score"],
            rows=rows,
        )

        section_tpl = self.env.get_template("components/section_wrapper.html.j2")
        html = section_tpl.render(title="Best Model by Metric", content=table_html)
        self.sections.append(html)


class _CallerComparisonMixin:
    """Mixin for caller vs ML model comparison sections."""

    def add_caller_comparison(
        self,
        caller_results: Dict[str, List[Any]],
        ml_results: Dict[str, List[Any]],
        bootstrap_n_iterations: int = 1000,
        bootstrap_ci_level: float = 0.95,
        bootstrap_ci_method: str = "bca",
        seed: int = 42,
    ):
        """
        Add caller comparison section with unified table showing callers and ML models.

        Displays simplified metrics: Recall (Class 0), Recall (Class 1), MCC
        with prediction-level bootstrap confidence intervals.

        Parameters
        ----------
        caller_results : dict
            Dict mapping caller/combination names to list of CallerResult.
        ml_results : dict
            Dict mapping ML model names to list of FoldMetrics.
        bootstrap_n_iterations : int
            Number of bootstrap iterations for CI.
        bootstrap_ci_level : float
            Confidence level (e.g., 0.95).
        bootstrap_ci_method : str
            Bootstrap CI method: "bca" or "percentile".
        seed : int
            Random seed for bootstrap.
        """
        if not caller_results:
            return

        import numpy as np

        from vartrustml.analysis.bootstrap import BootstrapAnalyzer, format_ci

        bootstrap = BootstrapAnalyzer(
            n_iterations=bootstrap_n_iterations,
            ci_level=bootstrap_ci_level,
            seed=seed,
            ci_method=bootstrap_ci_method,
        )

        # Metrics to display
        metrics_to_show = [
            "Recall (Class 0)",
            "Recall (Class 1)",
            "Matthews Corr. Coef.",
        ]

        html_parts = ['<div class="section">', "<h2>Caller vs ML Model Comparison</h2>"]
        html_parts.append(
            "<p><em>Metrics shown with bootstrap confidence intervals</em></p>"
        )

        # Build table header
        html_parts.append('<table class="info-table">')
        header_row = "<tr><th>Method</th><th>Type</th>"
        for metric in metrics_to_show:
            header_row += f"<th>{h(metric)}</th>"
        header_row += "</tr>"
        html_parts.append(header_row)

        # Add caller rows (individual callers first, then combinations)
        individual_callers = []
        combinations = []

        for name, results in caller_results.items():
            if " AND " in name or " OR " in name:
                combinations.append((name, results))
            else:
                individual_callers.append((name, results))

        # Process individual callers using prediction-level bootstrap
        for name, results in sorted(individual_callers):
            row = f"<tr><td><strong>{h(name)}</strong></td><td>Caller</td>"

            # Concatenate predictions from all folds
            y_true_all = np.concatenate([r.y_true for r in results])
            y_pred_all = np.concatenate([r.y_pred for r in results])

            # Compute CIs using prediction-level bootstrap (no probabilities for callers)
            ci_results = bootstrap.compute_all_cis_from_predictions(
                y_true_all, y_pred_all, y_prob=None
            )

            for metric in metrics_to_show:
                if metric in ci_results:
                    row += f"<td>{format_ci(ci_results[metric])}</td>"
                else:
                    row += "<td>N/A</td>"

            row += "</tr>"
            html_parts.append(row)

        # Process combinations using prediction-level bootstrap
        for name, results in sorted(combinations):
            row = f"<tr><td><strong>{h(name)}</strong></td><td>Combination</td>"

            if not results:
                for _ in metrics_to_show:
                    row += "<td>N/A</td>"
                row += "</tr>"
                html_parts.append(row)
                continue

            # Concatenate predictions from all folds
            y_true_all = np.concatenate([r.y_true for r in results])
            y_pred_all = np.concatenate([r.y_pred for r in results])

            # Compute CIs using prediction-level bootstrap
            ci_results = bootstrap.compute_all_cis_from_predictions(
                y_true_all, y_pred_all, y_prob=None
            )

            for metric in metrics_to_show:
                if metric in ci_results:
                    row += f"<td>{format_ci(ci_results[metric])}</td>"
                else:
                    row += "<td>N/A</td>"

            row += "</tr>"
            html_parts.append(row)

        # Add separator row (2 fixed cols + metrics)
        n_cols = 2 + len(metrics_to_show)
        html_parts.append(
            f'<tr style="background-color: #e9ecef;"><td colspan="{n_cols}" style="height: 2px; padding: 0;"></td></tr>'
        )

        # Add ML model rows using prediction-level bootstrap
        for model_name, fold_results in sorted(ml_results.items()):
            row = f"<tr><td><strong>{h(model_name)}</strong></td><td>ML Model</td>"

            # Concatenate OOF predictions from all folds using per-fold thresholds
            y_true_list = []
            y_prob_list = []
            y_pred_list = []

            for f in fold_results:
                if f.y_true_oof is not None and f.y_prob_oof is not None:
                    y_true_list.append(f.y_true_oof)
                    y_prob_list.append(f.y_prob_oof)
                    # Use per-fold optimal threshold if available
                    threshold = (
                        f.fold_optimal_threshold
                        if f.fold_optimal_threshold is not None
                        else 0.5
                    )
                    y_pred_list.append((f.y_prob_oof >= threshold).astype(int))

            if y_true_list and y_prob_list:
                y_true_all = np.concatenate(y_true_list)
                y_prob_all = np.concatenate(y_prob_list)
                y_pred_all = np.concatenate(y_pred_list)

                # Compute CIs using prediction-level bootstrap
                ci_results = bootstrap.compute_all_cis_from_predictions(
                    y_true_all, y_pred_all, y_prob_all
                )

                for metric in metrics_to_show:
                    if metric in ci_results:
                        row += f"<td>{format_ci(ci_results[metric])}</td>"
                    else:
                        row += "<td>N/A</td>"
            else:
                # Fallback if OOF predictions unavailable
                for _ in metrics_to_show:
                    row += "<td>N/A</td>"

            row += "</tr>"
            html_parts.append(row)

        html_parts.append("</table>")
        html_parts.append("</div>")
        self.sections.append("".join(html_parts))


class _StatisticalMixin:
    """Mixin for the paired pairwise statistical comparison section.

    Renders the paired comparison computed on pooled out-of-fold predictions:
    (A) the best ML model versus each variant caller / combination via McNemar's
    test, (B) ML-vs-ML model selection (McNemar q-value heatmap + DeLong AUROC
    table), (C) a threshold-free ROC/PR dominance figure, and (D) a per-classifier
    forest plot with bootstrap confidence intervals.
    """

    def add_pairwise_comparison(
        self,
        result,
        output_dir: Optional[str] = None,
        bootstrap_n_iterations: int = 1000,
        bootstrap_ci_level: float = 0.95,
        bootstrap_ci_method: str = "bca",
        seed: int = 42,
    ):
        """Add the paired pairwise statistical comparison section.

        Parameters
        ----------
        result : PairwiseComparisonResult or None
            Result from :func:`vartrustml.analysis.pairwise_comparison.compare_pairwise`.
        output_dir : str, optional
            Directory to also save the figures as PNGs.
        bootstrap_n_iterations, bootstrap_ci_level, bootstrap_ci_method, seed
            Bootstrap settings for the forest-plot confidence intervals (kept
            consistent with the rest of the report).
        """
        if result is None or not result.comparisons:
            logger.warning("No pairwise comparisons to render")
            return

        from vartrustml.analysis import pairwise_plots as pp
        from vartrustml.analysis.pairwise_comparison import (
            FAMILY_AUROC,
            FAMILY_OPERATING_POINT,
            TYPE_ML,
            format_pvalue,
        )

        alpha = result.alpha
        corr_label = (
            "Holm-Bonferroni (FWER)"
            if getattr(result, "correction_method", "bh") == "holm"
            else "Benjamini-Hochberg (FDR)"
        )
        corr_short = (
            "Holm" if getattr(result, "correction_method", "bh") == "holm" else "BH"
        )

        def _img(b64, alt):
            if not b64:
                return ""
            return (
                '<div style="text-align:center;margin:18px 0;">'
                f'<img src="data:image/png;base64,{b64}" alt="{h(alt)}" '
                'style="max-width:100%;height:auto;border:1px solid #dee2e6;'
                'border-radius:4px;"></div>'
            )

        def _fmt_or(c):
            if c.odds_ratio is None or not np.isfinite(c.odds_ratio):
                return "n/a"
            lo, hi = c.odds_ratio_ci_lower, c.odds_ratio_ci_upper
            if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
                return f"{c.odds_ratio:.2f}"
            return f"{c.odds_ratio:.2f} [{lo:.2f}, {hi:.2f}]"

        html_parts = [
            '<div class="section">',
            "<h2>Statistical comparison (paired, pooled out-of-fold)</h2>",
        ]
        html_parts.append(
            "<p>All comparisons are <strong>paired</strong> and computed on "
            "<strong>pooled out-of-fold</strong> predictions, where every variant is "
            "predicted exactly once at the operating point of its hold-out fold. "
            "Two classifiers at a fixed operating point are compared with "
            "<strong>McNemar's test</strong> (the appropriate paired test for callers, "
            "which emit hard binary calls); AUROC differences between ML models use "
            "<strong>DeLong's test</strong>. P-values are corrected for multiple "
            f"comparisons with the <strong>{corr_label}</strong> procedure "
            f"(&alpha;&nbsp;=&nbsp;{alpha:g}). Effect sizes are the paired accuracy "
            "difference (positive favours the ML model) with a 95% CI and the "
            "discordant odds ratio b/c.</p>"
        )

        # --- A. Best ML vs callers / combinations (main result) ---------------
        main = result.main_comparisons()
        if main and result.best_ml:
            html_parts.append(
                f"<h3>A. Best ML model (<em>{h(result.best_ml)}</em>) vs variant callers</h3>"
            )
            html_parts.append('<table class="info-table">')
            html_parts.append(
                "<tr><th>vs</th><th>Winner</th><th>&Delta; accuracy (95% CI)</th>"
                "<th>Odds ratio b/c</th><th>Discordant b / c</th>"
                f"<th>McNemar q ({corr_short})</th><th>Sig.</th></tr>"
            )
            for c in sorted(main, key=lambda x: abs(x.acc_diff or 0.0), reverse=True):
                sig = "&#10003;" if c.is_significant else ""
                winner = "tie" if c.better == "tie" else h(c.better)
                html_parts.append(
                    f"<tr><td>{h(c.name_b)}</td><td>{winner}</td>"
                    f"<td>{c.acc_diff:+.3f} [{c.acc_diff_ci_lower:+.3f}, {c.acc_diff_ci_upper:+.3f}]</td>"
                    f"<td>{_fmt_or(c)}</td>"
                    f"<td>{c.n_discordant_b} / {c.n_discordant_c}</td>"
                    f"<td>{h(format_pvalue(c.p_value_corrected))}</td><td>{sig}</td></tr>"
                )
            html_parts.append("</table>")
            html_parts.append(
                "<p><em>&Delta; accuracy is (best-ML accuracy &minus; caller accuracy); "
                "b = ML correct &amp; caller wrong, c = ML wrong &amp; caller correct. "
                "Full matrix in pairwise_mcnemar_full.csv.</em></p>"
            )

        # --- B. ML vs ML (model selection) ------------------------------------
        ml_op = [
            c
            for c in result.comparisons
            if c.family == FAMILY_OPERATING_POINT
            and result.entities.get(c.name_a)
            and result.entities.get(c.name_b)
            and result.entities[c.name_a].entity_type == TYPE_ML
            and result.entities[c.name_b].entity_type == TYPE_ML
        ]
        delong = [c for c in result.comparisons if c.family == FAMILY_AUROC]
        if ml_op or delong:
            html_parts.append("<h3>B. ML model selection (ML vs ML)</h3>")
            if ml_op:
                try:
                    heat = pp.plot_pvalue_heatmap(
                        result,
                        save_path=(
                            str(Path(output_dir) / "ml_vs_ml_mcnemar_qvalues.png")
                            if output_dir
                            else None
                        ),
                        return_base64=True,
                    )
                    html_parts.append(
                        _img(heat, f"ML vs ML McNemar q-values ({corr_short}-adjusted)")
                    )
                except Exception as e:
                    logger.warning(f"Could not generate ML-vs-ML heatmap: {e}")
            if delong:
                html_parts.append("<h4>AUROC differences (DeLong)</h4>")
                html_parts.append('<table class="info-table">')
                html_parts.append(
                    "<tr><th>Model A</th><th>Model B</th><th>AUROC A</th><th>AUROC B</th>"
                    f"<th>&Delta;AUROC</th><th>z</th><th>q ({corr_short})</th><th>Sig.</th></tr>"
                )
                for c in sorted(delong, key=lambda x: x.p_value_corrected):
                    sig = "&#10003;" if c.is_significant else ""
                    html_parts.append(
                        f"<tr><td>{h(c.name_a)}</td><td>{h(c.name_b)}</td>"
                        f"<td>{c.auroc_a:.4f}</td><td>{c.auroc_b:.4f}</td>"
                        f"<td>{c.auroc_diff:+.4f}</td><td>{c.statistic:.2f}</td>"
                        f"<td>{h(format_pvalue(c.p_value_corrected))}</td><td>{sig}</td></tr>"
                    )
                html_parts.append("</table>")
                html_parts.append(
                    "<p><em>Full DeLong results in auroc_delong.csv.</em></p>"
                )

        # --- C. Threshold-free dominance figure -------------------------------
        try:
            domin = pp.plot_roc_pr_dominance(
                result,
                save_path=(
                    str(Path(output_dir) / "roc_pr_dominance.png")
                    if output_dir
                    else None
                ),
                return_base64=True,
            )
            if domin:
                html_parts.append("<h3>C. Threshold-free dominance (ROC / PR)</h3>")
                html_parts.append(
                    "<p>ML models contribute full curves; each caller contributes its "
                    "single native operating point. A caller below/inside every ML "
                    "curve is dominated regardless of threshold.</p>"
                )
                html_parts.append(_img(domin, "ROC and PR dominance"))
        except Exception as e:
            logger.warning(f"Could not generate dominance figure: {e}")

        # --- D. Per-classifier forest plot (descriptive) ----------------------
        try:
            forest = pp.plot_metric_forest(
                result,
                bootstrap_n_iterations=bootstrap_n_iterations,
                ci_level=bootstrap_ci_level,
                ci_method=bootstrap_ci_method,
                seed=seed,
                save_path=(
                    str(Path(output_dir) / "metric_forest.png") if output_dir else None
                ),
                return_base64=True,
            )
            if forest:
                html_parts.append("<h3>D. Per-classifier performance (MCC)</h3>")
                html_parts.append(
                    "<p>Matthews' correlation coefficient per classifier on the pooled "
                    f"out-of-fold data, with {int(bootstrap_ci_level * 100)}% "
                    f"{bootstrap_ci_method.upper()} bootstrap confidence intervals.</p>"
                )
                html_parts.append(_img(forest, "Per-classifier MCC forest plot"))
        except Exception as e:
            logger.warning(f"Could not generate forest plot: {e}")

        html_parts.append("</div>")
        self.sections.append("".join(html_parts))


class _VisualizationMixin:
    """Mixin for visualization sections: heatmap, feature importance, confusion matrices, error analysis, correlation."""

    def add_metrics_heatmap(self, results_df: pd.DataFrame):
        """
        Add heatmap of all metrics across models.

        Parameters
        ----------
        results_df : pd.DataFrame
            DataFrame with model results
        """
        # Filter to mean columns only (exclude std columns)
        metric_cols = [col for col in results_df.columns if not col.endswith("_std")]

        if len(metric_cols) == 0:
            logger.warning("No metrics found for heatmap")
            return

        data = results_df[metric_cols].values

        fig = go.Figure(
            data=go.Heatmap(
                z=data,
                x=metric_cols,
                y=results_df.index,
                colorscale="Viridis",
                text=np.round(data, 3),
                texttemplate="%{text}",
                textfont={"size": 12},
                colorbar=dict(
                    title=dict(text="Score", font=dict(size=14)), tickfont=dict(size=12)
                ),
            )
        )

        fig.update_layout(
            title="Model Performance Heatmap",
            title_font_size=18,
            xaxis_title="Metric",
            yaxis_title="Model",
            yaxis=dict(autorange="reversed"),  # Invert y-axis to match PNG version
            height=max(600, len(results_df) * 60),
            font=dict(size=14),
        )

        # Update axes font sizes
        fig.update_xaxes(tickfont=dict(size=13), title_font_size=16)
        fig.update_yaxes(tickfont=dict(size=13), title_font_size=16)

        html = f"""
        <div class="section">
            <h2>Performance Heatmap</h2>
            {fig.to_html(include_plotlyjs="cdn", div_id="metrics_heatmap")}
        </div>
        """
        self.sections.append(html)

    def add_feature_importance(
        self,
        feature_importances: Dict[str, np.ndarray],
        feature_names: List[str],
        top_n: int = 20,
    ):
        """
        Add feature importance visualization.

        Parameters
        ----------
        feature_importances : Dict[str, np.ndarray]
            Dict mapping model names to importance arrays
        feature_names : List[str]
            List of feature names
        top_n : int
            Number of top features to display
        """
        if not feature_importances:
            return

        # Create subplots for each model
        n_models = len(feature_importances)
        fig = make_subplots(
            rows=n_models,
            cols=1,
            subplot_titles=list(feature_importances.keys()),
            vertical_spacing=0.15 / max(n_models, 1),
        )

        for i, (model_name, importances) in enumerate(feature_importances.items(), 1):
            if importances is None or len(importances) == 0:
                continue

            # Get top N features
            indices = np.argsort(importances)[-top_n:][::-1]
            top_features = [feature_names[idx] for idx in indices]
            top_importances = importances[indices]

            fig.add_trace(
                go.Bar(
                    x=top_importances,
                    y=top_features,
                    orientation="h",
                    marker_color="darkturquoise",
                    showlegend=False,
                ),
                row=i,
                col=1,
            )

        # Calculate height based on number of features and models
        # Ensure sufficient space for each feature bar (approx 30 pixels per feature)
        height_per_model = max(400, top_n * 30)
        fig.update_layout(
            title="Feature Importance by Model",
            title_font_size=18,
            height=max(600, n_models * height_per_model),
            showlegend=False,
            font=dict(size=14),
        )

        fig.update_xaxes(title_text="Importance", title_font_size=16)

        html = f"""
        <div class="section" data-initially-collapsed="true">
            <h2>Feature Importance Analysis</h2>
            {fig.to_html(include_plotlyjs="cdn", div_id="feature_importance")}
        </div>
        """
        self.sections.append(html)

    def add_confusion_matrices(self, confusion_matrices: Dict[str, np.ndarray]):
        """
        Add confusion matrix visualizations.

        Parameters
        ----------
        confusion_matrices : Dict[str, np.ndarray]
            Dict mapping model names to confusion matrices
        """
        if not confusion_matrices:
            return

        html_parts = [
            '<div class="section">',
            "<h2>Confusion Matrices</h2>",
            "<p>Normalized confusion matrices showing classification performance for each model.</p>",
            '<div style="display: flex; flex-wrap: wrap; justify-content: space-evenly; align-items: flex-start;">',
        ]

        # Calculate size for 2x2 confusion matrix cells
        cell_size = 80  # pixels per cell
        plot_size = max(300, 2 * cell_size)

        for i, (model_name, cm) in enumerate(confusion_matrices.items()):
            # Flip the confusion matrix vertically to match inverted y-axis labels
            cm_flipped = cm[::-1]

            fig = go.Figure(
                data=go.Heatmap(
                    z=cm_flipped,
                    x=["Predicted 0", "Predicted 1"],
                    y=["Actual 1", "Actual 0"],  # Inverted y-axis
                    colorscale="Viridis",
                    zmin=0,
                    zmax=1,
                    text=np.round(cm_flipped, 3),
                    texttemplate="%{text}",
                    textfont={"size": 14},
                    colorbar=dict(
                        tickfont=dict(size=12),
                    ),
                )
            )

            fig.update_layout(
                title=h(model_name),
                title_font_size=16,
                xaxis=dict(constrain="domain"),
                height=plot_size + 150,  # Extra for title and x-axis labels
                width=plot_size + 200,  # Extra for y-axis labels and colorbar
                font=dict(size=14),
                margin=dict(l=80, r=80, t=80, b=80),
            )

            fig.update_xaxes(tickfont=dict(size=12), title_font_size=14)
            fig.update_yaxes(tickfont=dict(size=12), title_font_size=14)

            div_id = f"confusion_matrix_{model_name.replace(' ', '_')}"
            html_parts.append(
                f'<div style="flex: 0 0 auto;">{fig.to_html(include_plotlyjs="cdn", div_id=div_id)}</div>'
            )

        html_parts.append("</div>")  # Close flex container
        html_parts.append("</div>")  # Close section
        self.sections.append("".join(html_parts))

    def add_error_analysis(self, error_analysis: Dict[str, Any]):
        """
        Add error analysis section.

        Parameters
        ----------
        error_analysis : Dict[str, Any]
            Dictionary with error analysis results
        """
        if not error_analysis:
            return

        html_parts = ['<div class="section">', "<h2>Error Analysis</h2>"]

        # Add confidence threshold analysis if available
        if "confidence_analysis" in error_analysis:
            conf_data = error_analysis["confidence_analysis"]

            fig = go.Figure()
            for model_name, thresholds in conf_data.items():
                fig.add_trace(
                    go.Scatter(
                        x=list(thresholds.keys()),
                        y=list(thresholds.values()),
                        mode="lines+markers",
                        name=model_name,
                    )
                )

            fig.update_layout(
                title="Prediction Confidence vs Threshold",
                title_font_size=18,
                xaxis_title="Confidence Threshold",
                yaxis_title="Percentage of Predictions",
                height=600,
                font=dict(size=14),
            )

            # Update axes font sizes
            fig.update_xaxes(title_font_size=16, tickfont=dict(size=13))
            fig.update_yaxes(title_font_size=16, tickfont=dict(size=13))

            html_parts.append(
                fig.to_html(include_plotlyjs="cdn", div_id="confidence_analysis")
            )

        # Add misclassification summary table
        if "misclassification_summary" in error_analysis:
            summary = error_analysis["misclassification_summary"]
            html_parts.append("<h3>Misclassification Summary</h3>")
            html_parts.append('<table class="info-table">')
            html_parts.append(
                "<tr><th>Model</th><th>Total Errors</th><th>False Positives</th><th>False Negatives</th></tr>"
            )

            for model_name, stats in summary.items():
                html_parts.append(f"""
                <tr>
                    <td>{h(model_name)}</td>
                    <td>{stats.get("total_errors", 0)}</td>
                    <td>{stats.get("false_positives", 0)}</td>
                    <td>{stats.get("false_negatives", 0)}</td>
                </tr>
                """)

            html_parts.append("</table>")

        html_parts.append("</div>")
        self.sections.append("".join(html_parts))

    def add_feature_correlation(self, correlation_matrix: pd.DataFrame):
        """
        Add feature correlation matrix visualization (triangular).

        Parameters
        ----------
        correlation_matrix : pd.DataFrame
            Correlation matrix DataFrame
        """
        if correlation_matrix is None or correlation_matrix.empty:
            return

        # Create a mask for the upper triangle
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)

        # Apply mask to get lower triangle only
        corr_masked = correlation_matrix.copy()
        corr_masked = corr_masked.where(~mask)

        # Create text array with NaN values replaced by empty strings
        text_values = np.round(corr_masked.values, 2).astype(str)
        text_values[text_values == "nan"] = ""

        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=corr_masked.values,
                x=correlation_matrix.columns.tolist(),
                y=correlation_matrix.index.tolist(),
                colorscale="RdBu_r",
                zmid=0,
                zmin=-1,
                zmax=1,
                text=text_values,
                texttemplate="%{text}",
                textfont={"size": 10},  # Increased font size
                colorbar=dict(
                    title=dict(text="Correlation", font=dict(size=14)),
                    tickfont=dict(size=12),
                ),
            )
        )

        # Calculate height based on number of features to ensure all labels are visible
        # Use approximately 40 pixels per feature for proper spacing
        fig.update_layout(
            title="Feature Correlation Matrix",
            title_font_size=18,
            xaxis_title="Features",
            yaxis_title="Features",
            height=max(1000, len(correlation_matrix) * 40),
            xaxis={"side": "bottom", "showgrid": False},
            yaxis={"autorange": "reversed", "showgrid": False},
            font=dict(size=14),
        )

        # Rotate x-axis labels for better readability
        fig.update_xaxes(tickangle=45, tickfont=dict(size=12), title_font_size=16)
        fig.update_yaxes(tickfont=dict(size=12), title_font_size=16)

        html = f"""
        <div class="section" data-initially-collapsed="true">
            <h2>Feature Correlation Matrix</h2>
            {fig.to_html(include_plotlyjs="cdn", div_id="correlation_matrix")}
        </div>
        """
        self.sections.append(html)

    def add_feature_target_correlation(self, feature_target_corr: pd.DataFrame):
        """
        Add feature-target correlation visualization as a single row heatmap.

        Parameters
        ----------
        feature_target_corr : pd.DataFrame
            DataFrame with features as index and 'Correlation' column
        """
        if feature_target_corr is None or feature_target_corr.empty:
            return

        # Prepare data for single row heatmap
        features = feature_target_corr.index.tolist()
        correlations = feature_target_corr["Correlation"].values.reshape(1, -1)

        # Create text values with 2 decimal places
        text_values = np.round(correlations, 2).astype(str)

        # Create single-row heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=correlations,
                x=features,
                y=["Target Correlation"],
                colorscale="RdBu_r",
                zmid=0,
                zmin=-1,
                zmax=1,
                text=text_values,
                texttemplate="%{text}",
                textfont={"size": 12},
                colorbar=dict(
                    title=dict(text="Correlation", font=dict(size=14)),
                    tickfont=dict(size=12),
                    orientation="h",
                    x=0.5,
                    y=-1.6,
                    xanchor="center",
                    yanchor="top",
                    len=0.7,
                    thickness=20,
                    ypad=60,
                ),
            )
        )

        fig.update_layout(
            title="Feature-Target Correlation",
            title_font_size=18,
            xaxis_title="",
            height=400,  # Increased to accommodate horizontal colorbar with spacing
            xaxis={"side": "bottom", "showgrid": False},
            yaxis={"showgrid": False},
            font=dict(size=14),
            margin=dict(b=250),  # Add bottom margin for colorbar with spacing
        )

        # Rotate x-axis labels for better readability
        fig.update_xaxes(tickangle=45, tickfont=dict(size=12), title_font_size=16)
        fig.update_yaxes(tickfont=dict(size=14))

        html = f"""
        <div class="section">
            <h2>Feature-Target Correlation</h2>
            {fig.to_html(include_plotlyjs="cdn", div_id="feature_target_correlation")}
        </div>
        """
        self.sections.append(html)


class _CVResultsMixin:
    """Mixin for cross-validation results and threshold optimization sections."""

    def add_cross_validation_results(self, cv_results: Dict[str, pd.DataFrame]):
        """
        Add cross-validation fold-by-fold results grouped under Metrics Comparison.

        Parameters
        ----------
        cv_results : Dict[str, pd.DataFrame]
            Dict mapping model names to DataFrames with fold results
        """
        if not cv_results:
            return

        # Create box plots for ALL metrics across folds
        all_metrics = set()
        for df in cv_results.values():
            all_metrics.update(df.columns)

        # Remove non-metric columns (like 'fold_id' if present)
        metrics_to_plot = sorted([m for m in all_metrics if m != "fold_id"])

        if not metrics_to_plot:
            return

        # Start building the Metrics Comparison section
        html_parts = ['<div class="section">', "<h2>Metrics Comparison</h2>"]

        for metric in metrics_to_plot:
            fig = go.Figure()

            for model_name, df in cv_results.items():
                if metric in df.columns:
                    fig.add_trace(go.Box(y=df[metric], name=model_name, boxmean="sd"))

            # Set appropriate y-axis range for known metrics
            yaxis_config = {"title": metric}
            if metric == "Matthews Corr. Coef.":
                # MCC is in [0, 1] range (showing significant range)
                yaxis_config["range"] = [0, 1]
            elif metric in ["Brier Score", "ECE", "MCE"]:
                # Calibration error metrics: 0 is perfect, typically < 0.3
                yaxis_config["range"] = [0, 0.5]
            elif metric in [
                "AUROC",
                "Balanced Accuracy",
                "F1 Score (Weighted)",
                "F1 Score (Class 0)",
                "F1 Score (Class 1)",
                "Precision (Class 0)",
                "Precision (Class 1)",
                "Recall (Class 0)",
                "Recall (Class 1)",
                "Precision",
                "Recall",
                "Accuracy",
            ]:
                # Most metrics: show significant range [0.5, 1]
                yaxis_config["range"] = [0.5, 1]

            fig.update_layout(
                title=h(metric),
                title_font_size=18,
                yaxis=yaxis_config,
                xaxis_title="Model",
                height=600,
                font=dict(size=14),
            )

            # Update axes font sizes
            fig.update_xaxes(title_font_size=16, tickfont=dict(size=13))
            fig.update_yaxes(title_font_size=16, tickfont=dict(size=13))

            html_parts.append(f"<h3>{h(metric)}</h3>")
            html_parts.append(
                fig.to_html(
                    include_plotlyjs="cdn",
                    div_id=f"cv_{metric.replace(' ', '_').replace('.', '_')}",
                )
            )

        html_parts.append("</div>")
        self.sections.append("".join(html_parts))

    def add_threshold_results(self, threshold_summary: Dict[str, Dict[str, Any]]):
        """Add threshold optimization results section for all models.

        Parameters
        ----------
        threshold_summary : dict
            Dictionary mapping model names to threshold info. Each model entry
            contains keys: fold_thresholds (list), mean_threshold (float),
            std_threshold (float), recommended_threshold (float),
            mean_youden_j (float), mean_sensitivity_at_threshold (float),
            mean_specificity_at_threshold (float), n_folds (int).
        """
        if not threshold_summary:
            return

        html_parts = [
            '<div class="section">',
            "<h2>Threshold Optimization Results</h2>",
        ]

        html_parts.append(
            "<p>Optimal thresholds were computed using Youden's J statistic on "
            "training data within each fold to avoid data leakage.</p>"
        )

        # Summary table
        html_parts.append("<h3>Summary by Model</h3>")
        html_parts.append('<table class="info-table" style="table-layout: fixed;">')
        html_parts.append(
            "<tr><th>Model</th><th>Mean Threshold</th><th>Std</th>"
            "<th>Youden's J</th></tr>"
        )

        for model_name, info in sorted(threshold_summary.items()):
            mean_t = info.get("mean_threshold", 0.5)
            std_t = info.get("std_threshold", 0)
            youden = info.get("mean_youden_j", 0)

            html_parts.append(
                f"<tr><td>{h(model_name)}</td><td>{mean_t:.4f}</td><td>{std_t:.4f}</td>"
                f"<td>{youden:.4f}</td></tr>"
            )

        html_parts.append("</table>")

        # Box plot of thresholds across folds
        model_names = list(threshold_summary.keys())
        has_fold_data = any(
            info.get("fold_thresholds") for info in threshold_summary.values()
        )

        if has_fold_data:
            fig = go.Figure()

            for model_name in sorted(model_names):
                fold_thresholds = threshold_summary[model_name].get(
                    "fold_thresholds", []
                )
                if fold_thresholds:
                    fig.add_trace(
                        go.Box(y=fold_thresholds, name=model_name, boxmean="sd")
                    )

            fig.update_layout(
                title="Optimal Threshold Distribution Across Folds",
                title_font_size=18,
                yaxis_title="Threshold",
                xaxis_title="Model",
                height=500,
                font=dict(size=14),
            )

            fig.update_xaxes(title_font_size=16, tickfont=dict(size=13))
            fig.update_yaxes(title_font_size=16, tickfont=dict(size=13))

            html_parts.append("<h3>Threshold Distribution</h3>")
            html_parts.append(
                fig.to_html(include_plotlyjs="cdn", div_id="threshold_boxplot")
            )

        html_parts.append("</div>")
        self.sections.append("".join(html_parts))
