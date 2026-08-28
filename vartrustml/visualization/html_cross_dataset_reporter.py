"""
HTML report generation for cross-dataset generalizability analysis.

:class:`HTMLCrossDatasetReporter` builds the interactive HTML report showing
how models generalize from one dataset to another.

Classes
-------
HTMLCrossDatasetReporter
    Generate interactive cross-dataset generalizability reports.

See Also
--------
vartrustml.core.cross_dataset.CrossDatasetEvaluator : Evaluation engine.
vartrustml.visualization.html_reporter.HTMLCompareReporter : Single-dataset reports.
"""

import logging
from datetime import datetime
from html import escape as h
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, PackageLoader, select_autoescape

from vartrustml._version import __version__ as VARTRUSTML_VERSION
from vartrustml.visualization._html_styles import REPORT_JS, get_report_css

logger = logging.getLogger(__name__)


class HTMLCrossDatasetReporter:
    """Generate interactive HTML reports for cross-dataset generalizability.

    Creates reports analyzing how models trained on one dataset
    generalize to others. Includes performance matrices, generalization gap
    analysis, and best/worst dataset combination rankings.

    Parameters
    ----------
    output_path : str, default="cross_dataset_report.html"
        Path to save the HTML report.

    Attributes
    ----------
    output_path : pathlib.Path
        Path for the HTML report output.
    sections : list of str
        List of report section HTML strings.

    Notes
    -----
    Key analyses included:
    - Performance matrices (train dataset × test dataset)
    - Generalization gap analysis (within vs cross-dataset)
    - Best/worst dataset combination rankings
    - Executive summary with key insights

    See Also
    --------
    CrossDatasetEvaluator : Generates data for this reporter.
    HTMLCompareReporter : Single-dataset comparison reports.

    Examples
    --------
    >>> reporter = HTMLCrossDatasetReporter("results/cross_dataset.html")
    >>> reporter.add_overview(config, datasets_info)
    >>> reporter.add_performance_matrices(results_mean, results_std, names)
    >>> reporter.add_generalization_gap_analysis(results_mean, names)
    >>> reporter.generate_report()
    """

    def __init__(self, output_path: str = "cross_dataset_report.html"):
        self.output_path = Path(output_path)
        self.sections = []
        self.env = Environment(
            loader=PackageLoader("vartrustml.visualization", "templates"),
            autoescape=select_autoescape(["html"]),
        )

    def add_overview(self, config: Dict[str, Any], datasets_info: List[Dict[str, Any]]):
        """Add experiment overview section.

        Parameters
        ----------
        config : dict
            Experiment configuration dictionary.
        datasets_info : list of dict
            List of dataset info dicts with keys:
            - 'name': str - dataset name
            - 'n_samples': int - number of samples
            - 'n_features': int - number of features
            - 'class_distribution': dict - class counts
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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

        # Format threshold optimization info
        threshold_enabled = config.get("optimize_threshold", False)
        if threshold_enabled:
            threshold_method = config.get("threshold_method", "N/A")
            threshold_info = f"Enabled ({threshold_method})"
        else:
            threshold_info = "Disabled"

        # Format HPO info
        hpo_method = config.get("hpo_method", "grid")
        if hpo_method == "optuna":
            optuna_trials = config.get("optuna_n_trials", 100)
            hpo_info = f"Optuna ({optuna_trials} trials)"
        else:
            hpo_info = "Grid Search"

        # Build datasets table
        datasets_html = """
            <h3>Datasets</h3>
            <table class="info-table">
                <tr><th>Dataset</th><th>Samples</th><th>Features</th><th>Class 0</th><th>Class 1</th><th>Class Balance</th></tr>
        """
        for ds_info in datasets_info:
            class_dist = ds_info.get("class_distribution", {})
            class_0 = class_dist.get(0, class_dist.get("0", "N/A"))
            class_1 = class_dist.get(1, class_dist.get("1", "N/A"))
            if isinstance(class_0, (int, float)) and isinstance(class_1, (int, float)):
                total = class_0 + class_1
                if total > 0:
                    class_0_pct = class_0 / total * 100
                    class_1_pct = class_1 / total * 100
                    balance = f"{class_0_pct:.2f}% : {class_1_pct:.2f}%"
                else:
                    balance = "N/A"
            else:
                balance = "N/A"

            datasets_html += f"""
                <tr>
                    <td><strong>{h(ds_info.get("name", "Unknown"))}</strong></td>
                    <td>{ds_info.get("n_samples", "N/A")}</td>
                    <td>{ds_info.get("n_features", "N/A")}</td>
                    <td>{class_0}</td>
                    <td>{class_1}</td>
                    <td>{balance}</td>
                </tr>
            """
        datasets_html += "</table>"

        html = f"""
        <div class="section">
            <h2>Experiment Overview</h2>

            <h3>Metadata</h3>
            <table class="info-table">
                <tr><th>Property</th><th>Value</th></tr>
                <tr><td>VarTrustML Version</td><td>{VARTRUSTML_VERSION}</td></tr>
                <tr><td>Report Generated</td><td>{timestamp}</td></tr>
                <tr><td>Number of Datasets</td><td>{len(datasets_info)}</td></tr>
            </table>

            <h3>Configuration</h3>
            <table class="info-table">
                <tr><th>Parameter</th><th>Value</th></tr>
                <tr><td>Random Seed</td><td>{config.get("seed", "N/A")}</td></tr>
                <tr><td>Outer CV Splits</td><td>{config.get("n_outer_splits", "N/A")}</td></tr>
                <tr><td>Inner CV Splits</td><td>{config.get("n_inner_splits", "N/A")}</td></tr>
                <tr><td>HPO Method</td><td>{hpo_info}</td></tr>
                <tr><td>Calibration</td><td>{calibration_info}</td></tr>
                <tr><td>Threshold Optimization</td><td>{threshold_info}</td></tr>
                <tr><td>Models</td><td>{h(", ".join(config.get("models_to_use", [])))}</td></tr>
            </table>

            {datasets_html}
        </div>
        """
        self.sections.append(html)

    def add_distribution_shift(self, class_priors, shift_heatmap_b64=None):
        """Section: class-prior and per-feature distribution shift between samples."""
        rows = "".join(
            f"<tr><td>{h(str(name))}</td><td>{r['n']}</td><td>{r['n_pos']}</td>"
            f"<td>{r['n_neg']}</td><td>{r['positive_rate']:.3f}</td></tr>"
            for name, r in class_priors.iterrows()
        )
        img = ""
        if shift_heatmap_b64:
            img = (
                '<div style="text-align:center;margin:18px 0;">'
                f'<img src="data:image/png;base64,{shift_heatmap_b64}" '
                'alt="Per-feature distribution shift" '
                'style="max-width:100%;height:auto;"></div>'
            )
        html = f"""
        <div class="section">
            <h2>Distribution shift between samples</h2>
            <p>The cross-dataset gap should be read against how different the samples
            are. Below: the class-prior (label) shift, and per-feature covariate shift
            measured per variable type (two-sample Kolmogorov-Smirnov for continuous
            features, absolute difference in positive proportion for binary callers;
            both in [0, 1]).</p>
            <h3>Class prior</h3>
            <table class="info-table">
                <tr><th>Dataset</th><th>n</th><th>Positives</th><th>Negatives</th><th>Positive rate</th></tr>
                {rows}
            </table>
            {img}
        </div>
        """
        self.sections.append(html)

    def add_caller_baseline(self, baseline_table, metric="Matthews Corr. Coef."):
        """Section: variant-caller baseline (operating-point metric) per test sample."""
        header = "".join(f"<th>{h(str(c))}</th>" for c in baseline_table.columns)
        body = ""
        for name, row in baseline_table.iterrows():
            cells = "".join(
                f"<td>{v:.3f}</td>" if pd.notna(v) else "<td>n/a</td>" for v in row
            )
            body += f"<tr><td>{h(str(name))}</td>{cells}</tr>"
        html = f"""
        <div class="section">
            <h2>Variant-caller baseline</h2>
            <p>Operating-point {h(metric)} of each caller and default AND/OR
            combination on each test sample (callers do not train, so the value
            depends only on the test sample). Read against the ML cross-sample
            cells: a model trained on one sample should still beat these baselines
            when applied to an unseen sample.</p>
            <table class="info-table">
                <tr><th>Caller / combination</th>{header}</tr>
                {body}
            </table>
        </div>
        """
        self.sections.append(html)

    def add_generalization_gap_ci(self, gap_df, metric="Matthews Corr. Coef."):
        """Section: per-source generalization gap (in-sample minus cross-sample) with CI."""
        if gap_df is None or gap_df.empty:
            return
        body = ""
        for _, r in gap_df.sort_values(["model", "source"]).iterrows():
            body += (
                f"<tr><td>{h(str(r['model']))}</td><td>{h(str(r['source']))}</td>"
                f"<td>{r['in_sample']:.3f}</td><td>{r['cross_sample']:.3f}</td>"
                f"<td>{r['gap']:+.3f} [{r['gap_ci_lower']:+.3f}, {r['gap_ci_upper']:+.3f}]</td></tr>"
            )
        html = f"""
        <div class="section">
            <h2>Generalization gap</h2>
            <p>Per training source: the drop in {h(metric)} when a model is applied
            to the other samples instead of its own (in-sample minus cross-sample),
            with a percentile bootstrap CI over the outer folds. A gap whose CI
            excludes 0 is a reliable loss under sample shift.</p>
            <table class="info-table">
                <tr><th>Model</th><th>Train source</th><th>In-sample</th><th>Cross-sample</th><th>Gap (95% CI)</th></tr>
                {body}
            </table>
        </div>
        """
        self.sections.append(html)

    def add_lodo(self, lodo_df, metric="Matthews Corr. Coef."):
        """Section: leave-one-dataset-out (train on N-1 pooled, test on held-out)."""
        if lodo_df is None or lodo_df.empty:
            return
        body = ""
        for _, r in lodo_df.sort_values(["model", "held_out"]).iterrows():
            pw = (
                f"{r['pairwise_cross']:.3f}" if pd.notna(r["pairwise_cross"]) else "n/a"
            )
            dl = f"{r['delta']:+.3f}" if pd.notna(r["delta"]) else "n/a"
            body += (
                f"<tr><td>{h(str(r['model']))}</td><td>{h(str(r['held_out']))}</td>"
                f"<td>{r['lodo']:.3f} [{r['lodo_ci_lower']:.3f}, {r['lodo_ci_upper']:.3f}]</td>"
                f"<td>{pw}</td><td>{dl}</td></tr>"
            )
        html = f"""
        <div class="section">
            <h2>Leave-one-dataset-out</h2>
            <p>Each model is trained on the pooled N-1 samples and tested on the
            held-out sample ({h(metric)} with a bootstrap CI over folds). The last
            columns compare against the average single-source cross-sample result:
            a positive delta means pooling several samples generalizes better than
            training on one.</p>
            <table class="info-table">
                <tr><th>Model</th><th>Held-out</th><th>LODO (95% CI)</th><th>Pairwise cross-sample</th><th>Delta</th></tr>
                {body}
            </table>
        </div>
        """
        self.sections.append(html)

    def add_performance_matrices(
        self,
        results_mean: Dict[str, Dict[str, pd.DataFrame]],
        results_std: Dict[str, Dict[str, pd.DataFrame]],
        dataset_names: List[str],
        primary_metric: str = "AUROC",
    ):
        """Add interactive heatmaps for each model/metric combination.

        Parameters
        ----------
        results_mean : dict of {str: dict of {str: pandas.DataFrame}}
            Nested dict ``[model_name][metric_name]`` mapping to a DataFrame
            with train datasets as rows and test datasets as columns.
        results_std : dict of {str: dict of {str: pandas.DataFrame}}
            Same structure as *results_mean* but containing standard deviations.
        dataset_names : list of str
            Dataset names used for axis labels.
        primary_metric : str, default="AUROC"
            Primary metric to show first.
        """
        if not results_mean:
            return

        html_parts = [
            '<div class="section">',
            "<h2>Cross-Dataset Performance Matrices</h2>",
        ]
        html_parts.append(
            "<p>Heatmaps showing performance when training on row dataset and testing on column dataset. "
            "Diagonal represents within-dataset performance (same train/test source).</p>"
        )

        # Get all models and metrics
        model_names = list(results_mean.keys())
        if not model_names:
            return

        # Get metrics from first model
        first_model = model_names[0]
        all_metrics = list(results_mean[first_model].keys())

        # Ensure primary metric is first if available
        if primary_metric in all_metrics:
            all_metrics.remove(primary_metric)
            all_metrics.insert(0, primary_metric)

        for model_name in model_names:
            html_parts.append(f"<h3>{h(model_name)}</h3>")
            html_parts.append(
                '<div style="display: flex; flex-wrap: wrap; justify-content: space-evenly; align-items: flex-start;">'
            )

            for metric_name in all_metrics:
                if metric_name not in results_mean[model_name]:
                    continue

                mean_df = results_mean[model_name][metric_name]
                std_df = results_std.get(model_name, {}).get(metric_name)

                # Create heatmap with hover showing mean ± std
                if std_df is not None:
                    hover_text = []
                    for i in range(len(mean_df)):
                        row_text = []
                        for j in range(len(mean_df.columns)):
                            mean_val = mean_df.iloc[i, j]
                            std_val = std_df.iloc[i, j]
                            row_text.append(
                                f"Train: {mean_df.index[i]}<br>Test: {mean_df.columns[j]}<br>"
                                f"{metric_name}: {mean_val:.3f} ± {std_val:.3f}"
                            )
                        hover_text.append(row_text)
                else:
                    hover_text = None

                # Create annotation text (mean ± std)
                if std_df is not None:
                    annotations = []
                    for i in range(len(mean_df)):
                        row_annot = []
                        for j in range(len(mean_df.columns)):
                            mean_val = mean_df.iloc[i, j]
                            std_val = std_df.iloc[i, j]
                            row_annot.append(f"{mean_val:.3f}<br>±{std_val:.3f}")
                        annotations.append(row_annot)
                else:
                    annotations = np.round(mean_df.values, 3).astype(str)

                zmin_val = 0 if metric_name == "Matthews Corr. Coef." else 0.5

                fig = go.Figure(
                    data=go.Heatmap(
                        z=mean_df.values,
                        x=mean_df.columns.tolist(),
                        y=mean_df.index.tolist(),
                        colorscale="Viridis",
                        zmin=zmin_val,
                        zmax=1,
                        text=annotations,
                        texttemplate="%{text}",
                        textfont={"size": 10},
                        hovertext=hover_text,
                        hovertemplate="%{hovertext}<extra></extra>"
                        if hover_text
                        else None,
                        colorbar=dict(
                            tickfont=dict(size=12),
                        ),
                    )
                )

                # Calculate size for square matrix cells
                n_datasets = len(dataset_names)
                cell_size = 80  # pixels per cell
                plot_size = max(300, n_datasets * cell_size)

                fig.update_layout(
                    title=f"{h(metric_name)}",
                    title_font_size=16,
                    xaxis_title="Test Dataset",
                    yaxis_title="Train Dataset",
                    yaxis=dict(autorange="reversed"),
                    xaxis=dict(constrain="domain"),
                    height=plot_size + 150,  # Extra for title and x-axis labels
                    width=plot_size + 200,  # Extra for y-axis labels and colorbar
                    font=dict(size=14),
                    margin=dict(l=100, r=100, t=80, b=100),
                )

                fig.update_xaxes(
                    tickfont=dict(size=12),
                    title_font_size=14,
                    tickangle=45,
                    constrain="domain",
                )
                fig.update_yaxes(
                    tickfont=dict(size=12), title_font_size=14, constrain="domain"
                )

                div_id = f"heatmap_{model_name.replace(' ', '_')}_{metric_name.replace(' ', '_').replace('.', '_')}"
                html_parts.append(
                    f'<div style="flex: 0 0 auto;">{fig.to_html(include_plotlyjs="cdn", div_id=div_id)}</div>'
                )

            # Close flex container for this model
            html_parts.append("</div>")

        html_parts.append("</div>")
        self.sections.append("".join(html_parts))

    def add_generalization_gap_analysis(
        self,
        results_mean: Dict[str, Dict[str, pd.DataFrame]],
        dataset_names: List[str],
        metrics: Optional[List[str]] = None,
    ):
        """Compare within-dataset vs cross-dataset (off-diagonal) performance.

        Parameters
        ----------
        results_mean : dict of {str: dict of {str: pandas.DataFrame}}
            Nested dict ``[model_name][metric_name]`` mapping to a DataFrame
            with train datasets as rows and test datasets as columns.
        dataset_names : list of str
            Dataset names.
        metrics : list of str, optional
            Metrics to analyze. Defaults to AUROC, F1 (Weighted), and MCC.
        """
        if not results_mean:
            return

        if metrics is None:
            metrics = ["AUROC", "F1 Score (Weighted)", "Matthews Corr. Coef."]

        html_parts = ['<div class="section">', "<h2>Generalization Gap Analysis</h2>"]
        html_parts.append(
            "<p>Comparing within-dataset performance (training and testing on same dataset) "
            "vs cross-dataset performance (training on one dataset, testing on another).</p>"
        )

        model_names = list(results_mean.keys())
        gap_data = []

        for model_name in model_names:
            for metric_name in metrics:
                if metric_name not in results_mean[model_name]:
                    continue

                mean_df = results_mean[model_name][metric_name]

                # Extract diagonal (within-dataset) values
                diagonal_vals = []
                for ds in dataset_names:
                    if ds in mean_df.index and ds in mean_df.columns:
                        diagonal_vals.append(mean_df.loc[ds, ds])

                # Extract off-diagonal (cross-dataset) values
                off_diag_vals = []
                for train_ds in dataset_names:
                    for test_ds in dataset_names:
                        if train_ds != test_ds:
                            if train_ds in mean_df.index and test_ds in mean_df.columns:
                                off_diag_vals.append(mean_df.loc[train_ds, test_ds])

                if diagonal_vals and off_diag_vals:
                    within_mean = np.mean(diagonal_vals)
                    cross_mean = np.mean(off_diag_vals)
                    gap = within_mean - cross_mean

                    gap_data.append(
                        {
                            "Model": model_name,
                            "Metric": metric_name,
                            "Within-Dataset": within_mean,
                            "Cross-Dataset": cross_mean,
                            "Gap": gap,
                            "Gap %": (gap / within_mean * 100)
                            if within_mean != 0
                            else 0,
                        }
                    )

        if not gap_data:
            html_parts.append("<p>No data available for gap analysis.</p>")
            html_parts.append("</div>")
            self.sections.append("".join(html_parts))
            return

        gap_df = pd.DataFrame(gap_data)

        # Create grouped bar chart
        for metric_name in metrics:
            metric_df = gap_df[gap_df["Metric"] == metric_name]
            if metric_df.empty:
                continue

            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    name="Within-Dataset",
                    x=metric_df["Model"],
                    y=metric_df["Within-Dataset"],
                    marker_color="#2ecc71",
                    text=metric_df["Within-Dataset"].round(3),
                    textposition="outside",
                )
            )

            fig.add_trace(
                go.Bar(
                    name="Cross-Dataset",
                    x=metric_df["Model"],
                    y=metric_df["Cross-Dataset"],
                    marker_color="#3498db",
                    text=metric_df["Cross-Dataset"].round(3),
                    textposition="outside",
                )
            )

            fig.update_layout(
                title=f"Generalization Gap: {h(metric_name)}",
                title_font_size=16,
                xaxis_title="Model",
                yaxis_title=metric_name,
                barmode="group",
                height=500,
                font=dict(size=14),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
            )

            # Set y-axis range based on metric
            if metric_name in ["AUROC", "F1 Score (Weighted)", "Balanced Accuracy"]:
                fig.update_yaxes(range=[0, 1])
            elif metric_name == "Matthews Corr. Coef.":
                fig.update_yaxes(range=[0, 1])

            fig.update_xaxes(tickfont=dict(size=12), title_font_size=14)
            fig.update_yaxes(tickfont=dict(size=12), title_font_size=14)

            div_id = f"gap_{metric_name.replace(' ', '_').replace('.', '_')}"
            html_parts.append(fig.to_html(include_plotlyjs="cdn", div_id=div_id))

        # Add summary table
        html_parts.append("<h3>Generalization Gap Summary</h3>")
        html_parts.append('<table class="info-table">')
        html_parts.append(
            "<tr><th>Model</th><th>Metric</th><th>Within-Dataset</th>"
            "<th>Cross-Dataset</th><th>Gap</th><th>Gap %</th></tr>"
        )

        for _, row in gap_df.iterrows():
            html_parts.append(f"""
                <tr>
                    <td><strong>{h(str(row["Model"]))}</strong></td>
                    <td>{h(str(row["Metric"]))}</td>
                    <td>{row["Within-Dataset"]:.3f}</td>
                    <td>{row["Cross-Dataset"]:.3f}</td>
                    <td>{row["Gap"]:.3f}</td>
                    <td>{row["Gap %"]:.1f}%</td>
                </tr>
            """)

        html_parts.append("</table>")
        html_parts.append("</div>")
        self.sections.append("".join(html_parts))

    def add_best_worst_combinations(
        self,
        results_mean: Dict[str, Dict[str, pd.DataFrame]],
        dataset_names: List[str],
        metric_name: str = "AUROC",
        n_top: int = 5,
    ):
        """Highlight best and worst train-test combinations per model.

        Parameters
        ----------
        results_mean : dict of {str: dict of {str: pandas.DataFrame}}
            Nested dict ``[model_name][metric_name]`` mapping to a DataFrame
            with train datasets as rows and test datasets as columns.
        dataset_names : list of str
            Dataset names.
        metric_name : str, default="AUROC"
            Metric to use for ranking.
        n_top : int, default=5
            Number of top/bottom combinations to show.
        """
        if not results_mean:
            return

        html_parts = [
            '<div class="section">',
            "<h2>Best & Worst Dataset Combinations</h2>",
        ]
        html_parts.append(
            f"<p>Top {n_top} best and worst train→test combinations based on {h(metric_name)}.</p>"
        )

        model_names = list(results_mean.keys())

        for model_name in model_names:
            if metric_name not in results_mean[model_name]:
                continue

            mean_df = results_mean[model_name][metric_name]

            html_parts.append(f"<h3>{h(model_name)}</h3>")

            # Flatten the matrix to get all combinations
            combinations = []
            for train_ds in mean_df.index:
                for test_ds in mean_df.columns:
                    score = mean_df.loc[train_ds, test_ds]
                    is_same = train_ds == test_ds
                    combinations.append(
                        {
                            "Train": train_ds,
                            "Test": test_ds,
                            "Score": score,
                            "Type": "Within-Dataset" if is_same else "Cross-Dataset",
                        }
                    )

            comb_df = pd.DataFrame(combinations)

            # Best combinations
            best = comb_df.nlargest(n_top, "Score")
            worst = comb_df.nsmallest(n_top, "Score")

            # Create side-by-side tables
            html_parts.append(
                '<div style="display: flex; gap: 40px; flex-wrap: wrap;">'
            )

            # Best table
            html_parts.append('<div style="flex: 1; min-width: 300px;">')
            html_parts.append(f"<h4>Top {n_top} Combinations</h4>")
            html_parts.append('<table class="info-table">')
            html_parts.append(
                f"<tr><th>Train → Test</th><th>{h(metric_name)}</th><th>Type</th></tr>"
            )

            for _, row in best.iterrows():
                html_parts.append(f"""
                    <tr>
                        <td><strong>{h(str(row["Train"]))}</strong> → <strong>{h(str(row["Test"]))}</strong></td>
                        <td>{row["Score"]:.3f}</td>
                        <td>{row["Type"]}</td>
                    </tr>
                """)

            html_parts.append("</table>")
            html_parts.append("</div>")

            # Worst table
            html_parts.append('<div style="flex: 1; min-width: 300px;">')
            html_parts.append(f"<h4>Bottom {n_top} Combinations</h4>")
            html_parts.append('<table class="info-table">')
            html_parts.append(
                f"<tr><th>Train → Test</th><th>{h(metric_name)}</th><th>Type</th></tr>"
            )

            for _, row in worst.iterrows():
                html_parts.append(f"""
                    <tr>
                        <td><strong>{h(str(row["Train"]))}</strong> → <strong>{h(str(row["Test"]))}</strong></td>
                        <td>{row["Score"]:.3f}</td>
                        <td>{row["Type"]}</td>
                    </tr>
                """)

            html_parts.append("</table>")
            html_parts.append("</div>")

            html_parts.append("</div>")  # Close flex container

        html_parts.append("</div>")
        self.sections.append("".join(html_parts))

    def add_cross_dataset_summary(
        self,
        results_mean: Dict[str, Dict[str, pd.DataFrame]],
        dataset_names: List[str],
        metric_name: str = "AUROC",
    ):
        """Add overall summary with key insights.

        Parameters
        ----------
        results_mean : dict of {str: dict of {str: pandas.DataFrame}}
            Nested dict ``[model_name][metric_name]`` mapping to a DataFrame
            with train datasets as rows and test datasets as columns.
        dataset_names : list of str
            Dataset names.
        metric_name : str, default="AUROC"
            Primary metric for summary.
        """
        if not results_mean:
            return

        html_parts = ['<div class="section">', "<h2>Executive Summary</h2>"]

        model_names = list(results_mean.keys())
        summary_stats = []

        for model_name in model_names:
            if metric_name not in results_mean[model_name]:
                continue

            mean_df = results_mean[model_name][metric_name]

            # Calculate key statistics
            diagonal_vals = [
                mean_df.loc[ds, ds]
                for ds in dataset_names
                if ds in mean_df.index and ds in mean_df.columns
            ]
            all_vals = mean_df.values.flatten()
            off_diag_vals = [
                v
                for i, v in enumerate(all_vals)
                if i // len(dataset_names) != i % len(dataset_names)
            ]

            if diagonal_vals and off_diag_vals:
                summary_stats.append(
                    {
                        "Model": model_name,
                        "Within-Dataset Mean": np.mean(diagonal_vals),
                        "Cross-Dataset Mean": np.mean(off_diag_vals),
                        "Overall Mean": np.mean(all_vals),
                        "Generalization Gap": np.mean(diagonal_vals)
                        - np.mean(off_diag_vals),
                        "Best Cross-Dataset": np.max(off_diag_vals),
                        "Worst Cross-Dataset": np.min(off_diag_vals),
                    }
                )

        if not summary_stats:
            html_parts.append("<p>No summary data available.</p>")
            html_parts.append("</div>")
            self.sections.append("".join(html_parts))
            return

        summary_df = pd.DataFrame(summary_stats)

        # Summary table
        html_parts.append("<h3>Model Summary</h3>")
        html_parts.append('<table class="info-table">')
        html_parts.append(
            "<tr><th>Model</th><th>Within-Dataset</th><th>Cross-Dataset</th>"
            "<th>Gap</th><th>Best Cross</th><th>Worst Cross</th></tr>"
        )

        for _, row in summary_df.iterrows():
            html_parts.append(f"""
                <tr>
                    <td><strong>{h(str(row["Model"]))}</strong></td>
                    <td>{row["Within-Dataset Mean"]:.3f}</td>
                    <td>{row["Cross-Dataset Mean"]:.3f}</td>
                    <td>{row["Generalization Gap"]:.3f}</td>
                    <td>{row["Best Cross-Dataset"]:.3f}</td>
                    <td>{row["Worst Cross-Dataset"]:.3f}</td>
                </tr>
            """)

        html_parts.append("</table>")
        html_parts.append("</div>")
        self.sections.insert(0, "".join(html_parts))  # Insert at beginning

    def generate_report(self) -> str:
        """Generate and save the complete HTML report.

        Returns
        -------
        str
            Path to the saved report file.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Use base template with shared CSS/JS (purple theme for cross-dataset)
        base_tpl = self.env.get_template("base.html.j2")
        html_content = base_tpl.render(
            title="Cross-Dataset Generalizability Report",
            subtitle="Generated automatically by VarTrustML (cross-dataset command)",
            timestamp=timestamp,
            css_content=get_report_css("mauve"),
            js_content=REPORT_JS,
            sections=self.sections,
        )

        # Save to file
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"Cross-dataset HTML report saved to: {self.output_path}")
        return str(self.output_path)
