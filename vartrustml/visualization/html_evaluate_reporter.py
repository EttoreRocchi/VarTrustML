"""HTML report generator for model evaluation experiments."""

import logging
from datetime import datetime
from html import escape as h
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from jinja2 import Environment, PackageLoader, select_autoescape

from vartrustml.visualization._html_mixins import render_confusion_matrix_html
from vartrustml.visualization._html_styles import REPORT_JS, get_report_css

logger = logging.getLogger(__name__)


class HTMLEvaluateReporter:
    """Generate interactive HTML reports for model evaluation.

    Creates HTML reports with Plotly visualizations for evaluating a saved
    model on a labeled dataset. Reports include metrics overview, confusion
    matrix, and per-class classification metrics.

    Parameters
    ----------
    output_path : str, default="evaluate_report.html"
        Path to save the HTML report.

    Attributes
    ----------
    output_path : pathlib.Path
        Path for the HTML report output.
    sections : list of str
        List of report section HTML strings.

    See Also
    --------
    HTMLTrainReporter : Reports for single model training.
    HTMLCompareReporter : Reports for multi-model comparison.

    Examples
    --------
    >>> reporter = HTMLEvaluateReporter("results/evaluate_report.html")
    >>> reporter.add_overview(model_path, data_path, target)
    >>> reporter.add_metrics_table(metrics)
    >>> reporter.add_confusion_matrix(cm)
    >>> reporter.generate_report()
    """

    def __init__(self, output_path: str = "evaluate_report.html"):
        self.output_path = Path(output_path)
        self.sections = []
        self.env = Environment(
            loader=PackageLoader("vartrustml.visualization", "templates"),
            autoescape=select_autoescape(["html"]),
        )

    def add_overview(
        self,
        model_path: str,
        data_path: str,
        target_column: str,
        n_samples: int,
        model_info: Optional[Dict[str, Any]] = None,
    ):
        """
        Add evaluation overview section.

        Parameters
        ----------
        model_path : str
            Path to the saved model
        data_path : str
            Path to the evaluation data
        target_column : str
            Name of the target column
        n_samples : int
            Number of samples in the evaluation dataset
        model_info : Optional[Dict[str, Any]]
            Optional dict with model metadata (type, threshold, etc.)
        """
        # Format model info if available
        model_type = "Unknown"
        optimal_threshold = "0.5 (default)"
        if model_info:
            model_type = model_info.get("model_type", "Unknown")
            if model_info.get("threshold_metadata"):
                optimal_threshold = (
                    f"{model_info.get('optimal_threshold', 0.5):.4f} (optimized)"
                )

        html = f"""
        <div class="section">
            <h2>Evaluation Overview</h2>
            <table class="info-table">
                <tr><th>Parameter</th><th>Value</th></tr>
                <tr><td>Model Path</td><td>{h(str(model_path))}</td></tr>
                <tr><td>Model Type</td><td>{h(str(model_type))}</td></tr>
                <tr><td>Classification Threshold</td><td>{h(str(optimal_threshold))}</td></tr>
                <tr><td>Data Path</td><td>{h(str(data_path))}</td></tr>
                <tr><td>Target Column</td><td>{h(str(target_column))}</td></tr>
                <tr><td>Evaluation Samples</td><td>{n_samples:,}</td></tr>
            </table>
        </div>
        """
        self.sections.append(html)

    def add_metrics_table(self, metrics: Dict[str, float]):
        """
        Add evaluation metrics table.

        Parameters
        ----------
        metrics : Dict[str, float]
            Dictionary mapping metric names to values
        """
        if not metrics:
            return

        html_parts = [
            '<div class="section">',
            "<h2>Evaluation Metrics</h2>",
            '<table class="info-table">',
            "<tr><th>Metric</th><th>Value</th></tr>",
        ]

        for name, value in metrics.items():
            if isinstance(value, float):
                html_parts.append(f"<tr><td>{h(name)}</td><td>{value:.4f}</td></tr>")
            else:
                html_parts.append(
                    f"<tr><td>{h(name)}</td><td>{h(str(value))}</td></tr>"
                )

        html_parts.append("</table>")
        html_parts.append("</div>")
        self.sections.append("".join(html_parts))

    def add_confusion_matrix(
        self, confusion_matrix: np.ndarray, normalize: bool = True
    ):
        """Add confusion matrix visualization."""
        if confusion_matrix is None or confusion_matrix.size == 0:
            return
        self.sections.append(
            render_confusion_matrix_html(
                confusion_matrix, normalize, div_id="eval_confusion_matrix"
            )
        )

    def add_classification_report(
        self,
        precision: Dict[str, float],
        recall: Dict[str, float],
        f1: Dict[str, float],
        support: Dict[str, int],
    ):
        """
        Add per-class classification metrics.

        Parameters
        ----------
        precision : Dict[str, float]
            Dict of class -> precision
        recall : Dict[str, float]
            Dict of class -> recall
        f1 : Dict[str, float]
            Dict of class -> f1-score
        support : Dict[str, int]
            Dict of class -> support count
        """
        html_parts = [
            '<div class="section">',
            "<h2>Per-Class Classification Metrics</h2>",
            '<table class="info-table">',
            "<tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr>",
        ]

        for cls in precision.keys():
            html_parts.append(
                f"<tr><td>{h(str(cls))}</td>"
                f"<td>{precision[cls]:.4f}</td>"
                f"<td>{recall[cls]:.4f}</td>"
                f"<td>{f1[cls]:.4f}</td>"
                f"<td>{support[cls]:,}</td></tr>"
            )

        html_parts.append("</table>")
        html_parts.append("</div>")
        self.sections.append("".join(html_parts))

    def generate_report(self):
        """
        Generate and save the complete HTML report.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Use base template with shared CSS/JS
        base_tpl = self.env.get_template("base.html.j2")
        html_content = base_tpl.render(
            title="VarTrustML Evaluation Report",
            subtitle="Generated automatically by VarTrustML (evaluate command)",
            timestamp=timestamp,
            css_content=get_report_css("taupe"),
            js_content=REPORT_JS,
            sections=self.sections,
        )

        # Save to file
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"HTML evaluation report saved to: {self.output_path}")
        return str(self.output_path)
