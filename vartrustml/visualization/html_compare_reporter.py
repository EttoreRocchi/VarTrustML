"""HTML report generator for multi-model comparison experiments."""

import logging
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape

from vartrustml.visualization._html_mixins import (
    _CallerComparisonMixin,
    _CVResultsMixin,
    _OverviewMixin,
    _StatisticalMixin,
    _VisualizationMixin,
)
from vartrustml.visualization._html_styles import REPORT_JS, get_report_css

logger = logging.getLogger(__name__)


class HTMLCompareReporter(
    _OverviewMixin,
    _CallerComparisonMixin,
    _StatisticalMixin,
    _VisualizationMixin,
    _CVResultsMixin,
):
    """Generate interactive HTML reports for model comparison.

    Creates HTML reports with Plotly visualizations for
    multi-model cross-validation experiments. Reports include performance
    comparisons, statistical tests, confusion matrices, and feature importance.

    Parameters
    ----------
    output_path : str, default="report.html"
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
    HTMLCrossDatasetReporter : Reports for cross-dataset evaluation.
    Visualizer : Static plot generation.

    Examples
    --------
    >>> reporter = HTMLCompareReporter("results/report.html")
    >>> reporter.add_overview(config_dict, dataset_info)
    >>> reporter.add_metrics_heatmap(results_df)
    >>> reporter.generate_report()
    """

    def __init__(self, output_path: str = "report.html"):
        self.output_path = Path(output_path)
        self.sections = []
        self.env = Environment(
            loader=PackageLoader("vartrustml.visualization", "templates"),
            autoescape=select_autoescape(["html"]),
        )

    def generate_report(self):
        """
        Generate and save the complete HTML report.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Use base template with shared CSS/JS
        base_tpl = self.env.get_template("base.html.j2")
        html_content = base_tpl.render(
            title="VarTrustML Experiment Report",
            subtitle="Generated automatically by VarTrustML (compare-models command)",
            timestamp=timestamp,
            css_content=get_report_css("slate_blue"),
            js_content=REPORT_JS,
            sections=self.sections,
        )

        # Save to file
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"HTML report saved to: {self.output_path}")
        return str(self.output_path)
