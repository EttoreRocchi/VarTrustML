"""
Evaluation CLI command for VarTrustML.

Contains:
- evaluate: Evaluate a saved model on a labeled dataset
"""

from io import StringIO
from pathlib import Path
from typing import Optional

import numpy as np
import typer

from vartrustml import DataLoader
from vartrustml.core.train_model import ModelTrainer

from vartrustml.cli.main import app


@app.command("evaluate", rich_help_panel="Model Application")
def evaluate(
    model_path: Path = typer.Argument(..., help="Path to saved model."),
    data: Path = typer.Argument(..., help="Path to test data."),
    target: str = typer.Option(
        ...,
        "--target",
        "-t",
        help="Target column (REQUIRED).",
        rich_help_panel="Data Options",
    ),
    output: Optional[Path] = typer.Option(
        Path("evaluation.txt"),
        "--output",
        "-o",
        help="Where to save text report.",
        rich_help_panel="Output",
    ),
    no_html_report: bool = typer.Option(
        False,
        "--no-html-report",
        help="Disable HTML report generation.",
        rich_help_panel="Reports",
    ),
    html_path: Optional[str] = typer.Option(
        None,
        "--html-path",
        help="Path for HTML report. [default: evaluate_report.html]",
        rich_help_panel="Reports",
    ),
    id_column: Optional[str] = typer.Option(
        None,
        "--id-column",
        help="Column to use as row identifier (excluded from features).",
        rich_help_panel="Data Options",
    ),
):
    """
    Evaluate a saved model on a labeled dataset.
    """
    from sklearn.metrics import (
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    typer.echo(f"Loading model from {model_path}")
    model_data = ModelTrainer.load_model(str(model_path))
    model = model_data["model"]
    optimal_threshold = model_data.get("optimal_threshold", 0.5)
    threshold_metadata = model_data.get("threshold_metadata")

    typer.echo(f"Loading test data from {data}")
    loader = DataLoader(str(data.parent))
    # Held-out rows are scored as given, matching `predict`: dropping duplicates
    # here would silently change the support behind every reported metric
    df = loader.load_dataset(data.name, id_column=id_column, drop_duplicates=False)
    X = df.drop(columns=[target])
    y = df[target]
    y_prob = model.predict_proba(X)

    # Predictions at default threshold (0.5)
    y_pred_default = model.predict(X)

    # Predictions at optimized threshold (if available)
    has_optimized_threshold = (
        threshold_metadata is not None and optimal_threshold != 0.5
    )
    if has_optimized_threshold:
        y_pred_optimized = (y_prob[:, 1] >= optimal_threshold).astype(int)
    else:
        y_pred_optimized = y_pred_default

    # Calculate metrics at default threshold
    balanced_acc_default = balanced_accuracy_score(y, y_pred_default)
    mcc_default = matthews_corrcoef(y, y_pred_default)
    f1_weighted_default = f1_score(y, y_pred_default, average="weighted")
    auroc = roc_auc_score(y, y_prob[:, 1]) if y_prob.shape[1] == 2 else None
    cm_default = confusion_matrix(y, y_pred_default)

    # Calculate metrics at optimized threshold (if different from default)
    if has_optimized_threshold:
        balanced_acc_optimized = balanced_accuracy_score(y, y_pred_optimized)
        mcc_optimized = matthews_corrcoef(y, y_pred_optimized)
        f1_weighted_optimized = f1_score(y, y_pred_optimized, average="weighted")
        cm_optimized = confusion_matrix(y, y_pred_optimized)

    # Text report
    buf = StringIO()
    buf.write("\n============================================================\n")
    buf.write("EVALUATION RESULTS\n")
    buf.write("============================================================\n")

    if has_optimized_threshold:
        buf.write("\nMetrics at DEFAULT threshold (0.5):\n")
        buf.write(f"  Balanced Accuracy: {balanced_acc_default:.4f}\n")
        buf.write(f"  Matthews Corr. Coef.: {mcc_default:.4f}\n")
        buf.write(f"  F1 Score (weighted): {f1_weighted_default:.4f}\n")
        buf.write(f"\nMetrics at OPTIMIZED threshold ({optimal_threshold:.4f}):\n")
        buf.write(f"  Balanced Accuracy: {balanced_acc_optimized:.4f}\n")
        buf.write(f"  Matthews Corr. Coef.: {mcc_optimized:.4f}\n")
        buf.write(f"  F1 Score (weighted): {f1_weighted_optimized:.4f}\n")
        buf.write("\nDifference (optimized - default):\n")
        buf.write(
            f"  Balanced Accuracy: {balanced_acc_optimized - balanced_acc_default:+.4f}\n"
        )
        buf.write(f"  Matthews Corr. Coef.: {mcc_optimized - mcc_default:+.4f}\n")
        buf.write(
            f"  F1 Score (weighted): {f1_weighted_optimized - f1_weighted_default:+.4f}\n"
        )
    else:
        buf.write(f"Balanced Accuracy: {balanced_acc_default:.4f}\n")
        buf.write(f"Matthews Corr. Coef.: {mcc_default:.4f}\n")
        buf.write(f"F1 Score (weighted): {f1_weighted_default:.4f}\n")

    if auroc is not None:
        buf.write(f"\nAUROC: {auroc:.4f}\n")

    # Use optimized threshold predictions for classification report if available
    y_pred_report = y_pred_optimized if has_optimized_threshold else y_pred_default
    cm_report = cm_optimized if has_optimized_threshold else cm_default

    buf.write("\nClassification Report")
    if has_optimized_threshold:
        buf.write(f" (at threshold {optimal_threshold:.4f})")
    buf.write(":\n")
    buf.write(classification_report(y, y_pred_report))
    buf.write("\nConfusion Matrix:\n")
    buf.write(f"{cm_report}\n")
    report = buf.getvalue()
    buf.close()
    typer.echo(report, nl=False)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report)
        typer.echo(f"\nSaved text report to {output.resolve()}")

    # Generate HTML report (enabled by default)
    if not no_html_report:
        from vartrustml.visualization.html_evaluate_reporter import HTMLEvaluateReporter

        # Determine HTML report path
        if html_path:
            html_output_path = Path(html_path)
        elif output:
            html_output_path = output.parent / "evaluate_report.html"
        else:
            html_output_path = Path("evaluate_report.html")

        reporter = HTMLEvaluateReporter(output_path=str(html_output_path))

        # Add overview
        model_info = {
            "model_type": type(model).__name__,
            "optimal_threshold": optimal_threshold,
            "threshold_metadata": threshold_metadata,
        }
        reporter.add_overview(
            model_path=str(model_path),
            data_path=str(data),
            target_column=target,
            n_samples=len(y),
            model_info=model_info,
        )

        # Add metrics table - show optimized metrics if available, otherwise default
        if has_optimized_threshold:
            metrics = {
                "Balanced Accuracy (optimized)": balanced_acc_optimized,
                "Matthews Corr. Coef. (optimized)": mcc_optimized,
                "F1 Score weighted (optimized)": f1_weighted_optimized,
                "Balanced Accuracy (default 0.5)": balanced_acc_default,
                "Matthews Corr. Coef. (default 0.5)": mcc_default,
                "F1 Score weighted (default 0.5)": f1_weighted_default,
            }
        else:
            metrics = {
                "Balanced Accuracy": balanced_acc_default,
                "Matthews Corr. Coef.": mcc_default,
                "F1 Score (weighted)": f1_weighted_default,
            }
        if auroc is not None:
            metrics["AUROC"] = auroc
        if threshold_metadata:
            metrics["Optimal Threshold"] = optimal_threshold
        reporter.add_metrics_table(metrics)

        # Add confusion matrix (normalized) - use optimized if available
        cm_normalized = cm_report.astype("float") / cm_report.sum(axis=1)[:, np.newaxis]
        reporter.add_confusion_matrix(cm_normalized, normalize=True)

        # Add per-class metrics - use optimized predictions if available
        precision, recall, f1, support = precision_recall_fscore_support(
            y, y_pred_report
        )
        classes = sorted(y.unique())
        precision_dict = {f"Class {c}": p for c, p in zip(classes, precision)}
        recall_dict = {f"Class {c}": r for c, r in zip(classes, recall)}
        f1_dict = {f"Class {c}": f for c, f in zip(classes, f1)}
        support_dict = {f"Class {c}": int(s) for c, s in zip(classes, support)}
        reporter.add_classification_report(
            precision_dict, recall_dict, f1_dict, support_dict
        )

        # Generate report
        report_path = reporter.generate_report()
        typer.echo(f"HTML report generated: {report_path}")
