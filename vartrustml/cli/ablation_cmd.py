"""
CLI command: ablation.

Run ablation study to measure feature importance via leave-one-out analysis.
"""

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from vartrustml import DataLoader
from vartrustml.analysis.ablation_config import ConfigAblationAnalyzer
from vartrustml.analysis.ablation_formatters import format_ablation_study
from vartrustml.utils.logging import setup_logging

from vartrustml.cli._constants import EXIT_ERROR, EXIT_VALIDATION_ERROR
from vartrustml.cli._shared import _parse_multi_value
from vartrustml.cli.main import app


@app.command("ablation", rich_help_panel="Model Comparison")
def ablation(
    dataset: str = typer.Argument(
        ...,
        help="Dataset file path (relative to --data-dir or absolute).",
    ),
    target_column: str = typer.Option(
        ...,
        "--target-column",
        "-t",
        help="Target column name (REQUIRED).",
        rich_help_panel="Data Configuration",
    ),
    model_name: str = typer.Option(
        ...,
        "--model",
        "-m",
        help="Model name to use (XGBoost, CatBoost, Random Forest, "
        "Logistic Regression, MLP, KNN). Fresh models are trained for each ablation.",
        rich_help_panel="Model",
    ),
    data_dir: Path = typer.Option(
        Path("."),
        "--data-dir",
        "-d",
        help="Directory containing datasets. [default: current directory]",
        rich_help_panel="Input/Output",
    ),
    output_dir: Path = typer.Option(
        Path("results/ablation"),
        "--output-dir",
        "-o",
        help="Output directory for ablation results. [default: results/ablation]",
        rich_help_panel="Input/Output",
    ),
    metric: str = typer.Option(
        "balanced_accuracy",
        "--metric",
        help="Metric to use for ablation analysis. Options: balanced_accuracy, f1, mcc, roc_auc.",
        rich_help_panel="Analysis",
    ),
    feature_groups: Optional[Path] = typer.Option(
        None,
        "--feature-groups",
        help="Path to YAML/JSON file defining feature groups for group ablation.",
        rich_help_panel="Analysis",
    ),
    features: Optional[str] = typer.Option(
        None,
        "--features",
        callback=_parse_multi_value,
        help="Comma-separated list of specific features to ablate. If not provided, all features are used.",
        rich_help_panel="Analysis",
    ),
    ablate_steps: bool = typer.Option(
        False,
        "--ablate-steps",
        help="Run pipeline step ablation (calibration, threshold, scaling) instead of feature ablation.",
        rich_help_panel="Analysis",
    ),
    n_splits: int = typer.Option(
        3,
        "--n-splits",
        help="Number of CV folds for ablation analysis. [default: 3]",
        rich_help_panel="Cross-Validation",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for reproducibility. [default: 42]",
        rich_help_panel="Cross-Validation",
    ),
    continuous_cols: Optional[str] = typer.Option(
        None,
        "--continuous",
        callback=_parse_multi_value,
        help="Continuous columns to scale (comma-separated or path to .txt file). "
        "REQUIRED when using --model.",
        rich_help_panel="Data Configuration",
    ),
    calibrate: bool = typer.Option(
        False,
        "--calibrate",
        help="Apply probability calibration during ablation. Only with --model.",
        rich_help_panel="Model Configuration",
    ),
    calibration_method: str = typer.Option(
        "isotonic",
        "--calibration-method",
        help="Calibration method: 'isotonic' or 'sigmoid'. [default: isotonic]",
        rich_help_panel="Model Configuration",
    ),
    optimize_threshold: bool = typer.Option(
        False,
        "--optimize-threshold",
        help="Apply threshold optimization during ablation. Only with --model.",
        rich_help_panel="Model Configuration",
    ),
    n_jobs: int = typer.Option(
        -1,
        "--n-jobs",
        help="Parallel jobs (-1 for all cores). [default: -1]",
        rich_help_panel="System",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be executed without running.",
        rich_help_panel="System",
    ),
    verbose: int = typer.Option(
        1,
        "--verbose",
        "-v",
        help="Verbosity level (0=WARNING, 1=INFO, 2=DEBUG). [default: 1]",
        rich_help_panel="System",
    ),
    id_column: Optional[str] = typer.Option(
        None,
        "--id-column",
        help="Column to use as row identifier (excluded from features).",
        rich_help_panel="Data Configuration",
    ),
) -> None:
    """
    Run ablation study to measure feature importance via leave-one-out analysis.

    Ablation studies systematically remove features (or feature groups) and measure
    the impact on model performance. This helps identify which features are most
    important for the model's predictions.

    Fresh models are trained from scratch for each ablation, ensuring accurate
    performance measurement without preprocessing pipeline issues.

    \b
    Examples:
        # Basic feature ablation
        vartrustml ablation data.csv -t state -m XGBoost --continuous feat1,feat2

        # With calibration and threshold optimization
        vartrustml ablation data.csv -t state -m CatBoost --continuous feat1,feat2 \\
            --calibrate --optimize-threshold

        # Feature group ablation
        vartrustml ablation data.csv -t state -m XGBoost --continuous feat1,feat2 \\
            --feature-groups groups.yaml

        # Pipeline step ablation (calibration/threshold/scaling impact)
        vartrustml ablation data.csv -t state -m XGBoost --continuous feat1,feat2 \\
            --ablate-steps
    """
    from sklearn.metrics import (
        balanced_accuracy_score,
        f1_score,
        matthews_corrcoef,
        roc_auc_score,
    )
    from vartrustml.analysis.ablation_config import SUPPORTED_MODELS

    # Setup logging
    setup_logging(verbose=verbose)

    # Validate model name
    if model_name not in SUPPORTED_MODELS:
        typer.echo(
            f"Error: Unsupported model '{model_name}'. "
            f"Supported models: {', '.join(SUPPORTED_MODELS)}",
            err=True,
        )
        raise typer.Exit(code=EXIT_VALIDATION_ERROR)

    # Validate continuous_cols
    if not continuous_cols:
        typer.echo(
            "Warning: --continuous not provided. No feature scaling will be applied.",
        )

    # Validate calibration method
    if calibration_method not in ("isotonic", "sigmoid"):
        typer.echo(
            f"Error: Invalid calibration method '{calibration_method}'. "
            "Options: isotonic, sigmoid",
            err=True,
        )
        raise typer.Exit(code=EXIT_VALIDATION_ERROR)

    # Validate metric
    valid_metrics = {
        "balanced_accuracy": balanced_accuracy_score,
        "f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="binary"),
        "mcc": matthews_corrcoef,
        "roc_auc": roc_auc_score,
    }
    if metric not in valid_metrics:
        typer.echo(
            f"Error: Invalid metric '{metric}'. "
            f"Valid options: {', '.join(valid_metrics.keys())}",
            err=True,
        )
        raise typer.Exit(code=EXIT_VALIDATION_ERROR)

    metric_func = valid_metrics[metric]

    # Load and validate dataset
    typer.echo("\n=== Ablation Study ===\n")
    typer.echo(f"Dataset: {dataset}")
    typer.echo(f"Model: {model_name}")
    if calibrate:
        typer.echo(f"Calibration: {calibration_method}")
    if optimize_threshold:
        typer.echo("Threshold optimization: enabled")
    typer.echo(f"Metric: {metric}")
    typer.echo(f"CV Splits: {n_splits}")
    typer.echo(f"Seed: {seed}")

    # Load dataset
    typer.echo("\n2) Loading dataset")
    try:
        loader = DataLoader(str(data_dir))
        df = loader.load_dataset(dataset, drop_duplicates=True, id_column=id_column)
        typer.echo(f"   Loaded dataset with shape: {df.shape}")
    except Exception as e:
        typer.echo(f"Error loading dataset: {e}", err=True)
        raise typer.Exit(code=EXIT_ERROR)

    # Validate target column
    if target_column not in df.columns:
        typer.echo(
            f"Error: Target column '{target_column}' not found in dataset. "
            f"Available columns: {', '.join(df.columns[:10])}...",
            err=True,
        )
        raise typer.Exit(code=EXIT_VALIDATION_ERROR)

    # Prepare X and y
    X = df.drop(columns=[target_column])
    y = df[target_column].values

    # Determine features to ablate
    if features:
        ablation_features = features
        # Validate features exist
        missing = set(ablation_features) - set(X.columns)
        if missing:
            typer.echo(
                f"Error: Features not found in dataset: {missing}",
                err=True,
            )
            raise typer.Exit(code=EXIT_VALIDATION_ERROR)
    else:
        ablation_features = list(X.columns)

    typer.echo(f"   Features to ablate: {len(ablation_features)}")

    # Load feature groups if provided
    feature_groups_dict = None
    if feature_groups:
        typer.echo(f"\n3) Loading feature groups from {feature_groups}")
        try:
            import yaml

            with open(feature_groups) as f:
                if feature_groups.suffix in [".yaml", ".yml"]:
                    feature_groups_dict = yaml.safe_load(f)
                else:
                    feature_groups_dict = json.load(f)
            typer.echo(f"   Loaded {len(feature_groups_dict)} feature groups")
            for group_name, group_features in feature_groups_dict.items():
                typer.echo(f"   - {group_name}: {len(group_features)} features")
        except Exception as e:
            typer.echo(f"Error loading feature groups: {e}", err=True)
            raise typer.Exit(code=EXIT_ERROR)

    # Display configuration
    console = Console()
    table = Table(title="Ablation Configuration", show_header=True)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Dataset", dataset)
    table.add_row("Target Column", target_column)
    table.add_row("Model", model_name)
    table.add_row("Calibration", f"{calibration_method}" if calibrate else "None")
    table.add_row("Threshold Opt", "Yes" if optimize_threshold else "No")
    table.add_row("Metric", metric)
    if ablate_steps:
        table.add_row("Mode", "Pipeline Step Ablation")
    else:
        table.add_row("Features", str(len(ablation_features)))
        table.add_row(
            "Feature Groups",
            str(len(feature_groups_dict)) if feature_groups_dict else "None",
        )
    table.add_row("CV Splits", str(n_splits))
    table.add_row("Seed", str(seed))
    table.add_row("Output", str(output_dir))
    console.print(table)

    # Dry run check
    if dry_run:
        typer.echo("\nDRY RUN MODE - No ablation will be performed")
        typer.echo("Configuration is valid. Run without --dry-run to execute.\n")
        return

    # Create analyzer
    step_num = 4 if feature_groups else 3
    typer.echo(f"\n{step_num}) Running ablation study...")
    analyzer = ConfigAblationAnalyzer(n_splits=n_splits, seed=seed, n_jobs=n_jobs)

    try:
        if ablate_steps:
            # Pipeline step ablation
            typer.echo("   Mode: Pipeline Step Ablation")
            results = analyzer.pipeline_step_ablation(
                X=X,
                y=y,
                model_name=model_name,
                metric_func=metric_func,
                metric_name=metric,
                continuous_cols=continuous_cols,  # type: ignore[arg-type]  # Typer callback returns List
            )
        elif feature_groups_dict:
            # Feature group ablation
            typer.echo("   Mode: Feature Group Ablation")
            results = analyzer.group_ablation_from_config(
                X=X,
                y=y,
                model_name=model_name,
                feature_groups=feature_groups_dict,
                metric_func=metric_func,
                metric_name=metric,
                continuous_cols=continuous_cols,  # type: ignore[arg-type]
                calibrate=calibrate,
                calibration_method=calibration_method,
                calibration_cv=3,
                optimize_threshold=optimize_threshold,
                threshold_method="auto",
            )
        else:
            # Feature ablation
            typer.echo("   Mode: Feature Ablation")
            results = analyzer.ablation_from_config(
                X=X,
                y=y,
                model_name=model_name,
                metric_func=metric_func,
                metric_name=metric,
                features_to_ablate=ablation_features,  # type: ignore[arg-type]
                continuous_cols=continuous_cols,  # type: ignore[arg-type]
                calibrate=calibrate,
                calibration_method=calibration_method,
                calibration_cv=3,
                optimize_threshold=optimize_threshold,
                threshold_method="auto",
            )
    except Exception as e:
        typer.echo(f"Error during ablation: {e}", err=True)
        raise typer.Exit(code=EXIT_ERROR)

    # Display results
    typer.echo("\n=== Ablation Results ===\n")
    typer.echo(format_ablation_study(results))

    # Display significant findings
    significant = results.get_significant_ablations()
    if significant:
        typer.echo(f"\nStatistically significant ablations ({len(significant)}):")
        for r in significant:
            direction = "decreases" if r.delta < 0 else "increases"
            typer.echo(
                f"  - {r.ablation_name}: Removing {direction} {metric} by "
                f"{abs(r.delta):.4f} ({abs(r.delta_pct):.1f}%), p={r.p_value:.4f}"
            )
    else:
        typer.echo("\nNo statistically significant ablations found at alpha=0.05")

    # Save results
    typer.echo(f"\n5) Saving results to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    csv_path = output_dir / "ablation_results.csv"
    results.summary_df.to_csv(csv_path, index=False)
    typer.echo(f"   Saved: {csv_path}")

    # Save text report
    report_path = output_dir / "ablation_report.txt"
    with open(report_path, "w") as f:
        f.write("=== ABLATION STUDY REPORT ===\n\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Model: {model_name}\n")
        if calibrate:
            f.write(f"Calibration: {calibration_method}\n")
        if optimize_threshold:
            f.write("Threshold Optimization: enabled\n")
        f.write(f"Metric: {metric}\n")
        f.write(f"Baseline Score: {results.baseline_score:.4f}\n")
        f.write(f"CV Splits: {n_splits}\n")
        f.write(f"Seed: {seed}\n\n")
        f.write(format_ablation_study(results))
        if significant:
            f.write(f"\n\nSignificant Findings ({len(significant)}):\n")
            for r in significant:
                direction = "decreases" if r.delta < 0 else "increases"
                f.write(
                    f"  - {r.ablation_name}: Removing {direction} {metric} by "
                    f"{abs(r.delta):.4f} ({abs(r.delta_pct):.1f}%), p={r.p_value:.4f}\n"
                )
    typer.echo(f"   Saved: {report_path}")

    typer.echo("\nAblation study complete!")
