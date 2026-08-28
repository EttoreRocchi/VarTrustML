"""
Utility CLI commands for VarTrustML.

Contains:
- list-models: List available models registered in VarTrustML
- validate: Validate input data and configuration before running analysis
- smoke-test: Quick import test to validate installation
- version: Display the VarTrustML version
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from vartrustml import ExperimentConfig, __version__

from vartrustml.cli._constants import EXIT_VALIDATION_ERROR
from vartrustml.cli._shared import _parse_multi_value
from vartrustml.cli.main import app


@app.command("list-models", rich_help_panel="Utility")
def fit_list_models() -> None:
    """
    List available models registered in VartrustML.
    """
    cfg = ExperimentConfig()
    from vartrustml.core.models import ModelEvaluator

    evaluator = ModelEvaluator(cfg)
    typer.echo("Available models:")
    for i, name in enumerate(evaluator.models.keys(), start=1):
        typer.echo(f"  {i}. {name}")
    typer.echo(
        "\nUse the exact model name with --model (quotes if it contains spaces)."
    )


@app.command("validate", rich_help_panel="Utility")
def validate(
    dataset: Path = typer.Argument(..., help="Path to dataset file to validate."),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to ExperimentConfig JSON file to validate.",
    ),
    target_column: str = typer.Option(
        "state",
        "--target",
        "-t",
        help="Name of target column for classification.",
    ),
    continuous: Optional[str] = typer.Option(
        None,
        "--continuous",
        help="Comma-separated list of continuous feature columns.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output validation result as JSON (for Nextflow integration).",
    ),
    id_column: Optional[str] = typer.Option(
        None,
        "--id-column",
        help="Column to use as row identifier (excluded from features).",
    ),
) -> None:
    """
    Validate input data and configuration before running analysis.

    Pre-flight validation for Nextflow pipelines. Checks:
    - File exists and is readable
    - Data format is valid (CSV/TSV/TXT)
    - Target column exists and is binary
    - Feature columns are present (if specified)
    - Config file is valid JSON (if provided)

    Returns exit code 0 if valid, 2 if validation fails.
    """
    import json as json_lib

    from vartrustml.io import DataLoader

    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "dataset": str(dataset),
        "n_samples": None,
        "n_features": None,
        "class_distribution": None,
    }

    if not dataset.exists():
        validation_result["valid"] = False
        validation_result["errors"].append(f"Dataset file not found: {dataset}")
        _output_validation_result(validation_result, json_output)
        raise typer.Exit(code=EXIT_VALIDATION_ERROR)

    try:
        loader = DataLoader(str(dataset.parent))
        df = loader.load_dataset(
            dataset.name, drop_duplicates=False, id_column=id_column
        )
        validation_result["n_samples"] = len(df)
        validation_result["n_features"] = len(df.columns) - 1  # Exclude target
    except Exception as e:
        validation_result["valid"] = False
        validation_result["errors"].append(f"Failed to load dataset: {str(e)}")
        _output_validation_result(validation_result, json_output)
        raise typer.Exit(code=EXIT_VALIDATION_ERROR)

    if target_column not in df.columns:
        validation_result["valid"] = False
        validation_result["errors"].append(
            f"Target column '{target_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    else:
        unique_values = df[target_column].dropna().unique()
        validation_result["class_distribution"] = (
            df[target_column].value_counts().to_dict()
        )
        if not set(unique_values).issubset({0, 1}):
            validation_result["valid"] = False
            validation_result["errors"].append(
                f"Target column must be binary (0/1). "
                f"Found values: {sorted(unique_values)}"
            )
        class_counts = df[target_column].value_counts()
        if len(class_counts) < 2:
            validation_result["valid"] = False
            validation_result["errors"].append(
                "Target column must have at least 2 classes."
            )
        elif class_counts.min() < 10:
            validation_result["warnings"].append(
                f"Minority class has only {class_counts.min()} samples. "
                "Consider collecting more data."
            )

    if continuous:
        continuous_cols = _parse_multi_value(continuous)
        missing_cols = [c for c in continuous_cols if c not in df.columns]
        if missing_cols:
            validation_result["valid"] = False
            validation_result["errors"].append(
                f"Continuous columns not found: {missing_cols}"
            )

    if config:
        if not config.exists():
            validation_result["valid"] = False
            validation_result["errors"].append(f"Config file not found: {config}")
        else:
            try:
                with open(config) as f:
                    config_data = json_lib.load(f)
                # Round-trips the dict through the dataclass to surface bad keys
                ExperimentConfig.from_dict(config_data)
                validation_result["config_valid"] = True
            except json_lib.JSONDecodeError as e:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Invalid JSON in config file: {e}")
            except Exception as e:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Invalid config: {e}")

    missing_pct = (df.isnull().sum().sum() / df.size) * 100
    if missing_pct > 0:
        validation_result["warnings"].append(
            f"Dataset contains {missing_pct:.2f}% missing values."
        )

    _output_validation_result(validation_result, json_output)

    if not validation_result["valid"]:
        raise typer.Exit(code=EXIT_VALIDATION_ERROR)


def _output_validation_result(result: dict, json_output: bool) -> None:
    """Output validation result in text or JSON format."""
    import json as json_lib

    if json_output:
        typer.echo(json_lib.dumps(result, indent=2, default=str))
        return

    console = Console()
    if result["valid"]:
        console.print("[green]Validation PASSED[/green]")
    else:
        console.print("[red]Validation FAILED[/red]")

    if result.get("n_samples"):
        console.print(f"  Samples: {result['n_samples']}")
    if result.get("n_features"):
        console.print(f"  Features: {result['n_features']}")
    if result.get("class_distribution"):
        console.print(f"  Class distribution: {result['class_distribution']}")

    for error in result.get("errors", []):
        console.print(f"  [red]ERROR:[/red] {error}")

    for warning in result.get("warnings", []):
        console.print(f"  [yellow]WARNING:[/yellow] {warning}")


@app.command("smoke-test", rich_help_panel="Utility")
def smoke_test() -> None:
    """
    Quick import test to validate installation.
    """
    typer.echo("Import successful!")


@app.command("version", rich_help_panel="Utility")
def version() -> None:
    """
    Display the VarTrustML version.
    """
    typer.echo(f"VarTrustML version {__version__}")
