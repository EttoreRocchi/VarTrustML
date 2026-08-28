"""
Shared helper functions for VarTrustML CLI commands.

These utilities are used across multiple sub-command modules for parsing
CLI options, loading/merging configurations, and displaying output.
"""

from pathlib import Path
from typing import Any, List, Optional

import typer
from rich.console import Console
from rich.table import Table

from vartrustml import ExperimentConfig


def _parse_multi_value(value: Optional[str]) -> Optional[List[str]]:
    """Parse comma-separated values or read from a text file.

    Used for --continuous and --categorical column flags.
    If the value ends with .txt and the file exists, reads column names from it
    (one per line or comma-separated). Otherwise, parses as a comma-separated string.
    """
    if value is None:
        return None

    # Only treat as file path if it has .txt extension
    # This avoids filesystem errors when passing long comma-separated lists
    if value.endswith(".txt"):
        path = Path(value)
        if path.exists() and path.is_file():
            content = path.read_text().strip()
            columns = []
            for line in content.splitlines():
                for col in line.split(","):
                    col = col.strip()
                    if col:
                        columns.append(col)
            return columns if columns else None

    # Parse as comma-separated string (for both --continuous and --categorical)
    return [v.strip() for v in value.split(",") if v.strip()]


def _parse_multi_float(value: Optional[str]) -> Optional[List[float]]:
    """Parse comma-separated float values."""
    if value is None:
        return None
    # Split by comma, strip whitespace, and convert to float
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def _load_config(config_path: Optional[Path]) -> Optional[ExperimentConfig]:
    if config_path:
        cfg = ExperimentConfig.load(str(config_path))
        typer.echo(f"Loaded ExperimentConfig from {config_path}")
        return cfg
    return None


_SUB_CONFIG_ROUTING = {
    # CVConfig
    "seed": "cv",
    "n_outer_splits": "cv",
    "n_inner_splits": "cv",
    # CalibrationConfig
    "calibrate_models": "calibration",
    "calibration_method": "calibration",
    "calibration_cv": "calibration",
    # ThresholdConfig
    "optimize_threshold": "threshold",
    "threshold_method": "threshold",
    # CallerComparisonConfig
    "compare_callers": "caller_comparison",
    "caller_columns": "caller_comparison",
    "caller_combinations": "caller_comparison",
    "include_default_combinations": "caller_comparison",
    # BootstrapConfig
    "bootstrap_n_iterations": "bootstrap",
    "bootstrap_ci_level": "bootstrap",
    "bootstrap_ci_method": "bootstrap",
    # VisualizationConfig
    "plot_top_n_features": "visualization",
    "figure_dpi": "visualization",
    "error_analysis_features": "visualization",
}


def _merge_experiment_config(
    base: Optional[ExperimentConfig] = None,
    **overrides: Any,
) -> ExperimentConfig:
    """Merge CLI overrides into an ExperimentConfig.

    Any keyword argument whose value is not None will be set on the config.
    Keys belonging to nested sub-configs are routed automatically.
    The ``output_dir`` key is automatically converted to ``str`` (CLI passes Path).
    """
    cfg = base or ExperimentConfig()
    for key, value in overrides.items():
        if value is not None:
            if key == "output_dir":
                value = str(value)
            if key in _SUB_CONFIG_ROUTING:
                sub_config = getattr(cfg, _SUB_CONFIG_ROUTING[key])
                setattr(sub_config, key, value)
            else:
                setattr(cfg, key, value)
    return cfg


def _save_config_if_requested(cfg: ExperimentConfig, out: Optional[Path]) -> None:
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        cfg.save(str(out))
        typer.echo(f"Saved ExperimentConfig to {out}")


def _display_parameters_table(
    params: dict, title: str = "Configuration Parameters"
) -> None:
    """Display parameters in a rich table format."""
    console = Console()
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan", width=30)
    table.add_column("Value", style="green")

    for key, value in params.items():
        # Format lists nicely
        if isinstance(value, list):
            value_str = ", ".join(str(v) for v in value)
        else:
            value_str = str(value)
        table.add_row(key, value_str)

    console.print(table)
