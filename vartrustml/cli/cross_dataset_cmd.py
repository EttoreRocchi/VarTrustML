"""
CLI command: cross-dataset.

Cross-dataset generalizability analysis.
"""

from pathlib import Path
from typing import List, Optional

import pandas as pd
import typer

from vartrustml import (
    CrossDatasetEvaluator,
    DataLoader,
)
from vartrustml.config.validation import validate_experiment_config
from vartrustml.core.missing import drop_missing_rows
from vartrustml.utils.logging import setup_logging

from vartrustml.cli._constants import EXIT_VALIDATION_ERROR
from vartrustml.cli._shared import (
    _display_parameters_table,
    _load_config,
    _merge_experiment_config,
    _parse_multi_value,
    _save_config_if_requested,
)
from vartrustml.cli.main import app


@app.command("cross-dataset", rich_help_panel="Model Comparison")
def cross_dataset(
    datasets: List[str] = typer.Argument(
        ...,
        help="Two or more dataset file names relative to --data-dir (REQUIRED).",
    ),
    data_dir: Path = typer.Option(
        Path("data"),
        "--data-dir",
        "-d",
        help="Directory containing datasets. [default: data]",
        rich_help_panel="Input/Output",
    ),
    output_dir: Path = typer.Option(
        Path("results/cross_dataset"),
        "--output-dir",
        "-o",
        help="Output directory for results. [default: results/cross_dataset]",
        rich_help_panel="Input/Output",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Path to ExperimentConfig JSON to load.",
        rich_help_panel="Input/Output",
    ),
    save_config: Optional[Path] = typer.Option(
        None,
        "--save-config",
        help="Write the effective ExperimentConfig JSON to this path.",
        rich_help_panel="Input/Output",
    ),
    log_file: Optional[Path] = typer.Option(
        None, help="Write logs to file.", rich_help_panel="Input/Output"
    ),
    # Cross-Validation settings
    seed: Optional[int] = typer.Option(
        None, help="Random seed.", rich_help_panel="Cross-Validation"
    ),
    n_outer_splits: Optional[int] = typer.Option(
        None, help="Number of outer CV splits.", rich_help_panel="Cross-Validation"
    ),
    n_inner_splits: Optional[int] = typer.Option(
        None, help="Number of inner CV splits.", rich_help_panel="Cross-Validation"
    ),
    # Hyperparameter Optimization
    hpo_method: Optional[str] = typer.Option(
        None,
        help="Hyperparameter optimization method: 'grid' or 'optuna'.",
        rich_help_panel="Hyperparameter Optimization",
    ),
    optuna_n_trials: Optional[int] = typer.Option(
        None,
        "--optuna-trials",
        help="Number of Optuna trials (when using --hpo-method optuna).",
        rich_help_panel="Hyperparameter Optimization",
    ),
    optuna_timeout: Optional[int] = typer.Option(
        None,
        help="Optuna timeout in seconds (when using --hpo-method optuna).",
        rich_help_panel="Hyperparameter Optimization",
    ),
    # Data Configuration
    target_column: Optional[str] = typer.Option(
        None, help="Target column name.", rich_help_panel="Data Configuration"
    ),
    continuous_cols: Optional[str] = typer.Option(
        None,
        "--continuous",
        callback=_parse_multi_value,
        help="Continuous columns to scale (comma-separated list or path to .txt file).",
        rich_help_panel="Data Configuration",
    ),
    categorical_cols: Optional[str] = typer.Option(
        None,
        "--categorical",
        callback=_parse_multi_value,
        help="Categorical columns (comma-separated list or path to .txt file). If provided without --continuous, continuous columns are inferred by exclusion.",
        rich_help_panel="Data Configuration",
    ),
    nan_strategy: Optional[str] = typer.Option(
        None,
        "--nan-strategy",
        help="Missing-value handling: 'median'/'mean'/'most_frequent' (impute in "
        "pipeline, fit per-fold) or 'drop' (remove rows with NaN). [default: median]",
        rich_help_panel="Data Configuration",
    ),
    # Model Selection
    models_to_use: Optional[str] = typer.Option(
        None,
        "--models",
        callback=_parse_multi_value,
        help="Model names to train (comma-separated).",
        rich_help_panel="Model Selection",
    ),
    # Calibration
    calibrate_model: bool = typer.Option(
        False,
        "--calibrate-model",
        help="Enable probability calibration.",
        rich_help_panel="Calibration",
    ),
    calibration_method: Optional[str] = typer.Option(
        None,
        "--calibration",
        help="Calibration method: 'isotonic' or 'sigmoid'.",
        rich_help_panel="Calibration",
    ),
    calibration_cv: Optional[int] = typer.Option(
        None,
        help="Number of folds for calibration cross-validation.",
        rich_help_panel="Calibration",
    ),
    # Visualization
    figure_dpi: Optional[int] = typer.Option(
        None, help="Figure DPI.", rich_help_panel="Visualization"
    ),
    no_html_report: bool = typer.Option(
        False,
        "--no-html-report",
        help="Disable HTML report generation.",
        rich_help_panel="Visualization",
    ),
    html_report_path: Optional[str] = typer.Option(
        None,
        "--html-path",
        help="Path for HTML report (relative to output dir).",
        rich_help_panel="Visualization",
    ),
    # Threshold Optimization
    optimize_threshold: bool = typer.Option(
        False,
        "--optimize-threshold",
        help="Enable threshold optimization using Youden's J statistic.",
        rich_help_panel="Threshold Optimization",
    ),
    threshold_method: Optional[str] = typer.Option(
        None,
        "--threshold-method",
        help="Threshold optimization method: 'oof', 'cv', or 'auto'.",
        rich_help_panel="Threshold Optimization",
    ),
    # Bootstrap
    bootstrap_n_iterations: Optional[int] = typer.Option(
        None,
        "--bootstrap-iters",
        help="Number of bootstrap resamples for confidence intervals. [default: 1000]",
        rich_help_panel="Bootstrap",
    ),
    bootstrap_ci_level: Optional[float] = typer.Option(
        None,
        "--ci-level",
        help="Confidence level for bootstrap CIs (e.g., 0.95 for 95% CI). [default: 0.95]",
        rich_help_panel="Bootstrap",
    ),
    bootstrap_ci_method: Optional[str] = typer.Option(
        None,
        "--ci-method",
        help="Bootstrap CI method: 'bca' (bias-corrected and accelerated) or "
        "'percentile'. [default: bca]",
        rich_help_panel="Bootstrap",
    ),
    # System settings
    n_jobs: Optional[int] = typer.Option(
        None, help="Parallel jobs (-1 for all cores).", rich_help_panel="System"
    ),
    verbose: Optional[int] = typer.Option(
        None,
        help="Verbosity level (1=tqdm only, 2=INFO, 3=DEBUG).",
        rich_help_panel="System",
    ),
    no_checkpoints: bool = typer.Option(
        False,
        "--no-checkpoints",
        help="Disable checkpoint saving.",
        rich_help_panel="System",
    ),
    checkpoint_dir: Optional[str] = typer.Option(
        None, help="Checkpoint subdirectory name.", rich_help_panel="System"
    ),
    # Control
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be executed without running.",
        rich_help_panel="System",
    ),
    id_column: Optional[str] = typer.Option(
        None,
        "--id-column",
        help="Column to use as row identifier (excluded from features).",
        rich_help_panel="Data Configuration",
    ),
    caller_columns: Optional[str] = typer.Option(
        None,
        "--callers",
        callback=_parse_multi_value,
        help="Caller column names (comma-separated) to evaluate as a baseline on each test sample.",
        rich_help_panel="Caller Baseline",
    ),
    include_default_combinations: bool = typer.Option(
        True,
        "--default-combinations/--no-default-combinations",
        help="Include default AND/OR caller combinations in the baseline.",
        rich_help_panel="Caller Baseline",
    ),
    cv_scheme: str = typer.Option(
        "pairwise",
        "--cv-scheme",
        help="Evaluation scheme: 'pairwise' (NxN matrix), 'lodo' (leave-one-dataset-out), or 'both'.",
        rich_help_panel="Cross-Validation",
    ),
):
    """
    Cross-dataset generalizability analysis.

    Train models on one dataset and test on all other datasets to evaluate
    how well models generalize across different data sources. Produces
    performance matrices, generalization gap analysis, and stability reports.

    Requires at least 2 datasets.

    Example:
        vartrustml cross-dataset HG002_DEL.csv HG002_DUP.csv HG002_INS.csv
    """
    # Validate minimum datasets
    if len(datasets) < 2:
        typer.echo("Error: cross-dataset requires at least 2 datasets.", err=True)
        raise typer.Exit(code=EXIT_VALIDATION_ERROR)

    base_cfg = _load_config(config)
    cfg = _merge_experiment_config(
        base_cfg,
        seed=seed,
        n_outer_splits=n_outer_splits,
        n_inner_splits=n_inner_splits,
        output_dir=output_dir,
        target_column=target_column,
        continuous_cols=continuous_cols,  # type: ignore[arg-type]  # Typer callback returns List
        categorical_cols=categorical_cols,  # type: ignore[arg-type]
        models_to_use=models_to_use,  # type: ignore[arg-type]
        calibrate_models=calibrate_model if calibrate_model else None,
        calibration_method=calibration_method,
        calibration_cv=calibration_cv,
        figure_dpi=figure_dpi,
        n_jobs=n_jobs,
        verbose=verbose,
        save_checkpoints=False if no_checkpoints else None,
        checkpoint_dir=checkpoint_dir,
        hpo_method=hpo_method,
        optuna_n_trials=optuna_n_trials,
        optuna_timeout=optuna_timeout,
        generate_html_report=False if no_html_report else None,
        html_report_path=html_report_path,
        optimize_threshold=optimize_threshold if optimize_threshold else None,
        threshold_method=threshold_method,
        bootstrap_n_iterations=bootstrap_n_iterations,
        bootstrap_ci_level=bootstrap_ci_level,
        bootstrap_ci_method=bootstrap_ci_method,
        nan_strategy=nan_strategy,
        caller_columns=caller_columns,
    )

    # Setup logging
    setup_logging(verbose=cfg.verbose, log_file=str(log_file) if log_file else None)

    # Validate configuration
    is_valid, errors = validate_experiment_config(cfg)
    if not is_valid:
        typer.echo("Configuration validation failed:", err=True)
        for error in errors:
            typer.echo(f"  - {error}", err=True)
        raise typer.Exit(code=EXIT_VALIDATION_ERROR)

    _save_config_if_requested(cfg, save_config)

    # Prepare parameters for display
    params = {
        "Output Directory": cfg.output_dir,
        "Models": cfg.models_to_use,
        "Datasets": datasets,
        "Number of Datasets": len(datasets),
        "Outer CV Folds": cfg.cv.n_outer_splits,
        "Inner CV Folds": cfg.cv.n_inner_splits,
        "Calibrate Models": cfg.calibration.calibrate_models,
        "Calibration Method": cfg.calibration.calibration_method
        if cfg.calibration.calibrate_models
        else "N/A",
        "Optimize Threshold": cfg.threshold.optimize_threshold,
        "Threshold Method": cfg.threshold.threshold_method
        if cfg.threshold.optimize_threshold
        else "N/A",
        "Save Checkpoints": cfg.save_checkpoints,
        "Parallel Jobs": cfg.n_jobs,
        "Random Seed": cfg.cv.seed,
        "Target Column": cfg.target_column,
        "NaN Strategy": cfg.nan_strategy,
        "Generate HTML Report": cfg.generate_html_report,
    }

    if cfg.continuous_cols:
        params["Continuous Columns"] = cfg.continuous_cols
    if cfg.hpo_method:
        params["HPO Method"] = cfg.hpo_method
        if cfg.hpo_method == "optuna":
            params["Optuna Trials"] = cfg.optuna_n_trials
            if cfg.optuna_timeout:
                params["Optuna Timeout (s)"] = cfg.optuna_timeout

    # Display parameters table
    typer.echo("")
    _display_parameters_table(params, "Cross-Dataset Configuration")
    typer.echo("")

    # Dry run mode - show what would be executed
    if dry_run:
        typer.echo("DRY RUN MODE - No models will be trained")
        typer.echo("Configuration is valid. Run without --dry-run to execute.\n")
        return

    typer.echo("Running cross-dataset generalizability analysis")

    # Load datasets
    loader = DataLoader(str(data_dir))
    dataset_list = []
    typer.echo("\n1) Loading datasets")
    for dataset in datasets:
        ds_name = Path(dataset).stem
        typer.echo(f"   Loading: {ds_name}")
        df = loader.load_dataset(dataset, drop_duplicates=True, id_column=id_column)
        typer.echo(f"     Shape: {df.shape}")
        if cfg.nan_strategy == "drop":
            df, n_dropped = drop_missing_rows(df, target_column=cfg.target_column)
            typer.echo(
                f"     Dropped {n_dropped} rows with missing values (nan-strategy=drop)"
            )
        dataset_list.append((df, ds_name))

    # Validate dataset compatibility
    typer.echo("\n2) Validating dataset compatibility")
    compat_report = loader.validate_datasets_compatibility(
        dataset_list, cfg.target_column
    )
    if not compat_report["compatible"]:
        typer.echo("Dataset compatibility validation failed:", err=True)
        for error in compat_report["issues"]:
            typer.echo(f"  - {error}", err=True)
        raise typer.Exit(code=EXIT_VALIDATION_ERROR)
    typer.echo("   Datasets are compatible")

    # Run cross-dataset evaluation
    typer.echo("\n3) Running cross-dataset evaluation")
    if cv_scheme not in ("pairwise", "lodo", "both"):
        typer.echo(
            f"Error: --cv-scheme must be 'pairwise', 'lodo' or 'both', got '{cv_scheme}'.",
            err=True,
        )
        raise typer.Exit(code=EXIT_VALIDATION_ERROR)
    evaluator = CrossDatasetEvaluator(cfg)
    results = evaluator.evaluate_cross_dataset(dataset_list, cv_scheme=cv_scheme)

    shift = None
    shift_heatmap_b64 = None
    caller_baseline = None
    if cfg.target_column and cfg.target_column in dataset_list[0][0].columns:
        typer.echo("\n   Quantifying distribution shift between samples")
        from vartrustml.analysis.distribution_shift import (
            compute_distribution_shift,
            plot_feature_shift_heatmap,
        )

        shift = compute_distribution_shift(
            dataset_list, cfg.target_column, continuous_cols=cfg.continuous_cols
        )
        shift_dir = Path(cfg.output_dir)
        shift_dir.mkdir(parents=True, exist_ok=True)
        shift["class_priors"].to_csv(shift_dir / "class_priors.csv")
        shift["feature_shift"].to_csv(shift_dir / "distribution_shift.csv", index=False)
        shift_heatmap_b64 = plot_feature_shift_heatmap(
            shift["feature_shift"],
            save_path=str(shift_dir / "distribution_shift_heatmap.png"),
            return_base64=cfg.generate_html_report,
        )

        if cfg.caller_comparison.caller_columns:
            from vartrustml.core.caller_evaluator import caller_baseline_table

            caller_baseline = caller_baseline_table(
                dataset_list,
                cfg.caller_comparison.caller_columns,
                cfg.target_column,
                include_combinations=include_default_combinations,
                metric=cfg.model_comparison_metric,
            )
            caller_baseline.to_csv(shift_dir / "caller_baseline_mcc.csv")
            typer.echo(
                f"   Caller baseline: {len(caller_baseline)} callers/combinations"
            )

    # Generate HTML report if requested
    html_path = None
    if cfg.generate_html_report:
        typer.echo("\n4) Generating HTML report")
        from vartrustml.visualization.html_cross_dataset_reporter import (
            HTMLCrossDatasetReporter,
        )

        html_path = cfg.html_report_path or "cross_dataset_report.html"
        full_html_path = Path(cfg.output_dir) / html_path

        reporter = HTMLCrossDatasetReporter(output_path=str(full_html_path))

        # Prepare dataset info
        datasets_info = []
        for df, name in dataset_list:
            y = df[cfg.target_column]
            datasets_info.append(
                {
                    "name": name,
                    "n_samples": len(df),
                    "n_features": len(df.columns) - 1,
                    "class_distribution": y.value_counts().to_dict(),
                }
            )

        # Add sections to the report
        reporter.add_overview(
            config={
                "n_outer_splits": cfg.cv.n_outer_splits,
                "n_inner_splits": cfg.cv.n_inner_splits,
                "seed": cfg.cv.seed,
                "models_to_use": cfg.models_to_use,
                "calibrate_models": cfg.calibration.calibrate_models,
                "calibration_method": cfg.calibration.calibration_method,
                "calibration_cv": cfg.calibration.calibration_cv,
                "optimize_threshold": cfg.threshold.optimize_threshold,
                "threshold_method": cfg.threshold.threshold_method,
            },
            datasets_info=datasets_info,
        )

        if shift is not None:
            reporter.add_distribution_shift(shift["class_priors"], shift_heatmap_b64)
        if caller_baseline is not None:
            reporter.add_caller_baseline(
                caller_baseline, metric=cfg.model_comparison_metric
            )
        reporter.add_generalization_gap_ci(
            getattr(evaluator, "generalization_gap", None),
            metric=cfg.model_comparison_metric,
        )
        reporter.add_lodo(
            getattr(evaluator, "lodo_results", None),
            metric=cfg.model_comparison_metric,
        )

        # Load results_std from saved CSV files for the reporter
        results_std = {}
        for model_name in results.keys():
            results_std[model_name] = {}
            model_dir = Path(cfg.output_dir) / model_name.replace(" ", "_")
            for metric_name in evaluator.METRICS_TO_TRACK:
                safe_metric = metric_name.replace(" ", "_").lower()
                std_path = model_dir / f"{safe_metric}_std.csv"
                if std_path.exists():
                    results_std[model_name][metric_name] = pd.read_csv(
                        std_path, index_col=0
                    )

        dataset_names = [name for _, name in dataset_list]

        reporter.add_performance_matrices(
            results_mean=results, results_std=results_std, dataset_names=dataset_names
        )

        reporter.add_generalization_gap_analysis(
            results_mean=results, dataset_names=dataset_names
        )

        report_path = reporter.generate_report()
        typer.echo(f"   HTML report generated: {report_path}")

    typer.echo("\n============================================================")
    typer.echo("CROSS-DATASET ANALYSIS COMPLETE")
    typer.echo("============================================================")
    typer.echo(f"\nResults saved to: {cfg.output_dir}/")
    typer.echo("\nOutput files:")
    typer.echo("  - cross_dataset_summary.txt")
    typer.echo("  - cross_dataset_results.json")
    if html_path:
        typer.echo(f"  - {html_path}")
    typer.echo("  - Per-model directories with heatmaps and CSV matrices")
