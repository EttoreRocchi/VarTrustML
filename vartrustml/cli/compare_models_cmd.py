"""
CLI command: compare-models.

Compare multiple ML models on a dataset using cross-validation.
"""

from pathlib import Path
from typing import List, Optional

import typer

from vartrustml import (
    CrossValidationPipeline,
    DataLoader,
    create_feature_importance_report,
)
from vartrustml.config.validation import validate_experiment_config
from vartrustml.core.missing import drop_missing_rows
from vartrustml.utils.logging import setup_logging

from vartrustml.cli._constants import EXIT_VALIDATION_ERROR
from vartrustml.cli._shared import (
    _display_parameters_table,
    _load_config,
    _merge_experiment_config,
    _parse_multi_float,
    _parse_multi_value,
    _save_config_if_requested,
)
from vartrustml.cli.main import app


@app.command("compare-models", rich_help_panel="Model Comparison")
def compare_models(
    datasets: List[str] = typer.Argument(
        ...,
        help="One or more dataset file names relative to --data-dir (REQUIRED).",
    ),
    target_column: str = typer.Option(
        ...,
        "--target-column",
        "-t",
        help="Target column name (REQUIRED).",
        rich_help_panel="Data Configuration",
    ),
    data_dir: Path = typer.Option(
        Path("data"),
        "--data-dir",
        "-d",
        help="Directory containing datasets. [default: data]",
        rich_help_panel="Input/Output",
    ),
    output_dir: Path = typer.Option(
        Path("results"),
        "--output-dir",
        "-o",
        help="Output directory for results. [default: results]",
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
    seed: Optional[int] = typer.Option(
        None, help="Random seed.", rich_help_panel="Cross-Validation"
    ),
    n_outer_splits: Optional[int] = typer.Option(
        None,
        help="Number of outer CV splits. [default: 10]",
        rich_help_panel="Cross-Validation",
    ),
    n_inner_splits: Optional[int] = typer.Option(
        None,
        help="Number of inner CV splits. [default: 5]",
        rich_help_panel="Cross-Validation",
    ),
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
    confidence_thresholds: Optional[str] = typer.Option(
        None,
        "--thresholds",
        callback=_parse_multi_float,
        help="Confidence thresholds for error analysis (comma-separated).",
        rich_help_panel="Data Configuration",
    ),
    models_to_use: Optional[str] = typer.Option(
        None,
        "--models",
        callback=_parse_multi_value,
        help="Model names to train (comma-separated).",
        rich_help_panel="Model Selection",
    ),
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
    plot_top_n_features: Optional[int] = typer.Option(
        None, help="Number of features to plot.", rich_help_panel="Visualization"
    ),
    figure_dpi: Optional[int] = typer.Option(
        None, help="Figure DPI.", rich_help_panel="Visualization"
    ),
    error_analysis_features: Optional[str] = typer.Option(
        None,
        "--error-features",
        callback=_parse_multi_value,
        help="Features to analyze in error distributions (comma-separated).",
        rich_help_panel="Data Configuration",
    ),
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
    # Caller comparison options
    compare_callers: bool = typer.Option(
        False,
        "--compare-callers",
        help="Compare ML models against variant callers as baselines.",
        rich_help_panel="Caller Comparison",
    ),
    caller_columns: Optional[str] = typer.Option(
        None,
        "--callers",
        callback=_parse_multi_value,
        help="Caller column names (comma-separated). Required if --compare-callers is set.",
        rich_help_panel="Caller Comparison",
    ),
    caller_combinations: Optional[str] = typer.Option(
        None,
        "--combinations",
        callback=_parse_multi_value,
        help="Custom caller combinations (comma-separated).",
        rich_help_panel="Caller Comparison",
    ),
    include_default_combinations: Optional[bool] = typer.Option(
        None,
        "--default-combinations/--no-default-combinations",
        help="Auto-generate default AND/OR combinations.",
        rich_help_panel="Caller Comparison",
    ),
    # Bootstrap CI options
    bootstrap_n_iterations: Optional[int] = typer.Option(
        None,
        "--bootstrap-iters",
        help="Number of bootstrap resamples for confidence intervals. [default: 1000]",
        rich_help_panel="Bootstrap & Statistics",
    ),
    bootstrap_ci_level: Optional[float] = typer.Option(
        None,
        "--ci-level",
        help="Confidence level for bootstrap CIs (e.g., 0.95 for 95% CI). [default: 0.95]",
        rich_help_panel="Bootstrap & Statistics",
    ),
    bootstrap_ci_method: Optional[str] = typer.Option(
        None,
        "--ci-method",
        help="Bootstrap CI method: 'bca' (bias-corrected and accelerated) or "
        "'percentile'. [default: bca]",
        rich_help_panel="Bootstrap & Statistics",
    ),
    # Model comparison options
    model_comparison_metric: Optional[str] = typer.Option(
        None,
        "--comparison-metric",
        help="Metric for model-level statistical comparison. [default: Matthews Corr. Coef.]",
        rich_help_panel="Bootstrap & Statistics",
    ),
    correction_method: Optional[str] = typer.Option(
        None,
        "--correction",
        help="Multiple-comparison correction for pairwise tests: 'holm' (FWER) or 'bh' (FDR). [default: holm]",
        rich_help_panel="Bootstrap & Statistics",
    ),
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
):
    """
    Compare multiple ML models on a dataset using cross-validation.
    """
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
        confidence_thresholds=confidence_thresholds,  # type: ignore[arg-type]
        models_to_use=models_to_use,  # type: ignore[arg-type]
        calibrate_models=calibrate_model if calibrate_model else None,
        calibration_method=calibration_method,
        calibration_cv=calibration_cv,
        plot_top_n_features=plot_top_n_features,
        figure_dpi=figure_dpi,
        error_analysis_features=error_analysis_features,  # type: ignore[arg-type]
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
        compare_callers=compare_callers if compare_callers else None,
        caller_columns=caller_columns,  # type: ignore[arg-type]
        caller_combinations=caller_combinations,  # type: ignore[arg-type]
        include_default_combinations=include_default_combinations,
        bootstrap_n_iterations=bootstrap_n_iterations,
        bootstrap_ci_level=bootstrap_ci_level,
        bootstrap_ci_method=bootstrap_ci_method,
        model_comparison_metric=model_comparison_metric,
        nan_strategy=nan_strategy,
        correction_method=correction_method,
    )

    # Validate caller comparison settings
    if (
        cfg.caller_comparison.compare_callers
        and not cfg.caller_comparison.caller_columns
    ):
        typer.echo(
            "Error: --callers is required when --compare-callers is enabled.", err=True
        )
        raise typer.Exit(code=EXIT_VALIDATION_ERROR)

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
        "Bootstrap CI Iterations": cfg.bootstrap.bootstrap_n_iterations,
        "Bootstrap CI Level": f"{cfg.bootstrap.bootstrap_ci_level * 100:.0f}%",
        "Bootstrap CI Method": cfg.bootstrap.bootstrap_ci_method.upper(),
        "Model Comparison Metric": cfg.model_comparison_metric,
    }

    # Caller comparison settings
    if cfg.caller_comparison.compare_callers:
        params["Compare Callers"] = cfg.caller_comparison.compare_callers
        params["Caller Columns"] = cfg.caller_comparison.caller_columns
        if cfg.caller_comparison.caller_combinations:
            params["Custom Combinations"] = cfg.caller_comparison.caller_combinations
        params["Include Default Combos"] = (
            cfg.caller_comparison.include_default_combinations
        )

    if cfg.continuous_cols:
        params["Continuous Columns"] = cfg.continuous_cols
    if cfg.visualization.error_analysis_features:
        params["Error Analysis Features"] = cfg.visualization.error_analysis_features
    if cfg.hpo_method:
        params["HPO Method"] = cfg.hpo_method
        if cfg.hpo_method == "optuna":
            params["Optuna Trials"] = cfg.optuna_n_trials
            if cfg.optuna_timeout:
                params["Optuna Timeout (s)"] = cfg.optuna_timeout

    # Display parameters table
    typer.echo("")
    _display_parameters_table(params, "Compare-Models Configuration")
    typer.echo("")

    # Dry run mode - show what would be executed
    if dry_run:
        typer.echo("DRY RUN MODE - No models will be trained")
        typer.echo("Configuration is valid. Run without --dry-run to execute.\n")
        return

    typer.echo("Comparing ML models")
    loader = DataLoader(str(data_dir))
    for dataset in datasets:
        ds_name = Path(dataset).stem
        typer.echo(f"\n1) Loading dataset: {ds_name}")
        df = loader.load_dataset(dataset, drop_duplicates=True, id_column=id_column)
        typer.echo(f"   Loaded dataset with shape: {df.shape}")
        if cfg.nan_strategy == "drop":
            df, n_dropped = drop_missing_rows(df, target_column=cfg.target_column)
            typer.echo(
                f"   Dropped {n_dropped} rows with missing values (nan-strategy=drop)"
            )
        typer.echo("\n2) Analyzing features")
        feature_report = loader.create_feature_report(
            df,
            continuous_cols=cfg.continuous_cols,
            target_col=cfg.target_column,
            output_path=str(Path(cfg.output_dir) / ds_name / "feature_report.json"),
        )
        typer.echo(f"   Number of features: {len(feature_report['columns']) - 1}")
        typer.echo(f"   Target distribution: {feature_report['target_distribution']}")
        typer.echo("\n3) Running cross-validation pipeline")

        # Set data file path for metadata tracking
        cfg.data_file_path = str((data_dir / dataset).resolve())

        pipeline = CrossValidationPipeline(cfg)
        results, caller_results = pipeline.run_cross_validation(df, ds_name)
        feature_names = [c for c in df.columns if c != cfg.target_column]
        create_feature_importance_report(
            results,
            feature_names,
            Path(cfg.output_dir) / ds_name / "feature_importance_summary.csv",
        )

        # Report caller comparison results if enabled
        if caller_results:
            typer.echo(
                f"\n   Caller comparison: {len(caller_results)} callers/combinations evaluated"
            )

        best_model, best_auroc = None, 0.0
        for model_name, fold_results in results.items():
            avg_auroc = sum(fr.metrics["AUROC"] for fr in fold_results) / max(
                len(fold_results), 1
            )
            if avg_auroc > best_auroc:
                best_auroc, best_model = avg_auroc, model_name
        typer.echo("\nAnalysis complete")
        typer.echo(f"Best performing model: {best_model} (AUROC: {best_auroc:.3f})")
        typer.echo(f"Results -> {cfg.output_dir}/{ds_name}")
