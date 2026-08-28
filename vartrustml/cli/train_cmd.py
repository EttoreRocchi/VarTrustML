"""
Training CLI command for VarTrustML.

Contains:
- train: Train a single model on a dataset
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from vartrustml import (
    DataLoader,
    ModelConfig,
    __version__,
)
from vartrustml.core.train_model import ModelTrainer, TrainConfig

from vartrustml.cli._shared import (
    _display_parameters_table,
    _parse_multi_value,
)
from vartrustml.cli.main import app


@app.command("train", rich_help_panel="Model Application")
def train(
    data: Path = typer.Argument(..., help="Path to training data file."),
    target: str = typer.Option(
        ...,
        "--target",
        "-t",
        help="Target column name (REQUIRED).",
        rich_help_panel="Data Configuration",
    ),
    model: str = typer.Option(
        "XGBoost",
        "--model",
        "-m",
        help="Model name. [default: XGBoost]",
        rich_help_panel="Model Selection",
    ),
    features: Optional[str] = typer.Option(
        None,
        "--features",
        "-f",
        callback=_parse_multi_value,
        help="Explicit feature columns (comma-separated).",
        rich_help_panel="Data Configuration",
    ),
    continuous: Optional[str] = typer.Option(
        None,
        "--continuous",
        callback=_parse_multi_value,
        help="Continuous columns to scale (comma-separated list or path to .txt file).",
        rich_help_panel="Data Configuration",
    ),
    categorical: Optional[str] = typer.Option(
        None,
        "--categorical",
        callback=_parse_multi_value,
        help="Categorical columns (comma-separated list or path to .txt file). If provided without --continuous, continuous columns are inferred by exclusion.",
        rich_help_panel="Data Configuration",
    ),
    test_data: Optional[Path] = typer.Option(
        None,
        "--test-data",
        help="Path to separate test data.",
        rich_help_panel="Data Configuration",
    ),
    test_size: float = typer.Option(
        0.0,
        "--test-size",
        help="Holdout test size (0 disables). [default: 0]",
        rich_help_panel="Data Configuration",
    ),
    cv_folds: int = typer.Option(
        5,
        "--cv-folds",
        help="CV folds for HPO. [default: 5]",
        rich_help_panel="Cross-Validation",
    ),
    scoring: str = typer.Option(
        "roc_auc",
        "--scoring",
        help="Scoring metric for CV. [default: roc_auc]",
        rich_help_panel="Cross-Validation",
    ),
    output_dir: Path = typer.Option(
        Path("results/trained_model"),
        "--output-dir",
        "-o",
        help="Output directory.",
        rich_help_panel="Input/Output",
    ),
    param_config: Optional[Path] = typer.Option(
        None,
        "--param-config",
        help="JSON file with custom hyperparameter search space.",
        rich_help_panel="Model Selection",
    ),
    seed: int = typer.Option(
        42, "--seed", help="Random seed.", rich_help_panel="Cross-Validation"
    ),
    n_jobs: int = typer.Option(
        -1, "--n-jobs", help="Parallel jobs. [default: -1]", rich_help_panel="System"
    ),
    calibrate_model: bool = typer.Option(
        False,
        "--calibrate-model",
        help="Enable probability calibration.",
        rich_help_panel="Calibration",
    ),
    calibration_method: str = typer.Option(
        "isotonic",
        "--calibration-method",
        help="Calibration method: 'isotonic' or 'sigmoid'.",
        rich_help_panel="Calibration",
    ),
    calibration_cv: int = typer.Option(
        3,
        "--calibration-cv",
        help="Number of CV folds for calibration.",
        rich_help_panel="Calibration",
    ),
    optimize_threshold: bool = typer.Option(
        False,
        "--optimize-threshold",
        help="Enable threshold optimization using Youden's J statistic.",
        rich_help_panel="Threshold Optimization",
    ),
    threshold_method: str = typer.Option(
        "auto",
        "--threshold-method",
        help="Threshold optimization method: 'oof', 'cv', or 'auto'.",
        rich_help_panel="Threshold Optimization",
    ),
    no_save_model: bool = typer.Option(
        False,
        "--no-save-model",
        help="Disable saving the fitted model.",
        rich_help_panel="Input/Output",
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
        help="Path for HTML report (relative to output dir). [default: train_report.html]",
        rich_help_panel="Reports",
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Disable output.", rich_help_panel="System"
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
    Train a single model on a dataset (with optional holdout or external test set).
    """
    if not quiet:
        typer.echo(f"Loading data from {data}")
    loader = DataLoader(str(data.parent))
    df = loader.load_dataset(data.name, id_column=id_column, drop_duplicates=False)
    n_raw = len(df)
    df = df.drop_duplicates(keep="first")
    if not quiet:
        typer.echo(f"Loaded dataset: {df.shape}")
        if len(df) < n_raw:
            typer.echo(f"Dropped {n_raw - len(df)} duplicate rows from training data.")
    X = df.drop(columns=[target])
    y = df[target]
    if not features:
        features = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[features]
    if not quiet:
        typer.echo(f"Using {len(features)} features.")
    if test_data:
        # Rooted at the test file's own directory: resolving its name against
        # the training directory would silently load a different file
        test_loader = DataLoader(str(test_data.parent))
        tdf = test_loader.load_dataset(
            test_data.name, id_column=id_column, drop_duplicates=False
        )
        y_test = tdf[target]
        X_test = tdf[features]
        X_train, y_train = X, y
        if not quiet:
            typer.echo(f"Using separate test set with {len(X_test)} samples.")
    elif test_size and test_size > 0:
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        if not quiet:
            typer.echo(f"Split data: train={len(X_train)}, test={len(X_test)}")
    else:
        X_train, y_train = X, y
        X_test, y_test = None, None
        if not quiet:
            typer.echo(f"Using all {len(X_train)} samples for training.")
    model_cfg = None
    if param_config:
        model_cfg = ModelConfig(**json.loads(Path(param_config).read_text()))
        if not quiet:
            typer.echo(f"Loaded hyperparameter config from {param_config}")
    # Build TrainConfig, only override continuous_cols if explicitly provided
    train_cfg_kwargs = {
        "model_name": model,
        "seed": seed,
        "n_cv_folds": cv_folds,
        "scoring": scoring,
        "n_jobs": n_jobs,
        "calibrate_model": calibrate_model,
        "calibration_method": calibration_method,
        "calibration_cv": calibration_cv,
        "optimize_threshold": optimize_threshold,
        "threshold_method": threshold_method,
        "target_column": target,
        "output_dir": str(output_dir),
        "save_model": not no_save_model,
    }
    if continuous:
        train_cfg_kwargs["continuous_cols"] = list(continuous)
    if categorical:
        train_cfg_kwargs["categorical_cols"] = list(categorical)
    train_cfg = TrainConfig(**train_cfg_kwargs)

    # Prepare parameters for display
    params = {
        "Data File": str(data),
        "Model": model,
        "Target Column": target,
        "Number of Features": len(features) if features else "All numeric",
        "Train Samples": len(X_train),
        "Test Samples": len(X_test) if X_test is not None else 0,
        "Test Strategy": "Separate test file"
        if test_data
        else (f"{test_size:.1%} holdout" if test_size > 0 else "No test set"),
        "CV Folds": cv_folds,
        "Scoring Metric": scoring,
        "Calibrate Model": calibrate_model,
        "Calibration Method": calibration_method if calibrate_model else "N/A",
        "Optimize Threshold": optimize_threshold,
        "Threshold Method": threshold_method if optimize_threshold else "N/A",
        "Parallel Jobs": n_jobs,
        "Random Seed": seed,
        "Output Directory": str(output_dir),
        "Save Model": not no_save_model,
        "Generate HTML Report": not no_html_report,
    }

    if continuous:
        params["Continuous Columns"] = continuous
    if categorical:
        params["Categorical Columns"] = categorical
    if features:
        params["Explicit Features"] = features
    if param_config:
        params["Custom Param Config"] = str(param_config)

    # Display parameters table
    if not quiet:
        typer.echo("")
        _display_parameters_table(params, "Train Configuration")
        typer.echo("")

    # Dry run mode - show what would be executed
    if dry_run:
        typer.echo("DRY RUN MODE - No model will be trained")
        typer.echo("Configuration is valid. Run without --dry-run to execute.\n")
        return

    if not quiet:
        typer.echo(f"Fitting {model} with {cv_folds}-fold CV")
    fitter = ModelTrainer(train_cfg, model_cfg)
    results = fitter.fit(X_train, y_train, X_test, y_test)
    if not quiet:
        typer.echo("\n============================================================")
        typer.echo("RESULTS")
        typer.echo("============================================================")
        typer.echo(f"Best parameters: {results['best_params']}")
        typer.echo(f"Best CV score ({scoring}): {results['best_score']:.4f}")
        if "test_results" in results:
            tr = results["test_results"]
            typer.echo("\nTest set performance:")
            typer.echo(f"  Matthews corr. coef.: {tr['matthews corr. coef.']:.4f}")
            typer.echo(f"  Balanced accuracy: {tr['balanced_accuracy']:.4f}")
            typer.echo(f"  F1 (weighted): {tr['f1_weighted']:.4f}")
            if tr["auroc"]:
                typer.echo(f"  AUROC: {tr['auroc']:.4f}")
        if "threshold_result" in results and results["threshold_result"] is not None:
            tr = results["threshold_result"]
            typer.echo("\nThreshold Optimization:")
            typer.echo(f"  Optimal threshold: {tr.optimal_threshold:.4f}")
            typer.echo(f"  Youden's J: {tr.youden_j:.4f}")
            typer.echo(f"  Sensitivity: {tr.sensitivity_at_threshold:.4f}")
            typer.echo(f"  Specificity: {tr.specificity_at_threshold:.4f}")
        typer.echo(f"\nOutputs saved to: {output_dir}/")

    # Generate HTML report (enabled by default)
    if not no_html_report:
        import sys

        from sklearn.metrics import confusion_matrix

        from vartrustml.visualization.html_train_reporter import HTMLTrainReporter

        # Determine HTML report path
        if html_path:
            html_output_path = output_dir / html_path
        else:
            html_output_path = output_dir / "train_report.html"

        reporter = HTMLTrainReporter(output_path=str(html_output_path))

        # Prepare run metadata
        run_metadata = {
            "vartrustml_version": __version__,
            "python_version": sys.version.split()[0],
            "input_file": str(data),
            "output_dir": str(output_dir),
        }

        # Prepare config dict
        config_dict = {
            "seed": seed,
            "n_cv_folds": cv_folds,
            "scoring": scoring,
            "calibrate_model": calibrate_model,
            "calibration_method": calibration_method,
            "calibration_cv": calibration_cv,
        }

        # Prepare dataset info
        # Compute class balance as "x% : y%" format
        train_class_0_count = int((y_train == 0).sum())
        train_class_1_count = int((y_train == 1).sum())
        train_class_0_pct = (
            (train_class_0_count / len(y_train) * 100) if len(y_train) > 0 else 0.0
        )
        train_class_1_pct = (
            (train_class_1_count / len(y_train) * 100) if len(y_train) > 0 else 0.0
        )
        train_class_balance = (
            f"{train_class_0_pct:.2f}% : {train_class_1_pct:.2f}%"
            if len(y_train) > 0
            else "N/A"
        )

        dataset_info = {
            "n_train_samples": len(X_train),
            "n_features": len(features) if features else X_train.shape[1],
            "n_continuous": len(train_cfg.continuous_cols)
            if train_cfg.continuous_cols
            else 0,
            "train_class_0_count": train_class_0_count,
            "train_class_1_count": train_class_1_count,
            "train_class_balance": train_class_balance,
            "has_test_set": X_test is not None,
        }

        if X_test is not None:
            dataset_info["test_size"] = len(X_test)
            if test_size > 0:
                dataset_info["test_split_ratio"] = test_size

        # Add training overview with run metadata
        reporter.add_training_overview(config_dict, dataset_info, model, run_metadata)

        # Add hyperparameter results
        cv_results_df = None
        if "cv_results" in results and results["cv_results"] is not None:
            cv_results_df = pd.DataFrame(results["cv_results"])

        reporter.add_hyperparameter_results(
            results["best_params"], results["best_score"], cv_results_df
        )

        # Add CV fold details
        if cv_results_df is not None:
            reporter.add_cv_fold_details(cv_results_df, scoring, cv_folds)

        # Add full search history
        if cv_results_df is not None:
            reporter.add_full_search_history(cv_results_df, scoring)

        # Add test results if available
        if "test_results" in results:
            reporter.add_test_results(results["test_results"])

        # Add confusion matrix if test set was used
        if X_test is not None and y_test is not None:
            try:
                y_pred = results.get("fitted_model", fitter.model).predict(X_test)
                cm = confusion_matrix(y_test, y_pred, normalize="true")
                reporter.add_confusion_matrix(cm, normalize=True)
            except Exception as e:
                if not quiet:
                    typer.echo(f"Warning: Could not generate confusion matrix: {e}")

        # Add feature importance if available
        if (
            "feature_importance" in results
            and results["feature_importance"] is not None
        ):
            feature_names_list = features if features else X_train.columns.tolist()
            reporter.add_feature_importance(
                results["feature_importance"],
                feature_names_list,
                top_n=min(20, len(feature_names_list)),
            )

        # Add threshold optimization results if available
        if optimize_threshold and "threshold_result" in results:
            threshold_result = results["threshold_result"]

            # Build test metrics comparison if test set was used
            test_metrics_comparison = None
            if X_test is not None and "test_results" in results:
                tr = results["test_results"]
                # Metrics at optimized threshold
                optimized_metrics = {
                    "Balanced Accuracy": tr.get(
                        "balanced_accuracy_threshold", tr.get("balanced_accuracy")
                    ),
                    "Matthews Corr. Coef.": tr.get(
                        "matthews_coef_threshold", tr.get("matthews corr. coef.")
                    ),
                    "F1 (Weighted)": tr.get(
                        "f1_weighted_threshold", tr.get("f1_weighted")
                    ),
                }
                # Metrics at default threshold (0.5)
                default_metrics = {
                    "Balanced Accuracy": tr.get("balanced_accuracy"),
                    "Matthews Corr. Coef.": tr.get("matthews corr. coef."),
                    "F1 (Weighted)": tr.get("f1_weighted"),
                }
                # Only include comparison if we have threshold-specific metrics
                if tr.get("balanced_accuracy_threshold") is not None:
                    test_metrics_comparison = {
                        "default": default_metrics,
                        "optimized": optimized_metrics,
                    }

            reporter.add_threshold_results(
                threshold_result=threshold_result.to_dict()
                if hasattr(threshold_result, "to_dict")
                else threshold_result,
                test_metrics_comparison=test_metrics_comparison,
            )

        # Generate and save report
        report_path = reporter.generate_report()
        if not quiet:
            typer.echo(f"\nHTML report generated: {report_path}")
