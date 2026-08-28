"""
Cross-dataset reporting helpers for VarTrustML.

Report-generation functions split out of ``CrossDatasetEvaluator`` so that
the evaluator itself only orchestrates.

Functions
---------
generate_cross_dataset_summary
    Build and save text + JSON summary reports for a cross-dataset run.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from vartrustml.utils.serialization import np_encoder

logger = logging.getLogger(__name__)


def _spread(values: List[float]) -> float:
    """Sample standard deviation of the cell means, 0.0 for a single cell."""
    if len(values) < 2:
        return 0.0
    return float(np.std(np.asarray(values, dtype=float), ddof=1))


def _gap_bootstrap_ci(
    diagonal_vals: List[float],
    off_diagonal_vals: List[float],
    n_iterations: int,
    ci_level: float,
    seed: int,
) -> tuple:
    """Percentile bootstrap interval for the same-minus-cross gap.

    Resamples the diagonal and off-diagonal cell means independently. Returns
    ``(nan, nan)`` when either group has fewer than two cells.
    """
    diag = np.asarray(diagonal_vals, dtype=float)
    off = np.asarray(off_diagonal_vals, dtype=float)
    if diag.size < 2 or off.size < 2:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    diag_means = diag[rng.integers(0, diag.size, (n_iterations, diag.size))].mean(
        axis=1
    )
    off_means = off[rng.integers(0, off.size, (n_iterations, off.size))].mean(axis=1)
    gaps = diag_means - off_means
    lo = float(np.percentile(gaps, 100 * (1 - ci_level) / 2))
    hi = float(np.percentile(gaps, 100 * (1 + ci_level) / 2))
    return lo, hi


def generate_cross_dataset_summary(
    results: Dict[str, Dict[str, pd.DataFrame]],
    results_std: Dict[str, Dict[str, pd.DataFrame]],
    dataset_names: List[str],
    output_dir: Path,
    metrics_to_track: List[str],
    n_outer_splits: int,
    n_inner_splits: int,
    optimize_threshold: bool,
    threshold_method: str,
    threshold_auto_n_samples: int,
    threshold_results: Optional[Dict[str, Dict[str, List[float]]]] = None,
    bootstrap_n_iterations: int = 2000,
    bootstrap_ci_level: float = 0.95,
    seed: int = 42,
) -> None:
    """Generate the text and JSON summary reports.

    This is a standalone version of the former
    ``CrossDatasetEvaluator._generate_summary_report`` method.

    Parameters
    ----------
    results : dict
        Mean results matrices.
        ``model_name -> metric_name -> DataFrame``.
    results_std : dict
        Std results matrices (same structure as *results*).
    dataset_names : list of str
        List of dataset names.
    output_dir : pathlib.Path
        Directory to save reports.
    metrics_to_track : list of str
        Metric names tracked during evaluation.
    n_outer_splits : int
        Number of outer CV folds (for the report header).
    n_inner_splits : int
        Number of inner CV folds (for the report header).
    optimize_threshold : bool
        Whether threshold optimization was enabled.
    threshold_method : str
        Threshold optimization method name.
    threshold_auto_n_samples : int
        Number of auto samples for threshold optimization.
    threshold_results : dict, optional
        Maps ``source_name -> model_name -> list`` of per-fold thresholds.
    bootstrap_n_iterations : int, default=2000
        Resamples used for the confidence interval of the generalization gap.
    bootstrap_ci_level : float, default=0.95
        Confidence level of that interval.
    seed : int, default=42
        Seed of the bootstrap resampling.

    Notes
    -----
    The dispersion reported next to the same-dataset and cross-dataset means is
    the spread of the cell means themselves, which is the quantity a reader
    interprets from that line. The per-cell fold standard deviations stay in the
    per-cell matrices. The gap is reported with a bootstrap interval over cells
    rather than a t-statistic, because cells that share training data are not
    independent and a t-test over them is anti-conservative.
    """
    report_lines: List[str] = []
    report_lines.append("CROSS-DATASET EVALUATION SUMMARY")
    report_lines.append("=" * 60)
    report_lines.append(f"Datasets: {', '.join(dataset_names)}")
    report_lines.append(f"Models: {', '.join(results.keys())}")
    report_lines.append(
        f"Cross-validation: {n_outer_splits} outer folds, {n_inner_splits} inner folds"
    )
    report_lines.append("")

    summary_data: List[dict] = []

    for model_name, model_results in results.items():
        report_lines.append(f"\n{model_name}")
        report_lines.append("-" * 40)

        model_summary: dict = {"Model": model_name}

        for metric_name, matrix in model_results.items():
            if matrix.empty:
                continue

            std_matrix = results_std[model_name][metric_name]

            # Calculate diagonal (same-dataset) statistics
            diagonal_vals: List[float] = []
            for ds in dataset_names:
                if pd.notna(matrix.loc[ds, ds]):
                    diagonal_vals.append(matrix.loc[ds, ds])

            # Calculate off-diagonal (cross-dataset) statistics
            off_diagonal_vals: List[float] = []
            best_cross_pair = None
            worst_cross_pair = None
            best_cross_val = -np.inf
            worst_cross_val = np.inf

            for train_ds in dataset_names:
                for test_ds in dataset_names:
                    if train_ds != test_ds and pd.notna(matrix.loc[train_ds, test_ds]):
                        val = matrix.loc[train_ds, test_ds]
                        off_diagonal_vals.append(val)

                        if val > best_cross_val:
                            best_cross_val = val
                            best_cross_pair = (train_ds, test_ds)

                        if val < worst_cross_val:
                            worst_cross_val = val
                            worst_cross_pair = (train_ds, test_ds)

            if diagonal_vals and off_diagonal_vals:
                same_mean = np.mean(diagonal_vals)
                cross_mean = np.mean(off_diagonal_vals)
                same_std = _spread(diagonal_vals)
                cross_std = _spread(off_diagonal_vals)
                gap = same_mean - cross_mean

                report_lines.append(f"\n  {metric_name}:")
                report_lines.append(
                    f"    Same dataset:  {same_mean:.3f} +/- {same_std:.3f} "
                    f"(SD across {len(diagonal_vals)} cells)"
                )
                report_lines.append(
                    f"    Cross dataset: {cross_mean:.3f} +/- {cross_std:.3f} "
                    f"(SD across {len(off_diagonal_vals)} cells)"
                )
                report_lines.append(f"    Generalization gap: {gap:.3f}")

                if best_cross_pair:
                    best_std = std_matrix.loc[best_cross_pair[0], best_cross_pair[1]]
                    report_lines.append(
                        f"    Best cross-dataset: {best_cross_val:.3f} +/- {best_std:.3f} "
                        f"({best_cross_pair[0]} -> {best_cross_pair[1]})"
                    )

                if worst_cross_pair:
                    worst_std = std_matrix.loc[worst_cross_pair[0], worst_cross_pair[1]]
                    report_lines.append(
                        f"    Worst cross-dataset: {worst_cross_val:.3f} +/- {worst_std:.3f} "
                        f"({worst_cross_pair[0]} -> {worst_cross_pair[1]})"
                    )

                # Bootstrap interval over cells for the gap
                gap_lo, gap_hi = _gap_bootstrap_ci(
                    diagonal_vals,
                    off_diagonal_vals,
                    bootstrap_n_iterations,
                    bootstrap_ci_level,
                    seed,
                )
                if not np.isnan(gap_lo):
                    report_lines.append(
                        f"    Gap {bootstrap_ci_level:.0%} CI: "
                        f"[{gap_lo:.3f}, {gap_hi:.3f}] "
                        f"(bootstrap over cells)"
                    )

                if metric_name == "AUROC":
                    model_summary["AUROC_same"] = f"{same_mean:.3f}+/-{same_std:.3f}"
                    model_summary["AUROC_cross"] = f"{cross_mean:.3f}+/-{cross_std:.3f}"
                    model_summary["AUROC_gap"] = f"{gap:.3f}"
                    model_summary["AUROC_gap_ci_lower"] = gap_lo
                    model_summary["AUROC_gap_ci_upper"] = gap_hi

        summary_data.append(model_summary)

    # AUROC Summary Table
    report_lines.append("\n\nSUMMARY TABLE (AUROC)")
    report_lines.append("=" * 60)
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        report_lines.append(summary_df.to_string(index=False))

    # Dataset-specific insights
    report_lines.append("\n\nDATASET-SPECIFIC INSIGHTS")
    report_lines.append("=" * 60)

    for model_name, model_results in results.items():
        if "AUROC" not in model_results:
            continue

        matrix = model_results["AUROC"]

        # Which dataset is easiest/hardest to predict?
        test_performance: Dict[str, float] = {}
        for test_ds in dataset_names:
            perfs: List[float] = []
            for train_ds in dataset_names:
                if pd.notna(matrix.loc[train_ds, test_ds]):
                    perfs.append(matrix.loc[train_ds, test_ds])
            if perfs:
                test_performance[test_ds] = np.mean(perfs)

        if test_performance:
            easiest = max(test_performance, key=test_performance.get)
            hardest = min(test_performance, key=test_performance.get)

            report_lines.append(f"\n{model_name}:")
            report_lines.append(
                f"  Easiest to predict: {easiest} (avg AUROC: {test_performance[easiest]:.3f})"
            )
            report_lines.append(
                f"  Hardest to predict: {hardest} (avg AUROC: {test_performance[hardest]:.3f})"
            )

    # Stability analysis
    report_lines.append("\n\nSTABILITY ANALYSIS (CV Standard Deviations)")
    report_lines.append("=" * 60)

    for model_name in results.keys():
        all_stds: List[float] = []
        for metric_name in ["AUROC", "F1 Score (Weighted)", "Balanced Accuracy"]:
            if metric_name in results_std[model_name]:
                std_matrix = results_std[model_name][metric_name]
                valid_stds = std_matrix.values.flatten()
                valid_stds = valid_stds[~np.isnan(valid_stds)]
                all_stds.extend(valid_stds)

        if all_stds:
            report_lines.append(f"\n{model_name}:")
            report_lines.append(f"  Mean CV std: {np.mean(all_stds):.4f}")
            report_lines.append(f"  Max CV std: {np.max(all_stds):.4f}")
            stability_score = 1 / (1 + np.mean(all_stds))
            report_lines.append(
                f"  Stability score: {stability_score:.3f} (higher is better)"
            )

    # Threshold Optimization Results (if enabled)
    # Now uses per-fold thresholds (like compare-models) instead of global threshold
    if threshold_results:
        report_lines.append("\n\nTHRESHOLD OPTIMIZATION RESULTS (Per-Fold)")
        report_lines.append("=" * 60)
        report_lines.append(
            f"Method: {threshold_method} (auto_n_samples={threshold_auto_n_samples})"
        )
        report_lines.append(
            "Note: Thresholds are optimized per-fold (like compare-models) for "
            "fair comparison."
        )

        for src_name in dataset_names:
            if src_name not in threshold_results:
                continue

            report_lines.append(f"\n{src_name} (Training Dataset):")
            report_lines.append("-" * 40)

            for model_name in results.keys():
                if model_name not in threshold_results[src_name]:
                    continue

                fold_thresholds = threshold_results[src_name][model_name]
                if fold_thresholds:
                    fold_mean = np.mean(fold_thresholds)
                    fold_std = np.std(fold_thresholds)
                    report_lines.append(f"  {model_name}:")
                    report_lines.append(
                        f"    Mean threshold: {fold_mean:.4f} +/- {fold_std:.4f}"
                    )
                    report_lines.append(
                        f"    Per-fold thresholds: "
                        f"[{', '.join(f'{t:.3f}' for t in fold_thresholds)}]"
                    )

    # Save text report
    with open(output_dir / "cross_dataset_summary.txt", "w") as f:
        f.write("\n".join(report_lines))

    # Save JSON results
    json_summary = {
        "config": {
            "datasets": dataset_names,
            "models": list(results.keys()),
            "n_outer_splits": n_outer_splits,
            "n_inner_splits": n_inner_splits,
            "metrics": metrics_to_track,
            "optimize_threshold": optimize_threshold,
            "threshold_method": threshold_method,
        },
        "results": {},
    }

    for model_name in results.keys():
        json_summary["results"][model_name] = {}
        for metric_name in results[model_name].keys():
            json_summary["results"][model_name][metric_name] = {
                "mean_matrix": results[model_name][metric_name].to_dict(),
                "std_matrix": results_std[model_name][metric_name].to_dict(),
            }

    # Add threshold results to JSON if present (per-fold thresholds)
    if threshold_results:
        json_summary["threshold_optimization"] = {
            src_name: {
                model_name: {
                    "per_fold_thresholds": fold_thresholds,
                    "mean": float(np.mean(fold_thresholds))
                    if fold_thresholds
                    else None,
                    "std": float(np.std(fold_thresholds)) if fold_thresholds else None,
                }
                for model_name, fold_thresholds in model_thresholds.items()
            }
            for src_name, model_thresholds in threshold_results.items()
        }

    with open(output_dir / "cross_dataset_results.json", "w") as f:
        json.dump(json_summary, f, indent=2, default=np_encoder)

    logger.info(f"Cross-dataset evaluation complete. Results saved to {output_dir}")
