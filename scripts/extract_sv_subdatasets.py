"""
Extract subdatasets for each SVTYPE_CALLER_* from structural variant datasets.

This script processes SV data files and creates subdatasets for each SV type:
- DEL (deletions)
- DUP (duplications)
- INS (insertions)
- INV (inversions)
"""

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


class Report:
    """Handle reporting with different verbosity levels."""

    def __init__(self, verbosity: int = 1, output_file: Optional[str] = None):
        """
        Initialize report handler.

        Parameters
        ----------
        verbosity : int
            Verbosity level (0=minimal, 1=normal, 2=detailed)
        output_file : str, optional
            Path to output file for report
        """
        self.verbosity = verbosity
        self.output_file = output_file
        self.messages = []

    def log(self, message: str, level: int = 1, end: str = "\n"):
        """
        Log a message at specified verbosity level.

        Parameters
        ----------
        message : str
            Message to log
        level : int
            Minimum verbosity level required to display this message
        end : str
            Line ending character
        """
        if self.verbosity >= level:
            print(message, end=end)
        self.messages.append(message + end)

    def section(self, title: str, level: int = 0):
        """Print a section header."""
        separator = "=" * 70
        self.log(f"\n{separator}", level)
        self.log(title, level)
        self.log(separator, level)

    def subsection(self, title: str, level: int = 1):
        """Print a subsection header."""
        self.log(f"\n{title}", level)
        self.log("-" * len(title), level)

    def save(self):
        """Save report to file if output_file is specified."""
        if self.output_file:
            output_path = Path(self.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.writelines(self.messages)
            print(f"\nReport saved to: {self.output_file}")


# =============================================================================
# Helper functions
# =============================================================================


def get_subdataset(
    results: Dict[str, Dict[str, pd.DataFrame]], dataset_name: str, sv_type: str
) -> pd.DataFrame:
    """
    Convenience function to get a specific subdataset.

    Parameters
    ----------
    results : Dict[str, Dict[str, pd.DataFrame]]
        Results from process_all_datasets
    dataset_name : str
        Name of the dataset (e.g., 'HG002')
    sv_type : str
        SV type (e.g., 'DEL', 'DUP', 'INS', 'INV')

    Returns
    -------
    pd.DataFrame
        The requested subdataset
    """
    return results[dataset_name][sv_type]


def filter_subdataset_by_callers(
    subdataset: pd.DataFrame,
    callers: List[str],
    require_all: bool = False,
    report: Optional[Report] = None,
) -> pd.DataFrame:
    """
    Filter a subdataset by caller tools.

    Parameters
    ----------
    subdataset : pd.DataFrame
        SV subdataset
    callers : List[str]
        List of caller names (e.g., ['MANTA', 'DELLY'])
    require_all : bool, optional
        If True, require all callers to have called the variant.
        If False, require at least one caller, by default False
    report : Report, optional
        Report handler for logging

    Returns
    -------
    pd.DataFrame
        Filtered subdataset
    """
    # Check which caller columns exist
    available_callers = [c for c in callers if c in subdataset.columns]

    if not available_callers:
        msg = "Warning: None of the specified callers found in columns"
        if report:
            report.log(msg, level=1)
        else:
            print(msg)
        return subdataset

    if require_all:
        # All specified callers must be 1
        mask = (subdataset[available_callers] == 1).all(axis=1)
    else:
        # At least one caller must be 1
        mask = (subdataset[available_callers] == 1).any(axis=1)

    return subdataset[mask].copy()


# =============================================================================
# Core data processing functions
# =============================================================================


def load_sv_dataset(filepath: str) -> pd.DataFrame:
    """
    Load a structural variant dataset from a text file.

    Parameters
    ----------
    filepath : str
        Path to the dataset file

    Returns
    -------
    pd.DataFrame
        Loaded dataset
    """
    # Use regex pattern to match comma or tab as delimiter
    return pd.read_csv(filepath, sep="[,\t]", engine="python")


def extract_sv_subdatasets(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Extract subdatasets for each SVTYPE_CALLER_* column.

    Parameters
    ----------
    df : pd.DataFrame
        Full structural variant dataset

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary with SV types as keys and filtered dataframes as values
        Note: SVTYPE_CALLER_* columns are removed from subdatasets
    """
    # Identify SVTYPE_CALLER_* columns
    svtype_columns = [col for col in df.columns if col.startswith("SVTYPE_CALLER_")]

    subdatasets = {}

    for col in svtype_columns:
        # Extract SV type name (e.g., 'DEL' from 'SVTYPE_CALLER_DEL')
        sv_type = col.replace("SVTYPE_CALLER_", "")

        # Filter rows where this SV type is 1 (handle both 1.0 and 1)
        # Convert to numeric to handle potential string values
        col_values = pd.to_numeric(df[col], errors="coerce")
        subdataset = df[col_values == 1].copy()

        # Remove all SVTYPE_CALLER_* columns from the subdataset
        subdataset = subdataset.drop(columns=svtype_columns)

        subdatasets[sv_type] = subdataset

    return subdatasets


# =============================================================================
# Analysis functions
# =============================================================================


def generate_subdataset_report(
    subdataset: pd.DataFrame,
    dataset_name: str,
    sv_type: str,
    target_column: str = "state",
    verbosity: int = 1,
    output_file: Optional[str] = None,
) -> Report:
    """
    Generate a detailed report for a specific subdataset.

    Parameters
    ----------
    subdataset : pd.DataFrame
        The subdataset to analyze
    dataset_name : str
        Name of the parent dataset
    sv_type : str
        SV type (e.g., 'DEL', 'DUP')
    target_column : str, optional
        Name of the binary target column
    verbosity : int, optional
        Verbosity level (0=minimal, 1=normal, 2=detailed)
    output_file : str, optional
        Path to save the report

    Returns
    -------
    Report
        Report object with analysis
    """
    report = Report(verbosity=verbosity, output_file=output_file)

    report.section(f"SUBDATASET REPORT: {dataset_name} - {sv_type}")

    # Basic information
    report.log(f"\nDataset: {dataset_name}", level=0)
    report.log(f"SV Type: {sv_type}", level=0)
    report.log(f"Total records: {len(subdataset):,}", level=0)
    report.log(f"Total features: {len(subdataset.columns)}", level=1)

    # Column information
    if verbosity >= 2:
        report.subsection("Column Information", level=2)
        report.log(f"  Columns ({len(subdataset.columns)}):", level=2)
        for i, col in enumerate(subdataset.columns, 1):
            dtype = subdataset[col].dtype
            null_count = subdataset[col].isna().sum()
            null_pct = (null_count / len(subdataset)) * 100
            report.log(
                f"    {i:2d}. {col:30s} ({dtype}, {null_count} nulls, {null_pct:.1f}%)",
                level=2,
            )

    # Target distribution
    if target_column in subdataset.columns:
        report.subsection("Target Distribution", level=0)
        analyze_target_distribution(subdataset, target_column, report, label="")

    # SVLEN statistics
    if "SVLEN_CALLER" in subdataset.columns:
        report.subsection("Structural Variant Length Statistics", level=0)
        compute_summary_statistics(subdataset, report, indent="  ")

    # Caller information
    caller_columns = [
        col
        for col in subdataset.columns
        if col.isupper()
        and col not in ["CHROM", "POS", "END", "SVLEN_CALLER", target_column]
    ]

    if caller_columns and verbosity >= 1:
        report.subsection("Caller Support", level=1)
        report.log(f"  Available callers: {len(caller_columns)}", level=1)

        for caller in caller_columns:
            if subdataset[caller].dtype in ["int64", "float64", "int32", "float32"]:
                support = (subdataset[caller] == 1).sum()
                support_pct = (support / len(subdataset)) * 100
                report.log(f"    {caller}: {support:,} ({support_pct:.1f}%)", level=1)

    # Statistical summary
    if verbosity >= 2:
        report.subsection("Numerical Feature Summary", level=2)
        numeric_cols = subdataset.select_dtypes(
            include=["int64", "float64", "int32", "float32"]
        ).columns
        numeric_cols = [col for col in numeric_cols if col != target_column]

        if len(numeric_cols) > 0:
            summary = subdataset[numeric_cols].describe()
            report.log(
                f"  Summary statistics for {len(numeric_cols)} numerical features:",
                level=2,
            )
            report.log("  (Showing count, mean, std, min, 25%, 50%, 75%, max)", level=2)
            # Show only selected columns to avoid clutter
            for col in numeric_cols[:5]:  # Show first 5
                report.log(f"\n  {col}:", level=2)
                for stat in ["count", "mean", "std", "min", "50%", "max"]:
                    report.log(
                        f"    {stat:6s}: {summary.loc[stat, col]:>12.2f}", level=2
                    )

            if len(numeric_cols) > 5:
                report.log(
                    f"\n  ... and {len(numeric_cols) - 5} more numerical features",
                    level=2,
                )

    report.section("REPORT COMPLETE")
    report.save()

    return report


def analyze_target_distribution(
    df: pd.DataFrame, target_column: str, report: Report, label: str = ""
) -> Dict:
    """
    Analyze distribution of binary target column.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to analyze
    target_column : str
        Name of the binary target column
    report : Report
        Report handler
    label : str, optional
        Label for the analysis (e.g., dataset name or SV type)

    Returns
    -------
    Dict
        Dictionary with distribution statistics
    """
    stats = {}

    if target_column not in df.columns:
        report.log(
            f"  Warning: Target column '{target_column}' not found in dataset", level=1
        )
        return stats

    # Get value counts
    value_counts = df[target_column].value_counts().sort_index()
    total = len(df)
    missing = df[target_column].isna().sum()

    stats["total"] = total
    stats["missing"] = missing
    stats["value_counts"] = value_counts.to_dict()

    prefix = f"    {label} - " if label else "  "
    report.log(f"{prefix}Target column: '{target_column}'", level=1)

    for value, count in value_counts.items():
        percentage = (count / total) * 100
        stats[f"pct_{value}"] = percentage
        report.log(f"{prefix}  {value}: {count:,} ({percentage:.2f}%)", level=1)

    # Additional statistics for detailed report
    if report.verbosity >= 2:
        report.log(
            f"{prefix}  Missing values: {missing} ({missing / total * 100:.2f}%)",
            level=2,
        )
        report.log(f"{prefix}  Unique values: {df[target_column].nunique()}", level=2)

        if df[target_column].nunique() == 2 and len(value_counts) == 2:
            # Calculate imbalance ratio for binary targets
            sorted_counts = sorted(value_counts.values, reverse=True)
            imbalance_ratio = sorted_counts[0] / sorted_counts[1]
            stats["imbalance_ratio"] = imbalance_ratio
            report.log(
                f"{prefix}  Imbalance ratio: {imbalance_ratio:.2f}:1 (majority:minority)",
                level=2,
            )

    return stats


def compute_summary_statistics(
    subdf: pd.DataFrame, report: Report, indent: str = "    "
) -> Dict:
    """
    Compute summary statistics for a subdataset.

    Parameters
    ----------
    subdf : pd.DataFrame
        Subdataset to analyze
    report : Report
        Report handler
    indent : str
        Indentation for log messages

    Returns
    -------
    Dict
        Dictionary with summary statistics
    """
    stats = {}
    stats["count"] = len(subdf)

    report.log(f"{indent}Count: {len(subdf):,}", level=1)

    if "SVLEN_CALLER" in subdf.columns:
        mean_svlen = subdf["SVLEN_CALLER"].mean()
        median_svlen = subdf["SVLEN_CALLER"].median()

        stats["mean_svlen"] = mean_svlen
        stats["median_svlen"] = median_svlen

        report.log(f"{indent}Mean SVLEN: {mean_svlen:,.0f}", level=1)
        report.log(f"{indent}Median SVLEN: {median_svlen:,.0f}", level=1)

        if report.verbosity >= 2:
            min_svlen = subdf["SVLEN_CALLER"].min()
            max_svlen = subdf["SVLEN_CALLER"].max()
            std_svlen = subdf["SVLEN_CALLER"].std()

            stats["std_svlen"] = std_svlen
            stats["min_svlen"] = min_svlen
            stats["max_svlen"] = max_svlen

            report.log(f"{indent}Std SVLEN: {std_svlen:,.0f}", level=2)
            report.log(f"{indent}Min SVLEN: {min_svlen:,.0f}", level=2)
            report.log(f"{indent}Max SVLEN: {max_svlen:,.0f}", level=2)

    return stats


def compare_sv_types_across_datasets(
    results: Dict[str, Dict[str, pd.DataFrame]], sv_type: str
) -> pd.DataFrame:
    """
    Compare a specific SV type across all datasets.

    Parameters
    ----------
    results : Dict[str, Dict[str, pd.DataFrame]]
        Results from process_all_datasets
    sv_type : str
        SV type to compare

    Returns
    -------
    pd.DataFrame
        Comparison dataframe with statistics
    """
    comparison = []

    for dataset_name, subdatasets in results.items():
        if sv_type in subdatasets:
            subdf = subdatasets[sv_type]
            stats = {"Dataset": dataset_name, "Count": len(subdf)}

            if "SVLEN_CALLER" in subdf.columns:
                stats.update(
                    {
                        "Mean_SVLEN": subdf["SVLEN_CALLER"].mean(),
                        "Median_SVLEN": subdf["SVLEN_CALLER"].median(),
                        "Min_SVLEN": subdf["SVLEN_CALLER"].min(),
                        "Max_SVLEN": subdf["SVLEN_CALLER"].max(),
                    }
                )

            comparison.append(stats)

    return pd.DataFrame(comparison)


# =============================================================================
# Main processing function
# =============================================================================


def process_all_datasets(
    filepaths: List[str],
    target_column: str,
    save_subdatasets: bool = False,
    output_dir: Optional[str] = None,
    report: Optional[Report] = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Process multiple SV datasets and extract subdatasets for each.

    Parameters
    ----------
    filepaths : List[str]
        List of paths to dataset files
    target_column : str
        Name of the binary target column
    save_subdatasets : bool, optional
        Whether to save subdatasets to CSV files, by default False
    output_dir : str, optional
        Directory to save subdatasets, by default None
    report : Report, optional
        Report handler for logging

    Returns
    -------
    Dict[str, Dict[str, pd.DataFrame]]
        Nested dictionary: {dataset_name: {sv_type: dataframe}}
    """
    if report is None:
        report = Report()

    all_results = {}
    all_target_stats = {}

    report.section("EXTRACTING SV SUBDATASETS")

    for filepath in filepaths:
        # Get dataset name from filename
        dataset_name = Path(filepath).stem

        # Load dataset
        report.log(f"\nProcessing: {dataset_name}", level=0)
        report.log(f"  File: {filepath}", level=1)

        try:
            df = load_sv_dataset(filepath)
            report.log(f"  Total records: {len(df):,}", level=0)

            if report.verbosity >= 2:
                report.log(f"  Total columns: {len(df.columns)}", level=2)
                report.log(
                    f"  Detected SVTYPE_CALLER columns: {len([c for c in df.columns if c.startswith('SVTYPE_CALLER_')])}",
                    level=2,
                )

            # Analyze overall target distribution
            if report.verbosity >= 1:
                report.log("  Overall target distribution:", level=1)
                target_stats = analyze_target_distribution(df, target_column, report)
                all_target_stats[dataset_name] = {"overall": target_stats}

            # Extract subdatasets
            subdatasets = extract_sv_subdatasets(df)

            if not subdatasets:
                report.log(
                    "  WARNING: No SVTYPE_CALLER_* columns found in dataset", level=0
                )
                continue

            # Summary of extracted subdatasets
            report.log(f"  Subdatasets extracted: {len(subdatasets)}", level=0)
            for sv_type, subdf in subdatasets.items():
                percentage = (len(subdf) / len(df)) * 100
                report.log(
                    f"    {sv_type}: {len(subdf):,} records ({percentage:.1f}%)",
                    level=0,
                )

                # Detailed info for each subdataset
                if report.verbosity >= 2:
                    report.log(f"      Columns: {len(subdf.columns)}", level=2)

                    # Target distribution per SV type
                    if target_column in subdf.columns:
                        subtype_stats = analyze_target_distribution(
                            subdf, target_column, report, label=sv_type
                        )
                        if dataset_name not in all_target_stats:
                            all_target_stats[dataset_name] = {}
                        all_target_stats[dataset_name][sv_type] = subtype_stats

            all_results[dataset_name] = subdatasets

            # Save if requested
            if save_subdatasets and output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)

                report.log(f"  Saving subdatasets to: {output_dir}", level=1)
                for sv_type, subdf in subdatasets.items():
                    output_file = output_path / f"{dataset_name}_{sv_type}.csv"
                    subdf.to_csv(output_file, index=False)
                    report.log(f"    Saved: {output_file.name}", level=1)

        except FileNotFoundError:
            report.log(f"  ERROR: File not found: {filepath}", level=0)
            continue
        except pd.errors.EmptyDataError:
            report.log(f"  ERROR: File is empty: {filepath}", level=0)
            continue
        except pd.errors.ParserError as e:
            report.log(f"  ERROR: Failed to parse file {dataset_name}", level=0)
            report.log(f"  Details: {e}", level=0)
            report.log("  Hint: Check file format and delimiter", level=0)
            continue
        except KeyError as e:
            report.log(f"  ERROR: Missing expected column in {dataset_name}", level=0)
            report.log(f"  Details: {e}", level=0)
            continue
        except (ValueError, TypeError, AttributeError) as e:
            report.log(
                f"  ERROR processing {dataset_name}: {type(e).__name__}", level=0
            )
            report.log(f"  Details: {e}", level=0)
            if report.verbosity >= 2:
                report.log("  Traceback:", level=2)
                report.log(traceback.format_exc(), level=2)
            continue

    # Generate summary statistics
    if all_results and report.verbosity >= 1:
        report.section("SUMMARY STATISTICS", level=1)

        for dataset_name, subdatasets in all_results.items():
            report.subsection(f"Dataset: {dataset_name}", level=1)

            for sv_type, subdf in subdatasets.items():
                report.log(f"  {sv_type}:", level=1)
                compute_summary_statistics(subdf, report, indent="    ")

    # Cross-dataset comparison
    if len(all_results) > 1 and report.verbosity >= 2:
        report.section("CROSS-DATASET COMPARISON", level=2)

        # Get all unique SV types
        all_sv_types = set()
        for subdatasets in all_results.values():
            all_sv_types.update(subdatasets.keys())

        for sv_type in sorted(all_sv_types):
            report.subsection(f"SV Type: {sv_type}", level=2)
            comparison_df = compare_sv_types_across_datasets(all_results, sv_type)
            if not comparison_df.empty:
                for _, row in comparison_df.iterrows():
                    report.log(f"  {row['Dataset']}: {row['Count']:,} records", level=2)
                    if "Mean_SVLEN" in row:
                        report.log(f"    Mean SVLEN: {row['Mean_SVLEN']:,.0f}", level=2)

    report.section("PROCESSING COMPLETE")
    report.save()

    return all_results


# =============================================================================
# Command-line interface
# =============================================================================


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract subdatasets for each SVTYPE_CALLER_* from structural variant datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default verbosity
  python %(prog)s HG002.txt NA12878.txt REACH.txt

  # Save subdatasets to specific directory
  python %(prog)s --output-dir ./sv_subdatasets --save-subdatasets HG002.txt NA12878.txt

  # Detailed report with custom target column
  python %(prog)s --verbosity 2 --target-column label HG002.txt

  # Save report to file with minimal console output
  python %(prog)s --verbosity 0 --report-output report.txt HG002.txt NA12878.txt

  # Full detailed analysis with saved outputs
  python %(prog)s -v 2 -o ./output -r report.txt --save-subdatasets --target-column state *.txt

Verbosity Levels:
  0 - Minimal: Only essential information (dataset names, record counts)
  1 - Normal: Standard processing information + summary statistics (default)
  2 - Detailed: All information including target distributions per SV type and cross-dataset comparisons
        """,
    )

    parser.add_argument(
        "input_files",
        nargs="+",
        help="Input SV dataset files to process (supports wildcards)",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for subdatasets (default: current directory if --save-subdatasets is used)",
    )

    parser.add_argument(
        "-v",
        "--verbosity",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="Verbosity level: 0=minimal, 1=normal (default), 2=detailed",
    )

    parser.add_argument(
        "-t",
        "--target-column",
        type=str,
        default="state",
        help="Name of the binary target column for distribution analysis (default: state)",
    )

    parser.add_argument(
        "-r",
        "--report-output",
        type=str,
        default=None,
        help="Save full report to specified file",
    )

    parser.add_argument(
        "--save-subdatasets",
        action="store_true",
        help="Save extracted subdatasets as CSV files",
    )

    parser.add_argument("--version", action="version", version="%(prog)s 2.0")

    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_arguments()

    # Validate input files
    valid_files = []
    for filepath in args.input_files:
        path = Path(filepath)
        if not path.exists():
            print(f"Warning: File not found: {filepath}")
        else:
            valid_files.append(filepath)

    if not valid_files:
        print("Error: No valid input files found")
        sys.exit(1)

    # Set output directory
    output_dir = args.output_dir if args.save_subdatasets else None
    if args.save_subdatasets and output_dir is None:
        output_dir = "."

    # Create report handler
    report = Report(verbosity=args.verbosity, output_file=args.report_output)

    # Add header with metadata
    report.log("SV Subdataset Extraction Report", level=0)
    report.log(f"{'=' * 70}", level=0)
    report.log(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", level=0)
    report.log(f"Input files: {len(valid_files)}", level=0)
    report.log(f"Target column: {args.target_column}", level=1)
    report.log(f"Verbosity level: {args.verbosity}", level=1)
    if args.save_subdatasets:
        report.log(f"Output directory: {output_dir or '.'}", level=1)
    if args.report_output:
        report.log(f"Report output: {args.report_output}", level=1)

    # Process datasets
    try:
        results = process_all_datasets(
            filepaths=valid_files,
            target_column=args.target_column,
            save_subdatasets=args.save_subdatasets,
            output_dir=output_dir,
            report=report,
        )
        return results
    except Exception as e:
        print(f"\nFATAL ERROR: {type(e).__name__}")
        print(f"Details: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    results = main()

    # Example: Generate individual subdataset report (commented out by default)
    # Uncomment these lines to generate detailed reports for specific subdatasets
    # if results and 'HG002' in results and 'DEL' in results['HG002']:
    #     generate_subdataset_report(
    #         subdataset=results['HG002']['DEL'],
    #         dataset_name='HG002',
    #         sv_type='DEL',
    #         target_column='state',
    #         verbosity=2,
    #         output_file='HG002_DEL_detailed_report.txt'
    #     )
