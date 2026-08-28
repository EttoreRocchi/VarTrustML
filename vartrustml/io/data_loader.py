"""
Data loading and preprocessing utilities.

Loads datasets from several file formats, checks that datasets are
compatible with each other, and generates feature reports.

Classes
-------
DataLoader
    Handle data loading and preprocessing for ML experiments.

See Also
--------
vartrustml.core.pipeline.CrossValidationPipeline : Uses DataLoader for input.
vartrustml.core.cross_dataset.CrossDatasetEvaluator : Cross-dataset evaluation.
"""

import logging
import os
from csv import Sniffer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """Handle data loading and preprocessing for ML experiments.

    Supports loading from CSV, TSV, TXT (auto-detected delimiter), and
    Excel files. Provides utilities for duplicate removal, column subsetting,
    and feature analysis.

    Parameters
    ----------
    data_directory : str
        Path to directory containing data files.

    Attributes
    ----------
    data_directory : pathlib.Path
        Path object for the data directory.

    Raises
    ------
    ValueError
        If data_directory does not exist.

    See Also
    --------
    CrossValidationPipeline : Uses DataLoader for data input.
    CrossDatasetEvaluator : Uses DataLoader for multi-dataset experiments.

    Examples
    --------
    >>> loader = DataLoader("data/")
    >>> df = loader.load_dataset("dataset.csv")
    >>> print(f"Loaded {len(df)} samples with {len(df.columns)} features")
    """

    def __init__(self, data_directory: str):
        self.data_directory = Path(data_directory).resolve()
        if not self.data_directory.exists():
            raise ValueError(f"Data directory {data_directory} does not exist")

    def _sanitize_path(self, filename: str) -> Path:
        """Sanitize and validate file path to prevent directory traversal.

        Parameters
        ----------
        filename : str
            Filename or relative path to validate.

        Returns
        -------
        pathlib.Path
            Validated absolute path within data_directory.

        Raises
        ------
        ValueError
            If the path attempts to escape data_directory (path traversal attack).
        """
        # Resolve the path to get absolute path
        filepath = Path(os.path.normpath(self.data_directory / filename))

        # Verify the resolved path is within data_directory
        try:
            filepath.relative_to(self.data_directory)
        except ValueError:
            raise ValueError(
                f"Path traversal detected: '{filename}' resolves outside data directory. "
                f"File must be within '{self.data_directory}'."
            )

        return filepath

    def load_dataset(
        self,
        filename: str,
        drop_duplicates: bool = True,
        subset_cols: Optional[List[str]] = None,
        id_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load a single dataset from file.

        Parameters
        ----------
        filename : str
            Name of the file to load (relative to data_directory).
        drop_duplicates : bool, default=True
            Whether to drop duplicate rows (keeps first occurrence).
        subset_cols : list of str, optional
            List of columns to keep. If None, keeps all columns.
        id_column : str, optional
            Column to use as the DataFrame index (row identifier).
            When set, the column is removed from the feature columns
            and preserved as the index. If None (default), uses the
            standard 0-based integer index.

        Returns
        -------
        pandas.DataFrame
            Loaded and optionally preprocessed dataframe.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        ValueError
            If the file type is not supported, path traversal detected,
            or id_column is not found in the dataset.
        """
        # Sanitize path to prevent directory traversal
        filepath = self._sanitize_path(filename)

        if not filepath.exists():
            raise FileNotFoundError(f"File {filepath} not found")

        if filepath.suffix == ".csv":
            df = pd.read_csv(filepath)
        elif filepath.suffix == ".txt":
            sniffer = Sniffer()
            with open(filepath) as f:
                sample = "".join(f.readline() for _ in range(5))
            if not sample.strip():
                raise ValueError(
                    f"File {filepath} is empty or contains only whitespace"
                )
            try:
                dialect = sniffer.sniff(sample, delimiters=",\t;")
            except Exception:
                logger.warning(
                    f"Could not detect delimiter for {filepath}, falling back to tab"
                )
                dialect = None
            sep = dialect.delimiter if dialect else "\t"
            df = pd.read_csv(filepath, sep=sep)
        elif filepath.suffix == ".tsv":
            df = pd.read_csv(filepath, sep="\t")
        elif filepath.suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(filepath)
        else:
            raise ValueError(f"Unsupported file type: {filepath.suffix}")

        logger.info(f"Loaded {filename}: shape {df.shape}")

        if id_column is not None:
            if id_column not in df.columns:
                raise ValueError(
                    f"id_column '{id_column}' not found in dataset. "
                    f"Available columns: {', '.join(df.columns[:10])}..."
                )
            df.set_index(id_column, inplace=True)
            logger.info(f"Set '{id_column}' as index")

        if drop_duplicates:
            original_shape = df.shape
            df.drop_duplicates(keep="first", inplace=True)
            if df.shape != original_shape:
                logger.info(f"Dropped duplicates: {original_shape} -> {df.shape}")

        if subset_cols:
            missing_cols = set(subset_cols) - set(df.columns)
            if missing_cols:
                logger.warning(f"Missing columns: {missing_cols}")

            available_cols = [col for col in subset_cols if col in df.columns]
            df = df[available_cols]
            logger.info(f"Subset to {len(available_cols)} columns")

        return df

    def load_multiple_datasets(
        self, dataset_configs: List[Dict[str, Any]]
    ) -> List[Tuple[pd.DataFrame, str]]:
        """Load multiple datasets with individual configurations.

        Parameters
        ----------
        dataset_configs : list of dict
            List of configuration dictionaries. Each dict must contain:
            - 'filename': str - Name of file to load
            - 'name': str - Display name for the dataset
            Optional keys:
            - 'drop_duplicates': bool - Whether to remove duplicates
            - 'subset_cols': list of str - Columns to keep

        Returns
        -------
        list of tuple
            List of (dataframe, name) tuples in the order specified.

        Raises
        ------
        Exception
            Propagates any exception from load_dataset with logging.
        """
        datasets = []

        for config in dataset_configs:
            filename = config["filename"]
            name = config["name"]
            drop_duplicates = config.get("drop_duplicates", True)
            subset_cols = config.get("subset_cols", None)
            id_column = config.get("id_column", None)

            try:
                df = self.load_dataset(
                    filename, drop_duplicates, subset_cols, id_column
                )
                datasets.append((df, name))
                logger.info(f"Successfully loaded {name}")
            except Exception as e:
                logger.error(f"Failed to load {name}: {e}")
                raise

        return datasets

    def create_feature_report(
        self,
        df: pd.DataFrame,
        continuous_cols: Optional[List[str]] = None,
        target_col: str = "state",
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a feature analysis report.

        Analyzes feature distributions, missing values, and correlations
        with the target variable.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataframe to analyze.
        continuous_cols : list of str, optional
            List of continuous feature columns. If None, auto-detects
            float64 columns.
        target_col : str, default="state"
            Name of the target column.
        output_path : str, optional
            Path to save report as JSON. If None, report is not saved.

        Returns
        -------
        dict
            Report dictionary containing:
            - 'shape': Tuple of (n_rows, n_cols)
            - 'columns': List of column names
            - 'dtypes': Dict mapping column names to dtypes
            - 'missing_values': Dict of missing value counts
            - 'feature_stats': Dict of per-feature statistics
            - 'target_distribution': Class counts (if target exists)
            - 'target_correlations': Feature-target correlations
        """
        report = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "feature_stats": {},
        }

        if continuous_cols is None:
            continuous_cols = df.select_dtypes(include=[np.float64]).columns.tolist()
            if target_col in continuous_cols:
                continuous_cols.remove(target_col)
        categorical_cols = set(df.columns) - set(continuous_cols) - {target_col}
        for col in continuous_cols:
            if col != target_col:
                report["feature_stats"][col] = {
                    "mean": df[col].mean(),
                    "std": df[col].std(),
                    "min": df[col].min(),
                    "max": df[col].max(),
                    "median": df[col].median(),
                    "q25": df[col].quantile(0.25),
                    "q75": df[col].quantile(0.75),
                }

        for col in categorical_cols:
            report["feature_stats"][col] = {
                "unique_values": df[col].nunique(),
                "top_value": df[col].mode()[0] if not df[col].mode().empty else None,
                "top_count": df[col].value_counts().iloc[0]
                if not df[col].value_counts().empty
                else 0,
            }

        if target_col in df.columns:
            report["target_distribution"] = df[target_col].value_counts().to_dict()
            report["target_balance"] = (
                df[target_col].value_counts(normalize=True).to_dict()
            )

        if target_col in df.columns and df[target_col].dtype in [np.int64, np.float64]:
            # Suppress warnings for zero-variance features
            with np.errstate(invalid="ignore", divide="ignore"):
                corr_series = df.drop(columns=[target_col]).corrwith(df[target_col])
            correlations = corr_series.dropna().to_dict()

            report["target_correlations"] = dict(
                sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
            )

        if output_path:
            import json
            import os

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Feature report saved to {output_path}")

        return report

    def validate_datasets_compatibility(
        self, datasets: List[Tuple[pd.DataFrame, str]], target_col: str = "state"
    ) -> Dict[str, Any]:
        """Validate that datasets are compatible for cross-dataset evaluation.

        Checks for matching target columns, consistent class labels, and
        sufficient feature overlap between datasets.

        Parameters
        ----------
        datasets : list of tuple
            List of (dataframe, name) tuples to validate.
        target_col : str, default="state"
            Name of the target column.

        Returns
        -------
        dict
            Validation report containing:
            - 'compatible': bool - Overall compatibility status
            - 'issues': list of str - Detected problems
            - 'feature_overlap': dict - Feature count per dataset
            - 'target_classes': dict - Class labels per dataset
            - 'common_features': list - Features present in all datasets
            - 'n_common_features': int - Count of common features
        """
        report: Dict[str, Any] = {
            "compatible": True,
            "issues": [],
            "feature_overlap": {},
            "target_classes": {},
        }

        for df, name in datasets:
            if target_col not in df.columns:
                report["compatible"] = False
                report["issues"].append(f"{name}: Missing target column '{target_col}'")
            else:
                classes = sorted(df[target_col].unique())
                report["target_classes"][name] = classes

        all_classes = list(report["target_classes"].values())
        if all_classes and not all(
            classes == all_classes[0] for classes in all_classes
        ):
            report["compatible"] = False
            report["issues"].append("Datasets have different target classes")

        all_features = []
        for df, name in datasets:
            features = [col for col in df.columns if col != target_col]
            all_features.append(set(features))
            report["feature_overlap"][name] = len(features)

        if all_features:
            common_features = set.intersection(*all_features)
            report["common_features"] = list(common_features)
            report["n_common_features"] = len(common_features)

            if len(common_features) < 5:
                report["compatible"] = False
                report["issues"].append(
                    f"Only {len(common_features)} common features found"
                )

        return report
