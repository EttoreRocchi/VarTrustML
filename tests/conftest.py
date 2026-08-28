"""
Shared pytest fixtures for VarTrustML tests.

Fixtures shared across several test files.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_binary_data():
    """Generate sample binary classification data."""
    np.random.seed(42)
    n_samples = 200
    X = np.random.randn(n_samples, 10)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


@pytest.fixture
def sample_predictions():
    """Generate sample predictions for testing."""
    np.random.seed(42)
    n_samples = 200
    y_true = np.random.randint(0, 2, n_samples)
    # Create predictions that are somewhat correlated with true values
    y_pred = y_true.copy()
    # Flip 20% of predictions to simulate errors
    flip_idx = np.random.choice(n_samples, size=int(n_samples * 0.2), replace=False)
    y_pred[flip_idx] = 1 - y_pred[flip_idx]
    y_prob = np.clip(y_pred + np.random.normal(0, 0.1, n_samples), 0.05, 0.95)
    return y_true, y_pred, y_prob


@pytest.fixture
def sample_fold_metrics():
    """Generate sample fold-level metrics for 10 folds."""
    return [0.85, 0.87, 0.83, 0.86, 0.84, 0.88, 0.82, 0.85, 0.87, 0.84]


@pytest.fixture
def sample_cv_results():
    """Generate sample CV results dictionary for multiple models."""
    np.random.seed(42)
    n_folds = 10
    return {
        "XGBoost": pd.DataFrame(
            {
                "Matthews Corr. Coef.": np.random.uniform(0.7, 0.9, n_folds),
                "AUROC": np.random.uniform(0.85, 0.95, n_folds),
                "Balanced Accuracy": np.random.uniform(0.80, 0.90, n_folds),
            }
        ),
        "Random Forest": pd.DataFrame(
            {
                "Matthews Corr. Coef.": np.random.uniform(0.65, 0.85, n_folds),
                "AUROC": np.random.uniform(0.80, 0.90, n_folds),
                "Balanced Accuracy": np.random.uniform(0.75, 0.85, n_folds),
            }
        ),
    }


@pytest.fixture
def sample_dataframe():
    """Generate a sample DataFrame for testing."""
    np.random.seed(42)
    n_samples = 100
    return pd.DataFrame(
        {
            "feature1": np.random.randn(n_samples),
            "feature2": np.random.randn(n_samples),
            "feature3": np.random.randn(n_samples),
            "state": np.random.randint(0, 2, n_samples),
        }
    )


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary directory with test data files."""
    # Create CSV file
    csv_data = pd.DataFrame(
        {
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature2": [0.5, 1.5, 2.5, 3.5, 4.5],
            "state": [0, 1, 0, 1, 0],
        }
    )
    csv_path = tmp_path / "test_data.csv"
    csv_data.to_csv(csv_path, index=False)

    # Create TSV file
    tsv_path = tmp_path / "test_data.tsv"
    csv_data.to_csv(tsv_path, sep="\t", index=False)

    # Create TXT file with comma delimiter
    txt_path = tmp_path / "test_data.txt"
    csv_data.to_csv(txt_path, index=False)

    # Create file with duplicates
    dup_data = pd.concat([csv_data, csv_data.iloc[[0, 1]]], ignore_index=True)
    dup_path = tmp_path / "test_duplicates.csv"
    dup_data.to_csv(dup_path, index=False)

    return tmp_path


@pytest.fixture
def cross_dataset_data(tmp_path):
    """Create datasets suitable for cross-dataset testing."""
    np.random.seed(42)

    # Dataset 1: Standard size
    df1 = pd.DataFrame(
        {
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "feature3": np.random.randn(100),
            "feature4": np.random.randn(100),
            "state": np.array([0] * 50 + [1] * 50),
        }
    )

    # Dataset 2: Slightly different size
    df2 = pd.DataFrame(
        {
            "feature1": np.random.randn(80),
            "feature2": np.random.randn(80),
            "feature3": np.random.randn(80),
            "feature4": np.random.randn(80),
            "state": np.array([0] * 40 + [1] * 40),
        }
    )

    # Dataset 3: Another size
    df3 = pd.DataFrame(
        {
            "feature1": np.random.randn(90),
            "feature2": np.random.randn(90),
            "feature3": np.random.randn(90),
            "feature4": np.random.randn(90),
            "state": np.array([0] * 45 + [1] * 45),
        }
    )

    # Save to CSV files
    df1.to_csv(tmp_path / "dataset1.csv", index=False)
    df2.to_csv(tmp_path / "dataset2.csv", index=False)
    df3.to_csv(tmp_path / "dataset3.csv", index=False)

    return {
        "data_dir": tmp_path,
        "datasets": [
            (df1, "dataset1"),
            (df2, "dataset2"),
            (df3, "dataset3"),
        ],
        "filenames": ["dataset1.csv", "dataset2.csv", "dataset3.csv"],
    }


@pytest.fixture
def sample_cross_dataset_results():
    """Create sample cross-dataset results for testing."""
    dataset_names = ["Dataset1", "Dataset2", "Dataset3"]

    results_mean = {
        "XGBoost": {
            "AUROC": pd.DataFrame(
                [[0.90, 0.85, 0.82], [0.84, 0.88, 0.80], [0.83, 0.81, 0.87]],
                index=dataset_names,
                columns=dataset_names,
            ),
            "F1 Score (Weighted)": pd.DataFrame(
                [[0.85, 0.80, 0.78], [0.79, 0.83, 0.76], [0.78, 0.77, 0.82]],
                index=dataset_names,
                columns=dataset_names,
            ),
        }
    }

    results_std = {
        "XGBoost": {
            "AUROC": pd.DataFrame(
                [[0.03, 0.04, 0.05], [0.04, 0.03, 0.05], [0.05, 0.05, 0.03]],
                index=dataset_names,
                columns=dataset_names,
            ),
            "F1 Score (Weighted)": pd.DataFrame(
                [[0.04, 0.05, 0.06], [0.05, 0.04, 0.06], [0.06, 0.06, 0.04]],
                index=dataset_names,
                columns=dataset_names,
            ),
        }
    }

    return results_mean, results_std, dataset_names
