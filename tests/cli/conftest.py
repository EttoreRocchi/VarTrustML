"""
Shared fixtures for CLI tests.
"""

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner


@pytest.fixture
def cli_runner():
    """Create a CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def temp_data_files(tmp_path):
    """Create temporary dataset files for CLI testing."""
    np.random.seed(42)

    # Create Dataset1
    df1 = pd.DataFrame(
        {
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "feature3": np.random.randn(100),
            "state": np.array([0] * 50 + [1] * 50),
        }
    )
    path1 = tmp_path / "dataset1.csv"
    df1.to_csv(path1, index=False)

    # Create Dataset2
    df2 = pd.DataFrame(
        {
            "feature1": np.random.randn(80),
            "feature2": np.random.randn(80),
            "feature3": np.random.randn(80),
            "state": np.array([0] * 40 + [1] * 40),
        }
    )
    path2 = tmp_path / "dataset2.csv"
    df2.to_csv(path2, index=False)

    # Create Dataset3
    df3 = pd.DataFrame(
        {
            "feature1": np.random.randn(90),
            "feature2": np.random.randn(90),
            "feature3": np.random.randn(90),
            "state": np.array([0] * 45 + [1] * 45),
        }
    )
    path3 = tmp_path / "dataset3.csv"
    df3.to_csv(path3, index=False)

    return tmp_path, ["dataset1.csv", "dataset2.csv", "dataset3.csv"]


@pytest.fixture
def sample_dataset(tmp_path):
    """Create a single sample dataset for CLI testing."""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "feature3": np.random.randn(100),
            "state": np.array([0] * 50 + [1] * 50),
        }
    )
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    return tmp_path, "sample.csv"
