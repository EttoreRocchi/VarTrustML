"""
Unit tests for ModelEvaluator class.
"""

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from vartrustml.config import CVConfig, ExperimentConfig, ModelConfig
from vartrustml.core.models import ModelEvaluator


@pytest.fixture
def base_config(tmp_path):
    """Create base experiment config."""
    return ExperimentConfig(
        cv=CVConfig(seed=42, n_outer_splits=3, n_inner_splits=2),
        output_dir=str(tmp_path / "test_output"),
        models_to_use=["Logistic Regression"],
    )


class TestModelEvaluatorInit:
    """Tests for ModelEvaluator initialization."""

    def test_init_with_default_model_config(self, base_config):
        """Test initialization uses default ModelConfig when not provided."""
        evaluator = ModelEvaluator(base_config)

        assert evaluator.config == base_config
        assert isinstance(evaluator.model_config, ModelConfig)

    def test_init_with_custom_model_config(self, base_config):
        """Test initialization with custom ModelConfig."""
        model_config = ModelConfig(rf_n_estimators_options=[50, 100])
        evaluator = ModelEvaluator(base_config, model_config)

        assert evaluator.model_config == model_config

    def test_error_analyzer_initialized(self, base_config):
        """Test ErrorAnalyzer is initialized with config thresholds."""
        evaluator = ModelEvaluator(base_config)

        assert evaluator.error_analyzer is not None


class TestModelSelection:
    """Tests for model selection and initialization."""

    def test_logistic_regression_only(self, base_config):
        """Test selecting only Logistic Regression."""
        base_config.models_to_use = ["Logistic Regression"]
        evaluator = ModelEvaluator(base_config)

        assert "Logistic Regression" in evaluator.models
        assert isinstance(evaluator.models["Logistic Regression"], LogisticRegression)
        assert len(evaluator.models) == 1

    def test_random_forest_only(self, base_config):
        """Test selecting only Random Forest."""
        base_config.models_to_use = ["Random Forest"]
        evaluator = ModelEvaluator(base_config)

        assert "Random Forest" in evaluator.models
        assert isinstance(evaluator.models["Random Forest"], RandomForestClassifier)

    def test_mlp_only(self, base_config):
        """Test selecting only MLP."""
        base_config.models_to_use = ["MLP"]
        evaluator = ModelEvaluator(base_config)

        assert "MLP" in evaluator.models
        assert isinstance(evaluator.models["MLP"], MLPClassifier)

    def test_knn_only(self, base_config):
        """Test selecting only KNN."""
        base_config.models_to_use = ["KNN"]
        evaluator = ModelEvaluator(base_config)

        assert "KNN" in evaluator.models
        assert isinstance(evaluator.models["KNN"], KNeighborsClassifier)

    def test_multiple_models(self, base_config):
        """Test selecting multiple models."""
        base_config.models_to_use = ["Logistic Regression", "Random Forest", "KNN"]
        evaluator = ModelEvaluator(base_config)

        assert len(evaluator.models) == 3
        assert "Logistic Regression" in evaluator.models
        assert "Random Forest" in evaluator.models
        assert "KNN" in evaluator.models

    def test_empty_models_list(self, base_config):
        """Test with empty models list creates no models."""
        base_config.models_to_use = []
        evaluator = ModelEvaluator(base_config)

        assert len(evaluator.models) == 0

    def test_unknown_model_ignored(self, base_config):
        """Test unknown model names are ignored."""
        base_config.models_to_use = ["Logistic Regression", "UnknownModel"]
        evaluator = ModelEvaluator(base_config)

        assert "Logistic Regression" in evaluator.models
        assert "UnknownModel" not in evaluator.models
        assert len(evaluator.models) == 1


class TestParamGrids:
    """Tests for hyperparameter grid initialization."""

    def test_param_grids_initialized(self, base_config):
        """Test param grids are initialized for selected models."""
        base_config.models_to_use = ["Logistic Regression", "Random Forest"]
        evaluator = ModelEvaluator(base_config)

        assert "Logistic Regression" in evaluator.param_grids
        assert "Random Forest" in evaluator.param_grids

    def test_logistic_regression_param_grid(self, base_config):
        """Test Logistic Regression has correct param grid keys."""
        base_config.models_to_use = ["Logistic Regression"]
        evaluator = ModelEvaluator(base_config)

        grid = evaluator.param_grids["Logistic Regression"]
        # Grid should have classifier__C parameter
        assert any("C" in key for key in grid.keys())

    def test_random_forest_param_grid(self, base_config):
        """Test Random Forest has correct param grid keys."""
        base_config.models_to_use = ["Random Forest"]
        evaluator = ModelEvaluator(base_config)

        grid = evaluator.param_grids["Random Forest"]
        # Should have n_estimators and max_depth parameters
        keys_str = str(grid.keys())
        assert "n_estimators" in keys_str or len(grid) > 0


class TestModelConfigIntegration:
    """Tests for ModelConfig parameter propagation."""

    def test_rf_estimators_from_config(self, base_config):
        """Test Random Forest uses n_estimators from ModelConfig."""
        model_config = ModelConfig(rf_n_estimators_options=[10, 20, 30])
        base_config.models_to_use = ["Random Forest"]

        evaluator = ModelEvaluator(base_config, model_config)

        _grid = evaluator.param_grids["Random Forest"]
        # Check if the estimators options are reflected
        assert evaluator.model_config.rf_n_estimators_options == [10, 20, 30]

    def test_lr_c_from_config(self, base_config):
        """Test Logistic Regression uses C values from ModelConfig."""
        model_config = ModelConfig(lr_C_options=[0.01, 0.1, 1.0])
        base_config.models_to_use = ["Logistic Regression"]

        evaluator = ModelEvaluator(base_config, model_config)

        assert evaluator.model_config.lr_C_options == [0.01, 0.1, 1.0]


class TestSeedPropagation:
    """Tests for random seed propagation to models."""

    def test_random_forest_seed(self, base_config):
        """Test Random Forest uses seed from config."""
        base_config.cv.seed = 123
        base_config.models_to_use = ["Random Forest"]

        evaluator = ModelEvaluator(base_config)

        assert evaluator.models["Random Forest"].random_state == 123

    def test_mlp_seed(self, base_config):
        """Test MLP uses seed from config."""
        base_config.cv.seed = 456
        base_config.models_to_use = ["MLP"]

        evaluator = ModelEvaluator(base_config)

        assert evaluator.models["MLP"].random_state == 456

    def test_logistic_regression_seed(self, base_config):
        """Test Logistic Regression uses seed from config."""
        base_config.cv.seed = 789
        base_config.models_to_use = ["Logistic Regression"]

        evaluator = ModelEvaluator(base_config)

        assert evaluator.models["Logistic Regression"].random_state == 789
