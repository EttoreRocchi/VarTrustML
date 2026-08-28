<div align="center">

# VarTrustML

<img src="https://raw.githubusercontent.com/EttoreRocchi/VarTrustML/main/docs/logo.png" alt="VarTrustML Logo" width="280">

### *A Machine Learning Framework for Reliable Structural Variant Classification*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/EttoreRocchi/VarTrustML/blob/main/LICENSE)
[![Documentation](https://img.shields.io/readthedocs/vartrustml)](https://vartrustml.readthedocs.io)
[![CI](https://github.com/EttoreRocchi/VarTrustML/actions/workflows/ci.yaml/badge.svg)](https://github.com/EttoreRocchi/VarTrustML/actions/workflows/ci.yaml)

</div>

---

## Overview

Short-read SV callers disagree with each other and with the truth, so a call alone says little about whether a variant is real. **VarTrustML** trains classifiers on caller output plus alignment-derived features and evaluates them under nested cross-validation, with the confidence intervals, paired significance tests and cross-dataset checks a published comparison needs. Every run records its seeds, library versions and configuration.

### Key Capabilities

| Category | Features |
|----------|----------|
| **Model Suite** | Random Forest, XGBoost, CatBoost, MLP, Logistic Regression, KNN |
| **Optimization** | Bayesian hyperparameter tuning via Optuna; grid search support |
| **Calibration** | Isotonic/Platt scaling with Brier score, ECE, MCE, and reliability diagrams |
| **Statistical Tests** | Paired McNemar (pooled out-of-fold), DeLong test, Holm-Bonferroni (FWER, default) or Benjamini-Hochberg (FDR) correction, paired effect sizes |
| **Ablation Studies** | Feature ablation, feature group ablation, statistical significance testing |
| **Interpretability** | SHAP values, feature importance rankings, error pattern analysis |
| **Generalization** | Cross-dataset matrices, per-source generalization gap with bootstrap CIs, distribution-shift quantification, variant-caller baseline, leave-one-dataset-out |
| **Reproducibility** | Full seed management, library version logging, checkpoint system |
| **Reporting** | Interactive HTML reports, structured outputs |

## Installation

VarTrustML requires Python 3.10 or newer.

### From source

```bash
git clone https://github.com/EttoreRocchi/VarTrustML.git
cd VarTrustML
pip install .
```

Optional extras: `[dev]` for the test and lint toolchain, `[docs]` to build the documentation, and `[dev,docs]` for both.

```bash
pip install ".[dev,docs]"
```

### Developer Installation

For an editable install, from the cloned repository:

```bash
pip install -e ".[dev]"
```

### Requirements

- Python >= 3.10

After installation, the `vartrustml` command will be available in your terminal.

## Quick Start

### Using the CLI

```bash
# Compare ML models across one or more datasets
vartrustml compare-models HG002_DEL.csv HG002_INS.csv -t state

# Train a custom model
vartrustml train data/HG002_DEL.csv -t state --model "XGBoost"

# List available models
vartrustml list-models

# Make predictions with a saved model
vartrustml predict model.joblib data.csv --output predictions.csv

# Evaluate a saved model
vartrustml evaluate model.joblib test_data.csv -t state

# Cross-dataset generalization analysis
vartrustml cross-dataset HG002_DEL.csv HG002_DUP.csv HG002_INS.csv

# Run ablation study to measure feature importance
vartrustml ablation data.csv -t state -m XGBoost --continuous "SVLEN_CALLER,CG_CONTENT" --output-dir results/ablation

# Check version
vartrustml version

# Installation test
vartrustml smoke-test
```

### Using Python API

```python
from vartrustml import ExperimentConfig, CrossValidationPipeline, DataLoader
from vartrustml.config.experiment import CVConfig, CalibrationConfig

# Load your data
loader = DataLoader("data/")
df = loader.load_dataset("HG002_DEL.csv")

# Configure experiment
config = ExperimentConfig(
    cv=CVConfig(seed=42, n_outer_splits=10, n_inner_splits=5),
    output_dir="results",
    continuous_cols=[
        "SVLEN_CALLER",
        "CG_CONTENT",
        "COVERAGE_MOSDEPTH",
        "coverage_inside",
        "mean_insert_inside",
        "sd_insert_inside",
        "mean_mapq_inside",
        "coverage_left",
        "coverage_right",
        "mean_flank_insert",
        "mean_flank_mapq",
        "DELTA_insert",
        "DELTA_mapq",
        "clipped_ratio",
        "split_ratio",
        "discordant_ratio",
    ],
    target_column="state",
    confidence_thresholds=[0.6, 0.7, 0.8, 0.9, 0.95],
    calibration=CalibrationConfig(calibrate_models=True, calibration_cv=3),
)

# Run cross-validation
pipeline = CrossValidationPipeline(config)
results, _ = pipeline.run_cross_validation(df, "HG002")
```

## Documentation

Full documentation lives at **[vartrustml.readthedocs.io](https://vartrustml.readthedocs.io)**

### Documentation Structure

| Section | Description |
|---------|-------------|
| [Getting Started](https://vartrustml.readthedocs.io/en/latest/getting-started/installation.html) | Installation and quickstart guide |
| [User Guide](https://vartrustml.readthedocs.io/en/latest/user-guide/index.html) | Detailed usage instructions |
| [CLI Reference](https://vartrustml.readthedocs.io/en/latest/cli-reference/index.html) | Complete command-line interface documentation |
| [API Reference](https://vartrustml.readthedocs.io/en/latest/api/index.html) | Python API documentation |
| [Changelog](https://vartrustml.readthedocs.io/en/latest/changelog.html) | Version history and changes |

### User Guide Topics

- **[Model Comparison](https://vartrustml.readthedocs.io/en/latest/user-guide/compare-models.html)** - Model comparison with cross-validation
- **[Training](https://vartrustml.readthedocs.io/en/latest/user-guide/training.html)** - Standalone model training and evaluation
- **[Ablation Studies](https://vartrustml.readthedocs.io/en/latest/user-guide/ablation-studies.html)** - Feature importance via systematic ablation
- **[Calibration](https://vartrustml.readthedocs.io/en/latest/user-guide/calibration.html)** - Probability calibration methods
- **[Threshold Optimization](https://vartrustml.readthedocs.io/en/latest/user-guide/threshold-optimization.html)** - Youden's J threshold selection
- **[Hyperparameter Optimization](https://vartrustml.readthedocs.io/en/latest/user-guide/hyperparameter-optimization.html)** - Grid search and Optuna
- **[Statistical Tests](https://vartrustml.readthedocs.io/en/latest/user-guide/statistical-tests.html)** - Paired McNemar / DeLong on pooled out-of-fold predictions, Holm-Bonferroni (FWER) or Benjamini-Hochberg (FDR) correction
- **[HTML Reports](https://vartrustml.readthedocs.io/en/latest/user-guide/html-reports.html)** - Interactive report generation

### Package Structure

```
vartrustml/
├── cli/           # Command-line interface
├── config/        # Configuration classes
├── core/          # Core pipeline, models, and threshold optimization
├── analysis/      # Error analysis and metrics
├── visualization/ # Plots and HTML reports
├── io/            # Data loading and checkpoints
└── utils/         # Helper functions
```

## Usage Examples

### Model Comparison with Cross-Validation

**CLI:**

```bash
# Basic analysis on a single dataset
vartrustml compare-models HG002_DEL.csv -t state

# Custom configuration with specific datasets
vartrustml compare-models HG002_DEL.csv HG002_INS.csv -t state \
  --continuous "SVLEN_CALLER,coverage_inside" \
  --models "XGBoost,Random Forest" \
  --calibrate-model \
  --calibration-cv 5 \
  --n-outer-splits 15

# Missing-value (NaN) handling: 'median' (default) / 'mean' / 'most_frequent'
# impute inside the per-fold pipeline; 'drop' removes rows with NaN before CV
vartrustml compare-models HG002_DEL.csv -t state --nan-strategy median
```

**Python:**

```python
from vartrustml import ExperimentConfig, CrossValidationPipeline, DataLoader
from vartrustml.config.experiment import CVConfig, CalibrationConfig

# Initialize data loader
loader = DataLoader("data/")
df = loader.load_dataset("HG002_DEL.csv")

# Create feature report
feature_report = loader.create_feature_report(
    df,
    continuous_cols=[
        "SVLEN_CALLER",
        "CG_CONTENT",
        "COVERAGE_MOSDEPTH",
        "coverage_inside",
        "mean_insert_inside",
        "sd_insert_inside",
        "mean_mapq_inside",
        "coverage_left",
        "coverage_right",
        "mean_flank_insert",
        "mean_flank_mapq",
        "DELTA_insert",
        "DELTA_mapq",
        "clipped_ratio",
        "split_ratio",
        "discordant_ratio",
    ],
    target_col="state",
    output_path="results/HG002/feature_report.json",
)

# Configure experiment with custom settings
config = ExperimentConfig(
    cv=CVConfig(seed=42, n_outer_splits=10, n_inner_splits=5),
    output_dir="results/HG002",
    continuous_cols=[
        "SVLEN_CALLER",
        "CG_CONTENT",
        "COVERAGE_MOSDEPTH",
        "coverage_inside",
        "mean_insert_inside",
        "sd_insert_inside",
        "mean_mapq_inside",
        "coverage_left",
        "coverage_right",
        "mean_flank_insert",
        "mean_flank_mapq",
        "DELTA_insert",
        "DELTA_mapq",
        "clipped_ratio",
        "split_ratio",
        "discordant_ratio",
    ],
    target_column="state",
    nan_strategy="median",  # 'median'/'mean'/'most_frequent' (impute) or 'drop'
    confidence_thresholds=[0.6, 0.7, 0.8, 0.9, 0.95],
    models_to_use=["Random Forest", "XGBoost"],
    calibration=CalibrationConfig(calibrate_models=True, calibration_cv=3),
    save_checkpoints=True,
)

# Run pipeline
pipeline = CrossValidationPipeline(config)
results, _ = pipeline.run_cross_validation(df, "HG002")
```

For further details, see the [model comparison documentation](https://vartrustml.readthedocs.io/en/latest/user-guide/compare-models.html).

### Training Framework

**CLI:**

```bash
# List available models
vartrustml list-models

# Train a model with cross-validation
vartrustml train data/HG002_DEL.csv -t state --model "Random Forest" --cv-folds 10

# Train model with calibration and holdout test
vartrustml train data/HG002_DEL.csv -t state \
  --model "XGBoost" \
  --test-size 0.2 \
  --calibrate-model

# Train and evaluate on external test set
vartrustml train data/HG002_DEL.csv -t state \
  --model "XGBoost" \
  --test-data data/REACH_DEL.csv

# Make predictions
vartrustml predict results/trained_model/XGBoost_model.joblib data/new_data.csv --proba

# Evaluate model
vartrustml evaluate results/trained_model/XGBoost_model.joblib data/NA12878_DEL.csv -t state
```

**Python:**

```python
from vartrustml import TrainConfig, ModelTrainer, DataLoader

# Load data
loader = DataLoader("data/")
df = loader.load_dataset("HG002_DEL.csv")

X = df.drop(columns=["state"])
y = df["state"]

# Configure fitting
train_config = TrainConfig(
    model_name="Random Forest",
    continuous_cols=[
        "SVLEN_CALLER",
        "CG_CONTENT",
        "COVERAGE_MOSDEPTH",
        "coverage_inside",
        "mean_insert_inside",
        "sd_insert_inside",
        "mean_mapq_inside",
        "coverage_left",
        "coverage_right",
        "mean_flank_insert",
        "mean_flank_mapq",
        "DELTA_insert",
        "DELTA_mapq",
        "clipped_ratio",
        "split_ratio",
        "discordant_ratio",
    ],
    n_cv_folds=5,
    scoring="roc_auc",
    calibrate_model=True,
    output_dir="results/trained_model",
)

# Fit model
trainer = ModelTrainer(train_config)
results = trainer.fit(X, y)

print(f"Best CV score: {results['best_score']:.4f}")
print(f"Best parameters: {results['best_params']}")

# Make predictions
predictions = trainer.predict(X_new)
probabilities = trainer.predict_proba(X_new)
```

For further details, see the [training documentation](https://vartrustml.readthedocs.io/en/latest/user-guide/training.html).

### Ablation Studies

Ablation studies systematically measure the impact of removing features or feature groups on model performance, providing rigorous feature importance analysis.

**CLI:**

```bash
# Basic feature ablation (removes each feature one at a time)
vartrustml ablation data.csv -t state -m XGBoost --continuous "SVLEN_CALLER,CG_CONTENT"

# Ablate specific features only
vartrustml ablation data.csv -t state -m XGBoost --features "feat1,feat2,feat3"

# Feature group ablation with YAML config
vartrustml ablation data.csv -t state -m XGBoost --feature-groups groups.yaml

# Use different metric
vartrustml ablation data.csv -t state -m XGBoost --metric roc_auc --n-splits 10
```

**Python:**

```python
from vartrustml.analysis import AblationAnalyzer
from sklearn.metrics import balanced_accuracy_score

# Initialize analyzer
analyzer = AblationAnalyzer(n_splits=5, seed=42)

# Run feature ablation
results = analyzer.feature_ablation(
    X=df.drop(columns=["state"]),
    y=df["state"].values,
    model=trained_model,
    metric_func=balanced_accuracy_score,
    metric_name="Balanced Accuracy",
)

# View results
print(results.summary_df)

# Get statistically significant features
for r in results.get_significant_ablations():
    print(f"{r.ablation_name}: Δ={r.delta:.4f} (p={r.p_value:.4f})")

# Feature group ablation
feature_groups = {
    "caller_features": ["manta_call", "delly_call", "lumpy_call"],
    "size_features": ["sv_length", "sv_size_category"],
}
group_results = analyzer.feature_group_ablation(
    X=X,
    y=y,
    model=model,
    feature_groups=feature_groups,
    metric_func=balanced_accuracy_score,
    metric_name="Balanced Accuracy",
)
```

For further details, see the [ablation studies documentation](https://vartrustml.readthedocs.io/en/latest/user-guide/ablation-studies.html).

## Output Structure

VarTrustML produces a structured output hierarchy designed for both human inspection and programmatic access:

```
results/
├── <dataset_name>/
│   ├── experiment_config.json
│   ├── reproducibility_info.json           # Seeds and library versions
│   ├── feature_correlation_matrix.csv
│   ├── feature_target_correlation.csv
│   ├── threshold_optimization.joblib     # If threshold optimization enabled
│   ├── <Model_Name>/
│   │   ├── metrics_summary.csv
│   │   ├── <Model_Name>_fold_statistics.csv
│   │   ├── best_parameters.csv
│   │   ├── all_misclassified_samples.csv
│   │   ├── detailed_error_summary.joblib
│   │   ├── <Model_Name>_confusion_matrix.png
│   │   ├── <Model_Name>_confidence_distribution.png
│   │   ├── <Model_Name>_error_analysis.png
│   │   ├── <Model_Name>_fold_consistency.png
│   │   ├── <Model_Name>_reliability_diagram.png  # Calibration curve
│   │   ├── <Model_Name>_shap_summary.png   # If applicable
│   │   └── folds/
│   │       └── fold_<N>/
│   │           ├── metrics.csv
│   │           ├── confusion_matrix.csv
│   │           ├── misclassified.csv
│   │           ├── error_analysis.joblib
│   │           └── best_params.joblib
│   ├── model_comparison.csv
│   ├── model_comparison.png
│   ├── report.html                        # Interactive HTML report
│   └── summary_report.txt
├── checkpoints/ (if enabled)
│   └── <run_key>/                         # digest of settings + data
│       └── <dataset_name>/
│           └── <Model_Name>/
│               └── fold_<N>/
│                   ├── fold_<N>_model.joblib
│                   └── fold_<N>_results.joblib
└── model_fitting/
    ├── <Model_name>_model.joblib
    ├── <Model_name>_report.joblib
    └── <Model_name>_cv_results.csv
```

### Key Outputs

**Metrics Files:**
- Performance metrics: AUROC, balanced accuracy, MCC, F1, Brier score, ECE, MCE
- Statistics with 95% confidence intervals
- Per-fold and aggregated results

**Error Analysis:**
- Misclassifications at multiple confidence thresholds
- Error patterns by class and features
- Confidence and margin distributions

**Visualizations:**
- Confusion matrices
- Feature importance plots
- ROC curves
- SHAP summary plots
- Error analysis charts
- Reliability diagrams (calibration curves)

**Reports:**
- Human-readable summaries
- JSON exports for programmatic access
- CSV files for further analysis

## Supported Models

VarTrustML implements a diverse suite of classification algorithms, spanning classical statistical methods to modern deep learning architectures:

| Model | Description | Reference |
|-------|-------------|-----------|
| **Logistic Regression** | Linear classifier with L1/L2 regularization | Hastie et al. (2009) |
| **KNN** | K-Nearest Neighbors distance-based classifier | Cover & Hart (1967) |
| **Random Forest** | Bagged ensemble of decision trees | Breiman (2001) |
| **XGBoost** | Gradient boosting with regularized tree learners | Chen & Guestrin (2016) |
| **CatBoost** | Gradient boosting with ordered boosting for categorical features | Prokhorenkova et al. (2018) |
| **MLP** | Multi-layer perceptron with configurable architecture |  -  |

All models share a unified interface supporting:
- Automated hyperparameter optimization (grid search or Bayesian via Optuna)
- Cross-validated probability calibration
- Native or permutation-based feature importance
- SHAP value computation for post-hoc interpretability

## Configuration

### Using JSON Configuration Files

Create an `experiment_config.json`:

```json
{
  "cv": {
    "seed": 42,
    "n_outer_splits": 10,
    "n_inner_splits": 5
  },
  "calibration": {
    "calibrate_models": true,
    "calibration_method": "isotonic",
    "calibration_cv": 3
  },
  "visualization": {
    "plot_top_n_features": 20,
    "figure_dpi": 300,
    "error_analysis_features": ["SVTYPE_CALLER_DEL", "SVTYPE_CALLER_DUP"]
  },
  "output_dir": "results",
  "target_column": "state",
  "continuous_cols": [
    "SVLEN_CALLER", "CG_CONTENT", "COVERAGE_MOSDEPTH",
    "coverage_inside", "mean_insert_inside", "sd_insert_inside",
    "mean_mapq_inside", "coverage_left", "coverage_right",
    "mean_flank_insert", "mean_flank_mapq", "DELTA_insert",
    "DELTA_mapq", "clipped_ratio", "split_ratio", "discordant_ratio"
  ],
  "confidence_thresholds": [0.6, 0.7, 0.8, 0.9, 0.95],
  "models_to_use": ["XGBoost", "Random Forest", "MLP"],
  "n_jobs": -1,
  "verbose": 1,
  "save_checkpoints": true,
  "checkpoint_dir": "checkpoints"
}
```

Load and use:

```bash
vartrustml compare-models HG002_DEL.csv -t state --config experiment_config.json
```

Override specific parameters:

```bash
vartrustml compare-models HG002_DEL.csv -t state --config base_config.json --models "XGBoost" --seed 123
```

### Key Configuration Parameters

| Parameter | Meaning |
|-----------|---------|
| `cv.seed` | Random seed, recorded with every run |
| `cv.n_outer_splits` | Outer folds, used for model evaluation |
| `cv.n_inner_splits` | Inner folds, used for hyperparameter tuning |
| `continuous_cols` | Features passed through StandardScaler (16 by default) |
| `confidence_thresholds` | Cut-points at which error analysis is stratified |
| `models_to_use` | Which of the six registered models to train |
| `calibration.calibrate_models` | Turn probability calibration on |
| `calibration.calibration_cv` | Folds used to fit the calibration mapping (default 3) |
| `calibration.calibration_method` | `isotonic` or `sigmoid` |
| `save_checkpoints` | Write per-fold checkpoints so a run can resume |

## Advanced Features

### Checkpointing

Resume interrupted analyses:

```bash
vartrustml compare-models HG002_DEL.csv -t state --checkpoint-dir checkpoints
```

Checkpoints save:
- Trained models per fold
- Predictions and metrics
- Error analysis results
- SHAP values (if computed)

Each run writes under a `<run_key>` directory: a digest of the settings that
determine what a fold contains (seed, split counts, feature roles, NaN
strategy, model list, calibration, threshold and HPO settings) plus a
fingerprint of the data. Changing any of them starts a fresh set of
checkpoints instead of resuming folds that belong to a different experiment;
rerunning the same command resumes as expected.

### Model Calibration

Improve probability reliability with cross-validated calibration:

```bash
vartrustml compare-models HG002_DEL.csv -t state --calibrate-model --calibration isotonic --calibration-cv 5
```

Two methods are available: isotonic regression, which is non-parametric and preserves the model's ranking, and sigmoid (Platt) scaling, which is parametric and holds up better on small calibration sets. Isotonic is the default.

Calibration runs after training through `CalibratedClassifierCV`, which wraps the fitted model and uses its own internal cross-validation, with `calibration_cv` folds, to fit the mapping.

### Error Analysis Features

Focus error analysis on specific features:

```json
{
  "error_analysis_features": [
    "SVTYPE_CALLER_DEL",
    "SVTYPE_CALLER_DUP",
    "SVLEN_CALLER",
    "MAXQV"
  ]
}
```

Generates feature-specific error distribution plots.

---

## Citation

Citation information will be available soon.

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes before submitting a pull request. We particularly welcome:

- Bug reports and fixes
- New model implementations
- Documentation improvements
- Performance optimizations

## License

This project is licensed under the MIT License. See [LICENSE](https://github.com/EttoreRocchi/VarTrustML/blob/main/LICENSE) for details.

## Contact

**Ettore Rocchi**
Department of Medical and Surgical Sciences
University of Bologna

- Email: [ettore.rocchi3@unibo.it](mailto:ettore.rocchi3@unibo.it)
- GitHub: [@EttoreRocchi](https://github.com/EttoreRocchi)
