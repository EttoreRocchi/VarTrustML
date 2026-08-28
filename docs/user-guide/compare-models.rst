================
Model Comparison
================

Overview
--------

The ``compare-models`` command puts a dataset through the full evaluation:

1. **Data Loading and Validation**
2. **Feature Summary and Statistics**
3. **Machine Learning Configuration**
4. **Cross-Validation and Model Evaluation**
5. **Feature Importance and Error Analysis**
6. **Visualization and Reporting**

It fits several models, scores them with confidence-aware metrics, and writes the results as CSV and HTML.

Directory Structure
-------------------

::

   VarTrustML/
   ├── data/
   │   ├── HG002.csv
   │   ├── NA12878.csv
   │   └── ... (SV type-specific subsets)
   ├── vartrustml/
   │   └── ... (package modules)
   ├── results/
   │   └── <dataset_name>/
   │       ├── feature_report.json
   │       ├── experiment_config.json
   │       ├── model_metrics_comparison.csv
   │       ├── <Model_Name>/
   │       │   ├── metrics_summary.csv
   │       │   ├── threshold.joblib (if --optimize-threshold)
   │       │   ├── error_analysis_summary.csv
   │       │   ├── best_parameters.csv
   │       │   └── folds/
   │       │       └── fold_<N>/
   │       ├── checkpoints/ (if enabled)
   │       └── report.html

Usage
-----

Prerequisites
^^^^^^^^^^^^^

* Python 3.9+
* VarTrustML installed (``pip install -e .``)
* Dataset files in ``data/`` directory

Run the Analysis
^^^^^^^^^^^^^^^^

.. tab-set::

   .. tab-item:: CLI (Recommended)

      .. code-block:: bash

         # Analyze one dataset
         vartrustml compare-models HG002.csv -t state

         # Analyze several datasets in one run
         vartrustml compare-models HG002.csv REACH.csv -t state

         # With Optuna optimization and HTML report
         vartrustml compare-models HG002.csv -t state \
           --hpo-method optuna \
           --optuna-trials 100

         # Full custom configuration
         vartrustml compare-models HG002.csv -t state \
           --continuous "SVLEN_CALLER,MAXQV" \
           --models "XGBoost,Random Forest" \
           --n-outer-splits 10 \
           --calibrate-model \
           --hpo-method optuna \
           --optuna-trials 50

   .. tab-item:: Python API

      .. code-block:: python

         from vartrustml import ExperimentConfig, CrossValidationPipeline, DataLoader
         from vartrustml.config.experiment import CVConfig, CalibrationConfig

         # Load data
         loader = DataLoader("data/")
         df = loader.load_dataset("HG002.csv")

         # Create feature report
         feature_report = loader.create_feature_report(
             df,
             continuous_cols=["SVLEN_CALLER", "MAXQV", "CG_CONTENT"],
             target_col="state",
             output_path="results/HG002/feature_report.json"
         )

         # Configure experiment with Optuna and HTML report
         config = ExperimentConfig(
             cv=CVConfig(seed=42, n_outer_splits=10, n_inner_splits=5),
             output_dir="results/HG002",
             models_to_use=["Random Forest", "XGBoost", "MLP"],
             calibration=CalibrationConfig(calibrate_models=True),
             save_checkpoints=True,
             hpo_method="optuna",
             optuna_n_trials=100,
             generate_html_report=True
         )

         # Run cross-validation pipeline
         pipeline = CrossValidationPipeline(config)
         results, caller_results = pipeline.run_cross_validation(df, "HG002")

Configuration
-------------

The analysis uses a configurable ``ExperimentConfig`` object with the following key settings:

Data Settings
^^^^^^^^^^^^^

* **Target column**: ``state`` (binary classification)
* **Continuous features**: Scaled using StandardScaler (20 features by default)

Cross-Validation Settings
^^^^^^^^^^^^^^^^^^^^^^^^^

* **Outer folds**: 10 (default) - For model evaluation
* **Inner folds**: 5 (default) - For hyperparameter tuning
* **Stratified sampling**: Maintains class balance across folds

Model Settings
^^^^^^^^^^^^^^

* **Models**: XGBoost, Random Forest, CatBoost, MLP, Logistic Regression, KNN
* **HPO Method**: Grid Search or Optuna (Bayesian)
* **Calibration**: Isotonic or Sigmoid (optional)

CLI Options Reference
---------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``-d, --data-dir PATH``
     - Directory containing datasets (default: ``data``)
   * - ``-o, --output-dir PATH``
     - Output directory (default: ``results``)
   * - ``--config FILE``
     - Load ``ExperimentConfig`` from JSON
   * - ``--seed INT``
     - Random seed
   * - ``--n-outer-splits INT``
     - Number of outer CV folds
   * - ``--n-inner-splits INT``
     - Number of inner CV folds
   * - ``--models TEXT``
     - Comma-separated list of models
   * - ``--hpo-method TEXT``
     - ``grid`` or ``optuna``
   * - ``--optuna-trials INT``
     - Number of Optuna trials
   * - ``--calibrate-model``
     - Enable probability calibration
   * - ``--compare-callers``
     - Compare ML models against variant callers
   * - ``--no-html-report``
     - Suppress HTML report generation

Output Files
------------

Dataset-Level Files
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - File
     - Description
   * - ``report.html``
     - Interactive visualizations with Plotly
   * - ``model_metrics_comparison.csv``
     - All models' metrics with mean, mean_bootstrap, std, and confidence intervals
   * - ``experiment_config.json``
     - Experiment configuration for reproducibility
   * - ``feature_report.json``
     - Feature statistics and summary
   * - ``pairwise_mcnemar_full.csv``
     - Full matrix of paired McNemar / DeLong comparisons (if callers enabled)
   * - ``auroc_delong.csv``
     - DeLong AUROC comparisons between ML models

Per-Model Files (in ``<Model_Name>/``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - File
     - Description
   * - ``metrics_summary.csv``
     - Detailed metrics with mean, mean_bootstrap, std, and CIs
   * - ``threshold.joblib``
     - Threshold optimization results (if ``--optimize-threshold``)
   * - ``error_analysis_summary.csv``
     - Error analysis by confidence level
   * - ``best_parameters.csv``
     - Optimal hyperparameters for each fold
   * - ``all_misclassified_samples.csv``
     - Misclassified samples from all folds

See Also
--------

- :doc:`training` -- Train individual models
- :doc:`hyperparameter-optimization` -- Master Optuna for efficient HPO
- :doc:`html-reports` -- Interactive HTML report details
- :doc:`threshold-optimization` -- Threshold optimization details
