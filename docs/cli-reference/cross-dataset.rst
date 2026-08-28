=============
cross-dataset
=============

Cross-dataset generalizability analysis. Train models on one dataset and test on all other datasets to evaluate how well models generalize across different data sources.

Synopsis
--------

.. code-block:: bash

   vartrustml cross-dataset [OPTIONS] DATASETS...

Description
-----------

The ``cross-dataset`` command evaluates model generalizability by training models on each dataset and testing on all other datasets using aligned outer cross-validation splits.

This analysis helps answer the question: *"How well does a model trained on one dataset generalize to other datasets?"*

**Key outputs:**

- Performance matrices (train dataset vs test dataset)
- Generalization gap analysis
- Model stability reports
- Interactive HTML report

**Requirements:**

- At least 2 datasets are required
- All datasets must have compatible features (same columns)
- All datasets must have the same target column

Basic Usage
-----------

.. code-block:: bash

   # Basic cross-dataset analysis with 2 datasets
   vartrustml cross-dataset HG002_DEL.csv HG002_DUP.csv

   # With 3 or more datasets
   vartrustml cross-dataset HG002_DEL.csv HG002_DUP.csv HG002_INS.csv

   # Specify data directory and output directory
   vartrustml cross-dataset HG002_DEL.csv HG002_DUP.csv \
       --data-dir data/sv \
       --output-dir results/generalization

   # Select specific models
   vartrustml cross-dataset HG002_DEL.csv HG002_DUP.csv \
       --models "XGBoost,Random Forest,CatBoost"

Advanced Examples
-----------------

Full analysis with all options:

.. code-block:: bash

   vartrustml cross-dataset HG002_DEL.csv HG002_DUP.csv HG002_INS.csv \
       --data-dir data \
       --output-dir results/cross_dataset \
       --models "XGBoost,Random Forest,CatBoost" \
       --n-outer-splits 5 \
       --n-inner-splits 3 \
       --seed 42 \
       --calibrate-model \
       --calibration isotonic \
       --optimize-threshold \
       --bootstrap-iters 1000 \
       --n-jobs -1

With Optuna hyperparameter optimization:

.. code-block:: bash

   vartrustml cross-dataset HG002_DEL.csv HG002_DUP.csv \
       --hpo-method optuna \
       --optuna-trials 50 \
       --optuna-timeout 3600

Dry run to validate configuration:

.. code-block:: bash

   vartrustml cross-dataset HG002_DEL.csv HG002_DUP.csv --dry-run

Arguments
---------

DATASETS
   Two or more dataset file names relative to ``--data-dir`` (REQUIRED).

Options
-------

**Input/Output:**

``-d, --data-dir PATH``
   Directory containing datasets. Default: ``data``

``-o, --output-dir PATH``
   Output directory for results. Default: ``results/cross_dataset``

``--config PATH``
   Path to ExperimentConfig JSON file to load.

``--save-config PATH``
   Write the effective ExperimentConfig JSON to this path.

``--log-file PATH``
   Write logs to file.

**Cross-Validation:**

``--seed INTEGER``
   Random seed for reproducibility.

``--n-outer-splits INTEGER``
   Number of outer CV splits (default from config).

``--n-inner-splits INTEGER``
   Number of inner CV splits (default from config).

**Hyperparameter Optimization:**

``--hpo-method TEXT``
   Hyperparameter optimization method: ``grid`` or ``optuna``.

``--optuna-trials INTEGER``
   Number of Optuna trials (when using ``--hpo-method optuna``).

``--optuna-timeout INTEGER``
   Optuna timeout in seconds.

**Data Configuration:**

``--target-column TEXT``
   Target column name (required for datasets without default).

``--continuous TEXT``
   Continuous columns to scale (comma-separated list or path to .txt file).

``--categorical TEXT``
   Categorical columns (comma-separated list or path to .txt file). If provided without ``--continuous``, continuous columns are inferred by exclusion. Categorical columns are not scaled, but they do receive a most-frequent imputer whenever ``--nan-strategy`` is an impute strategy.

``--nan-strategy TEXT``
   Missing-value (NaN) handling: ``median`` (default), ``mean`` or ``most_frequent`` (impute in the per-fold pipeline) or ``drop`` (remove rows with NaN before CV).

**Model Selection:**

``--models TEXT``
   Model names to train (comma-separated). Example: ``"XGBoost,Random Forest,MLP"``

**Caller Baseline:**

``--callers TEXT``
   Caller column names (comma-separated) to evaluate as a baseline on each test sample, e.g. ``MANTA,SMOOVE,DELLY``. Shows whether an ML model trained on one sample still beats the callers on an unseen sample.

``--default-combinations`` / ``--no-default-combinations``
   Include the default AND/OR caller combinations in the baseline. On by
   default; pass ``--no-default-combinations`` to report only the individual
   callers.

**Evaluation Scheme:**

``--cv-scheme TEXT``
   ``pairwise`` (default; NxN train-vs-test matrix), ``lodo`` (leave-one-dataset-out: train on the N-1 pooled samples, test on the held-out sample), or ``both``.

**Calibration:**

``--calibrate-model``
   Enable probability calibration.

``--calibration TEXT``
   Calibration method: ``isotonic`` or ``sigmoid``.

``--calibration-cv INTEGER``
   Number of folds for calibration cross-validation.

**Threshold Optimization:**

``--optimize-threshold``
   Enable threshold optimization using Youden's J statistic.

``--threshold-method TEXT``
   Threshold optimization method: ``oof``, ``cv``, or ``auto``.

**Bootstrap:**

``--bootstrap-iters INTEGER``
   Number of bootstrap resamples for confidence intervals (default: 1000).

``--ci-level FLOAT``
   Confidence level for bootstrap CIs, e.g., 0.95 for 95% CI.

``--ci-method TEXT``
   Bootstrap CI method: ``bca`` (bias-corrected and accelerated, default) or ``percentile``.

**Visualization:**

``--figure-dpi INTEGER``
   Figure DPI for saved plots.

``--no-html-report``
   Suppress HTML report generation.

``--html-path TEXT``
   Path for HTML report (relative to output dir).

**System:**

``--n-jobs INTEGER``
   Parallel jobs (-1 for all cores).

``--verbose INTEGER``
   Verbosity level (1=tqdm only, 2=INFO, 3=DEBUG).

``--no-checkpoints``
   Disable checkpoint saving.

``--checkpoint-dir TEXT``
   Checkpoint subdirectory name.

``--dry-run``
   Show what would be executed without running.

Output Files
------------

The command produces the following output structure:

.. code-block:: text

   results/cross_dataset/
   ├── cross_dataset_summary.txt       # Human-readable summary
   ├── cross_dataset_results.json      # Full results in JSON format
   ├── cross_dataset_report.html       # Interactive HTML report
   ├── class_priors.csv                # Per-sample class prior (label shift)
   ├── distribution_shift.csv          # Per-feature shift between samples
   ├── distribution_shift_heatmap.png
   ├── caller_baseline_mcc.csv         # Caller baseline MCC per test sample (with --callers)
   ├── generalization_gap.csv          # Per-source in-sample minus cross-sample, with CI
   ├── lodo_results.csv                # Leave-one-dataset-out (with --cv-scheme lodo/both)
   └── <ModelName>/                     # Per-model results
       ├── <Metric>_heatmap.png        # Heatmap visualizations
       └── <Metric>_matrix.csv         # Raw performance matrices

**Performance matrices** show train dataset (rows) vs test dataset (columns) with mean performance across CV folds.

Generalizability analyses
-------------------------

Alongside the performance matrices, the report contextualises the cross-dataset gap with four analyses:

- **Distribution shift** between samples, measured per variable type: two-sample Kolmogorov-Smirnov for continuous features and the absolute difference in positive proportion for binary callers (both in [0, 1]), plus the class-prior (label) shift.
- **Variant-caller baseline**: the operating-point metric of each caller and default AND/OR combination on each test sample (with ``--callers``), to read the ML cross-sample cells against.
- **Generalization gap**: per training source, the in-sample minus cross-sample performance with a percentile bootstrap CI over the outer folds.
- **Leave-one-dataset-out** (with ``--cv-scheme lodo`` or ``both``): train on the N-1 pooled samples and test on the held-out one, compared with the average single-source cross-sample result.

Python API
----------

For programmatic access, use the ``CrossDatasetEvaluator`` class:

.. code-block:: python

   from vartrustml import ExperimentConfig, CrossDatasetEvaluator, DataLoader
   from vartrustml.config.experiment import CVConfig

   # Load datasets
   loader = DataLoader("data/")
   datasets = [
       (loader.load_dataset("HG002_DEL.csv"), "DEL"),
       (loader.load_dataset("HG002_DUP.csv"), "DUP"),
       (loader.load_dataset("HG002_INS.csv"), "INS"),
   ]

   # Configure experiment
   config = ExperimentConfig(
       cv=CVConfig(n_outer_splits=5),
       models_to_use=["XGBoost", "Random Forest"],
       output_dir="results/cross_dataset",
   )

   # Run evaluation
   evaluator = CrossDatasetEvaluator(config)
   results = evaluator.evaluate_cross_dataset(datasets)

See Also
--------

- :doc:`compare-models` -- Single-dataset model comparison
- :doc:`train` -- Train a single model
- :doc:`/user-guide/index` -- User guide
- :doc:`/api/core` -- Core API reference (CrossDatasetEvaluator)
