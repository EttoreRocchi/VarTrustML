===========
Quick Start
===========

The basics for running a first experiment.

Prerequisites
-------------

Ensure VarTrustML is installed:

.. code-block:: bash

   pip install -r requirements.txt
   pip install -e .

Verify installation:

.. code-block:: bash

   vartrustml --help

Your First Experiment
---------------------

Step 1: Prepare Your Data
^^^^^^^^^^^^^^^^^^^^^^^^^

The labelled tables used in the project experiments are not distributed with the
repository. The examples below name them for consistency with the rest of the
documentation; any CSV with a binary target column works.

VarTrustML expects CSV or tab-separated files with:

- One row per sample
- One column as target (default: ``state``)
- Remaining columns as features

Example data structure:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - SVLEN_CALLER
     - MAXQV
     - CG_CONTENT
     - state
   * - 500
     - 30
     - 0.45
     - 1
   * - -300
     - 25
     - 0.52
     - 0

Step 2: Run Model Comparison
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tab-set::

   .. tab-item:: CLI (Simplest)

      .. code-block:: bash

         vartrustml compare-models data/HG002.csv -t state

      This uses default settings:

      - All available models
      - 10 outer folds, 5 inner folds
      - Grid search for hyperparameters
      - Model calibration disabled (enable with ``--calibrate-model``)

   .. tab-item:: With Optuna HPO

      .. code-block:: bash

         vartrustml compare-models data/HG002.csv -t state \
           --hpo-method optuna \
           --optuna-trials 100

   .. tab-item:: With Custom Models

      .. code-block:: bash

         vartrustml compare-models data/HG002.csv -t state \
           --models "XGBoost,Random Forest"

   .. tab-item:: Full Configuration

      .. code-block:: bash

         vartrustml compare-models data/HG002.csv -t state \
           --models "XGBoost,Random Forest,MLP" \
           --continuous "SVLEN_CALLER,MAXQV" \
           --hpo-method optuna \
           --optuna-trials 50 \
           --n-outer-splits 10 \
           --verbose 1

Step 3: View Results
^^^^^^^^^^^^^^^^^^^^

Results are saved to ``results/<dataset_name>/``:

.. code-block:: bash

   ls results/HG002/
   # Output:
   # - report.html (if enabled)
   # - experiment_config.json
   # - model_comparison.csv
   # - XGBoost/
   # - Random_Forest/
   # - ...

Open the HTML report:

.. code-block:: bash

   open results/HG002/report.html

Common Workflows
----------------

Quick Model Comparison
^^^^^^^^^^^^^^^^^^^^^^

Compare models with default settings:

.. code-block:: bash

   vartrustml compare-models data/HG002.csv -t state \
     --models "XGBoost,Random Forest,CatBoost"

Optimized Experiment
^^^^^^^^^^^^^^^^^^^^

Use Optuna for faster hyperparameter tuning:

.. code-block:: bash

   vartrustml compare-models data/HG002.csv -t state \
     --hpo-method optuna \
     --optuna-trials 100

Caller Comparison
^^^^^^^^^^^^^^^^^

Compare ML models against variant callers (MANTA, DELLY, etc.):

.. code-block:: bash

   vartrustml compare-models data/HG002_DEL.csv -t state \
     --compare-callers \
     --callers "MANTA,DELLY,SMOOVE"

This evaluates callers on the same CV folds as ML models for fair comparison, including logical combinations (AND/OR).

Dry-Run First
^^^^^^^^^^^^^

Preview the experiment:

.. code-block:: bash

   vartrustml compare-models data/HG002.csv -t state \
     --models "XGBoost,CatBoost,MLP" \
     --n-outer-splits 15 \
     --dry-run

Shows configuration without running training.

Python API Quick Start
----------------------

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   from vartrustml import ExperimentConfig, CrossValidationPipeline, DataLoader
   from vartrustml.config.experiment import CVConfig

   # Load data
   loader = DataLoader("data/")
   df = loader.load_dataset("HG002.csv")

   # Configure
   config = ExperimentConfig(
       cv=CVConfig(seed=42),
       models_to_use=["XGBoost", "Random Forest"]
   )

   # Run
   pipeline = CrossValidationPipeline(config)
   results, _ = pipeline.run_cross_validation(df, "HG002")

   # Access results
   for model_name, fold_results in results.items():
       print(f"{model_name} AUROC: {fold_results[0].metrics['AUROC']:.4f}")

With Optuna and HTML Report
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   config = ExperimentConfig(
       cv=CVConfig(seed=42, n_outer_splits=10, n_inner_splits=5),
       hpo_method="optuna",
       optuna_n_trials=100,
       generate_html_report=True,
       models_to_use=["XGBoost", "Random Forest", "CatBoost"]
   )

   pipeline = CrossValidationPipeline(config)
   results, _ = pipeline.run_cross_validation(df, "HG002")

Understanding Output
--------------------

Directory Structure
^^^^^^^^^^^^^^^^^^^

::

   results/HG002/
   ├── report.html                     # Interactive HTML report
   ├── experiment_config.json          # Configuration used
   ├── model_comparison.csv            # Summary of all models
   ├── XGBoost/
   │   ├── metrics_summary.csv        # Mean ± std for all metrics
   │   ├── best_parameters.csv        # Best hyperparameters per fold
   │   ├── confusion_matrix_aggregated.png
   │   ├── feature_importance.png
   │   └── folds/
   │       └── fold_0/
   │           ├── metrics.csv
   │           └── ...
   └── Random_Forest/
       └── ...

Key Files
^^^^^^^^^

``report.html``
    Interactive visualizations. Written by default; suppress with ``--no-html-report``.
``model_comparison.csv``
    One row per model, for a quick read of which won.
``metrics_summary.csv``
    Every metric with its bootstrap confidence interval.
``best_parameters.csv``
    The hyperparameters selected in each fold.
``pairwise_mcnemar_full.csv``
    The full matrix of paired McNemar and DeLong comparisons.
``auroc_delong.csv``
    DeLong AUROC comparisons between the ML models.

Next Steps
----------

Learn More
^^^^^^^^^^

- :doc:`../user-guide/hyperparameter-optimization` -- Master Optuna
- :doc:`../user-guide/html-reports` -- Explore interactive reports
- :doc:`../user-guide/compare-models` -- Detailed workflow

Try Advanced Features
^^^^^^^^^^^^^^^^^^^^^

**Model Calibration**: Improve probability estimates

.. code-block:: bash

   vartrustml compare-models data/HG002.csv -t state --calibrate-model

**Checkpointing**: Resume interrupted experiments (enabled by default; disable with ``--no-checkpoints``)

.. code-block:: bash

   vartrustml compare-models data/HG002.csv -t state --checkpoint-dir checkpoints

**Logging**: Save detailed logs

.. code-block:: bash

   vartrustml compare-models data/HG002.csv -t state --verbose 2 --log-file experiment.log

Troubleshooting
^^^^^^^^^^^^^^^

Command not found
"""""""""""""""""

Ensure VarTrustML is installed:

.. code-block:: bash

   pip install -e .

Out of memory
"""""""""""""

Reduce parallelization:

.. code-block:: bash

   vartrustml compare-models data/HG002.csv -t state --n-jobs 1

Slow training
"""""""""""""

Use Optuna with fewer trials:

.. code-block:: bash

   vartrustml compare-models data/HG002.csv -t state --hpo-method optuna --optuna-trials 30

Getting Help
^^^^^^^^^^^^

Bugs and feature requests go to
https://github.com/EttoreRocchi/VarTrustML/issues. For anything else, write to
ettore.rocchi3@unibo.it.
