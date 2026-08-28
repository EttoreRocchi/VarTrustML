==============
compare-models
==============

Compare multiple ML models on a dataset using nested cross-validation.

Synopsis
--------

.. code-block:: bash

   vartrustml compare-models [DATASETS...] [OPTIONS]

Description
-----------

This command runs nested cross-validation experiments comparing multiple machine learning models. It supports:

- Multiple datasets in a single run
- Grid search or Optuna hyperparameter optimization
- Model calibration
- Variant caller comparison
- HTML report generation

Examples
--------

Basic Usage
^^^^^^^^^^^

.. code-block:: bash

   # Compare models with defaults
   vartrustml compare-models data/HG002_DEL.csv -t state

   # Multiple datasets
   vartrustml compare-models data/HG002_DEL.csv data/HG002_DUP.csv -t state

   # Specific models
   vartrustml compare-models data/HG002_DEL.csv -t state \
     --models "XGBoost,Random Forest,CatBoost"

With Optuna Optimization
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   vartrustml compare-models data/HG002_DEL.csv -t state \
     --hpo-method optuna \
     --optuna-trials 100

With Caller Comparison
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   vartrustml compare-models data/HG002_DEL.csv -t state \
     --compare-callers \
     --callers "MANTA,DELLY,SMOOVE"

Full Configuration
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   vartrustml compare-models data/HG002_DEL.csv -t state \
     --models "XGBoost,Random Forest,MLP" \
     --n-outer-splits 10 \
     --n-inner-splits 5 \
     --hpo-method optuna \
     --optuna-trials 50 \
     --calibrate-model \
     --optimize-threshold \
     --compare-callers \
     --callers "MANTA,DELLY" \
     --verbose 2

Options Reference
-----------------

Data Configuration Options
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``-t, --target-column TEXT``
     - Target column name (**required**)
   * - ``--continuous TEXT``
     - Continuous columns to scale (comma-separated list or path to .txt file)
   * - ``--categorical TEXT``
     - Categorical columns (comma-separated list or path to .txt file). If provided without ``--continuous``, continuous columns are inferred by exclusion. Categorical columns are not scaled, but they do receive a most-frequent imputer whenever ``--nan-strategy`` is an impute strategy.
   * - ``--nan-strategy TEXT``
     - Missing-value (NaN) handling: ``median`` (default), ``mean`` or ``most_frequent`` (impute in the per-fold pipeline) or ``drop`` (remove rows with NaN before CV)

Input/Output Options
^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``DATASETS...``
     - Dataset files relative to ``--data-dir``
   * - ``-d, --data-dir PATH``
     - Directory containing datasets (default: ``data``)
   * - ``-o, --output-dir PATH``
     - Output directory (default: ``results``)
   * - ``--config FILE``
     - Load ``ExperimentConfig`` from JSON
   * - ``--save-config FILE``
     - Write effective config to JSON
   * - ``--log-file PATH``
     - Write logs to file

Cross-Validation Options
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``--seed INT``
     - Random seed (default: 42)
   * - ``--n-outer-splits INT``
     - Outer CV folds (default: 10)
   * - ``--n-inner-splits INT``
     - Inner CV folds (default: 5)

Model Options
^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``--models TEXT``
     - Comma-separated list of models
   * - ``--calibrate-model``
     - Enable probability calibration
   * - ``--calibration TEXT``
     - Method: ``isotonic`` or ``sigmoid``
   * - ``--calibration-cv INT``
     - Calibration CV folds

HPO Options
^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``--hpo-method TEXT``
     - ``grid`` or ``optuna``
   * - ``--optuna-trials INT``
     - Number of Optuna trials
   * - ``--optuna-timeout INT``
     - Optuna timeout in seconds

Caller Comparison Options
^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``--compare-callers``
     - Compare ML models against variant callers
   * - ``--callers TEXT``
     - Comma-separated list of caller column names
   * - ``--combinations TEXT``
     - Custom logical combinations
   * - ``--default-combinations`` / ``--no-default-combinations``
     - Auto-generate AND/OR combinations (on by default; pass ``--no-default-combinations`` to evaluate only the individual callers)

Output Options
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``--no-html-report``
     - Suppress HTML report generation
   * - ``--html-path TEXT``
     - Custom HTML report path
   * - ``--optimize-threshold``
     - Enable threshold optimization
   * - ``--bootstrap-iters INT``
     - Bootstrap iterations (default: 1000)
   * - ``--ci-level FLOAT``
     - Bootstrap CI confidence level (default: 0.95)
   * - ``--ci-method TEXT``
     - Bootstrap CI method: ``bca`` (default) or ``percentile``
   * - ``--correction TEXT``
     - Multiple-comparison correction for the pairwise tests: ``holm`` (FWER, default) or ``bh`` (FDR)

System Options
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``--n-jobs INT``
     - Parallel jobs (-1 for all cores)
   * - ``--verbose INT``
     - Verbosity level (1-3)
   * - ``--no-checkpoints``
     - Disable checkpoint saving
   * - ``--dry-run``
     - Preview experiment without running

See Also
--------

- :doc:`train` -- Train a single model
- :doc:`../user-guide/compare-models` -- Detailed user guide
- :doc:`../api/core` -- Python API for CrossValidationPipeline
