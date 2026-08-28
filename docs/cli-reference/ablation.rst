======================
ablation
======================

Run ablation study to measure feature importance via leave-one-out analysis.

Synopsis
--------

.. code-block:: bash

   vartrustml ablation DATASET [OPTIONS]

Description
-----------

Ablation studies systematically remove features (or feature groups) and measure the impact on model performance. This helps identify which features are most important for the model's predictions.

The command:

1. Loads the dataset
2. Trains a fresh baseline model with all features
3. Removes each feature/group one at a time
4. Re-trains and computes performance delta
5. Tests statistical significance (paired t-test)
6. Outputs results to CSV and text report

Arguments
---------

``DATASET``
   Dataset file path (relative to ``--data-dir`` or absolute). Required.

Options
-------

Data Configuration
^^^^^^^^^^^^^^^^^^

``-t``, ``--target-column`` TEXT
   Target column name. **Required.**

``--continuous`` TEXT
   Continuous columns to scale (comma-separated or path to .txt file).

Model
^^^^^

``-m``, ``--model`` TEXT
   Model name to use (XGBoost, CatBoost, Random Forest, Logistic Regression, MLP, KNN). Fresh models are trained for each ablation. **Required.**

Model Configuration
^^^^^^^^^^^^^^^^^^^

``--calibrate``
   Apply probability calibration during ablation.

``--calibration-method`` TEXT
   Calibration method: ``isotonic`` (default) or ``sigmoid``.

``--optimize-threshold``
   Apply threshold optimization during ablation.

Input/Output
^^^^^^^^^^^^

``-d``, ``--data-dir`` PATH
   Directory containing datasets. Default: current directory.

``-o``, ``--output-dir`` PATH
   Output directory for ablation results. Default: ``results/ablation``.

Analysis
^^^^^^^^

``--metric`` TEXT
   Metric to use for ablation analysis. Options:

   - ``balanced_accuracy`` (default)
   - ``f1``
   - ``mcc``
   - ``roc_auc``

``--feature-groups`` PATH
   Path to YAML/JSON file defining feature groups for group ablation.

``--features`` TEXT
   Comma-separated list of specific features to ablate. If not provided, all features are used.

``--ablate-steps``
   Run pipeline step ablation (calibration, threshold, scaling) instead of feature ablation.

Cross-Validation
^^^^^^^^^^^^^^^^

``--n-splits`` INTEGER
   Number of CV folds for ablation analysis. Default: 3.

``--seed`` INTEGER
   Random seed for reproducibility. Default: 42.

System
^^^^^^

``--n-jobs`` INTEGER
   Parallel jobs (-1 for all cores). Default: -1.

``--dry-run``
   Show what would be executed without running.

``-v``, ``--verbose`` INTEGER
   Verbosity level (0=WARNING, 1=INFO, 2=DEBUG). Default: 1.

Examples
--------

Basic Feature Ablation
^^^^^^^^^^^^^^^^^^^^^^

Remove each feature one at a time:

.. code-block:: bash

   vartrustml ablation data/HG002_DEL.csv \
       --target-column state \
       --model XGBoost \
       --continuous feature1,feature2,feature3

Specific Features
^^^^^^^^^^^^^^^^^

Test only certain features:

.. code-block:: bash

   vartrustml ablation data/HG002_DEL.csv \
       --target-column state \
       --model XGBoost \
       --continuous feature1,feature2,feature3 \
       --features "feature1,feature2"

Feature Group Ablation
^^^^^^^^^^^^^^^^^^^^^^

Create ``feature_groups.yaml``:

.. code-block:: yaml

   caller_features:
     - manta_call
     - delly_call
     - lumpy_call

   quality_features:
     - mapping_quality
     - base_quality

Run:

.. code-block:: bash

   vartrustml ablation data/HG002_DEL.csv \
       --target-column state \
       --model XGBoost \
       --continuous feature1,feature2 \
       --feature-groups feature_groups.yaml

With Calibration and Threshold Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Apply calibration and threshold optimization:

.. code-block:: bash

   vartrustml ablation data/HG002_DEL.csv \
       --target-column state \
       --model CatBoost \
       --continuous feature1,feature2 \
       --calibrate \
       --optimize-threshold

Pipeline Step Ablation
^^^^^^^^^^^^^^^^^^^^^^

Measure impact of calibration, threshold optimization, and scaling:

.. code-block:: bash

   vartrustml ablation data/HG002_DEL.csv \
       --target-column state \
       --model XGBoost \
       --continuous feature1,feature2 \
       --ablate-steps

Different Metric
^^^^^^^^^^^^^^^^

Use ROC AUC instead of balanced accuracy:

.. code-block:: bash

   vartrustml ablation data/HG002_DEL.csv \
       --target-column state \
       --model XGBoost \
       --continuous feature1,feature2 \
       --metric roc_auc

Dry Run
^^^^^^^

Preview configuration without running:

.. code-block:: bash

   vartrustml ablation data/HG002_DEL.csv \
       --target-column state \
       --model XGBoost \
       --dry-run

Output
------

The command produces:

``ablation_results.csv``
   Summary table with columns:

   - ``ablation_name``: Feature or group name
   - ``baseline_score``: Score with all features
   - ``ablated_score``: Score after removal
   - ``delta``: Score change (negative = feature helps)
   - ``delta_pct``: Percentage change
   - ``p_value``: Statistical significance
   - ``is_significant``: True if p < 0.05
   - ``effect_size``: Cohen's d

``ablation_report.txt``
   Human-readable text report with configuration and findings.

Console Output
^^^^^^^^^^^^^^

The command displays:

1. Configuration summary table
2. Progress during ablation
3. Results table with all ablations
4. Significant findings highlighted

Exit Codes
----------

.. list-table::
   :header-rows: 1
   :widths: 10 90

   * - Code
     - Meaning
   * - 0
     - Success
   * - 1
     - Runtime error (training failed, etc.)
   * - 2
     - Validation error (invalid config, missing files)

See Also
--------

- :doc:`train` - Train models
- :doc:`compare-models` - Model comparison with cross-validation
