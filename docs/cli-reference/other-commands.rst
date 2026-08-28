==============
Other Commands
==============

Additional utility commands available in VarTrustML.

predict
-------

Generate predictions from a saved model.

Synopsis
^^^^^^^^

.. code-block:: bash

   vartrustml predict MODEL_PATH DATA [OPTIONS]

Description
^^^^^^^^^^^

Load a trained model and generate predictions on new data. If the model was trained with threshold optimization, the optimized threshold is used by default. Use ``--default-threshold`` to override this and use the standard 0.5 threshold.

Examples
^^^^^^^^

.. code-block:: bash

   # Basic prediction (uses optimized threshold if available)
   vartrustml predict models/xgboost.joblib data/new_samples.csv

   # Force default 0.5 threshold
   vartrustml predict models/xgboost.joblib data/new_samples.csv \
     --default-threshold

   # With probability output (shows distance from threshold)
   vartrustml predict models/xgboost.joblib data/new_samples.csv \
     --proba \
     --output predictions.csv

Options
^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``MODEL_PATH``
     - Path to saved model file
   * - ``DATA``
     - Path to input data
   * - ``-o, --output PATH``
     - Output file path
   * - ``--proba``
     - Output probabilities (also shows distance from optimized threshold)
   * - ``--default-threshold``
     - Use default 0.5 threshold instead of model's optimized threshold

evaluate
--------

Evaluate a saved model on a labeled dataset.

Synopsis
^^^^^^^^

.. code-block:: bash

   vartrustml evaluate MODEL_PATH DATA [OPTIONS]

Description
^^^^^^^^^^^

Load a trained model and evaluate performance on labeled test data. If the model was trained with threshold optimization, metrics are computed at both the default threshold (0.5) and the optimized threshold, allowing comparison of the threshold's impact on performance.

Examples
^^^^^^^^

.. code-block:: bash

   # Basic evaluation
   vartrustml evaluate models/xgboost.joblib data/test_set.csv -t state

   # With custom target column
   vartrustml evaluate models/xgboost.joblib data/test_set.csv \
     --target label \
     --output evaluation_results.csv

   # Evaluate model with optimized threshold (shows both thresholds)
   vartrustml evaluate models/optimized_model.joblib data/test_set.csv \
     --target state

Options
^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``MODEL_PATH``
     - Path to saved model file
   * - ``DATA``
     - Path to labeled test data
   * - ``-t, --target TEXT``
     - Target column name (**required**)
   * - ``-o, --output PATH``
     - Output file for text results (default: ``evaluation.txt``)
   * - ``--no-html-report``
     - Disable HTML report generation (enabled by default)
   * - ``--html-path TEXT``
     - Custom HTML report path (default: ``evaluate_report.html``)

list-models
-----------

List available models.

Synopsis
^^^^^^^^

.. code-block:: bash

   vartrustml list-models

Description
^^^^^^^^^^^

Display all registered model types available for training.

Example Output
^^^^^^^^^^^^^^

.. code-block:: text

   Available models:
     1. MLP
     2. Random Forest
     3. XGBoost
     4. CatBoost
     5. Logistic Regression
     6. KNN

   Use the exact model name with --model (quotes if it contains spaces).

smoke-test
----------

Quick import test to validate installation.

Synopsis
^^^^^^^^

.. code-block:: bash

   vartrustml smoke-test

Description
^^^^^^^^^^^

Confirm that the package and its CLI entry point import cleanly. The command
succeeds only if every module imported by the CLI, and their third-party
dependencies, resolve. It does not exercise training or plotting.

Example Output
^^^^^^^^^^^^^^

.. code-block:: text

   Import successful!

version
-------

Display the VarTrustML version.

Synopsis
^^^^^^^^

.. code-block:: bash

   vartrustml version

Example Output
^^^^^^^^^^^^^^

.. code-block:: text

   VarTrustML version 0.1.0
