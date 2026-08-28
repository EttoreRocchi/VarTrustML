======================
Threshold Optimization
======================

VarTrustML provides optional threshold optimization using **Youden's J statistic** to find the optimal classification threshold that balances sensitivity and specificity.

Theory
------

Youden's J Statistic
^^^^^^^^^^^^^^^^^^^^

Youden's J statistic (also called Youden's index) is defined as:

.. math::

   J = \text{Sensitivity} + \text{Specificity} - 1 = \text{TPR} - \text{FPR}

where sensitivity, the true positive rate, is TP / (TP + FN); specificity, the
true negative rate, is TN / (TN + FP); and FPR is 1 minus specificity.

The optimal threshold is the one that **maximizes J**, providing the best trade-off between sensitivity and specificity.

.. note::

   Reference: Youden, W.J. (1950). "Index for rating diagnostic tests." Cancer, 3(1), 32-35.

Methodology
-----------

VarTrustML implements threshold optimization **within the inner cross-validation loop**, ensuring scientific rigor.

How It Works
^^^^^^^^^^^^

For each outer fold in nested cross-validation:

1. **Hyperparameter optimization** finds the best model configuration
2. **Model training** fits the best model on the training fold
3. **Calibration** (optional) calibrates probabilities
4. **OOF predictions**: Get out-of-fold predictions for training samples
5. **Threshold optimization**: Find threshold maximizing Youden's J
6. **Test prediction**: Apply optimized threshold to test fold
7. **Metrics computation**: All metrics use the optimized threshold

Because the threshold is derived only from out-of-fold predictions on the
training portion, the test fold never influences it, and every metric reported
for that fold is measured at the threshold the fold actually chose. Comparing
those per-fold thresholds also shows how stable the operating point is across
resamples.

Usage
-----

CLI Usage
^^^^^^^^^

.. code-block:: bash

   # Enable threshold optimization
   vartrustml compare-models data.csv -t state --optimize-threshold

   # With model calibration (recommended)
   vartrustml compare-models data.csv -t state --optimize-threshold --calibrate-model

   # Train command with threshold optimization
   vartrustml train data.csv -t state --optimize-threshold --calibrate-model

Python API Usage
^^^^^^^^^^^^^^^^

.. code-block:: python

   from vartrustml import ExperimentConfig, CrossValidationPipeline
   from vartrustml.config.experiment import ThresholdConfig, CalibrationConfig

   # Configure threshold optimization
   config = ExperimentConfig(
       threshold=ThresholdConfig(optimize_threshold=True),
       calibration=CalibrationConfig(calibrate_models=True),  # recommended
   )

   # Run pipeline
   pipeline = CrossValidationPipeline(config)
   results = pipeline.run_cross_validation(df, "my_dataset")

   # Access per-fold threshold results
   for model_name, fold_results in results.items():
       for fold in fold_results:
           print(f"{model_name} Fold {fold.fold_id}:")
           print(f"  Threshold: {fold.fold_optimal_threshold:.4f}")
           print(f"  Youden's J: {fold.fold_youden_j:.4f}")

Using ThresholdOptimizer Directly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from vartrustml.core.threshold import ThresholdOptimizer
   import numpy as np

   # Find optimal threshold
   y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1])
   y_prob = np.array([0.2, 0.8, 0.6, 0.3, 0.9, 0.1, 0.4, 0.7])

   optimizer = ThresholdOptimizer()
   result = optimizer.optimize_from_oof(y_true, y_prob)

   print(f"Optimal threshold: {result.optimal_threshold:.4f}")
   print(f"Youden's J: {result.youden_j:.4f}")
   print(f"Sensitivity: {result.sensitivity:.4f}")
   print(f"Specificity: {result.specificity:.4f}")

Configuration Parameters
------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 15 15 40

   * - Parameter
     - Type
     - Default
     - Description
   * - ``optimize_threshold``
     - bool
     - ``False``
     - Enable threshold optimization
   * - ``threshold_method``
     - str
     - ``"auto"``
     - Method: ``"oof"``, ``"cv"``, or ``"auto"``
   * - ``threshold_auto_n_samples``
     - int
     - ``1000``
     - Sample threshold for auto method

Threshold Methods
^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Method
     - Description
     - Recommended For
   * - ``"oof"``
     - Uses concatenated out-of-fold predictions
     - Datasets with < 1000 samples
   * - ``"cv"``
     - Finds threshold per fold, then averages; the reported sensitivity,
       specificity and Youden's J are then measured on the pooled fold
       predictions **at that averaged threshold**
     - Datasets with >= 1000 samples
   * - ``"auto"``
     - Automatically selects based on sample size
     - General use (default)

FoldMetrics Threshold Fields
----------------------------

When threshold optimization is enabled, each ``FoldMetrics`` object contains:

- ``fold_optimal_threshold``: The optimized threshold for this fold
- ``fold_youden_j``: Youden's J statistic at optimal threshold
- ``fold_sensitivity_at_threshold``: True positive rate
- ``fold_specificity_at_threshold``: True negative rate

Output Files
------------

Threshold optimization results are saved per-model:

.. code-block:: text

   results/dataset_name/
   ├── CatBoost/
   │   └── threshold.joblib
   ├── XGBoost/
   │   └── threshold.joblib
   ├── Random_Forest/
   │   └── threshold.joblib
   └── ...

Each ``threshold.joblib`` file contains a dictionary with:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Field
     - Description
   * - ``fold_thresholds``
     - List of optimized thresholds for each CV fold
   * - ``mean_threshold``
     - Average threshold across folds
   * - ``std_threshold``
     - Standard deviation of thresholds
   * - ``recommended_threshold``
     - Recommended threshold (same as mean_threshold)
   * - ``mean_youden_j``
     - Youden's J at the recommended threshold
   * - ``mean_sensitivity_at_threshold``
     - Sensitivity (TPR) at the recommended threshold
   * - ``mean_specificity_at_threshold``
     - Specificity (TNR) at the recommended threshold
   * - ``n_folds``
     - Number of CV folds

Loading threshold data:

.. code-block:: python

   import joblib

   threshold_data = joblib.load("results/HG002/CatBoost/threshold.joblib")
   print(f"Recommended threshold: {threshold_data['recommended_threshold']:.4f}")
   print(f"Youden's J: {threshold_data['mean_youden_j']:.4f}")

See Also
--------

- :doc:`calibration` -- Probability calibration
- :doc:`compare-models` -- Model comparison workflow
