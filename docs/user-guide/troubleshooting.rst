===============
Troubleshooting
===============

This guide covers common issues and their solutions when using VarTrustML.

Common Issues
-------------

Memory Issues with Large Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem**: Out of memory errors during training.

**Solutions**:

1. **Reduce parallelization**:

   .. code-block:: python

      config = ExperimentConfig(n_jobs=2)  # Instead of -1

2. **Use fewer models**:

   .. code-block:: python

      config = ExperimentConfig(
          models_to_use=["XGBoost", "Random Forest"]  # Lighter models
      )

3. **Reduce cross-validation folds**:

   .. code-block:: python

      from vartrustml.config.experiment import CVConfig

      config = ExperimentConfig(
          cv=CVConfig(
              n_outer_splits=5,   # Instead of 10
              n_inner_splits=3    # Instead of 5
          )
      )


Missing Values (NaN) in Features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem**: ``ValueError: Input X contains NaN`` during training (typically from MLP, KNN or Logistic Regression; tree models such as XGBoost / CatBoost tolerate NaN natively).

**Cause**: some feature columns contain missing values, for example insert-size or mapping-quality statistics that are undefined when no suitable reads cover a region.

**Solution**: pick a missing-value strategy with ``--nan-strategy`` (CLI) or ``ExperimentConfig.nan_strategy``:

.. code-block:: bash

   # Impute (default): 'median' / 'mean' / 'most_frequent', fit per fold (no leakage)
   vartrustml compare-models data.csv -t state --nan-strategy median

   # Or drop rows that contain any missing feature value
   vartrustml compare-models data.csv -t state --nan-strategy drop


Checkpoint Resume Failures
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem**: Checkpoints not loading correctly when resuming.

**Causes and Solutions**:

1. **Changed configuration**: Ensure ``ExperimentConfig`` matches the original run
2. **Corrupted checkpoint files**: Delete the affected checkpoint and let it regenerate
3. **Version mismatch**: Checkpoints may not be compatible across VarTrustML versions

.. code-block:: python

   # Clear all checkpoints to start fresh
   import shutil
   shutil.rmtree("checkpoints/dataset_name", ignore_errors=True)


Calibration Warnings
^^^^^^^^^^^^^^^^^^^^

**Problem**: Calibration produces warnings about poor fit.

**Solutions**:

1. **Increase calibration CV folds** (more data per fold):

   .. code-block:: python

      from vartrustml.config.experiment import CalibrationConfig

      config = ExperimentConfig(
          calibration=CalibrationConfig(calibration_cv=5)  # Instead of default 3
      )

2. **Try sigmoid calibration** (better for small datasets):

   .. code-block:: python

      config = ExperimentConfig(
          calibration=CalibrationConfig(calibration_method="sigmoid")  # Instead of "isotonic"
      )


SHAP Computation Failures
^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem**: SHAP values fail to compute for certain models.

**Solution**: This typically happens with complex models or edge cases. VarTrustML handles this gracefully by skipping SHAP when it fails:

.. code-block:: python

   # SHAP failures don't stop the pipeline
   # Check logs for warnings about SHAP computation


Import Errors
^^^^^^^^^^^^^

**Problem**: ``ModuleNotFoundError`` for optional dependencies.

**Solution**: Install the required optional packages:

.. code-block:: bash

   # For Optuna HPO
   pip install optuna

   # For full visualization
   pip install plotly


HTML Report Not Generating
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem**: Report generation fails or produces empty reports.

**Causes and Solutions**:

1. **Missing results**: Ensure pipeline ran successfully
2. **Permission issues**: Check write permissions for output directory
3. **Disk space**: Ensure sufficient disk space for report

.. code-block:: python

   # Verify results exist
   if results and any(len(v) > 0 for v in results.values()):
       print("Results available for report generation")


Frequently Asked Questions
--------------------------

How do I add a custom model?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

VarTrustML supports any scikit-learn compatible classifier:

.. code-block:: python

   from sklearn.ensemble import AdaBoostClassifier
   from vartrustml.core.models import ModelEvaluator

   # Get the evaluator
   evaluator = pipeline.evaluator

   # Add custom model
   evaluator.models["AdaBoost"] = AdaBoostClassifier()
   evaluator.param_grids["AdaBoost"] = {
       "n_estimators": [50, 100, 200],
       "learning_rate": [0.01, 0.1, 1.0]
   }


How do I use custom metrics?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``FoldMetrics`` dataclass stores all computed metrics. To add custom metrics:

.. code-block:: python

   from sklearn.metrics import cohen_kappa_score

   # After getting predictions from a fold
   y_true = fold_result.y_true_oof
   y_pred = (fold_result.y_prob_oof >= 0.5).astype(int)

   custom_metric = cohen_kappa_score(y_true, y_pred)


How do I disable certain features?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``ExperimentConfig`` to disable features:

.. code-block:: python

   from vartrustml.config.experiment import CalibrationConfig, ThresholdConfig

   config = ExperimentConfig(
       calibration=CalibrationConfig(calibrate_models=False),
       threshold=ThresholdConfig(optimize_threshold=False),
       generate_html_report=False,    # No HTML report
       save_checkpoints=False,        # No checkpoints
   )


How do I interpret Cliff's delta?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Cliff's delta effect size interpretation (absolute value):

- ``|delta| < 0.147``: Negligible
- ``0.147 <= |delta| < 0.33``: Small
- ``0.33 <= |delta| < 0.474``: Medium
- ``|delta| >= 0.474``: Large


Why are my bootstrap CIs very wide?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Wide confidence intervals typically indicate:

1. **High variance across folds**: Your model performance varies significantly
2. **Small sample size**: More data reduces CI width
3. **Few folds**: More CV folds provide more bootstrap samples

.. code-block:: python

   from vartrustml.config.experiment import BootstrapConfig

   # Increase bootstrap iterations for more stable CIs
   config = ExperimentConfig(
       bootstrap=BootstrapConfig(bootstrap_n_iterations=2000)  # Instead of default 1000
   )


Debug Mode
----------

Enable verbose logging for debugging:

.. code-block:: python

   import logging

   # Set VarTrustML to debug level
   logging.getLogger("vartrustml").setLevel(logging.DEBUG)

   # Or set globally
   config = ExperimentConfig(verbose=2)  # 0=WARNING, 1=INFO, 2=DEBUG


Getting Help
------------

If you encounter issues not covered here:

1. Check the `GitHub Issues <https://github.com/EttoreRocchi/VarTrustML/issues>`_
2. Search existing issues for similar problems
3. Open a new issue with:

   - VarTrustML version (``pip show vartrustml``)
   - Python version
   - Full error traceback
   - Minimal reproducible example
