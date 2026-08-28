======================
End-to-End Tutorial
======================

This tutorial walks through a complete VarTrustML workflow, from data preparation to interpreting results.

Prerequisites
-------------

Install VarTrustML:

.. code-block:: bash

   git clone https://github.com/EttoreRocchi/VarTrustML.git
   cd VarTrustML
   pip install .


Step 1: Prepare Your Data
-------------------------

VarTrustML expects a pandas DataFrame with:

- Feature columns (numeric)
- A binary target column (0/1)

.. code-block:: python

   import pandas as pd
   from vartrustml import DataLoader

   # Option 1: Load from CSV
   loader = DataLoader("data/")
   df = loader.load_dataset("my_dataset.csv")

   # Option 2: Use an existing DataFrame
   df = pd.read_csv("data/my_dataset.csv")

   # Verify data structure
   print(f"Shape: {df.shape}")
   print(f"Target distribution:\n{df['state'].value_counts()}")


Step 2: Configure the Experiment
--------------------------------

Create an ``ExperimentConfig`` to control the pipeline:

.. code-block:: python

   from vartrustml import ExperimentConfig
   from vartrustml.config.experiment import CVConfig, CalibrationConfig

   config = ExperimentConfig(
       # Reproducibility & cross-validation setup
       cv=CVConfig(
           seed=42,
           n_outer_splits=10,    # For unbiased performance estimates
           n_inner_splits=5,     # For hyperparameter tuning
       ),

       # Model selection
       models_to_use=[
           "XGBoost",
           "Random Forest",
           "Logistic Regression",
       ],

       # Calibration (recommended for probability estimates)
       calibration=CalibrationConfig(
           calibrate_models=True,
           calibration_method="isotonic",
       ),

       # Output settings
       output_dir="results",
       generate_html_report=True,
   )


Step 3: Run Cross-Validation
----------------------------

Execute the pipeline:

.. code-block:: python

   from vartrustml import CrossValidationPipeline

   # Create pipeline
   pipeline = CrossValidationPipeline(config)

   # Run cross-validation
   results, caller_results = pipeline.run_cross_validation(
       df,
       dataset_name="my_experiment"
   )

   print(f"Completed evaluation for {len(results)} models")


Step 4: Examine Results
-----------------------

Access per-fold and aggregate results:

.. code-block:: python

   import pandas as pd

   # Results structure: Dict[model_name, List[FoldMetrics]]
   for model_name, fold_results in results.items():
       print(f"\n{model_name}:")

       # Per-fold metrics
       for i, fold in enumerate(fold_results):
           print(f"  Fold {i}: MCC={fold.metrics['Matthews Corr. Coef.']:.4f}")

       # Aggregate across folds
       mcc_values = [f.metrics['Matthews Corr. Coef.'] for f in fold_results]
       print(f"  Mean MCC: {sum(mcc_values)/len(mcc_values):.4f}")


Step 5: View Statistical Comparisons
------------------------------------

Compare models statistically:

Models are compared with paired McNemar / DeLong tests on pooled out-of-fold predictions (this runs automatically in the HTML report; the API below lets you reproduce it):

.. code-block:: python

   from vartrustml.analysis.pairwise_comparison import (
       build_entities, compare_pairwise,
   )

   # oof_predictions is produced by the metric aggregator / report pipeline:
   #   {model: {"y_true", "y_pred", "y_prob", "sample_indices"}}
   # caller_results: {caller: [CallerResult, ...]} (optional)
   entities = build_entities(oof_predictions, caller_results)

   result = compare_pairwise(entities, primary_metric="Matthews Corr. Coef.")

   print("Best ML model:", result.best_ml)
   for c in result.main_comparisons():          # best ML vs each caller
       sig = "significant" if c.is_significant else "n.s."
       print(f"  vs {c.name_b}: winner={c.better}, "
             f"Δacc={c.acc_diff:+.3f}, q(BH)={c.p_value_corrected:.2e} ({sig})")


Step 6: Compute Confidence Intervals
------------------------------------

Get bootstrap confidence intervals:

.. code-block:: python

   from vartrustml.analysis.bootstrap import BootstrapAnalyzer

   analyzer = BootstrapAnalyzer(
       n_iterations=1000,
       ci_level=0.95,
       seed=42
   )

   # Compute CIs for each model using concatenated out-of-fold predictions
   import numpy as np

   for model_name, fold_results in results.items():
       y_true = np.concatenate([f.y_true_oof for f in fold_results if f.y_true_oof is not None])
       y_prob = np.concatenate([f.y_prob_oof for f in fold_results if f.y_prob_oof is not None])

       ci_result = analyzer.compute_ci_from_predictions(
           y_true, y_prob, metric_name="MCC"
       )

       print(f"{model_name}: {ci_result.point_estimate:.3f} "
             f"[{ci_result.ci_lower:.3f}, {ci_result.ci_upper:.3f}]")


Step 7: Explore the HTML Report
-------------------------------

Open the generated HTML report:

.. code-block:: python

   import webbrowser
   from pathlib import Path

   report_path = Path("results/my_experiment/report.html")
   if report_path.exists():
       webbrowser.open(str(report_path))

The report opens with dataset statistics and the configuration that produced
the run, then the metrics table with confidence intervals, then the paired
McNemar and DeLong comparisons with their effect sizes and the dominance
figure. Confusion matrices and ROC curves follow, and per-fold results sit
below in expandable sections.


Advanced: Threshold Optimization
--------------------------------

Enable threshold optimization for better decision thresholds:

.. code-block:: python

   from vartrustml.config.experiment import ThresholdConfig

   config = ExperimentConfig(
       threshold=ThresholdConfig(
           optimize_threshold=True,
           threshold_method="auto",  # "oof", "cv", or "auto"
       ),
       # ... other settings
   )

   pipeline = CrossValidationPipeline(config)
   results, _ = pipeline.run_cross_validation(df, "threshold_experiment")

   # Access optimized thresholds
   for model_name, fold_results in results.items():
       for i, fold in enumerate(fold_results):
           if fold.fold_optimal_threshold is not None:
               print(f"{model_name} Fold {i}: "
                     f"threshold={fold.fold_optimal_threshold:.3f}, "
                     f"Youden's J={fold.fold_youden_j:.3f}")


Advanced: Caller Comparison
---------------------------

Compare ML models against variant callers:

.. code-block:: python

   from vartrustml.config.experiment import CallerComparisonConfig

   config = ExperimentConfig(
       caller_comparison=CallerComparisonConfig(
           compare_callers=True,
           caller_columns=["MANTA", "DELLY", "LUMPY"],  # Binary columns
           include_default_combinations=True,  # Auto-generate AND/OR combos
       ),
       # ... other settings
   )

   pipeline = CrossValidationPipeline(config)
   results, caller_results = pipeline.run_cross_validation(df, "caller_comparison")

   # Caller results structure: Dict[caller_name, List[CallerResult]]
   if caller_results:
       for caller_name, fold_results in caller_results.items():
           mcc_values = [f.metrics['Matthews Corr. Coef.'] for f in fold_results]
           print(f"{caller_name}: Mean MCC = {sum(mcc_values)/len(mcc_values):.4f}")


Advanced: Cross-Dataset Evaluation
----------------------------------

Test model generalization across datasets:

.. code-block:: python

   from vartrustml import CrossDatasetEvaluator

   # Load multiple datasets
   datasets = [
       (pd.read_csv("data/dataset_A.csv"), "Dataset_A"),
       (pd.read_csv("data/dataset_B.csv"), "Dataset_B"),
       (pd.read_csv("data/dataset_C.csv"), "Dataset_C"),
   ]

   config = ExperimentConfig(
       models_to_use=["XGBoost", "Random Forest"],
       output_dir="results/cross_dataset",
   )

   evaluator = CrossDatasetEvaluator(config)
   cross_results = evaluator.evaluate_cross_dataset(datasets)

   # Results contain train×test performance matrices
   for model_name, metrics in cross_results.items():
       print(f"\n{model_name}:")
       print(metrics['Matthews Corr. Coef.'])


Summary
-------

This tutorial covered:

1. **Data preparation**: Loading and validating input data
2. **Configuration**: Setting up ExperimentConfig
3. **Execution**: Running the CrossValidationPipeline
4. **Results analysis**: Examining FoldMetrics
5. **Statistical testing**: Comparing models with paired McNemar / DeLong tests
6. **Confidence intervals**: Computing CIs with BootstrapAnalyzer
7. **Reporting**: Viewing HTML reports
8. **Advanced features**: Threshold optimization, caller comparison, cross-dataset evaluation

For more details, see:

- :doc:`../api/index` for complete API reference
- :doc:`../user-guide/architecture` for system design
- :doc:`../user-guide/troubleshooting` for common issues
