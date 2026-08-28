============
Core Modules
============

Core functionality for cross-validation pipelines, model training, and evaluation.

CrossValidationPipeline
-----------------------

Main orchestration class for nested cross-validation experiments.

.. autoclass:: vartrustml.core.pipeline.CrossValidationPipeline
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

   from vartrustml import ExperimentConfig, CrossValidationPipeline, DataLoader
   from vartrustml.config.experiment import CVConfig

   # Load data
   loader = DataLoader("data/")
   df = loader.load_dataset("HG002.csv")

   # Configure
   config = ExperimentConfig(
       cv=CVConfig(n_outer_splits=10),
       models_to_use=["XGBoost", "Random Forest"],
       generate_html_report=True
   )

   # Run pipeline
   pipeline = CrossValidationPipeline(config)
   results = pipeline.run_cross_validation(df, "HG002")

   # Access results
   for model_name, fold_results in results.items():
       print(f"{model_name}: {fold_results[0].metrics['AUROC']:.4f}")

CrossDatasetEvaluator
---------------------

Evaluator for cross-dataset generalizability experiments.

.. autoclass:: vartrustml.core.cross_dataset.CrossDatasetEvaluator
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
^^^^^^^^^^^^^

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

   # Run cross-dataset evaluation
   evaluator = CrossDatasetEvaluator(config)
   results = evaluator.evaluate_cross_dataset(datasets)

   # Results contain performance matrices per model and metric
   for model_name, metrics in results.items():
       for metric_name, matrix_df in metrics.items():
           print(f"{model_name} - {metric_name}:")
           print(matrix_df)

ModelEvaluator
--------------

Model training and evaluation for individual folds.

.. autoclass:: vartrustml.core.models.ModelEvaluator
   :members:
   :undoc-members:
   :show-inheritance:

ModelTrainer
------------

Standalone model fitting with hyperparameter tuning.

.. autoclass:: vartrustml.core.train_model.ModelTrainer
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

   from vartrustml import ModelTrainer, TrainConfig

   config = TrainConfig(
       model_name="XGBoost",
       n_cv_folds=5,
       calibrate_model=True
   )

   trainer = ModelTrainer(config)
   results = trainer.fit(X_train, y_train, X_test, y_test)

   print(f"Best score: {results['best_score']:.4f}")
   print(f"Test AUROC: {results['test_results']['auroc']:.4f}")

ThresholdOptimizer
------------------

Threshold optimization using Youden's J statistic.

.. autoclass:: vartrustml.core.threshold.ThresholdOptimizer
   :members:
   :undoc-members:
   :show-inheritance:

ThresholdResult
---------------

Result container for threshold optimization.

.. autoclass:: vartrustml.core.threshold.ThresholdResult
   :show-inheritance:

ThresholdMethod
---------------

Enumeration of threshold optimization methods.

.. autoclass:: vartrustml.core.threshold.ThresholdMethod
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

   from vartrustml.core.threshold import ThresholdOptimizer
   import numpy as np

   y_true = np.array([0, 1, 1, 0, 1, 0])
   y_prob = np.array([0.2, 0.8, 0.6, 0.3, 0.9, 0.1])

   optimizer = ThresholdOptimizer()
   result = optimizer.optimize_from_oof(y_true, y_prob)

   print(f"Optimal threshold: {result.optimal_threshold:.4f}")
   print(f"Youden's J: {result.youden_j:.4f}")
