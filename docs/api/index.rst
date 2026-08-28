=============
API Reference
=============

Every public class and function is documented here, for use when the CLI does not cover the workflow you need.

Installation
------------

.. code-block:: bash

   git clone https://github.com/EttoreRocchi/VarTrustML.git
   cd VarTrustML
   pip install .

Quick Start
-----------

.. code-block:: python

   from vartrustml import (
       ExperimentConfig,
       CrossValidationPipeline,
       DataLoader,
       ThresholdOptimizer,
   )

   # Load data
   loader = DataLoader("data/")
   df = loader.load_dataset("my_dataset.csv")

   from vartrustml.config.experiment import CVConfig, ThresholdConfig

   # Configure experiment
   config = ExperimentConfig(
       cv=CVConfig(n_outer_splits=5, n_inner_splits=3),
       models_to_use=["XGBoost", "Random Forest"],
       threshold=ThresholdConfig(optimize_threshold=True),
   )

   # Run cross-validation
   pipeline = CrossValidationPipeline(config)
   results = pipeline.run_cross_validation(df, "my_experiment")

Public API
----------

The following classes and functions are exported from the ``vartrustml`` package:

**Core Classes:**

- :class:`~vartrustml.CrossValidationPipeline` -- Main orchestration for nested CV
- :class:`~vartrustml.ModelEvaluator` -- Model training and evaluation
- :class:`~vartrustml.ModelTrainer` -- Standalone model fitting
- :class:`~vartrustml.ThresholdOptimizer` -- Youden's J threshold optimization
- :class:`~vartrustml.CrossDatasetEvaluator` -- Cross-dataset generalization testing

**Configuration Classes:**

- :class:`~vartrustml.ExperimentConfig` -- Main experiment configuration
- :class:`~vartrustml.ModelConfig` -- Hyperparameter search spaces
- :class:`~vartrustml.TrainConfig` -- Training configuration

**Analysis Classes:**

- :class:`~vartrustml.BootstrapAnalyzer` -- Bootstrap confidence intervals
- :func:`~vartrustml.compare_pairwise` -- Paired McNemar / DeLong comparison
- :class:`~vartrustml.ErrorAnalyzer` -- Error analysis by confidence
- :class:`~vartrustml.FoldMetrics` -- Per-fold results storage

**I/O Classes:**

- :class:`~vartrustml.DataLoader` -- Data loading and preprocessing

**Visualization Classes:**

- :class:`~vartrustml.Visualizer` -- Plot generation
- :class:`~vartrustml.HTMLCompareReporter` -- Model comparison reports
- :class:`~vartrustml.HTMLTrainReporter` -- Training reports
- :class:`~vartrustml.HTMLCrossDatasetReporter` -- Cross-dataset evaluation reports

Module Reference
----------------

.. toctree::
   :maxdepth: 2

   config
   core
   analysis
   visualization
   io
