======================
Configuration Classes
======================

VarTrustML uses dataclasses for configuration management, providing type-safe and well-documented settings.

ExperimentConfig
----------------

Main configuration class for cross-validation experiments.

.. autoclass:: vartrustml.config.experiment.ExperimentConfig
   :members: save, load
   :exclude-members: __init__, __new__
   :show-inheritance:

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

   from vartrustml import ExperimentConfig
   from vartrustml.config.experiment import CVConfig, CalibrationConfig

   config = ExperimentConfig(
       cv=CVConfig(seed=42, n_outer_splits=10, n_inner_splits=5),
       models_to_use=["XGBoost", "Random Forest"],
       calibration=CalibrationConfig(calibrate_models=True),
       hpo_method="optuna",
       optuna_n_trials=100,
       generate_html_report=True
   )

   # Save to JSON
   config.save("experiment_config.json")

   # Load from JSON
   loaded_config = ExperimentConfig.load("experiment_config.json")

ModelConfig
-----------

Configuration for hyperparameter search spaces.

.. autoclass:: vartrustml.config.model.ModelConfig
   :show-inheritance:

VisualizationConfig
-------------------

Configuration for plot generation settings.

.. autoclass:: vartrustml.config.visualization.VisualizationConfig
   :show-inheritance:

CallerConfig
------------

Configuration for variant caller comparison settings.

.. autoclass:: vartrustml.config.caller.CallerConfig
   :show-inheritance:

TrainConfig
-----------

Configuration for standalone model training.

.. autoclass:: vartrustml.core.train_model.TrainConfig
   :show-inheritance:

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

   from vartrustml import TrainConfig, ModelTrainer

   config = TrainConfig(
       model_name="XGBoost",
       n_cv_folds=5,
       calibrate_model=True,
       scoring="roc_auc"
   )

   trainer = ModelTrainer(config)
   results = trainer.fit(X, y)
