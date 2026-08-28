============================
Hyperparameter Optimization
============================

VarTrustML supports two methods for hyperparameter optimization: **Grid Search** (exhaustive) and **Optuna** (Bayesian).

Overview
--------

Two search strategies are available: exhaustive grid search, which is the
default, and Bayesian optimization through Optuna. Both run inside the inner
cross-validation loop and both take their randomness from the run's seed, so a
search can be repeated exactly.

Methods Comparison
------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Feature
     - Grid Search
     - Optuna
   * - **Search Strategy**
     - Exhaustive
     - Bayesian (TPE)
   * - **Speed**
     - Slower
     - Faster
   * - **Best for**
     - Small search spaces
     - Large search spaces
   * - **Trials**
     - All combinations
     - Configurable (default: 50)
   * - **Early Stopping**
     - No
     - Yes (timeout)
   * - **Smart Search**
     - No
     - Yes (learns from previous trials)

Using Grid Search (Default)
---------------------------

Grid Search exhaustively tries all parameter combinations.

.. tab-set::

   .. tab-item:: CLI

      .. code-block:: bash

         vartrustml compare-models HG002.csv -t state --hpo-method grid

   .. tab-item:: Python API

      .. code-block:: python

         from vartrustml import ExperimentConfig, CrossValidationPipeline

         config = ExperimentConfig(
             hpo_method="grid",  # Default
             models_to_use=["XGBoost", "Random Forest"]
         )

         pipeline = CrossValidationPipeline(config)
         results = pipeline.run_cross_validation(df, "HG002")

When to Use Grid Search
^^^^^^^^^^^^^^^^^^^^^^^

Grid search is the right choice when the space is small enough to enumerate,
when you need every combination visited rather than sampled, or when you want
a ceiling to measure a cheaper search against.

Using Optuna (Recommended)
--------------------------

Optuna uses Tree-structured Parzen Estimator (TPE) for intelligent search.

.. tab-set::

   .. tab-item:: CLI

      .. code-block:: bash

         # Basic usage
         vartrustml compare-models HG002.csv -t state \
           --hpo-method optuna \
           --optuna-trials 100

         # With timeout
         vartrustml compare-models HG002.csv -t state \
           --hpo-method optuna \
           --optuna-trials 200 \
           --optuna-timeout 7200  # 2 hours

   .. tab-item:: Python API

      .. code-block:: python

         from vartrustml import ExperimentConfig, CrossValidationPipeline

         config = ExperimentConfig(
             hpo_method="optuna",
             optuna_n_trials=100,
             optuna_timeout=3600,  # 1 hour timeout
             models_to_use=["XGBoost"]
         )

         pipeline = CrossValidationPipeline(config)
         results = pipeline.run_cross_validation(df, "HG002")

Configuration Parameters
^^^^^^^^^^^^^^^^^^^^^^^^

``optuna_n_trials`` (default: 50)
"""""""""""""""""""""""""""""""""

Number of hyperparameter combinations to try.

.. code-block:: bash

   --optuna-trials 100

**Recommendations:**

- Quick experiments: 20-50 trials
- Standard experiments: 50-100 trials
- Thorough search: 100-200 trials

``optuna_timeout`` (default: 3600 seconds)
""""""""""""""""""""""""""""""""""""""""""

Maximum time for optimization in seconds.

.. code-block:: bash

   --optuna-timeout 7200  # 2 hours

.. note::

   Optimization stops when either ``n_trials`` or ``timeout`` is reached, whichever comes first.

When to Use Optuna
^^^^^^^^^^^^^^^^^^

Reach for Optuna once the grid stops being enumerable: many hyperparameters,
wide ranges, or a compute budget that rules out exhaustive search. That is
usually the case for XGBoost, CatBoost and the MLP.

How Optuna Works
----------------

Tree-structured Parzen Estimator (TPE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Optuna uses TPE to intelligently select hyperparameters:

1. Tries initial random configurations
2. Builds probabilistic model of parameter performance
3. Suggests parameters likely to improve performance
4. Updates model based on results
5. Repeats until stopping criterion

Trial Management
^^^^^^^^^^^^^^^^

Each trial:

- Selects hyperparameters based on previous trials
- Trains model with cross-validation
- Records performance metric (e.g., AUROC)
- Updates internal model

Best Parameters Selection
^^^^^^^^^^^^^^^^^^^^^^^^^

After optimization:

- Best trial is identified
- Final model is trained with best parameters
- Results include best parameters and score

Parameter Spaces
----------------

VarTrustML defines parameter search spaces for each model:

**XGBoost:**

- ``n_estimators``: 50-200
- ``max_depth``: 3-5
- ``learning_rate``: 0.01-0.1

**Random Forest:**

- ``n_estimators``: 50, 100, 200
- ``max_depth``: 3, 5

**MLP:**

- ``hidden_layers``: Various architectures
- ``activation``: relu, tanh
- ``alpha``: 0.0001-0.001

Both methods search the same declared candidate values, configured once per model in :class:`~vartrustml.config.model.ModelConfig`. Grid search evaluates every combination, while Optuna samples from those same values with its TPE sampler, so the two methods are directly comparable and only differ in how many combinations they try.

See Also
--------

- :doc:`compare-models` -- Full model comparison workflow
- :doc:`training` -- Single model training
- :doc:`calibration` -- Probability calibration
