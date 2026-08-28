========
Training
========

Overview
--------

**VarTrustML** provides a flexible training module for training, tuning, evaluating, and saving machine learning models, ideal for:

* Rapid experimentation
* Simplified model evaluation
* Pipeline integration

Features
--------

* Cross-validated hyperparameter tuning
* Python API & CLI support
* Model calibration support
* The full metric set

Python API
----------

Quick Start
^^^^^^^^^^^

.. code-block:: python

   from vartrustml import ModelTrainer, TrainConfig, DataLoader

   # Load data
   loader = DataLoader("data/")
   df = loader.load_dataset("HG002.csv")

   X = df.drop(columns=["state"])
   y = df["state"]

   # Configure fitting
   train_config = TrainConfig(
       model_name="Random Forest",
       continuous_cols=["SVLEN_CALLER", "MAXQV", "CG_CONTENT"],
       n_cv_folds=5,
       scoring="roc_auc"
   )

   # Fit model
   trainer = ModelTrainer(train_config)
   results = trainer.fit(X, y)

   print(f"Best CV score: {results['best_score']:.4f}")
   print(f"Best parameters: {results['best_params']}")

TrainConfig Parameters
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 15 20 40

   * - Parameter
     - Type
     - Default
     - Description
   * - ``model_name``
     - str
     - "XGBoost"
     - Model to fit
   * - ``seed``
     - int
     - 42
     - Random seed
   * - ``n_cv_folds``
     - int
     - 5
     - Number of CV folds
   * - ``scoring``
     - str
     - "roc_auc"
     - Scikit-learn scoring metric
   * - ``n_jobs``
     - int
     - -1
     - CPU cores for parallelization
   * - ``calibrate_model``
     - bool
     - False
     - Enable probability calibration
   * - ``calibration_method``
     - str
     - "isotonic"
     - Calibration method
   * - ``output_dir``
     - str
     - "results/model_fitting"
     - Where to save output

Advanced Usage
^^^^^^^^^^^^^^

**With Train/Test Split:**

.. code-block:: python

   from sklearn.model_selection import train_test_split

   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42, stratify=y
   )

   # Fit with evaluation on test set
   train_config = TrainConfig(model_name="XGBoost")
   trainer = ModelTrainer(train_config)
   results = trainer.fit(X_train, y_train, X_test, y_test)

   print(f"Test AUROC: {results['test_results']['auroc']:.4f}")

**With Calibration:**

.. code-block:: python

   train_config = TrainConfig(
       model_name="Random Forest",
       calibrate_model=True,
       calibration_method="isotonic",
       calibration_cv=3
   )

   trainer = ModelTrainer(train_config)
   results = trainer.fit(X, y)

Command-Line Interface
----------------------

List Available Models
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   vartrustml list-models

Basic Training
^^^^^^^^^^^^^^

.. code-block:: bash

   # Fit XGBoost (default)
   vartrustml train data/HG002.csv -t state

   # Fit specific model
   vartrustml train data/HG002.csv -t state --model "Random Forest"

   # With custom CV folds and scoring
   vartrustml train data/HG002.csv -t state \
     --model "XGBoost" \
     --cv-folds 10 \
     --scoring balanced_accuracy

Training with Holdout Test Set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Internal holdout split
   vartrustml train data/HG002.csv -t state \
     --model "Random Forest" \
     --test-size 0.2

   # External test dataset
   vartrustml train data/HG002.csv -t state \
     --model "XGBoost" \
     --test-data data/REACH.csv

With Calibration and HTML Report
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   vartrustml train data/HG002.csv -t state \
     --model "XGBoost" \
     --test-size 0.2 \
     --calibrate-model \
     --calibration-method isotonic

The HTML report is written by default; ``--no-html-report`` disables it.

Output Files
------------

Training produces the following files in the output directory:

::

   results/model_fitting/
   ├── <Model_Name>_model.joblib     # Serialized model with metadata
   ├── <Model_Name>_report.joblib    # Training report
   └── <Model_Name>_cv_results.csv   # CV results

See Also
--------

- :doc:`compare-models` -- Compare multiple models
- :doc:`calibration` -- Model calibration details
- :doc:`hyperparameter-optimization` -- HPO methods
