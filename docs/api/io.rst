===========
I/O Modules
===========

Modules for data loading, preprocessing, and checkpoint management.

DataLoader
----------

Data loading and preprocessing utilities.

.. autoclass:: vartrustml.io.data_loader.DataLoader
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

   from vartrustml import DataLoader

   # Initialize loader
   loader = DataLoader("data/")

   # Load dataset
   df = loader.load_dataset("HG002.csv", drop_duplicates=True)

   # Create feature report
   feature_report = loader.create_feature_report(
       df,
       continuous_cols=["SVLEN_CALLER", "MAXQV", "CG_CONTENT"],
       target_col="state",
       output_path="results/feature_report.json"
   )

Checkpoint Functions
--------------------

Functions for saving and loading experiment checkpoints.

save_fold_results
^^^^^^^^^^^^^^^^^

.. autofunction:: vartrustml.io.checkpoint.save_fold_results

load_checkpoint_model
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: vartrustml.io.checkpoint.load_checkpoint_model

list_checkpoints
^^^^^^^^^^^^^^^^

.. autofunction:: vartrustml.io.checkpoint.list_checkpoints

cleanup_checkpoints
^^^^^^^^^^^^^^^^^^^

.. autofunction:: vartrustml.io.checkpoint.cleanup_checkpoints

get_checkpoint_summary
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: vartrustml.io.checkpoint.get_checkpoint_summary

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

   from vartrustml import (
       save_fold_results,
       load_checkpoint_model,
       list_checkpoints,
       get_checkpoint_summary
   )

   # List available checkpoints
   checkpoints = list_checkpoints("results/HG002/checkpoints")

   # Get summary
   summary = get_checkpoint_summary("results/HG002/checkpoints")
   print(summary[["Dataset", "Model", "Completed Folds"]])

   # Load a specific checkpoint
   model_data = load_checkpoint_model(
       "results/HG002/checkpoints/XGBoost/fold_0"
   )
   if model_data is not None:
       model = model_data["model"]

Utility Functions
-----------------

Helper functions exported from the main package.

create_summary_report
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: vartrustml.utils.reporting.create_summary_report

create_feature_importance_report
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: vartrustml.utils.reporting.create_feature_importance_report

calculate_minimum_samples_for_cv
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: vartrustml.utils.validation.calculate_minimum_samples_for_cv

validate_target_for_cv
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: vartrustml.utils.validation.validate_target_for_cv
