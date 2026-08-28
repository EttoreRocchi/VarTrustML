=============
CLI Reference
=============

Unified command-line interface for running **model comparison** and **training** workflows in VarTrustML.

Installation
------------

.. code-block:: bash

   pip install -e .

After installation, the command ``vartrustml`` is available.

Quick Start
-----------

Model comparison with defaults:

.. code-block:: bash

   vartrustml compare-models HG002.csv -t state

Training:

.. code-block:: bash

   vartrustml train data/HG002.csv -t state --model "XGBoost" --cv-folds 5

Smoke test:

.. code-block:: bash

   vartrustml smoke-test

Command Overview
----------------

.. code-block:: text

   Usage: vartrustml [OPTIONS] COMMAND [ARGS]...

     VarTrustML unified CLI. Run model comparison and training workflows.

   Options:
     -h, --help  Show this message and exit.

   Commands:
     compare-models  Compare multiple ML models on a dataset using cross-validation.
     cross-dataset   Cross-dataset generalizability analysis.
     train           Train a single model with hyperparameter tuning.
     ablation        Run ablation study to measure feature importance.
     predict         Generate predictions from a saved model.
     evaluate        Evaluate a saved model on a labeled dataset.
     list-models     List available models.
     smoke-test      Quick import test to validate installation.
     version         Display the VarTrustML version.

.. toctree::
   :maxdepth: 2

   compare-models
   cross-dataset
   train
   ablation
   other-commands
