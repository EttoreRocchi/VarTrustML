=========================
Variant Caller Comparison
=========================

Overview
--------

VarTrustML scores variant callers such as MANTA, DELLY and SMOOVE on the same
cross-validation folds as the ML models, so both sides of the comparison see
identical train and test splits.

Why Compare Against Callers?
----------------------------

A caller is a heuristic that emits call or no-call, which makes it exactly the
baseline a reviewer will ask about. Running one through the same evaluation
puts a number on how much the model actually adds, shows where the two differ
in precision against recall rather than only in aggregate, and extends to
consensus rules: an AND or OR of two callers is itself a well-defined
non-ML baseline, and often a stronger one than either caller alone.

Quick Start
-----------

.. tab-set::

   .. tab-item:: CLI

      .. code-block:: bash

         vartrustml compare-models data/HG002_DEL.csv -t state \
           --compare-callers \
           --callers "MANTA,DELLY,SMOOVE"

   .. tab-item:: Python API

      .. code-block:: python

         from vartrustml import ExperimentConfig, CrossValidationPipeline, DataLoader
         from vartrustml.config.experiment import CallerComparisonConfig

         loader = DataLoader("data/")
         df = loader.load_dataset("HG002_DEL.csv")

         config = ExperimentConfig(
             caller_comparison=CallerComparisonConfig(
                 compare_callers=True,
                 caller_columns=["MANTA", "DELLY", "SMOOVE"],
                 include_default_combinations=True,
             ),
             generate_html_report=True,
             models_to_use=["XGBoost", "Random Forest", "CatBoost"]
         )

         pipeline = CrossValidationPipeline(config)
         results, caller_results = pipeline.run_cross_validation(df, "HG002_DEL")

Configuration Options
---------------------

``--compare-callers``
^^^^^^^^^^^^^^^^^^^^^

Enable caller comparison mode.

.. code-block:: bash

   vartrustml compare-models data.csv -t state --compare-callers

``--callers`` (Required)
^^^^^^^^^^^^^^^^^^^^^^^^

Specify which columns contain caller predictions. These must be binary (0/1) columns.

.. code-block:: bash

   --callers "MANTA,DELLY,SMOOVE"

``--combinations``
^^^^^^^^^^^^^^^^^^

Add custom logical combinations to evaluate:

.. code-block:: bash

   --combinations "MANTA AND DELLY,DELLY OR SMOOVE"

``--default-combinations``
^^^^^^^^^^^^^^^^^^^^^^^^^^

Automatically generate all pairwise and all-caller combinations (default: enabled).

**Default combinations generated** (for 3 callers):

- Pairwise AND: ``MANTA AND DELLY``, ``MANTA AND SMOOVE``, ``DELLY AND SMOOVE``
- Pairwise OR: ``MANTA OR DELLY``, ``MANTA OR SMOOVE``, ``DELLY OR SMOOVE``
- All callers: ``MANTA AND DELLY AND SMOOVE``, ``MANTA OR DELLY OR SMOOVE``

Logical Operations
------------------

AND Combinations
^^^^^^^^^^^^^^^^

Both callers must agree for a positive prediction:

::

   MANTA AND DELLY: Positive only if MANTA=1 AND DELLY=1

**Effect**: Higher precision, lower recall (more conservative)

OR Combinations
^^^^^^^^^^^^^^^

Either caller being positive triggers a positive prediction:

::

   MANTA OR DELLY: Positive if MANTA=1 OR DELLY=1

**Effect**: Higher recall, lower precision (more sensitive)

Output
------

When caller comparison is enabled, results include:

- ``caller_comparison.csv``, holding metrics for every caller and combination
- an HTML report section placing the ML models next to the callers
- paired McNemar tests of each ML model against each caller and combination

Callers emit hard 0/1 calls rather than scores, so the probability-based
metrics (AUROC, Brier score, ECE, MCE) are reported as ``NaN`` for caller
rows. Only threshold-based metrics are meaningful for them; see
:func:`vartrustml.core.metrics.calculate_classification_metrics`.

Example Output
^^^^^^^^^^^^^^

Caller baselines on the bundled ``data/HG002.csv`` (7168 variants, 80.8 %
positive):

.. code-block:: text

   Caller/Combination            Prec(1)    Rec(1)     F1(w)       MCC    BalAcc     AUROC
   ----------------------------------------------------------------------------------------
   MANTA                           0.875     0.930     0.826     0.419     0.685       NaN
   DELLY                           0.769     0.426     0.489    -0.090     0.443       NaN
   SMOOVE                          0.832     0.456     0.537     0.054     0.534       NaN
   MANTA AND DELLY                 0.881     0.382     0.500     0.137     0.583       NaN
   MANTA OR DELLY                  0.823     0.974     0.757     0.175     0.545       NaN
   MANTA AND DELLY AND SMOOVE      0.917     0.356     0.487     0.187     0.610       NaN
   MANTA OR DELLY OR SMOOVE        0.808     1.000     0.722     0.000     0.500       NaN

The AND and OR rows show the expected trade: unanimous consensus buys
precision at heavy cost in recall, while any-caller agreement reaches
perfect recall but collapses to MCC 0 because it calls every variant
positive. MCC and balanced accuracy separate the callers here in a way
that raw precision does not.

See Also
--------

- :doc:`compare-models` -- Full model comparison workflow
- :doc:`statistical-tests` -- Statistical comparison methods
- :doc:`bootstrap-confidence-intervals` -- Confidence intervals
