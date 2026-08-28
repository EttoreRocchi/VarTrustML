==============================
Bootstrap Confidence Intervals
==============================

Overview
--------

Every metric VarTrustML reports carries a bootstrap confidence interval. Bootstrapping replaces the Student's t interval, which would require the metric to be normally distributed across folds, an assumption that MCC and AUROC do not satisfy.

Why Bootstrap CIs?
------------------

The bootstrap makes no distributional assumption, so it applies to any metric
that can be computed from a set of predictions, including bounded and heavily
skewed ones like MCC. It is also less sensitive to outliers than a parametric
interval.

The gain specific to VarTrustML is where the resampling happens: predictions
are resampled individually rather than fold summaries, so an interval rests on
thousands of observations instead of the 5 to 10 a fold-level bootstrap would
give it.

How It Works
------------

VarTrustML resamples at the prediction level:

1. **Concatenate** out-of-fold predictions from all CV folds
2. **Resample** individual predictions with replacement (n_iterations times)
3. **Compute** the metric for each resample
4. **Calculate** the interval, by default the **bias-corrected and accelerated (BCa)** interval, which adjusts the percentile bounds for bias and skewness of the bootstrap distribution

This approach is more powerful than fold-level resampling because it uses all individual predictions (hundreds or thousands of samples) rather than just the fold-level summary statistics (5-10 values).

Confidence interval method (BCa vs percentile)
----------------------------------------------

The interval method is controlled by ``ci_method`` (``--ci-method`` on the CLI):

- ``bca`` (default): **bias-corrected and accelerated**. Corrects the percentile bounds for bias (``z0``) and skewness/acceleration (``a``), giving second-order accurate coverage. This is the recommended default, especially for bounded or skewed metrics (MCC, AUROC near 1, ECE). The acceleration is estimated with a leave-one-block-out jackknife (delete-d on large samples) so the cost stays bounded.
- ``percentile``: the raw bootstrap percentiles (2.5th / 97.5th for a 95% CI).

BCa falls back to the percentile interval on a per-metric basis when it is not applicable (for example a point estimate at the edge of the bootstrap distribution, such as recall = 1.0). The method actually used is recorded on each result (``BootstrapCIResult.ci_method``) and stated in the HTML report.

Configuration
-------------

CLI Options
^^^^^^^^^^^

.. code-block:: bash

   vartrustml compare-models data.csv -t state \
     --bootstrap-iters 1000 \
     --ci-level 0.95 \
     --ci-method bca       # bca (default) or percentile

Python API
^^^^^^^^^^

.. code-block:: python

   from vartrustml import ExperimentConfig
   from vartrustml.config.experiment import BootstrapConfig

   config = ExperimentConfig(
       bootstrap=BootstrapConfig(
           bootstrap_n_iterations=1000,  # Number of bootstrap resamples
           bootstrap_ci_level=0.95,      # 95% confidence level
           bootstrap_ci_method="bca",    # "bca" (default) or "percentile"
       )
   )

Parameters
^^^^^^^^^^

``--bootstrap-iters`` (default: 1000)
"""""""""""""""""""""""""""""""""""""

Number of bootstrap resamples to generate.

Values below 100 are rejected by config validation. The default of 1000 is
enough for reported intervals; push to 2000 or beyond when the interval bounds
themselves need to be stable to three decimals.

.. code-block:: bash

   --bootstrap-iters 2000

``--ci-level`` (default: 0.95)
""""""""""""""""""""""""""""""

Confidence level for the interval.

Use 0.95 unless you have a reason not to. Lower it to 0.90 for a narrower
interval, raise it to 0.99 for a wider one.

.. code-block:: bash

   --ci-level 0.95

``--ci-method`` (default: bca)
""""""""""""""""""""""""""""""

Confidence interval method.

``bca`` applies the bias and acceleration corrections and is the recommended
setting; ``percentile`` takes the raw bootstrap quantiles.

.. code-block:: bash

   --ci-method bca

Output Format
-------------

CSV Files
^^^^^^^^^

Metrics are reported with CI bounds in ``metrics_summary.csv``. The values
below are illustrative and show the column layout only; they are not results
from any VarTrustML run.

.. list-table::
   :header-rows: 1
   :widths: 22 13 13 13 13 13 13

   * - Metric
     - mean
     - mean_bootstrap
     - std
     - ci_lower
     - ci_upper
     - median
   * - AUROC
     - 0.923
     - 0.921
     - 0.015
     - 0.901
     - 0.945
     - 0.924
   * - Balanced Acc.
     - 0.887
     - 0.885
     - 0.018
     - 0.862
     - 0.912
     - 0.888
   * - MCC
     - 0.774
     - 0.772
     - 0.021
     - 0.742
     - 0.806
     - 0.775

**Column descriptions:**

- ``mean``: Average of fold-level metrics
- ``mean_bootstrap``: OOF-based point estimate (the value that ``ci_lower``/``ci_upper`` bracket)
- ``std``: Standard deviation across folds
- ``ci_lower``, ``ci_upper``: Bootstrap confidence interval bounds
- ``median``: Median across folds

HTML Report
^^^^^^^^^^^

In the HTML report, metrics are displayed as point estimate followed by the
interval, for example:

::

   AUROC: 0.923 [0.901, 0.945]

Using BootstrapAnalyzer Directly
--------------------------------

.. code-block:: python

   from vartrustml.analysis.bootstrap import BootstrapAnalyzer, format_ci

   # Create analyzer
   analyzer = BootstrapAnalyzer(
       n_iterations=1000,
       ci_level=0.95,
       ci_method="bca",  # "bca" (default) or "percentile"
       seed=42
   )

   # Compute CIs from concatenated predictions
   # y_true, y_pred: concatenated from all test folds
   # y_prob: predicted probabilities (optional, enables AUROC CI)
   ci_results = analyzer.compute_all_cis_from_predictions(
       y_true, y_pred, y_prob
   )

   # Access individual metric results
   mcc_result = ci_results["Matthews Corr. Coef."]
   print(f"MCC: {format_ci(mcc_result)}")
   # Prints the point estimate and interval, e.g. MCC: 0.774 [0.742, 0.806]

   # Available metrics
   for metric_name, result in ci_results.items():
       print(f"{metric_name}: {format_ci(result)}")

Supported Metrics
-----------------

The ``compute_all_cis_from_predictions`` method computes CIs for:

- Precision (Class 0, Class 1)
- Recall (Class 0, Class 1)
- F1 Score (Class 0, Class 1, Weighted)
- Matthews Correlation Coefficient
- Balanced Accuracy
- AUROC, Brier Score, ECE, MCE (when probabilities are provided)

See Also
--------

- :doc:`statistical-tests` -- Statistical comparison methods
- :doc:`compare-models` -- Model comparison workflow
- :doc:`html-reports` -- Interactive HTML reports
