================
Analysis Modules
================

Modules for error analysis, bootstrap confidence intervals, and statistical testing.

ErrorAnalyzer
-------------

Error analysis by confidence thresholds.

.. autoclass:: vartrustml.analysis.error_analysis.ErrorAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:

FoldMetrics
-----------

Container for per-fold evaluation results.

.. autoclass:: vartrustml.analysis.error_analysis.FoldMetrics
   :show-inheritance:

BootstrapAnalyzer
-----------------

Bootstrap confidence interval computation. The default interval is the bias-corrected and accelerated (BCa) interval; ``ci_method="percentile"`` selects the raw percentile interval.

.. autoclass:: vartrustml.analysis.bootstrap.BootstrapAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

   from vartrustml.analysis.bootstrap import BootstrapAnalyzer, format_ci

   analyzer = BootstrapAnalyzer(
       n_iterations=1000,
       ci_level=0.95,
       ci_method="bca",  # default; or "percentile"
       seed=42,
   )

   # y_true, y_pred, y_prob: concatenated out-of-fold predictions
   ci_results = analyzer.compute_all_cis_from_predictions(y_true, y_pred, y_prob)

   mcc = ci_results["Matthews Corr. Coef."]
   print(format_ci(mcc))          # e.g. "0.774 [0.742, 0.806]"
   print(mcc.ci_method)            # "bca" (or "percentile" on per-metric fallback)

BootstrapCIResult
-----------------

Result container for bootstrap confidence intervals.

.. autoclass:: vartrustml.analysis.bootstrap.BootstrapCIResult
   :show-inheritance:

Pairwise comparison (paired, pooled out-of-fold)
------------------------------------------------

Paired McNemar / DeLong comparison of classifiers (ML models and variant callers) on pooled out-of-fold predictions, with multiple-comparison correction (Holm-Bonferroni by default, Benjamini-Hochberg optional) and paired effect sizes.

.. autofunction:: vartrustml.analysis.pairwise_comparison.compare_pairwise

.. autofunction:: vartrustml.analysis.pairwise_comparison.build_entities

.. autofunction:: vartrustml.analysis.pairwise_comparison.comparisons_to_dataframe

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

   from vartrustml.analysis.pairwise_comparison import (
       build_entities, compare_pairwise, comparisons_to_dataframe,
   )

   # oof_predictions: {model: {"y_true", "y_pred", "y_prob", "sample_indices"}}
   # caller_results:  {caller: [CallerResult, ...]} (per fold)
   entities = build_entities(oof_predictions, caller_results)
   result = compare_pairwise(entities, primary_metric="Matthews Corr. Coef.")

   for c in result.main_comparisons():       # best ML vs each caller
       print(c.name_b, c.better, c.acc_diff, c.p_value_corrected)

   comparisons_to_dataframe(result).to_csv("pairwise_mcnemar_full.csv", index=False)

.. autoclass:: vartrustml.analysis.pairwise_comparison.Entity
   :show-inheritance:

.. autoclass:: vartrustml.analysis.pairwise_comparison.PairwiseComparison
   :show-inheritance:

.. autoclass:: vartrustml.analysis.pairwise_comparison.PairwiseComparisonResult
   :members:
   :show-inheritance:

Paired tests (McNemar / DeLong)
-------------------------------

.. autofunction:: vartrustml.analysis.delong_mcnemar.mcnemar_test

.. autofunction:: vartrustml.analysis.delong_mcnemar.delong_test

.. autofunction:: vartrustml.analysis.delong_mcnemar.benjamini_hochberg_correction

Comparison plots
----------------

.. automodule:: vartrustml.analysis.pairwise_plots
   :members: plot_metric_forest, plot_roc_pr_dominance, plot_pvalue_heatmap
