=================
Statistical Tests
=================

VarTrustML compares classifiers with **paired tests on pooled out-of-fold (OOF) predictions**. Every variant is predicted exactly once, at the operating point of the cross-validation fold in which it was held out, so the comparison is a single paired test over all variants. Because each instance contributes one paired observation, there is no cross-fold dependence to correct for (the inflated Type-I error of fold-level tests; Nadeau & Bengio, 2003), and the comparison does not rely on aggregating heterogeneous methods into "families".

Two classifiers are compared with:

- **McNemar's test** at a fixed operating point. This is the appropriate paired test for two classifiers on the same test set (Dietterich, 1998) and is the only inferential comparison available for variant callers, which emit hard binary calls (no probability scores).
- **DeLong's test** for the AUROC of two ML models (probability scores required, so this is restricted to ML-vs-ML comparisons).

What is compared
----------------

The analysis builds one *entity* per classifier from the pooled OOF predictions:

- **ML models**: full curves and hard predictions at the per-fold operating point;
- **single callers** (e.g. MANTA, DELLY, SMOOVE);
- **caller combinations** (logical ``AND`` / ``OR``).

It then computes the **full matrix** of comparisons:

- every ML model vs every caller and combination (McNemar);
- every ML-vs-ML pair (McNemar at the operating point **and** DeLong on AUROC).

The best ML model (by the primary metric on the pooled OOF data) is highlighted in a focused table of best-ML-vs-callers, while the full matrix is exported as a CSV.

McNemar's Test (operating point)
--------------------------------

For two classifiers evaluated on the same instances, only the **discordant** predictions matter:

- ``b`` = A correct **and** B wrong,
- ``c`` = A wrong **and** B correct.

Under the null hypothesis ``b`` is distributed Binomial(``b + c``, 0.5). An **exact binomial** test is used when ``b + c < 25`` and a chi-square approximation with Edwards' continuity correction otherwise.

DeLong's Test (AUROC, ML vs ML)
-------------------------------

The DeLong test compares two correlated AUROC values from the same sample, accounting for the correlation between predictions. It is more powerful than treating fold-level AUROCs as independent observations. Callers have no probability scores, so DeLong is only applied between ML models.

Effect sizes
------------

Every comparison reports paired effect sizes alongside the p-value:

- **discordant counts** ``b`` and ``c``;
- **paired accuracy difference** ``(b - c) / n`` (A minus B; positive favours A) with a 95% confidence interval from the standard error of correlated proportions;
- **discordant odds ratio** ``b / c`` with a 95% confidence interval (a Haldane-Anscombe 0.5 correction is applied when a discordant cell is zero).

For DeLong comparisons the effect size is the AUROC difference.

Operating point
---------------

ML probability scores are binarised at each fold's operating point (the per-fold optimised threshold when ``--optimize-threshold`` is enabled, otherwise 0.5), consistent with the rest of the report. McNemar compares at this single operating point, which is the like-for-like comparison against a fixed binary caller. The threshold-free ROC/PR **dominance** figure complements it: a caller whose operating point lies inside/below every ML curve is dominated regardless of threshold.

Multiple Testing Correction
---------------------------

P-values are corrected within each comparison family (all operating-point McNemar tests; all AUROC DeLong tests). The procedure is selectable with ``--correction``:

- **Holm-Bonferroni** (default): controls the Family-Wise Error Rate (the probability of any false positive). More conservative; preferred when even a single false claim is costly, e.g. definitive pairwise statements.
- **Benjamini-Hochberg**: controls the False Discovery Rate. More powerful; appropriate when many comparisons are made and a few false positives are tolerable.

.. code-block:: python

   from vartrustml.analysis import correct_pvalues

   # method="holm" (default) or method="bh"
   q_values, significant = correct_pvalues(p_values, method="holm", alpha=0.05)

CLI Usage
---------

The comparison runs automatically when callers are supplied to ``compare-models``:

.. code-block:: bash

   vartrustml compare-models data/HG002_DEL.csv -t state \
       --models "XGBoost,Random Forest,CatBoost" \
       --optimize-threshold \
       --compare-callers \
       --callers "MANTA,DELLY,SMOOVE" \
       --default-combinations

   # Choose the metric used to rank ML models / pick the best ML
   vartrustml compare-models data/HG002_DEL.csv -t state \
       --models "XGBoost,Random Forest,CatBoost,MLP" \
       --comparison-metric "Matthews Corr. Coef."

Python API
----------

.. code-block:: python

   from vartrustml.analysis.pairwise_comparison import (
       build_entities, compare_pairwise, comparisons_to_dataframe,
   )

   # oof_predictions: {model: {"y_true", "y_pred", "y_prob", "sample_indices"}}
   # caller_results:  {caller: [CallerResult, ...]}  (per fold)
   entities = build_entities(oof_predictions, caller_results)

   result = compare_pairwise(
       entities,
       primary_metric="Matthews Corr. Coef.",
       alpha=0.05,
   )

   print("Best ML model:", result.best_ml)
   for c in result.main_comparisons():            # best ML vs each caller
       print(f"{c.name_b}: winner={c.better}, "
             f"Δacc={c.acc_diff:+.3f}, q(BH)={c.p_value_corrected:.2e}")

   # Full matrix to CSV
   comparisons_to_dataframe(result).to_csv("pairwise_mcnemar_full.csv", index=False)

Output Files
------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - File
     - Content
   * - ``report.html``
     - Paired comparison section: (A) best-ML-vs-callers table, (B) ML-vs-ML q-value heatmap + DeLong table, (C) ROC/PR dominance figure, (D) per-classifier MCC forest plot with bootstrap CIs
   * - ``pairwise_mcnemar_full.csv``
     - Full comparison matrix (test, raw + BH p-values, winner, effect sizes)
   * - ``auroc_delong.csv``
     - DeLong AUROC comparisons between ML models

Example Output
--------------

Best ML model vs variant callers (main table). The figures below illustrate
the layout; they are not results from a VarTrustML run:

.. code-block:: text

   Best ML model: XGBoost
   vs MANTA            winner=XGBoost  Δacc=+0.238 [0.223, 0.253]  OR=80.3   q=2.0e-200  ✓
   vs DELLY            winner=XGBoost  Δacc=+0.333 [0.316, 0.350]  OR=125.9  q=1.1e-285  ✓
   vs MANTA AND DELLY  winner=XGBoost  Δacc=+0.294 [0.278, 0.311]  OR=111.4  q=1.7e-251  ✓

References
----------

- Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation*, 10(7), 1895-1923.
- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika*, 12(2), 153-157.
- DeLong, E.R., DeLong, D.M., & Clarke-Pearson, D.L. (1988). Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. *Biometrics*, 44(3), 837-845.
- Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate. *Journal of the Royal Statistical Society B*, 57(1), 289-300.
- Nadeau, C. & Bengio, Y. (2003). Inference for the generalization error. *Machine Learning*, 52(3), 239-281.

See Also
--------

- :doc:`bootstrap-confidence-intervals` -- Bias-corrected (BCa) confidence intervals
- :doc:`caller-comparison` -- Caller comparison setup
- :doc:`compare-models` -- Model comparison workflow
