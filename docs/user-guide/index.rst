==========
User Guide
==========

This section provides detailed guides for all VarTrustML features and workflows.

.. toctree::
   :maxdepth: 2

   architecture
   compare-models
   training
   ablation-studies
   hyperparameter-optimization
   calibration
   threshold-optimization
   html-reports
   caller-comparison
   bootstrap-confidence-intervals
   statistical-tests
   troubleshooting

Core Workflows
--------------

- :doc:`compare-models` -- Compare multiple ML models with nested cross-validation
- :doc:`training` -- Train individual models with hyperparameter tuning
- :doc:`ablation-studies` -- Measure feature importance via systematic ablation

Optimization & Calibration
--------------------------

- :doc:`hyperparameter-optimization` -- Master Optuna for efficient HPO
- :doc:`calibration` -- Improve probability reliability with calibration
- :doc:`threshold-optimization` -- Optimize classification thresholds

Analysis & Reporting
--------------------

- :doc:`html-reports` -- Generate interactive visualizations
- :doc:`caller-comparison` -- Compare ML models to variant callers
- :doc:`bootstrap-confidence-intervals` -- Statistically rigorous metrics
- :doc:`statistical-tests` -- Paired McNemar / DeLong on pooled out-of-fold predictions
