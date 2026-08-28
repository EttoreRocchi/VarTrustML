================
Ablation Studies
================

Ablation studies are a fundamental technique for understanding feature importance by systematically removing features and measuring the impact on model performance. VarTrustML pairs that with significance testing and multiple-comparison correction.

What Are Ablation Studies?
--------------------------

An ablation study measures the contribution of each feature (or feature group) by:

1. Training the model with all features (baseline)
2. Removing one feature at a time
3. Re-training and measuring performance
4. Computing the performance delta with statistical significance

This provides a rigorous, model-agnostic measure of feature importance that is often more reliable than built-in feature importance metrics.

When to Use Ablation Studies
----------------------------

Ablation gives a model-agnostic measure of what each feature contributes,
which is what makes it useful for more than one purpose: deciding which
features can be dropped without cost, finding the ones that hurt
generalization rather than help it, and explaining to a reader which inputs
the predictions actually rest on.

Using the CLI
-------------

Basic Feature Ablation
^^^^^^^^^^^^^^^^^^^^^^

Remove each feature one at a time and measure impact:

.. code-block:: bash

   vartrustml ablation data.csv \
       --target-column state \
       --model XGBoost \
       --continuous feature1,feature2,feature3 \
       --output-dir results/ablation

Ablate Specific Features
^^^^^^^^^^^^^^^^^^^^^^^^

Test only specific features:

.. code-block:: bash

   vartrustml ablation data.csv \
       --target-column state \
       --model XGBoost \
       --continuous feature1,feature2,feature3 \
       --features "feature1,feature2"

Feature Group Ablation
^^^^^^^^^^^^^^^^^^^^^^

Create a YAML file defining feature groups:

.. code-block:: yaml

   # feature_groups.yaml
   caller_features:
     - manta_call
     - delly_call
     - lumpy_call

   size_features:
     - sv_length
     - sv_size_category

   quality_features:
     - mapping_quality
     - base_quality
     - read_depth

Run group ablation:

.. code-block:: bash

   vartrustml ablation data.csv \
       --target-column state \
       --model XGBoost \
       --continuous feature1,feature2 \
       --feature-groups feature_groups.yaml

With Calibration and Threshold Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Apply calibration and threshold optimization for better probability estimates:

.. code-block:: bash

   vartrustml ablation data.csv \
       --target-column state \
       --model CatBoost \
       --continuous feature1,feature2 \
       --calibrate \
       --optimize-threshold

Pipeline Step Ablation
^^^^^^^^^^^^^^^^^^^^^^

Measure the impact of calibration, threshold optimization, and scaling:

.. code-block:: bash

   vartrustml ablation data.csv \
       --target-column state \
       --model XGBoost \
       --continuous feature1,feature2 \
       --ablate-steps

Different Metrics
^^^^^^^^^^^^^^^^^

Use different evaluation metrics:

.. code-block:: bash

   # ROC AUC
   vartrustml ablation data.csv -t state -m XGBoost --metric roc_auc

   # Matthew's Correlation Coefficient
   vartrustml ablation data.csv -t state -m XGBoost --metric mcc

   # F1 Score
   vartrustml ablation data.csv -t state -m XGBoost --metric f1

CLI Options
^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Option
     - Default
     - Description
   * - ``--target-column``, ``-t``
     - Required
     - Target column name
   * - ``--model``, ``-m``
     - Required
     - Model name (XGBoost, CatBoost, Random Forest, etc.)
   * - ``--continuous``
     - None
     - Continuous columns to scale
   * - ``--output-dir``, ``-o``
     - results/ablation
     - Output directory
   * - ``--metric``
     - balanced_accuracy
     - Metric: balanced_accuracy, f1, mcc, roc_auc
   * - ``--feature-groups``
     - None
     - YAML/JSON file defining feature groups
   * - ``--features``
     - All
     - Comma-separated features to ablate
   * - ``--calibrate``
     - False
     - Apply probability calibration
   * - ``--optimize-threshold``
     - False
     - Optimize classification threshold
   * - ``--ablate-steps``
     - False
     - Run pipeline step ablation
   * - ``--n-splits``
     - 3
     - CV folds for ablation
   * - ``--seed``
     - 42
     - Random seed
   * - ``--n-jobs``
     - -1
     - Parallel jobs (-1 for all cores)

Using the Python API
--------------------

Basic Feature Ablation
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from vartrustml.analysis import AblationAnalyzer
   from sklearn.metrics import balanced_accuracy_score
   import pandas as pd

   # Load data and model
   df = pd.read_csv("data.csv")
   X = df.drop(columns=['state'])
   y = df['state'].values

   # Load trained model
   from vartrustml.core.train_model import ModelTrainer
   model_data = ModelTrainer.load_model("trained_model.joblib")
   model = model_data['model']

   # Initialize analyzer
   analyzer = AblationAnalyzer(n_splits=5, seed=42, n_jobs=-1)

   # Run ablation
   results = analyzer.feature_ablation(
       X=X,
       y=y,
       model=model,
       metric_func=balanced_accuracy_score,
       metric_name='Balanced Accuracy'
   )

   # View summary
   print(results.summary_df)

Feature Group Ablation
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Define feature groups
   feature_groups = {
       'caller_features': ['manta_call', 'delly_call', 'lumpy_call'],
       'size_features': ['sv_length', 'sv_size_category'],
       'quality_features': ['mapping_quality', 'base_quality'],
   }

   # Run group ablation
   group_results = analyzer.feature_group_ablation(
       X=X,
       y=y,
       model=model,
       feature_groups=feature_groups,
       metric_func=balanced_accuracy_score,
       metric_name='Balanced Accuracy'
   )

   # Print results
   for r in group_results.results:
       print(f"{r.ablation_name}: Δ={r.delta:.4f} ({r.delta_pct:.1f}%)")

Interpreting Results
--------------------

The ``AblationResult`` dataclass contains:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Field
     - Description
   * - ``ablation_name``
     - Name of the ablated feature/group
   * - ``baseline_score``
     - Performance with all features
   * - ``ablated_score``
     - Performance after removing the feature
   * - ``delta``
     - Score change (negative = feature helps)
   * - ``delta_pct``
     - Percentage change
   * - ``p_value``
     - Statistical significance (paired t-test)
   * - ``is_significant``
     - True if p < 0.05
   * - ``effect_size``
     - Cohen's d effect size

Significant Ablations
^^^^^^^^^^^^^^^^^^^^^

Filter to statistically significant results:

.. code-block:: python

   # Get significant ablations
   significant = results.get_significant_ablations()

   print(f"Found {len(significant)} significant features:")
   for r in significant:
       direction = "hurts" if r.delta < 0 else "improves"
       print(f"  Removing {r.ablation_name} {direction} performance by {abs(r.delta_pct):.1f}%")

Formatting for Reports
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from vartrustml.analysis import format_ablation_study

   # Get formatted report string
   report = format_ablation_study(results)
   print(report)

   # Save to CSV
   results.summary_df.to_csv("ablation_results.csv", index=False)

Complete Example Workflow
-------------------------

A full ablation workflow:

.. code-block:: python

   from vartrustml.analysis import AblationAnalyzer, format_ablation_study
   from vartrustml.core.train_model import ModelTrainer
   from sklearn.metrics import balanced_accuracy_score, roc_auc_score
   import pandas as pd

   # 1. Load your best model from cross-validation
   model_data = ModelTrainer.load_model("results/best_model.joblib")
   model = model_data['model']

   # 2. Load data
   df = pd.read_csv("data/HG002_DEL.csv")
   X = df.drop(columns=['state'])
   y = df['state'].values

   # 3. Ablate against several metrics
   analyzer = AblationAnalyzer(n_splits=10, seed=42)

   # Balanced accuracy ablation
   ba_results = analyzer.feature_ablation(
       X, y, model, balanced_accuracy_score, 'Balanced Accuracy'
   )

   # 4. Define biologically meaningful feature groups
   feature_groups = {
       'Caller Consensus': ['manta_call', 'delly_call', 'lumpy_call', 'svim_call'],
       'Size Features': ['SVLEN_CALLER', 'sv_size_category'],
       'Mapping Quality': ['mean_mapq_inside', 'mean_flank_mapq', 'DELTA_mapq'],
       'Coverage': ['coverage_inside', 'coverage_left', 'coverage_right'],
       'Read Support': ['clipped_ratio', 'split_ratio', 'discordant_ratio'],
   }

   group_results = analyzer.feature_group_ablation(
       X, y, model, feature_groups, balanced_accuracy_score, 'Balanced Accuracy'
   )

   # 5. Report significant findings
   print("=== Feature Ablation ===")
   print(f"Baseline: {ba_results.baseline_score:.4f}")
   print(f"\nSignificant features ({len(ba_results.get_significant_ablations())}):")
   for r in ba_results.get_significant_ablations()[:10]:
       print(f"  {r.ablation_name}: Δ={r.delta:.4f} (p={r.p_value:.4f})")

   print("\n=== Feature Group Ablation ===")
   for r in sorted(group_results.results, key=lambda x: abs(x.delta), reverse=True):
       sig = "*" if r.is_significant else ""
       print(f"  {r.ablation_name}: Δ={r.delta:.4f} ({r.delta_pct:+.1f}%){sig}")

   # 6. Save results to CSV
   ba_results.summary_df.to_csv("results/feature_ablation.csv", index=False)
   group_results.summary_df.to_csv("results/group_ablation.csv", index=False)

See Also
--------

- :doc:`compare-models` - Model comparison with cross-validation
- :doc:`statistical-tests` - Statistical significance testing
- :doc:`training` - Training individual models
