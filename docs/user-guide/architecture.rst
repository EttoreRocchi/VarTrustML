==================
Architecture Guide
==================

This guide provides an overview of VarTrustML's architecture, explaining how the components fit together and how data flows through the system.

Pipeline Overview
-----------------

VarTrustML implements a nested cross-validation pipeline for rigorous model evaluation. The architecture follows a sequential workflow:

.. graphviz::

   digraph pipeline {
       rankdir=TB;
       node [shape=box, style="rounded,filled", fillcolor="#f5f5f5", fontname="sans-serif"];
       edge [fontname="sans-serif", fontsize=10];

       // Configuration
       ExperimentConfig [label="ExperimentConfig\n(seed, CV splits, models, options)"];

       // Main pipeline
       CrossValidationPipeline [label="CrossValidationPipeline\n(orchestrates the workflow)"];

       // Training phase
       subgraph cluster_training {
           label="Training Phase (per outer fold)";
           labelloc="b";
           fontcolor="#0066CC";
           fontname="sans-serif";
           style=dashed;
           color="#666666";
           margin="16";

           ModelEvaluator [label="ModelEvaluator"];
           InnerCV [label="Inner CV\n(hyperparameter tuning)"];
           FitModel [label="Fit best model\non training fold"];
           Calibration [label="Calibration\n(CalibratedClassifierCV)", style="rounded,filled,dashed", fillcolor="#ffffcc"];
           ThresholdOpt [label="Threshold Optimization\n(Youden's J)", style="rounded,filled,dashed", fillcolor="#ffffcc"];
       }

       // Evaluation phase
       subgraph cluster_evaluation {
           label="Evaluation Phase (per outer fold)";
           labelloc="b";
           fontcolor="#0066CC";
           fontname="sans-serif";
           style=dashed;
           color="#666666";
           margin="16";

           Evaluate [label="Evaluate on\ntest fold"];
           FoldMetrics [label="FoldMetrics\n(predictions, metrics, SHAP)"];
       }

       // Aggregation phase
       Aggregate [label="Aggregate results\nacross all folds"];
       CallerEvaluator [label="CallerEvaluator\n(optional: per-caller analysis)", style="rounded,filled,dashed", fillcolor="#ffffcc"];

       // Post-processing phase
       subgraph cluster_postprocessing {
           label="Post-Processing";
           labelloc="b";
           fontcolor="#0066CC";
           fontname="sans-serif";
           style=dashed;
           color="#666666";
           margin="16";

           PairwiseComparison [label="compare_pairwise\n(McNemar / DeLong tests)"];
           BootstrapAnalyzer [label="BootstrapAnalyzer\n(confidence intervals)"];
           Visualizer [label="Visualizer\n(static plots)"];
           HTMLReporter [label="HTMLCompareReporter\n(interactive report)"];
       }

       // Flow
       ExperimentConfig -> CrossValidationPipeline;
       CrossValidationPipeline -> ModelEvaluator;
       ModelEvaluator -> InnerCV;
       InnerCV -> FitModel;
       FitModel -> Calibration [style=dashed, label="optional"];
       Calibration -> ThresholdOpt [style=dashed];
       FitModel -> Evaluate [label="  or direct"];
       ThresholdOpt -> Evaluate [style=dashed];
       Evaluate -> FoldMetrics;
       FoldMetrics -> Aggregate [label="repeat for\nall folds"];
       Aggregate -> CallerEvaluator [style=dashed, label="optional"];
       CallerEvaluator -> PairwiseComparison [style=dashed];
       Aggregate -> PairwiseComparison;
       PairwiseComparison -> BootstrapAnalyzer;
       BootstrapAnalyzer -> Visualizer;
       Visualizer -> HTMLReporter;
   }


Core Components
---------------

ExperimentConfig
^^^^^^^^^^^^^^^^

The central configuration dataclass that controls all pipeline behavior. It stores settings for:

- Cross-validation parameters (n_outer_splits, n_inner_splits)
- Model selection and hyperparameter optimization
- Calibration and threshold optimization
- Report generation

.. code-block:: python

   from vartrustml import ExperimentConfig
   from vartrustml.config.experiment import CVConfig, CalibrationConfig, ThresholdConfig

   config = ExperimentConfig(
       cv=CVConfig(seed=42, n_outer_splits=10, n_inner_splits=5),
       models_to_use=["XGBoost", "Random Forest"],
       calibration=CalibrationConfig(calibrate_models=True),
       threshold=ThresholdConfig(optimize_threshold=True),
   )

CrossValidationPipeline
^^^^^^^^^^^^^^^^^^^^^^^

The main orchestration class that coordinates the entire workflow:

1. **Data Validation**: Checks target variable for binary labels
2. **Stratified Splitting**: Creates outer CV folds with class balance
3. **Model Training**: Delegates to ModelEvaluator for each fold
4. **Result Aggregation**: Computes statistics across folds
5. **Report Generation**: Creates CSV, HTML, and visualization outputs

ModelEvaluator
^^^^^^^^^^^^^^

Handles individual model training within each fold:

1. **Preprocessing Pipeline**: StandardScaler for continuous features
2. **Hyperparameter Search**: GridSearchCV or OptunaSearchCV
3. **Calibration** (optional): CalibratedClassifierCV wrapper
4. **Threshold Optimization** (optional): Youden's J on training data
5. **Evaluation**: metrics computed on the held-out test fold
6. **Interpretability**: SHAP values for feature importance


Data Flow
---------

Input Processing
^^^^^^^^^^^^^^^^

.. graphviz::

   digraph input_processing {
       rankdir=TB;
       node [shape=box, style="rounded,filled", fillcolor="#f5f5f5", fontname="sans-serif"];
       edge [fontname="sans-serif", fontsize=10];

       RawData [label="Raw CSV/DataFrame"];
       DataLoader [label="DataLoader.load_dataset()"];

       subgraph cluster_validation {
           label="Validation & Preprocessing";
           labelloc="b";
           fontcolor="#0066CC";
           fontname="sans-serif";
           style=dashed;
           color="#666666";
           margin="16";

           Validation [label="Data validation\n(check columns, types)"];
           Missing [label="Missing value\nhandling"];
           Types [label="Type conversions\n(numeric, categorical)"];
       }

       Ready [label="Preprocessed DataFrame\n(ready for CV pipeline)"];

       RawData -> DataLoader;
       DataLoader -> Validation;
       Validation -> Missing;
       Missing -> Types;
       Types -> Ready;
   }


Training Pipeline
^^^^^^^^^^^^^^^^^

For each outer fold:

.. graphviz::

   digraph training_pipeline {
       rankdir=TB;
       node [shape=box, style="rounded,filled", fillcolor="#f5f5f5", fontname="sans-serif"];
       edge [fontname="sans-serif", fontsize=10];

       TrainingData [label="Training Data\n(Outer Fold k)"];

       subgraph cluster_hpo {
           label="Hyperparameter Optimization";
           labelloc="b";
           fontcolor="#0066CC";
           fontname="sans-serif";
           style=dashed;
           color="#666666";
           margin="16";

           InnerCV [label="Inner CV\n(GridSearchCV / OptunaSearchCV)"];
           BestParams [label="Best hyperparameters"];
       }

       FitModel [label="Fit model on\nfull training fold"];

       subgraph cluster_optional {
           label="Optional Enhancements";
           labelloc="b";
           fontcolor="#0066CC";
           fontname="sans-serif";
           style=dashed;
           color="#666666";
           margin="16";

           Calibration [label="Calibration\n(CalibratedClassifierCV)", style="rounded,filled,dashed", fillcolor="#ffffcc"];
           Threshold [label="Threshold Optimization\n(Youden's J)", style="rounded,filled,dashed", fillcolor="#ffffcc"];
       }

       subgraph cluster_eval {
           label="Evaluation on Test Fold";
           labelloc="b";
           fontcolor="#0066CC";
           fontname="sans-serif";
           style=dashed;
           color="#666666";
           margin="16";

           Evaluate [label="Generate predictions\n& probabilities"];
           Metrics [label="Compute metrics\n(AUROC, MCC, F1, etc.)"];
           SHAP [label="SHAP values\n(feature importance)"];
       }

       FoldMetrics [label="FoldMetrics\n(store results)"];

       TrainingData -> InnerCV;
       InnerCV -> BestParams;
       BestParams -> FitModel;
       FitModel -> Calibration [style=dashed, label="optional"];
       Calibration -> Threshold [style=dashed];
       FitModel -> Evaluate [label="  or direct  "];
       Threshold -> Evaluate [style=dashed];
       Evaluate -> Metrics;
       Evaluate -> SHAP;
       Metrics -> FoldMetrics;
       SHAP -> FoldMetrics;
   }


Result Aggregation
^^^^^^^^^^^^^^^^^^

.. graphviz::

   digraph result_aggregation {
       rankdir=TB;
       node [shape=box, style="rounded,filled", fillcolor="#f5f5f5", fontname="sans-serif"];
       edge [fontname="sans-serif", fontsize=10];

       FoldMetrics [label="FoldMetrics\n(from all K outer folds)"];

       subgraph cluster_aggregate {
           label="Aggregation";
           labelloc="b";
           fontcolor="#0066CC";
           fontname="sans-serif";
           style=dashed;
           color="#666666";
           margin="16";

           Aggregate [label="Aggregate across folds\n(per model)"];
           Stats [label="Summary statistics\n(mean, std, median, min, max)"];
       }

       subgraph cluster_analysis {
           label="Analysis (parallel)";
           labelloc="b";
           fontcolor="#0066CC";
           fontname="sans-serif";
           style=dashed;
           color="#666666";
           margin="16";

           StatAnalyzer [label="compare_pairwise\n(McNemar / DeLong, effect sizes)"];
           Bootstrap [label="BootstrapAnalyzer\n(95% CI for metrics)"];
       }

       subgraph cluster_output {
           label="Output Generation";
           labelloc="b";
           fontcolor="#0066CC";
           fontname="sans-serif";
           style=dashed;
           color="#666666";
           margin="16";

           Visualizer [label="Visualizer\n(confusion matrices, ROC, SHAP)"];
           HTMLReporter [label="HTMLCompareReporter\n(interactive report)"];
       }

       FoldMetrics -> Aggregate;
       Aggregate -> Stats;
       Stats -> StatAnalyzer;
       Stats -> Bootstrap;
       StatAnalyzer -> Visualizer;
       Bootstrap -> Visualizer;
       Visualizer -> HTMLReporter;
   }


Analysis Components
-------------------

BootstrapAnalyzer
^^^^^^^^^^^^^^^^^

Computes confidence intervals without normality assumptions:

- **Prediction-level resampling**: Resamples the pooled out-of-fold predictions
- **BCa interval**: Bias-corrected and accelerated by default (or percentile)

.. code-block:: python

   from vartrustml.analysis.bootstrap import BootstrapAnalyzer

   analyzer = BootstrapAnalyzer(n_iterations=1000, ci_level=0.95, ci_method="bca")
   ci_results = analyzer.compute_all_cis_from_predictions(y_true, y_pred, y_prob)

Pairwise comparison
^^^^^^^^^^^^^^^^^^^

Paired hypothesis testing between classifiers on pooled out-of-fold predictions:

- **McNemar's test**: Paired comparison at a fixed operating point
- **DeLong's test**: AUROC comparison between ML models
- **Holm-Bonferroni / Benjamini-Hochberg**: selectable FWER or FDR correction across the comparison family
- **Paired effect sizes**: Accuracy difference with CI, discordant odds ratio

.. code-block:: python

   from vartrustml.analysis.pairwise_comparison import (
       build_entities, compare_pairwise,
   )

   entities = build_entities(oof_predictions, caller_results)
   result = compare_pairwise(entities, primary_metric="Matthews Corr. Coef.")


Visualization Components
------------------------

Visualizer
^^^^^^^^^^

Static plot generation using Matplotlib/Seaborn:

- Confusion matrices
- ROC curves
- Feature importance plots
- Calibration plots

HTMLCompareReporter
^^^^^^^^^^^^^^^^^^^

Interactive HTML reports with:

- Model comparison tables
- Embedded visualizations
- Statistical test results
- Expandable fold details


Cross-Dataset Evaluation
------------------------

For testing model generalization across different datasets:

.. graphviz::

   digraph cross_dataset {
       rankdir=TB;
       node [shape=box, style="rounded,filled", fillcolor="#f5f5f5", fontname="sans-serif"];
       edge [fontname="sans-serif", fontsize=10];

       subgraph cluster_datasets {
           label="Input Datasets";
           labelloc="b";
           fontcolor="#0066CC";
           fontname="sans-serif";
           style=dashed;
           color="#666666";
           margin="16";

           DatasetA [label="Dataset A\n(e.g., HG002_DEL)"];
           DatasetB [label="Dataset B\n(e.g., HG002_INS)"];
           DatasetC [label="Dataset C\n(e.g., HG002_DUP)"];
       }

       AlignedCV [label="Aligned CV Splits\n(same seed for fair comparison)", style="rounded,filled", fillcolor="#ffffcc"];

       subgraph cluster_training {
           label="Cross-Dataset Training";
           labelloc="b";
           fontcolor="#0066CC";
           fontname="sans-serif";
           style=dashed;
           color="#666666";
           margin="16";

           Evaluator [label="CrossDatasetEvaluator"];
           TrainTest [label="Train on each dataset\nTest on all datasets"];
       }

       subgraph cluster_results {
           label="Results";
           labelloc="b";
           fontcolor="#0066CC";
           fontname="sans-serif";
           style=dashed;
           color="#666666";
           margin="16";

           Matrix [label="N x N Performance Matrix\n(train rows × test cols)"];
           Gap [label="Generalization gap analysis\n(diagonal vs off-diagonal)"];
       }

       subgraph cluster_output {
           label="Output";
           labelloc="b";
           fontcolor="#0066CC";
           fontname="sans-serif";
           style=dashed;
           color="#666666";
           margin="16";

           Bootstrap [label="BootstrapAnalyzer\n(confidence intervals)"];
           Reporter [label="HTMLCrossDatasetReporter\n(interactive report)"];
       }

       DatasetA -> AlignedCV;
       DatasetB -> AlignedCV;
       DatasetC -> AlignedCV;
       AlignedCV -> Evaluator;
       Evaluator -> TrainTest;
       TrainTest -> Matrix;
       Matrix -> Gap;
       Gap -> Bootstrap;
       Bootstrap -> Reporter;
   }

**CrossDatasetEvaluator produces:**

- Performance matrices (N × N: each train dataset vs each test dataset)
- **Aligned CV splits**: same fold indices across all datasets for fair comparison
- Generalization gap analysis (diagonal vs off-diagonal)
- Bootstrap confidence intervals for cross-dataset metrics
- HTMLCrossDatasetReporter output


Directory Structure
-------------------

Typical output organization:

.. code-block:: text

   results/
   └── dataset_name/
       ├── experiment_config.json
       ├── model_metrics_comparison.csv
       ├── report.html
       ├── XGBoost/
       │   ├── metrics_summary.csv
       │   ├── best_parameters.csv
       │   ├── error_analysis_summary.csv
       │   ├── all_misclassified_samples.csv
       │   └── folds/
       │       └── fold_0/
       │           └── ...
       └── Random_Forest/
           └── ...


Checkpoint System
-----------------

VarTrustML supports checkpoint/resume for long-running experiments:

.. code-block:: text

   checkpoints/
   └── dataset_name/
       └── XGBoost/
           ├── fold_0/
           │   └── fold_0_model.joblib
           ├── fold_1/
           │   └── fold_1_model.joblib
           └── ...

Checkpoints are saved after each fold completes, allowing interrupted experiments to resume without recomputing completed folds.

Enable with:

.. code-block:: python

   config = ExperimentConfig(
       save_checkpoints=True,
       checkpoint_dir="checkpoints"
   )


Best Practices
--------------

1. **Start with default models**: Use the default ``models_to_use`` list before customizing

2. **Enable calibration for probability estimates**: Set ``calibration=CalibrationConfig(calibrate_models=True)`` if you need well-calibrated probabilities

3. **Use threshold optimization carefully**: Only enable when you have a specific decision threshold need; default 0.5 works well for many cases

4. **Monitor memory usage**: Large datasets with many models can consume significant memory; consider reducing ``n_jobs`` if needed

5. **Use checkpoints for large experiments**: Enable ``save_checkpoints=True`` for experiments that may need to be resumed
