=====================
Visualization Modules
=====================

Modules for generating static plots and interactive HTML reports.

Visualizer
----------

Static plot generation using Matplotlib and Seaborn.

.. autoclass:: vartrustml.visualization.plots.Visualizer
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

   from vartrustml.visualization.plots import Visualizer

   visualizer = Visualizer(output_dir="results/plots")

   # Plot confusion matrix
   visualizer.plot_confusion_matrix(
       y_true, y_pred,
       title="XGBoost Confusion Matrix",
       filename="confusion_matrix.png"
   )

   # Plot feature importance
   visualizer.plot_feature_importance(
       importance_dict,
       title="Feature Importance",
       filename="feature_importance.png"
   )

HTMLCompareReporter
-------------------

Interactive HTML report generation for model comparison.

.. autoclass:: vartrustml.visualization.html_compare_reporter.HTMLCompareReporter
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

   from vartrustml.visualization.html_compare_reporter import HTMLCompareReporter

   reporter = HTMLCompareReporter(output_path="report.html")

   # Add experiment overview
   reporter.add_overview(config, dataset_stats)

   # Add model results
   for model_name, results in cv_results.items():
       reporter.add_model_results(model_name, results)

   # Add confusion matrices
   reporter.add_confusion_matrices(confusion_data)

   # Generate report
   report_path = reporter.generate_report()
   print(f"Report saved to: {report_path}")

HTMLTrainReporter
-----------------

Interactive HTML report generation for training workflows.

.. autoclass:: vartrustml.visualization.html_train_reporter.HTMLTrainReporter
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

   from vartrustml.visualization.html_train_reporter import HTMLTrainReporter

   reporter = HTMLTrainReporter(output_path="training_report.html")

   # Add training overview
   reporter.add_training_overview(train_config)

   # Add hyperparameter results
   reporter.add_hyperparameter_results(cv_results)

   # Add test metrics (if available)
   reporter.add_test_metrics(test_results)

   # Generate report
   report_path = reporter.generate_report()

HTMLCrossDatasetReporter
------------------------

Interactive HTML report generation for cross-dataset evaluation.

.. autoclass:: vartrustml.visualization.html_cross_dataset_reporter.HTMLCrossDatasetReporter
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

   from vartrustml.visualization.html_cross_dataset_reporter import HTMLCrossDatasetReporter

   reporter = HTMLCrossDatasetReporter(output_path="cross_dataset_report.html")

   # Add experiment overview
   reporter.add_overview(config_dict, training_info, test_info)

   # Add cross-dataset results
   reporter.add_cross_dataset_results(results)

   # Generate report
   report_path = reporter.generate_report()
