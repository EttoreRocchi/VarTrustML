=================
Model Calibration
=================

Overview
--------

A calibrated model is one whose probabilities mean what they say: among the
variants it scores at 0.8, close to 80 % should be true positives. VarTrustML
can calibrate any of its models and reports how far from that ideal they land.

Why Calibration Matters
-----------------------

Most of the models here produce probabilities that should not be read at face
value. Random forests and MLPs push scores toward the extremes and end up
overconfident; boosted trees such as XGBoost and CatBoost drift in whichever
direction the class balance pulls them, and drift furthest on the minority
class.

That matters as soon as a probability is used as a number rather than a
ranking: choosing an operating threshold, comparing scores across models,
feeding a downstream risk calculation, or showing a confidence to whoever
reads the call.

.. note::

   Traditional variant callers (e.g., MANTA, DELLY, SMOOVE) produce binary outputs (0/1) rather than probabilities, so calibration does not apply to them.

Calibration Methods
-------------------

VarTrustML supports two calibration methods from scikit-learn:

Sigmoid (Platt Scaling)
^^^^^^^^^^^^^^^^^^^^^^^

Fits a logistic regression to the model's raw outputs. Being parametric, it
stays stable on small calibration sets, at the cost of assuming the
miscalibration really is sigmoidal. Prefer it when calibration data is scarce.

.. code-block:: python

   from vartrustml.config.experiment import CalibrationConfig

   config = ExperimentConfig(
       calibration=CalibrationConfig(
           calibrate_models=True,
           calibration_method="sigmoid",
       )
   )

Isotonic Regression
^^^^^^^^^^^^^^^^^^^

Fits an arbitrary monotonic function instead, so it can correct shapes Platt
scaling cannot. The flexibility costs data: on small calibration sets it
overfits. This is the default, and the right choice whenever folds are large
enough to support it.

.. code-block:: python

   config = ExperimentConfig(
       calibration=CalibrationConfig(
           calibrate_models=True,
           calibration_method="isotonic",  # Default
       )
   )

Configuration
-------------

``calibrate_models`` (default: False)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Enable/disable calibration:

.. tab-set::

   .. tab-item:: Python API

      .. code-block:: python

         config = ExperimentConfig(
             calibration=CalibrationConfig(calibrate_models=True)
         )

   .. tab-item:: CLI

      .. code-block:: bash

         vartrustml compare-models HG002_DEL.csv -t state --calibrate-model

``calibration_method`` (default: "isotonic")
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Choose calibration method:

.. code-block:: bash

   vartrustml compare-models HG002_DEL.csv -t state \
     --calibrate-model \
     --calibration isotonic

``calibration_cv`` (default: 3)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Number of cross-validation folds for calibration:

.. code-block:: bash

   vartrustml compare-models HG002_DEL.csv -t state \
     --calibrate-model \
     --calibration-cv 3

**Recommendations:**

- Standard: 3 folds (fast and reliable for most cases)
- Small datasets: Increase to 5 or 10 for better calibration

Usage Examples
--------------

Model Comparison with Calibration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from vartrustml import ExperimentConfig, CrossValidationPipeline, DataLoader

   # Load data
   loader = DataLoader("data/")
   df = loader.load_dataset("HG002_DEL.csv")

   # Configure with calibration
   config = ExperimentConfig(
       calibration=CalibrationConfig(
           calibrate_models=True,
           calibration_method="isotonic",
           calibration_cv=3,
       ),
       models_to_use=["XGBoost", "Random Forest"]
   )

   # Run
   pipeline = CrossValidationPipeline(config)
   results = pipeline.run_cross_validation(df, "HG002_DEL")

CLI Example
^^^^^^^^^^^

.. code-block:: bash

   vartrustml compare-models data/HG002_DEL.csv -t state \
     --models "XGBoost,Random Forest,CatBoost" \
     --calibrate-model \
     --calibration isotonic \
     --calibration-cv 3

Calibration Quality Metrics
---------------------------

VarTrustML automatically computes calibration quality metrics to assess how well-calibrated your model's probabilities are.

Brier Score
^^^^^^^^^^^

The Brier score measures the mean squared error of predicted probabilities:

.. math::

   BS = \frac{1}{N} \sum_{i=1}^{N} (p_i - y_i)^2

Lower is better, bounded in [0, 1]. A perfect model scores 0; guessing 0.5 on
a balanced problem scores 0.25, which is the number to beat.

Expected Calibration Error (ECE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ECE measures the average gap between predicted confidence and actual accuracy across probability bins:

.. math::

   ECE = \sum_{b=1}^{B} \frac{|B_b|}{N} |acc(B_b) - conf(B_b)|

Lower is better, and 0 means the bins line up exactly. Because it averages
over bins weighted by occupancy, ECE describes the typical prediction, not the
worst one.

Maximum Calibration Error (MCE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MCE measures the worst-case calibration error across all probability bins:

.. math::

   MCE = \max_{b \in \{1, ..., B\}} |acc(B_b) - conf(B_b)|

Lower is better, and 0 again means perfect agreement. MCE reports the single
worst bin rather than the average, so it is the metric to quote when a
decision hangs on the least reliable region of the probability range.

All three are computed for every model and appear in the per-fold metrics CSV,
the model comparison tables, the HTML report, and the reliability diagrams.

Reliability Diagrams
--------------------

A reliability diagram plots mean predicted probability per bin against the
fraction of positives actually observed in that bin. Perfect calibration is
the diagonal, and the area between the curve and the diagonal is the
calibration error made visible. One is generated for every model.

.. code-block:: text

   Output: results/<dataset>/<Model_Name>/<Model_Name>_reliability_diagram.png

The top panel holds the calibration curve, annotated with Brier score, ECE
and MCE. The bottom panel is a histogram of how many predictions fall in each
bin, which is what tells you whether a badly calibrated bin holds enough
variants to worry about.

See Also
--------

- :doc:`threshold-optimization` -- Optimize classification thresholds
- :doc:`compare-models` -- Full model comparison workflow
