"""
Ablation study framework for systematic feature and component analysis.

:class:`AblationAnalyzer` measures the contribution of individual features,
feature groups, and pipeline components by removing them and re-scoring.

Examples
--------
Feature ablation study with model object:

>>> from vartrustml.analysis.ablation import AblationAnalyzer
>>> analyzer = AblationAnalyzer(n_splits=5, seed=42)
>>> results = analyzer.feature_ablation(
...     X, y, model, features_to_ablate=['feature1', 'feature2'],
...     metric_func=balanced_accuracy_score, metric_name='Balanced Accuracy'
... )
>>> for r in results:
...     print(f"{r.ablation_name}: delta={r.delta:.4f}, p={r.p_value:.4f}")

For config-driven ablation with fresh model training, see
:class:`~vartrustml.analysis.ablation_config.ConfigAblationAnalyzer`.
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


@dataclass
class AblationResult:
    """Container for single ablation result.

    Stores the performance comparison between baseline and ablated conditions.

    Attributes
    ----------
    ablation_name : str
        Name identifying what was ablated (feature name, component name, etc.).
    baseline_score : float
        Mean cross-validation score with all features/components.
    baseline_std : float
        Standard deviation of baseline scores across folds.
    ablated_score : float
        Mean cross-validation score after ablation.
    ablated_std : float
        Standard deviation of ablated scores across folds.
    delta : float
        Change in performance (ablated - baseline). Negative means ablation hurt.
    delta_pct : float
        Percent change in performance relative to baseline.
    p_value : float
        P-value from paired t-test comparing fold scores.
    is_significant : bool
        Whether the difference is statistically significant at alpha level.
    effect_size : float
        Cohen's d effect size for the difference.
    baseline_scores : List[float]
        Per-fold baseline scores.
    ablated_scores : List[float]
        Per-fold ablated scores.
    metric_name : str
        Name of the metric used.
    alpha : float
        Significance level used for testing.

    Notes
    -----
    A negative delta indicates that removing the feature/component decreased
    performance, suggesting it was beneficial. A positive delta indicates
    the ablated component may have been harmful or redundant.
    """

    ablation_name: str
    baseline_score: float
    baseline_std: float
    ablated_score: float
    ablated_std: float
    delta: float
    delta_pct: float
    p_value: float
    is_significant: bool
    effect_size: float
    baseline_scores: List[float]
    ablated_scores: List[float]
    metric_name: str
    alpha: float = 0.05
    p_value_corrected: Optional[float] = None


@dataclass
class AblationStudyResult:
    """Container for complete ablation study results.

    Attributes
    ----------
    study_type : str
        Type of ablation study ('feature', 'feature_group', 'component').
    results : List[AblationResult]
        Individual ablation results.
    baseline_score : float
        Overall baseline performance.
    metric_name : str
        Metric used for evaluation.
    n_splits : int
        Number of CV splits used.
    seed : int
        Random seed for reproducibility.
    summary_df : pd.DataFrame
        Summary DataFrame with all results.
    """

    study_type: str
    results: List[AblationResult]
    baseline_score: float
    metric_name: str
    n_splits: int
    seed: int
    summary_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self):
        """Generate summary DataFrame if not provided."""
        if self.summary_df.empty and self.results:
            self.summary_df = self._create_summary_df()

    def _create_summary_df(self) -> pd.DataFrame:
        """Create summary DataFrame from results."""
        data = []
        for r in self.results:
            row = {
                "ablation_name": r.ablation_name,
                "baseline_score": r.baseline_score,
                "ablated_score": r.ablated_score,
                "delta": r.delta,
                "delta_pct": r.delta_pct,
                "p_value": r.p_value,
                "p_value_corrected": r.p_value_corrected,
                "is_significant": r.is_significant,
                "effect_size": r.effect_size,
            }
            data.append(row)
        df = pd.DataFrame(data)
        # Sort by absolute delta (most impactful first)
        df = df.sort_values("delta", key=abs, ascending=False)
        return df

    def get_significant_ablations(self) -> List[AblationResult]:
        """Return only statistically significant ablation results."""
        return [r for r in self.results if r.is_significant]

    def get_top_k_features(self, k: int = 10) -> List[AblationResult]:
        """Return top-k most impactful ablations by absolute delta."""
        sorted_results = sorted(self.results, key=lambda x: abs(x.delta), reverse=True)
        return sorted_results[:k]


class AblationAnalyzer:
    """Analyzer for systematic ablation studies.

    Conducts ablation studies to measure the contribution of features
    and components to model performance using cross-validation.

    Parameters
    ----------
    n_splits : int, default=3
        Number of cross-validation splits. Default is 3 for faster ablation
        studies while maintaining statistical validity.
    seed : int, default=42
        Random seed for reproducibility.
    alpha : float, default=0.05
        Significance level for statistical tests.
    n_jobs : int, default=-1
        Number of parallel jobs for cross-validation. Default is -1 to use
        all available cores for faster execution.

    Attributes
    ----------
    n_splits : int
        Number of CV splits.
    seed : int
        Random seed.
    alpha : float
        Significance level.
    n_jobs : int
        Number of parallel jobs.

    Examples
    --------
    Using a pre-existing model object:

    >>> analyzer = AblationAnalyzer(n_splits=3, seed=42)
    >>> results = analyzer.feature_ablation(
    ...     X, y, model,
    ...     metric_func=balanced_accuracy_score,
    ...     metric_name='Balanced Accuracy'
    ... )
    """

    def __init__(
        self,
        n_splits: int = 3,
        seed: int = 42,
        alpha: float = 0.05,
        n_jobs: int = -1,
    ):
        """Initialize the ablation analyzer."""
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        self.n_splits = n_splits
        self.seed = seed
        self.alpha = alpha
        self.n_jobs = n_jobs
        self._cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def _apply_holm_correction(
        self, results: List[AblationResult]
    ) -> List[AblationResult]:
        """Apply Holm-Bonferroni correction to ablation p-values.

        Controls the family-wise error rate (FWER) when testing multiple
        features simultaneously. Updates ``p_value_corrected`` and
        ``is_significant`` on each result in-place.

        Parameters
        ----------
        results : list of AblationResult
            Ablation results with uncorrected p-values.

        Returns
        -------
        list of AblationResult
            Same results with corrected p-values and updated significance.

        References
        ----------
        - Holm, S. (1979). A simple sequentially rejective multiple test
          procedure. Scandinavian Journal of Statistics, 6(2), 65-70.
        """
        if not results:
            return results

        # NaN p-values (paired differences with zero variance) carry no
        # evidence: they are excluded from the correction rather than ranked,
        # otherwise max(cumulative_max, nan) returns the left operand and the
        # NaN both becomes a corrected p of 0.0 and consumes a Holm rank slot
        indexed = [
            (r.p_value, i)
            for i, r in enumerate(results)
            if r.p_value is not None and np.isfinite(r.p_value)
        ]

        corrected = [float("nan")] * len(results)

        n = len(indexed)
        if n == 1:
            corrected[indexed[0][1]] = indexed[0][0]
        elif n > 1:
            indexed.sort(key=lambda x: x[0])

            # Apply Holm correction with monotonicity enforcement
            cumulative_max = 0.0
            for rank, (p, orig_idx) in enumerate(indexed):
                adjusted = p * (n - rank)
                cumulative_max = max(cumulative_max, adjusted)
                corrected[orig_idx] = min(cumulative_max, 1.0)

        # Update results
        for i, result in enumerate(results):
            result.p_value_corrected = corrected[i]
            result.is_significant = bool(
                np.isfinite(corrected[i]) and corrected[i] < self.alpha
            )

        return results

    def _compute_cv_scores(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: np.ndarray,
        model: BaseEstimator,
        metric_func: Callable,
    ) -> np.ndarray:
        """Compute cross-validation scores for a model.

        Parameters
        ----------
        X : DataFrame or ndarray
            Feature matrix.
        y : ndarray
            Target labels.
        model : BaseEstimator
            Scikit-learn compatible model.
        metric_func : callable
            Scoring function (y_true, y_pred) -> float.

        Returns
        -------
        ndarray
            Array of per-fold scores.
        """
        scores = []
        for train_idx, test_idx in self._cv.split(X, y):
            if isinstance(X, pd.DataFrame):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            else:
                X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Clone and fit model
            model_clone = clone(model)
            model_clone.fit(X_train, y_train)
            y_pred = model_clone.predict(X_test)

            score = metric_func(y_test, y_pred)
            scores.append(score)

        return np.array(scores)

    def _compute_ablation_result(
        self,
        ablation_name: str,
        baseline_scores: np.ndarray,
        ablated_scores: np.ndarray,
        metric_name: str,
    ) -> AblationResult:
        """Compute ablation result from baseline and ablated scores.

        Parameters
        ----------
        ablation_name : str
            Name of what was ablated.
        baseline_scores : ndarray
            Per-fold baseline scores.
        ablated_scores : ndarray
            Per-fold ablated scores.
        metric_name : str
            Name of the metric.

        Returns
        -------
        AblationResult
            Complete ablation result with statistics.
        """
        baseline_mean = float(np.mean(baseline_scores))
        baseline_std = float(np.std(baseline_scores, ddof=1))
        ablated_mean = float(np.mean(ablated_scores))
        ablated_std = float(np.std(ablated_scores, ddof=1))

        delta = ablated_mean - baseline_mean
        delta_pct = (delta / baseline_mean * 100) if baseline_mean != 0 else 0.0

        # Paired t-test
        if len(baseline_scores) > 1:
            n = len(baseline_scores)
            if n < 5:
                logger.warning(
                    f"Paired t-test for '{ablation_name}' has only {n - 1} degrees "
                    f"of freedom (n_splits={n}). Statistical power is extremely low "
                    f"and the normality assumption on paired differences is "
                    f"untestable at this sample size. Consider using n_splits >= 5."
                )
            t_stat, p_value = stats.ttest_rel(ablated_scores, baseline_scores)
            p_value = float(p_value)
        else:
            p_value = 1.0

        # Cohen's d_av effect size (average variance pooling).
        # Uses d_av = delta / sqrt((sd1^2 + sd2^2) / 2) rather than d_z
        # (which uses SD of paired differences). d_av is a conservative
        # choice for paired designs as it does not benefit from within-pair
        # correlation. See Lakens (2013) for discussion of effect size
        # variants in paired designs.
        pooled_std = np.sqrt((baseline_std**2 + ablated_std**2) / 2)
        effect_size = delta / pooled_std if pooled_std > 0 else 0.0

        return AblationResult(
            ablation_name=ablation_name,
            baseline_score=baseline_mean,
            baseline_std=baseline_std,
            ablated_score=ablated_mean,
            ablated_std=ablated_std,
            delta=delta,
            delta_pct=delta_pct,
            p_value=p_value,
            is_significant=p_value < self.alpha,
            effect_size=float(effect_size),
            baseline_scores=baseline_scores.tolist(),
            ablated_scores=ablated_scores.tolist(),
            metric_name=metric_name,
            alpha=self.alpha,
        )

    def feature_ablation(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        model: BaseEstimator,
        metric_func: Callable,
        metric_name: str = "metric",
        features_to_ablate: Optional[List[str]] = None,
    ) -> AblationStudyResult:
        """Perform leave-one-out feature ablation study.

        For each feature, trains the model without that feature and measures
        the impact on performance.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with named columns.
        y : ndarray
            Target labels.
        model : BaseEstimator
            Scikit-learn compatible model.
        metric_func : callable
            Scoring function (y_true, y_pred) -> float.
        metric_name : str, default="metric"
            Name of the metric for reporting.
        features_to_ablate : List[str], optional
            Specific features to ablate. If None, ablates all features.

        Returns
        -------
        AblationStudyResult
            Complete ablation study results.

        Examples
        --------
        >>> results = analyzer.feature_ablation(
        ...     X, y, model,
        ...     metric_func=balanced_accuracy_score,
        ...     metric_name='Balanced Accuracy',
        ...     features_to_ablate=['COVERAGE', 'MAPQ']
        ... )
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame for feature ablation")

        features = features_to_ablate or list(X.columns)
        logger.info(f"Starting feature ablation study with {len(features)} features")

        logger.info("Computing baseline scores with all features")
        baseline_scores = self._compute_cv_scores(X, y, model, metric_func)

        results = []
        for i, feature in enumerate(features):
            logger.debug(f"Ablating feature {i + 1}/{len(features)}: {feature}")

            X_ablated = X.drop(columns=[feature])

            ablated_scores = self._compute_cv_scores(X_ablated, y, model, metric_func)

            result = self._compute_ablation_result(
                ablation_name=feature,
                baseline_scores=baseline_scores,
                ablated_scores=ablated_scores,
                metric_name=metric_name,
            )
            results.append(result)

        logger.info(f"Feature ablation complete. {len(results)} features analyzed.")

        self._apply_holm_correction(results)

        return AblationStudyResult(
            study_type="feature",
            results=results,
            baseline_score=float(np.mean(baseline_scores)),
            metric_name=metric_name,
            n_splits=self.n_splits,
            seed=self.seed,
        )

    def feature_group_ablation(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        model: BaseEstimator,
        feature_groups: Dict[str, List[str]],
        metric_func: Callable,
        metric_name: str = "metric",
    ) -> AblationStudyResult:
        """Perform feature group ablation study.

        For each feature group, trains the model without that group and
        measures the impact on performance.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with named columns.
        y : ndarray
            Target labels.
        model : BaseEstimator
            Scikit-learn compatible model.
        feature_groups : Dict[str, List[str]]
            Dictionary mapping group names to lists of feature names.
        metric_func : callable
            Scoring function (y_true, y_pred) -> float.
        metric_name : str, default="metric"
            Name of the metric for reporting.

        Returns
        -------
        AblationStudyResult
            Complete ablation study results.

        Examples
        --------
        >>> feature_groups = {
        ...     'coverage': ['COVERAGE', 'coverage_left', 'coverage_right'],
        ...     'mapping_quality': ['mean_mapq_inside', 'mean_flank_mapq'],
        ... }
        >>> results = analyzer.feature_group_ablation(
        ...     X, y, model, feature_groups,
        ...     metric_func=balanced_accuracy_score
        ... )
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame for feature group ablation")

        logger.info(
            f"Starting feature group ablation study with {len(feature_groups)} groups"
        )

        logger.info("Computing baseline scores with all features")
        baseline_scores = self._compute_cv_scores(X, y, model, metric_func)

        results = []
        for group_name, features in feature_groups.items():
            # Validate features exist
            missing = [f for f in features if f not in X.columns]
            if missing:
                logger.warning(
                    f"Group '{group_name}' contains missing features: {missing}"
                )
                features = [f for f in features if f in X.columns]
                if not features:
                    continue

            logger.debug(f"Ablating group '{group_name}': {features}")

            X_ablated = X.drop(columns=features)

            ablated_scores = self._compute_cv_scores(X_ablated, y, model, metric_func)

            result = self._compute_ablation_result(
                ablation_name=group_name,
                baseline_scores=baseline_scores,
                ablated_scores=ablated_scores,
                metric_name=metric_name,
            )
            results.append(result)

        logger.info(f"Feature group ablation complete. {len(results)} groups analyzed.")

        self._apply_holm_correction(results)

        return AblationStudyResult(
            study_type="feature_group",
            results=results,
            baseline_score=float(np.mean(baseline_scores)),
            metric_name=metric_name,
            n_splits=self.n_splits,
            seed=self.seed,
        )

    def component_ablation(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: np.ndarray,
        model_configs: Dict[str, BaseEstimator],
        metric_func: Callable,
        metric_name: str = "metric",
        baseline_name: str = "full_pipeline",
    ) -> AblationStudyResult:
        """Perform component ablation study.

        Compares different model configurations to understand the contribution
        of various pipeline components (e.g., calibration, threshold optimization).

        Parameters
        ----------
        X : DataFrame or ndarray
            Feature matrix.
        y : ndarray
            Target labels.
        model_configs : Dict[str, BaseEstimator]
            Dictionary mapping configuration names to models.
            Must include a baseline configuration.
        metric_func : callable
            Scoring function (y_true, y_pred) -> float.
        metric_name : str, default="metric"
            Name of the metric for reporting.
        baseline_name : str, default="full_pipeline"
            Name of the baseline configuration in model_configs.

        Returns
        -------
        AblationStudyResult
            Complete ablation study results.

        Examples
        --------
        >>> model_configs = {
        ...     'full_pipeline': CalibratedClassifierCV(RandomForestClassifier()),
        ...     'no_calibration': RandomForestClassifier(),
        ...     'no_feature_scaling': Pipeline([('clf', RandomForestClassifier())]),
        ... }
        >>> results = analyzer.component_ablation(
        ...     X, y, model_configs,
        ...     metric_func=balanced_accuracy_score
        ... )
        """
        if baseline_name not in model_configs:
            raise ValueError(
                f"Baseline '{baseline_name}' not found in model_configs. "
                f"Available: {list(model_configs.keys())}"
            )

        logger.info(
            f"Starting component ablation study with {len(model_configs)} configurations"
        )

        baseline_model = model_configs[baseline_name]
        logger.info(f"Computing baseline scores with '{baseline_name}'")
        baseline_scores = self._compute_cv_scores(X, y, baseline_model, metric_func)

        results = []
        for config_name, model in model_configs.items():
            if config_name == baseline_name:
                continue

            logger.debug(f"Testing configuration: {config_name}")

            ablated_scores = self._compute_cv_scores(X, y, model, metric_func)

            result = self._compute_ablation_result(
                ablation_name=config_name,
                baseline_scores=baseline_scores,
                ablated_scores=ablated_scores,
                metric_name=metric_name,
            )
            results.append(result)

        logger.info(
            f"Component ablation complete. {len(results)} configurations analyzed."
        )

        self._apply_holm_correction(results)

        return AblationStudyResult(
            study_type="component",
            results=results,
            baseline_score=float(np.mean(baseline_scores)),
            metric_name=metric_name,
            n_splits=self.n_splits,
            seed=self.seed,
        )
