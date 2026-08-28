"""
Missing-value (NaN) handling utilities.

VarTrustML supports two families of NaN-handling strategies, selected via the
``nan_strategy`` config field (CLI: ``--nan-strategy``):

- **Impute in pipeline** (``"median"``, ``"mean"``, ``"most_frequent"``): a
  :class:`sklearn.impute.SimpleImputer` is added *before* scaling inside the
  per-fold preprocessing pipeline, so imputation is fit only on training data
  (no leakage) and applies uniformly to every model.
- **Drop** (``"drop"``): rows containing any missing feature value are removed
  from the dataset before cross-validation.

Tree models (XGBoost, CatBoost) tolerate NaN natively, but MLP / KNN /
Logistic Regression do not; a strategy is therefore needed for a fair,
all-models comparison.
"""

from typing import List, Optional, Tuple

import pandas as pd

#: Strategies handled by imputation inside the modelling pipeline.
IMPUTE_STRATEGIES: Tuple[str, ...] = ("median", "mean", "most_frequent")

#: All supported NaN-handling strategies.
NAN_STRATEGIES: Tuple[str, ...] = IMPUTE_STRATEGIES + ("drop",)


def make_imputer(strategy: str):
    """Return a :class:`SimpleImputer` for an impute strategy, else ``None``.

    Parameters
    ----------
    strategy : str
        One of :data:`NAN_STRATEGIES`. For ``"drop"`` (or anything not in
        :data:`IMPUTE_STRATEGIES`) this returns ``None`` (no in-pipeline imputer).

    Returns
    -------
    sklearn.impute.SimpleImputer or None
    """
    if strategy in IMPUTE_STRATEGIES:
        from sklearn.impute import SimpleImputer

        return SimpleImputer(strategy=strategy)
    return None


def drop_missing_rows(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, int]:
    """Drop rows that contain any missing value in the feature columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    target_column : str, optional
        Target column name; excluded when ``feature_cols`` is not given so that
        a missing target (which should not happen) does not silently drop rows
        for the wrong reason.
    feature_cols : list of str, optional
        Explicit feature columns to check. Defaults to every column except
        ``target_column``.

    Returns
    -------
    tuple of (pandas.DataFrame, int)
        The filtered dataframe (index reset) and the number of dropped rows.
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_column]
    feature_cols = [c for c in feature_cols if c in df.columns]

    before = len(df)
    out = df.dropna(subset=feature_cols).reset_index(drop=True)
    return out, before - len(out)
