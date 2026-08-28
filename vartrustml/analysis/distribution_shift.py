"""
Distribution shift between datasets, measured per variable type.

Continuous features use the two-sample Kolmogorov-Smirnov statistic
(sup distance between empirical CDFs); binary/categorical features use the
absolute difference in positive proportion (the total-variation distance for a
Bernoulli). Both live on [0, 1], so a continuous KS and a binary proportion gap
are directly comparable. The target (class prior) shift is reported separately.
"""

from typing import Dict, Optional, Sequence, Tuple

import pandas as pd
from scipy.stats import ks_2samp

KS = "KS"
PROP_DIFF = "abs_prop_diff"


def _is_binary(s: pd.Series) -> bool:
    """True only for genuine 0/1 columns.

    Cardinality alone is not enough: the proportion-difference metric is a
    total-variation distance only when the values are 0 and 1, and is unbounded
    for any other two-valued encoding.
    """
    values = pd.unique(s.dropna())
    return len(values) <= 2 and set(values).issubset({0, 1})


def class_prior_table(
    datasets: Sequence[Tuple[pd.DataFrame, str]], target_column: str
) -> pd.DataFrame:
    """Per-dataset positive rate and class counts."""
    rows = []
    for df, name in datasets:
        y = df[target_column]
        rows.append(
            {
                "dataset": name,
                "n": int(len(y)),
                "n_pos": int((y == 1).sum()),
                "n_neg": int((y == 0).sum()),
                "positive_rate": float((y == 1).mean()),
            }
        )
    return pd.DataFrame(rows).set_index("dataset")


def _feature_types(
    datasets: Sequence[Tuple[pd.DataFrame, str]],
    feature_cols: Sequence[str],
    continuous_cols: Optional[Sequence[str]],
) -> Dict[str, bool]:
    """Map feature -> is_continuous, decided across all datasets.

    A column counts as binary only when it is genuinely 0/1 in every dataset,
    so a column that is merely absent from ``continuous_cols`` never falls into
    the proportion-difference branch by default. Listing a column in
    ``continuous_cols`` forces it to continuous, but omitting one does not
    force it to binary. Deciding on the union rather than on the first dataset
    keeps a column that is constant in one sample from being typed binary for
    every pair.
    """
    cont = set(continuous_cols or [])
    types = {}
    for col in feature_cols:
        if col in cont:
            types[col] = True
            continue
        present = [df[col] for df, _ in datasets if col in df.columns]
        types[col] = not (present and all(_is_binary(s) for s in present))
    return types


def compute_pairwise_feature_shift(
    datasets: Sequence[Tuple[pd.DataFrame, str]],
    feature_cols: Sequence[str],
    continuous_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Long-form per-feature shift for every dataset pair.

    Returns columns: dataset_a, dataset_b, feature, var_type, metric, shift.
    """
    types = _feature_types(datasets, feature_cols, continuous_cols)
    rows = []
    for i in range(len(datasets)):
        for j in range(i + 1, len(datasets)):
            df_a, name_a = datasets[i]
            df_b, name_b = datasets[j]
            for col in feature_cols:
                a = df_a[col].dropna().to_numpy()
                b = df_b[col].dropna().to_numpy()
                if a.size == 0 or b.size == 0:
                    continue
                if types[col]:
                    shift = float(ks_2samp(a, b).statistic)
                    metric = KS
                    var_type = "continuous"
                else:
                    shift = float(abs(a.mean() - b.mean()))
                    metric = PROP_DIFF
                    var_type = "binary"
                rows.append(
                    {
                        "dataset_a": name_a,
                        "dataset_b": name_b,
                        "feature": col,
                        "var_type": var_type,
                        "metric": metric,
                        "shift": shift,
                    }
                )
    return pd.DataFrame(rows)


def compute_distribution_shift(
    datasets: Sequence[Tuple[pd.DataFrame, str]],
    target_column: str,
    feature_cols: Optional[Sequence[str]] = None,
    continuous_cols: Optional[Sequence[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Class-prior shift + per-feature pairwise shift (type-appropriate)."""
    if feature_cols is None:
        feature_cols = [c for c in datasets[0][0].columns if c != target_column]
    return {
        "class_priors": class_prior_table(datasets, target_column),
        "feature_shift": compute_pairwise_feature_shift(
            datasets, feature_cols, continuous_cols
        ),
    }


def plot_feature_shift_heatmap(
    feature_shift: pd.DataFrame,
    save_path: Optional[str] = None,
    return_base64: bool = False,
) -> Optional[str]:
    """Heatmap of per-feature shift (rows) across dataset pairs (columns).

    Continuous features (KS) are grouped above binary features (prop-diff),
    separated by a line; both share the [0, 1] colour scale.
    """
    import base64
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if feature_shift.empty:
        return None

    fs = feature_shift.copy()
    fs["pair"] = fs["dataset_a"] + " vs " + fs["dataset_b"]
    type_rank = {"continuous": 0, "binary": 1}
    order_df = fs[["feature", "var_type"]].drop_duplicates()
    order_df = order_df.assign(_r=order_df["var_type"].map(type_rank)).sort_values(
        ["_r", "feature"]
    )
    feat_order = order_df["feature"].tolist()
    mat = fs.pivot(index="feature", columns="pair", values="shift").reindex(feat_order)
    n_cont = int((order_df["var_type"] == "continuous").sum())
    rows, cols = mat.shape

    fig, ax = plt.subplots(figsize=(1.7 * cols + 4, 0.34 * rows + 1.5))
    im = ax.imshow(mat.values, cmap="magma", aspect="auto", vmin=0.0, vmax=1.0)
    ax.grid(False)
    ax.set_xticks(range(cols))
    ax.set_xticklabels(mat.columns, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(rows))
    ax.set_yticklabels(mat.index, fontsize=8)
    if 0 < n_cont < rows:
        ax.axhline(n_cont - 0.5, color="white", linewidth=2)
    for i in range(rows):
        for j in range(cols):
            v = mat.values[i, j]
            if pd.notna(v):
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if v < 0.6 else "black",
                )
    ax.set_title("Per-feature distribution shift between samples")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("shift (KS continuous, prop-diff binary)")
    fig.tight_layout()

    encoded = None
    if return_base64:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return encoded
