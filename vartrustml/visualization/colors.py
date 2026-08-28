"""
Fixed colour assignments for ML models.

Centralising the palette here guarantees that every figure draws a given model
in the *same* colour (e.g. CatBoost is always the same blue in the ROC/PR plot,
the boxplot comparison, etc.), so the plots can be cross-referenced at a glance.

Keys are the registered model display names (see
``vartrustml.config.experiment.SUPPORTED_MODELS``).
"""

from typing import Dict

#: Stable, well-separated colours (matplotlib "tab10" family), one per model.
MODEL_COLORS: Dict[str, str] = {
    "CatBoost": "#1f77b4",  # blue
    "XGBoost": "#ff7f0e",  # orange
    "Random Forest": "#2ca02c",  # green
    "MLP": "#d62728",  # red
    "Logistic Regression": "#9467bd",  # purple
    "KNN": "#8c564b",  # brown
}

#: Fallback colour for any name not in :data:`MODEL_COLORS`.
DEFAULT_MODEL_COLOR = "#7f8c8d"  # grey


def model_color(name: str) -> str:
    """Return the fixed colour for a model name (grey fallback if unknown)."""
    return MODEL_COLORS.get(name, DEFAULT_MODEL_COLOR)


def model_palette(names) -> Dict[str, str]:
    """Build a ``{name: colour}`` palette for the given model names."""
    return {n: model_color(n) for n in names}
