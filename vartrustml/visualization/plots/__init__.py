"""
Visualization package for ML pipeline results.

Standalone plotting functions plus a thin Visualizer facade, covering model
performance, feature importance, SHAP values, and error analysis.

See Also
--------
vartrustml.visualization.plots._facade.Visualizer : Facade class.
"""

from vartrustml.visualization.plots._facade import Visualizer

__all__ = ["Visualizer"]
