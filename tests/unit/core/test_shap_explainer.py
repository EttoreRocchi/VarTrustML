"""
Unit tests for SHAPExplainer.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from vartrustml.core.shap_explainer import SHAPExplainer

N_FEATURES = 4


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(80, N_FEATURES)), columns=[f"f{i}" for i in range(N_FEATURES)])
    y = (X["f0"] + 0.3 * X["f1"] > 0).astype(int)
    return X.iloc[:60], X.iloc[60:], y.iloc[:60]


def fit_pipeline(estimator, X, y, calibrated):
    clf = CalibratedClassifierCV(estimator, method="isotonic", cv=3) if calibrated else estimator
    return Pipeline([("preprocessor", StandardScaler()), ("clf", clf)]).fit(X, y)


@pytest.mark.parametrize("calibrated", [False, True])
@pytest.mark.parametrize(
    "name, estimator",
    [
        ("KNN", KNeighborsClassifier(n_neighbors=5)),
        ("MLP", MLPClassifier(hidden_layer_sizes=(8,), max_iter=500, random_state=0)),
        ("Random Forest", RandomForestClassifier(n_estimators=20, random_state=0)),
    ],
)
def test_shap_values_are_positive_class_matrix(data, name, estimator, calibrated):
    X_train, X_test, y_train = data
    model = fit_pipeline(estimator, X_train, y_train, calibrated)
    explainer = SHAPExplainer(shap_cache_enabled=False)
    values = explainer.compute_shap_values(model, name, X_train, X_test, background_sample_size=20)
    assert values is not None
    assert values.shape == (len(X_test), N_FEATURES)


def test_kernel_shap_explains_positive_class(data):
    X_train, X_test, y_train = data
    model = fit_pipeline(KNeighborsClassifier(n_neighbors=5), X_train, y_train, calibrated=False)
    explainer = SHAPExplainer(shap_cache_enabled=False)
    values = explainer.compute_shap_values(model, "KNN", X_train, X_test, background_sample_size=20)
    background = model.named_steps["preprocessor"].transform(X_train)[:20]
    p1 = model.named_steps["clf"].predict_proba(model.named_steps["preprocessor"].transform(X_test))[:, 1]
    base = model.named_steps["clf"].predict_proba(background)[:, 1].mean()
    np.testing.assert_allclose(values.sum(axis=1) + base, p1, atol=1e-6)
