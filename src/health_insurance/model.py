"""Models, baseline and preprocessing for the medical-cost regression.

Each model is a scikit-learn ``Pipeline`` that wraps the preprocessing
(one-hot on categorical features, standardisation of numeric ones where useful)
together with the estimator. This ensures the transformation is fitted on the
training set only and re-applied to the test set without leakage.
"""

from __future__ import annotations

import os
import random

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from health_insurance.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def set_seed(seed: int = 0) -> int:
    """Set the global seeds for reproducibility. Returns the seed used."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed


def _preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """Build the preprocessing: one-hot on categorical features, optional scaling.

    Scaling the numeric features helps linear models; for tree-based
    ensembles it makes no difference, so it is turned off for clarity.
    """
    numeric = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def build_models(seed: int = 0) -> dict[str, Pipeline]:
    """Return the models to compare, as pipelines ready to fit.

    - ``linear_regression``: linear regression (scaled features).
    - ``random_forest``: tree ensemble, captures non-linearities and
      interactions (e.g. smoking × BMI).
    """
    return {
        "linear_regression": Pipeline(
            steps=[
                ("prep", _preprocessor(scale_numeric=True)),
                ("model", LinearRegression()),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("prep", _preprocessor(scale_numeric=False)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=None,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def build_baseline() -> Pipeline:
    """Non-informative baseline: always predicts the mean of the training target.

    It serves as the minimum reference: any useful model has to beat it.
    """
    return Pipeline(
        steps=[
            ("prep", _preprocessor(scale_numeric=False)),
            ("model", DummyRegressor(strategy="mean")),
        ]
    )
