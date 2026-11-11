"""Small clinical ML project: predicting individual medical costs.

Regression of the `charges` target of the Medical Cost Personal dataset from
demographic and lifestyle variables (age, sex, bmi, children, smoker, region).

Exposes the main pipeline functions: data loading/generation, model
construction and evaluation metrics.
"""

from health_insurance.data import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    load_insurance,
    make_synthetic,
)
from health_insurance.evaluate import regression_metrics
from health_insurance.model import build_baseline, build_models, set_seed

__all__ = [
    "CATEGORICAL_FEATURES",
    "FEATURES",
    "NUMERIC_FEATURES",
    "TARGET",
    "build_baseline",
    "build_models",
    "load_insurance",
    "make_synthetic",
    "regression_metrics",
    "set_seed",
]
