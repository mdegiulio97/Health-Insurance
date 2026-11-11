"""Evaluation metrics for the medical-cost regression."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Compute R2, RMSE and MAE between actual and predicted values.

    - ``r2``: share of variance explained (1.0 = perfect, 0.0 = same as the mean).
    - ``rmse``: root mean squared error, in target units (USD).
    - ``mae``: mean absolute error, robust to outliers, in USD.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }
