"""Metriche di valutazione per la regressione delle spese sanitarie."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Calcola R2, RMSE e MAE fra valori reali e predetti.

    - ``r2``: quota di varianza spiegata (1.0 = perfetto, 0.0 = come la media).
    - ``rmse``: errore quadratico medio radice, nelle unità del target (USD).
    - ``mae``: errore assoluto medio, robusto agli outlier, in USD.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }
