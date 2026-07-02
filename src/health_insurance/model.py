"""Modelli, baseline e preprocessing per la regressione delle spese sanitarie.

Ogni modello è una ``Pipeline`` scikit-learn che incapsula il preprocessing
(one-hot sulle categoriche, standardizzazione delle numeriche dove serve) più
lo stimatore. Questo garantisce che la trasformazione sia appresa solo sul
train e riapplicata al test senza leakage.
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
    """Imposta i seed globali per riproducibilità. Restituisce il seed usato."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed


def _preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """Costruisce il preprocessing: one-hot sulle categoriche, scaling opzionale.

    Lo scaling delle numeriche giova ai modelli lineari; per gli ensemble
    basati su alberi è ininfluente, quindi lo si disattiva per chiarezza.
    """
    numeric = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def build_models(seed: int = 0) -> dict[str, Pipeline]:
    """Restituisce i modelli da confrontare, come pipeline pronte al fit.

    - ``linear_regression``: regressione lineare (feature scalate).
    - ``random_forest``: ensemble di alberi, cattura le non linearità e le
      interazioni (es. fumo × BMI).
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
    """Baseline non informativa: predice sempre la media del target di train.

    Serve come riferimento minimo: qualunque modello utile deve batterla.
    """
    return Pipeline(
        steps=[
            ("prep", _preprocessor(scale_numeric=False)),
            ("model", DummyRegressor(strategy="mean")),
        ]
    )
