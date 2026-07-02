"""Mini-progetto ML clinico: predizione delle spese sanitarie individuali.

Regressione del target `charges` del dataset Medical Cost Personal a partire da
variabili demografiche e di stile di vita (age, sex, bmi, children, smoker, region).

Espone le funzioni principali della pipeline: caricamento/generazione dei dati,
costruzione dei modelli e metriche di valutazione.
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
