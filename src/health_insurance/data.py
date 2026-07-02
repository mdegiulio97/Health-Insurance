"""Caricamento del dataset reale e generatore sintetico con ground truth.

Due sorgenti di dati:

- ``load_insurance(data_dir)`` legge il CSV reale (Medical Cost Personal)
  precedentemente scaricato in ``data/`` da ``scripts/download_data.py``.
- ``make_synthetic(n, seed)`` genera un dataset con le stesse colonne e una
  relazione NOTA fra feature e target. Serve ai test, che così girano
  completamente offline (la CI non ha accesso di rete).

Colonne del dataset: age, sex, bmi, children, smoker, region, charges.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

#: Nome del file CSV atteso nella cartella dati.
CSV_NAME = "insurance.csv"

#: Colonna target (regressione): spesa sanitaria annua in USD.
TARGET = "charges"

#: Feature numeriche.
NUMERIC_FEATURES = ["age", "bmi", "children"]

#: Feature categoriche (verranno codificate one-hot nella pipeline).
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]

#: Tutte le feature usate come input del modello.
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

#: Regioni del dataset originale.
REGIONS = ["southwest", "southeast", "northwest", "northeast"]


def load_insurance(data_dir: str | Path) -> pd.DataFrame:
    """Carica il dataset reale ``insurance.csv`` dalla cartella ``data_dir``.

    Restituisce un DataFrame con le colonne canoniche. Solleva un errore
    esplicito se il file non è presente (va scaricato prima con
    ``scripts/download_data.py``).
    """
    path = Path(data_dir) / CSV_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"'{path}' non trovato. Esegui prima 'uv run python scripts/download_data.py' "
            "per scaricare il dataset (i dati non sono committati)."
        )
    df = pd.read_csv(path)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Colonne mancanti nel CSV: {sorted(missing)}")
    return df[FEATURES + [TARGET]].copy()


def make_synthetic(n: int = 1338, seed: int = 0) -> pd.DataFrame:
    """Genera un dataset sintetico con la stessa struttura di quello reale.

    Il target ``charges`` è costruito da una relazione nota (ground truth):
    cresce con età e BMI, è fortemente amplificato dallo status di fumatore e
    perturbato da rumore gaussiano. Serve ai test per verificare che la
    pipeline apprenda un segnale reale senza dipendere dalla rete.
    """
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 65, size=n)
    bmi = np.round(rng.normal(30.0, 6.0, size=n).clip(15, 53), 1)
    children = rng.integers(0, 6, size=n)
    sex = rng.choice(["male", "female"], size=n)
    smoker = rng.choice(["yes", "no"], size=n, p=[0.2, 0.8])
    region = rng.choice(REGIONS, size=n)

    smoker_flag = (smoker == "yes").astype(float)
    # Ground truth: contributi lineari + forte effetto interazione fumo/BMI.
    charges = (
        250.0 * age
        + 340.0 * bmi
        + 450.0 * children
        + 20000.0 * smoker_flag
        + 1200.0 * smoker_flag * (bmi - 30.0)
        + rng.normal(0.0, 3000.0, size=n)
    )
    charges = np.clip(charges, 1000.0, None)

    return pd.DataFrame(
        {
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region,
            "charges": np.round(charges, 2),
        }
    )
