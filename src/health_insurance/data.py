"""Loading of the real dataset and a synthetic generator with ground truth.

Two data sources:

- ``load_insurance(data_dir)`` reads the real CSV (Medical Cost Personal)
  previously downloaded into ``data/`` by ``scripts/download_data.py``.
- ``make_synthetic(n, seed)`` generates a dataset with the same columns and a
  KNOWN relationship between features and target. It is used by the tests, which
  therefore run fully offline (CI has no network access).

Dataset columns: age, sex, bmi, children, smoker, region, charges.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

#: Name of the CSV file expected in the data folder.
CSV_NAME = "insurance.csv"

#: Target column (regression): annual medical costs in USD.
TARGET = "charges"

#: Numeric features.
NUMERIC_FEATURES = ["age", "bmi", "children"]

#: Categorical features (one-hot encoded in the pipeline).
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]

#: All features used as model input.
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

#: Regions of the original dataset.
REGIONS = ["southwest", "southeast", "northwest", "northeast"]


def load_insurance(data_dir: str | Path) -> pd.DataFrame:
    """Load the real ``insurance.csv`` dataset from the ``data_dir`` folder.

    Returns a DataFrame with the canonical columns. Raises an explicit error
    if the file is missing (download it first with
    ``scripts/download_data.py``).
    """
    path = Path(data_dir) / CSV_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"'{path}' not found. First run 'uv run python scripts/download_data.py' "
            "to download the dataset (the data are not committed)."
        )
    df = pd.read_csv(path)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in the CSV: {sorted(missing)}")
    return df[FEATURES + [TARGET]].copy()


def make_synthetic(n: int = 1338, seed: int = 0) -> pd.DataFrame:
    """Generate a synthetic dataset with the same structure as the real one.

    The ``charges`` target is built from a known relationship (ground truth):
    it grows with age and BMI, is strongly amplified by smoking status and is
    perturbed by Gaussian noise. The tests use it to check that the pipeline
    learns a real signal without depending on the network.
    """
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 65, size=n)
    bmi = np.round(rng.normal(30.0, 6.0, size=n).clip(15, 53), 1)
    children = rng.integers(0, 6, size=n)
    sex = rng.choice(["male", "female"], size=n)
    smoker = rng.choice(["yes", "no"], size=n, p=[0.2, 0.8])
    region = rng.choice(REGIONS, size=n)

    smoker_flag = (smoker == "yes").astype(float)
    # Ground truth: linear terms + a strong smoking/BMI interaction effect.
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
