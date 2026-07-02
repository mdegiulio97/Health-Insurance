"""App Streamlit: mostra metriche e figure generate dalla pipeline.

Legge gli artefatti da results/ (prodotti da scripts/train.py) e li presenta.
Avvia con:  uv run streamlit run app/main.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from health_insurance import set_seed

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

st.set_page_config(page_title="Health Insurance — Spese sanitarie", layout="wide")

set_seed(0)

st.title("🏥 Predizione delle spese sanitarie individuali")
st.write(
    "Regressione del target `charges` (spesa sanitaria annua) dal dataset "
    "Medical Cost Personal, a partire da età, sesso, BMI, numero di figli, "
    "status di fumatore e regione."
)

metrics_path = RESULTS_DIR / "metrics.json"
if not metrics_path.exists():
    st.warning(
        "Nessun risultato trovato. Genera gli artefatti con:\n\n"
        "```bash\n"
        "uv run python scripts/download_data.py\n"
        "uv run python scripts/train.py\n"
        "```"
    )
    st.stop()

metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

st.subheader("Metriche sul test set")
table = pd.DataFrame(metrics).T[["r2", "rmse", "mae"]]
table = table.rename(columns={"r2": "R2", "rmse": "RMSE (USD)", "mae": "MAE (USD)"})
st.dataframe(table.style.format({"R2": "{:.3f}", "RMSE (USD)": "{:.0f}", "MAE (USD)": "{:.0f}"}))

st.caption("Il `baseline_mean` predice sempre la media: ogni modello utile deve batterlo.")

col1, col2 = st.columns(2)
eda_fig = RESULTS_DIR / "eda_charges_by_smoker.png"
pred_fig = RESULTS_DIR / "pred_vs_true.png"
if eda_fig.exists():
    col1.image(str(eda_fig), caption="EDA: spese per status di fumatore")
if pred_fig.exists():
    col2.image(str(pred_fig), caption="Predetto vs reale (modello migliore)")
