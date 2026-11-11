"""Streamlit app: shows the metrics and figures produced by the pipeline.

Reads the artifacts from results/ (produced by scripts/train.py) and displays them.
Run with:  uv run streamlit run app/main.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from health_insurance import set_seed

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

st.set_page_config(page_title="Health Insurance — Medical costs", layout="wide")

set_seed(0)

st.title("🏥 Predicting individual medical costs")
st.write(
    "Regression of the `charges` target (annual medical costs) from the "
    "Medical Cost Personal dataset, using age, sex, BMI, number of children, "
    "smoking status and region."
)

metrics_path = RESULTS_DIR / "metrics.json"
if not metrics_path.exists():
    st.warning(
        "No results found. Generate the artifacts with:\n\n"
        "```bash\n"
        "uv run python scripts/download_data.py\n"
        "uv run python scripts/train.py\n"
        "```"
    )
    st.stop()

metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

st.subheader("Test-set metrics")
table = pd.DataFrame(metrics).T[["r2", "rmse", "mae"]]
table = table.rename(columns={"r2": "R2", "rmse": "RMSE (USD)", "mae": "MAE (USD)"})
st.dataframe(table.style.format({"R2": "{:.3f}", "RMSE (USD)": "{:.0f}", "MAE (USD)": "{:.0f}"}))

st.caption("`baseline_mean` always predicts the mean: any useful model has to beat it.")

col1, col2 = st.columns(2)
eda_fig = RESULTS_DIR / "eda_charges_by_smoker.png"
pred_fig = RESULTS_DIR / "pred_vs_true.png"
if eda_fig.exists():
    col1.image(str(eda_fig), caption="EDA: costs by smoking status")
if pred_fig.exists():
    col2.image(str(pred_fig), caption="Predicted vs actual (best model)")
