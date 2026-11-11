# health-insurance

> **University coursework — archived.** Kept as an example of a clean regression pipeline with a baseline; not maintained.

> Small **ML project on clinical data**: regression of **individual medical costs** (Medical Cost Personal dataset) with a reproducible pipeline, a baseline and a Streamlit app.

Predicts `charges` (annual medical costs, USD) from demographic and lifestyle variables: age, sex, BMI, number of children, smoking status, region. It started as a rework of an old exercise and was restructured to follow the portfolio conventions (uv + `src/` + tests + CI + reproducible pipeline).

## What it shows

- **scikit-learn pipeline without leakage**: preprocessing (one-hot on categorical features, scaling on numeric ones) is wrapped in the `Pipeline` and fitted on the training set only.
- **Comparison against a baseline**: every model is measured against a `DummyRegressor(strategy="mean")`, so the gain is quantified rather than just claimed.
- **Reproducibility**: fixed seeds, `uv.lock` under version control, data never committed but downloadable with an idempotent script.
- **Offline tests**: the test suite runs on a synthetic dataset with known ground truth, so CI does not need network access.

## Pipeline

```
download_data.py         train.py
   (data/insurance.csv)     |
        |                    v
        |            load_insurance ──> train/test split (seed=42)
        v                    |
   [ raw data ]              v
                    ColumnTransformer (one-hot + scaling)
                             |
              ┌──────────────┼───────────────┐
              v              v                v
        baseline_mean   LinearRegression  RandomForest
              └──────────────┼───────────────┘
                             v
                 metrics (R2, RMSE, MAE) ──> results/metrics.json
                             + figures ────> results/*.png
                             |
                             v
                    Streamlit app (app/main.py)
```

## Data and licenses

- **Name**: Medical Cost Personal Dataset (`insurance.csv`), 1338 rows.
- **Target**: `charges` (annual medical costs in USD).
- **Features**: `age`, `sex`, `bmi`, `children`, `smoker`, `region`.
- **Source**: dataset from the book *Machine Learning with R* (Brett Lantz), publicly redistributed in the repository [`stedy/Machine-Learning-with-R-datasets`](https://github.com/stedy/Machine-Learning-with-R-datasets).
- **Data licence**: the dataset has its own terms (see the source); the code in this repo is MIT. The data are **not** committed: they are downloaded into `data/` (gitignored) with the provided script.

## How to run (uv)

```bash
uv sync --extra dev                          # creates .venv (Python 3.12) and installs everything
uv run python scripts/download_data.py       # downloads insurance.csv into data/ (idempotent)
uv run python scripts/train.py               # trains, evaluates, writes results/
uv run ruff check .                          # lint
uv run pytest -q                             # tests (offline, synthetic data)
uv run streamlit run app/main.py             # app showing metrics + figures
```

## Results

Test-set metrics (20% of the data, seed=42), from `results/metrics.json`:

| Model | R2 | RMSE (USD) | MAE (USD) |
|---|---|---|---|
| baseline (mean) | -0.001 | 12466 | 9593 |
| Linear Regression | 0.784 | 5796 | 4181 |
| Random Forest | **0.863** | **4609** | **2529** |

The Random Forest is the best model: it can capture non-linearities and the smoking × BMI interaction. Both models clearly beat the baseline (R2 ≈ 0), which stands for "predicting nothing useful".

Figures generated in `results/`:
- `eda_charges_by_smoker.png` — smokers' costs are much higher (the dominant signal).
- `pred_vs_true.png` — predicted vs actual for the best model.

## Limitations

- Small dataset (1338 rows), synthetic but realistic in structure: the results are for teaching purposes, not clinically actionable.
- No systematic hyperparameter tuning and no nested cross-validation: models are compared with reasonable default settings, on a single train/test split.
- `charges` is strongly skewed and driven by smoking status; the other features explain a residual share of the variance.
- No fairness/bias analysis: variables such as `sex` and `region` are used for demonstration of prediction only.
- The dataset contains 1 duplicated row out of 1338; with a random 80/20 split it may end up on both sides, a negligible effect.
