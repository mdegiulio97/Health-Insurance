# health-insurance

> Mini-progetto di **ML su dati clinici**: regressione delle **spese sanitarie individuali** (dataset Medical Cost Personal) con pipeline riproducibile, baseline e app Streamlit.

Predice `charges` (spesa sanitaria annua, USD) a partire da variabili demografiche e di stile di vita: età, sesso, BMI, numero di figli, status di fumatore, regione. Nato dalla rielaborazione di un vecchio esercizio, è ristrutturato secondo lo standard del portfolio (uv + `src/` + test + CI + pipeline riproducibile).

## Cosa dimostra

- **Pipeline scikit-learn senza leakage**: preprocessing (one-hot sulle categoriche, scaling sulle numeriche) incapsulato nella `Pipeline`, appreso solo sul train.
- **Confronto contro baseline**: ogni modello è misurato contro un `DummyRegressor(strategy="mean")`, così il guadagno è quantificato e non solo asserito.
- **Riproducibilità**: seed fissi, `uv.lock` versionato, dati mai committati ma scaricabili in modo idempotente.
- **Test offline**: la suite gira su un dataset sintetico con ground truth nota, quindi la CI non richiede rete.

## Pipeline

```
download_data.py         train.py
   (data/insurance.csv)     |
        |                    v
        |            load_insurance ──> split train/test (seed=42)
        v                    |
   [ dati grezzi ]           v
                    ColumnTransformer (one-hot + scaling)
                             |
              ┌──────────────┼───────────────┐
              v              v                v
        baseline_mean   LinearRegression  RandomForest
              └──────────────┼───────────────┘
                             v
                 metriche (R2, RMSE, MAE) ──> results/metrics.json
                             + figure ──────> results/*.png
                             |
                             v
                    app Streamlit (app/main.py)
```

## Dataset

- **Nome**: Medical Cost Personal Dataset (`insurance.csv`), 1338 righe.
- **Target**: `charges` (spesa sanitaria annua in USD).
- **Feature**: `age`, `sex`, `bmi`, `children`, `smoker`, `region`.
- **Fonte**: dataset del libro *Machine Learning with R* (Brett Lantz), redistribuito pubblicamente nel repository [`stedy/Machine-Learning-with-R-datasets`](https://github.com/stedy/Machine-Learning-with-R-datasets).
- **Licenza dati**: il dataset ha termini propri (vedi la fonte); il codice di questo repo è MIT. I dati **non** sono committati: si scaricano con lo script fornito in `data/` (gitignored).

## Come eseguire (uv)

```bash
uv sync --extra dev                          # crea .venv (Python 3.12) e installa tutto
uv run python scripts/download_data.py       # scarica insurance.csv in data/ (idempotente)
uv run python scripts/train.py               # addestra, valuta, scrive results/
uv run ruff check .                          # lint
uv run pytest -q                             # test (offline, dati sintetici)
uv run streamlit run app/main.py             # app che mostra metriche + figure
```

## Risultati

Metriche sul test set (20% dei dati, seed=42), da `results/metrics.json`:

| Modello | R2 | RMSE (USD) | MAE (USD) |
|---|---|---|---|
| baseline (media) | -0.001 | 12466 | 9593 |
| Linear Regression | 0.784 | 5796 | 4181 |
| Random Forest | **0.863** | **4609** | **2529** |

Il Random Forest è il modello migliore: cattura le non linearità e l'interazione fumo×BMI. Entrambi i modelli battono nettamente la baseline (R2 ≈ 0), che rappresenta il "non predire nulla di utile".

Figure generate in `results/`:
- `eda_charges_by_smoker.png` — le spese dei fumatori sono nettamente più alte (segnale dominante).
- `pred_vs_true.png` — predetto vs reale del modello migliore.

## Limiti

- Dataset piccolo (1338 righe) e sintetico-realistico nella struttura: i risultati sono didattici, non clinicamente attuabili.
- Nessun tuning sistematico degli iperparametri né validazione incrociata annidata: si confrontano modelli con impostazioni ragionevoli di default.
- `charges` è fortemente asimmetrica e guidata dallo status di fumatore; il resto delle feature spiega una quota residua di varianza.
- Nessuna nozione di equità/bias: variabili come `sex` e `region` sono usate solo a scopo predittivo dimostrativo.
