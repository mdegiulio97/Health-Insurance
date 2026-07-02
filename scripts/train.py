"""Pipeline riproducibile: carica -> addestra -> valuta -> scrive in results/.

Passi:
1. Carica il dataset reale da data/ (scaricalo prima con download_data.py).
2. Split train/test con seed fisso.
3. Addestra baseline + modelli, valuta su test (R2, RMSE, MAE).
4. Scrive results/metrics.json e due figure (EDA + predetto-vs-reale).

Esegui con:  uv run python scripts/train.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend non interattivo, adatto agli script
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from health_insurance import (
    TARGET,
    build_baseline,
    build_models,
    load_insurance,
    regression_metrics,
    set_seed,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
SEED = 42


def _plot_eda(df, path: Path) -> None:
    """Boxplot delle spese per status di fumatore: la separazione più netta."""
    fig, ax = plt.subplots(figsize=(7, 5))
    groups = [df.loc[df["smoker"] == v, TARGET].to_numpy() for v in ("no", "yes")]
    ax.boxplot(groups, tick_labels=["non fumatore", "fumatore"])
    ax.set_ylabel("Spese sanitarie annue (USD)")
    ax.set_title("Spese sanitarie per status di fumatore")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_pred_vs_true(y_true, y_pred, name: str, path: Path) -> None:
    """Scatter predetto-vs-reale con la bisettrice ideale."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=12, alpha=0.5, color="#2c7fb8")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Spese reali (USD)")
    ax.set_ylabel("Spese predette (USD)")
    ax.set_title(f"Predetto vs reale — {name}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    set_seed(SEED)
    RESULTS_DIR.mkdir(exist_ok=True)

    df = load_insurance(DATA_DIR)
    print(f"Dataset caricato: {len(df)} righe, {df.shape[1]} colonne.")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    results: dict[str, dict[str, float]] = {}

    # Baseline: media del target di train.
    baseline = build_baseline().fit(X_train, y_train)
    results["baseline_mean"] = regression_metrics(y_test, baseline.predict(X_test))

    # Modelli veri e propri.
    best_name, best_r2, best_pred = None, -float("inf"), None
    for name, model in build_models(seed=SEED).items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = regression_metrics(y_test, pred)
        print(f"{name:20s} R2={results[name]['r2']:.3f}  RMSE={results[name]['rmse']:.0f}")
        if results[name]["r2"] > best_r2:
            best_name, best_r2, best_pred = name, results[name]["r2"], pred

    metrics_path = RESULTS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Metriche scritte in: {metrics_path}")

    _plot_eda(df, RESULTS_DIR / "eda_charges_by_smoker.png")
    _plot_pred_vs_true(
        y_test.to_numpy(),
        best_pred,
        best_name,
        RESULTS_DIR / "pred_vs_true.png",
    )
    print(f"Figure salvate in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
