"""Reproducible pipeline: load -> train -> evaluate -> write to results/.

Steps:
1. Load the real dataset from data/ (download it first with download_data.py).
2. Train/test split with a fixed seed.
3. Train the baseline + models, evaluate on the test set (R2, RMSE, MAE).
4. Write results/metrics.json and two figures (EDA + predicted-vs-actual).

Run with:  uv run python scripts/train.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend, suitable for scripts
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
    """Boxplot of costs by smoking status: the clearest separation."""
    fig, ax = plt.subplots(figsize=(7, 5))
    groups = [df.loc[df["smoker"] == v, TARGET].to_numpy() for v in ("no", "yes")]
    ax.boxplot(groups, tick_labels=["non-smoker", "smoker"])
    ax.set_ylabel("Annual medical costs (USD)")
    ax.set_title("Medical costs by smoking status")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_pred_vs_true(y_true, y_pred, name: str, path: Path) -> None:
    """Predicted-vs-actual scatter with the identity line."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=12, alpha=0.5, color="#2c7fb8")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Actual costs (USD)")
    ax.set_ylabel("Predicted costs (USD)")
    ax.set_title(f"Predicted vs actual — {name}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    set_seed(SEED)
    RESULTS_DIR.mkdir(exist_ok=True)

    df = load_insurance(DATA_DIR)
    print(f"Dataset loaded: {len(df)} rows, {df.shape[1]} columns.")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    results: dict[str, dict[str, float]] = {}

    # Baseline: mean of the training target.
    baseline = build_baseline().fit(X_train, y_train)
    results["baseline_mean"] = regression_metrics(y_test, baseline.predict(X_test))

    # Actual models.
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
    print(f"Metrics written to: {metrics_path}")

    _plot_eda(df, RESULTS_DIR / "eda_charges_by_smoker.png")
    _plot_pred_vs_true(
        y_test.to_numpy(),
        best_pred,
        best_name,
        RESULTS_DIR / "pred_vs_true.png",
    )
    print(f"Figures saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
