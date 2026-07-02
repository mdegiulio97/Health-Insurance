"""Smoke test OFFLINE: girano su dati sintetici, senza rete (compatibili CI).

Verificano che: il generatore sintetico sia coerente e riproducibile, il
preprocessing/pipeline funzioni end-to-end, le metriche siano ben formate e i
modelli battano la baseline su una relazione nota.
"""

from __future__ import annotations

import numpy as np

from health_insurance import (
    FEATURES,
    TARGET,
    build_baseline,
    build_models,
    make_synthetic,
    regression_metrics,
    set_seed,
)


def test_synthetic_schema_and_reproducibility():
    a = make_synthetic(n=200, seed=1)
    b = make_synthetic(n=200, seed=1)
    assert set(a.columns) == set(FEATURES + [TARGET])
    assert len(a) == 200
    assert a.equals(b)  # stesso seed -> stesso dataset


def test_set_seed_is_reproducible():
    set_seed(42)
    x = np.random.rand(5)
    set_seed(42)
    y = np.random.rand(5)
    assert np.allclose(x, y)


def test_metrics_are_well_formed():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    m = regression_metrics(y_true, y_true)
    assert set(m) == {"r2", "rmse", "mae"}
    assert m["r2"] == 1.0
    assert m["rmse"] == 0.0
    assert m["mae"] == 0.0


def test_baseline_pipeline_runs():
    df = make_synthetic(n=300, seed=0)
    X, y = df[FEATURES], df[TARGET]
    baseline = build_baseline().fit(X, y)
    pred = baseline.predict(X)
    assert pred.shape == (300,)


def test_models_beat_baseline_on_known_signal():
    df = make_synthetic(n=600, seed=7)
    X, y = df[FEATURES], df[TARGET]
    split = 480
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    baseline = build_baseline().fit(X_train, y_train)
    base_r2 = regression_metrics(y_test, baseline.predict(X_test))["r2"]

    for name, model in build_models(seed=0).items():
        model.fit(X_train, y_train)
        r2 = regression_metrics(y_test, model.predict(X_test))["r2"]
        assert r2 > base_r2, f"{name} non batte la baseline (r2={r2:.3f})"
        assert r2 > 0.5, f"{name} non recupera il segnale noto (r2={r2:.3f})"
