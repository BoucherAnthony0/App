"""Tests de l'API FastAPI : /health et /predict (avec modèle injecté)."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

import app.service as service
from app.api import app
from src.models.model_factory import build_full_pipeline


@pytest.fixture(scope="module")
def client(synthetic_df, config):
    """Entraîne un petit modèle et l'injecte dans le service (pas de fichier requis)."""
    numerical = [c for c in config["features"]["numerical"] if c in synthetic_df.columns]
    categorical = config["features"]["categorical"]
    engineered = config["features"]["engineered"]
    target = config["target"]["column"]

    X = synthetic_df[numerical + categorical]
    y = np.log1p(synthetic_df[target])
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=0)

    pipe = build_full_pipeline(Ridge(alpha=10.0), numerical + engineered, categorical)
    pipe.fit(X_tr, y_tr)
    X_bg = pipe[:-1].transform(X_tr.sample(n=50, random_state=0))

    # Injection directe : évite la dépendance à un .joblib sur disque.
    service._artifact = {
        "pipeline": pipe,
        "best_model_name": "ridge",
        "numerical_features": numerical + engineered,
        "categorical_features": categorical,
        "log_transform": True,
        "X_background": X_bg,
    }
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_returns_positive_value(client):
    payload = {"overall": 82, "potential": 90, "age": 20, "height_cm": 185, "weight_kg": 78}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["estimated_value_eur"] > 0
    assert body["model_name"] == "ridge"
    assert body["potential_gap"] == 8  # 90 - 82


def test_predict_rejects_out_of_bounds(client):
    # overall > 99 viole la contrainte Pydantic -> 422.
    r = client.post("/predict", json={"overall": 150, "potential": 90, "age": 20,
                                      "height_cm": 185, "weight_kg": 78})
    assert r.status_code == 422
