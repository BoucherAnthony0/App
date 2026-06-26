"""Serialization helpers."""
import os
import joblib


def save_model(artifact: dict, path: str = "models/best_model.joblib") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(artifact, path)


def load_model(path: str = "models/best_model.joblib") -> dict:
    return joblib.load(path)
