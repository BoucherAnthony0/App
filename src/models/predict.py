"""Model prediction helpers."""
import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


def predict_value(model_artifact: dict, player_features: dict) -> float:
    pipeline = model_artifact["pipeline"]
    numerical_features = model_artifact["numerical_features"]
    categorical_features = model_artifact["categorical_features"]
    log_transform = model_artifact.get("log_transform", True)

    all_features = numerical_features + categorical_features
    row = {}
    for feat in numerical_features:
        row[feat] = float(player_features.get(feat, 0))
    for feat in categorical_features:
        defaults = {"preferred_foot": "Right", "work_rate": "Medium/Medium"}
        row[feat] = player_features.get(feat, defaults.get(feat, "Unknown"))

    df = pd.DataFrame([row])

    raw_pred = pipeline.predict(df)[0]
    value = float(np.expm1(raw_pred)) if log_transform else float(raw_pred)
    return max(0.0, value)
