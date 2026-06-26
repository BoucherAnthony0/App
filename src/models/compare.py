"""Multi-model comparison engine."""
import os
import numpy as np
import pandas as pd
from sklearn.base import clone
from src.models.model_factory import get_models, build_full_pipeline
from src.models.train import train_model
from src.models.evaluate import compute_metrics
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_all_models(
    models_dict: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numerical_features: list,
    categorical_features: list,
) -> dict:
    trained = {}
    for name, model in models_dict.items():
        logger.info(f"[{name}] Starting training…")
        pipeline = build_full_pipeline(
            clone(model), numerical_features, categorical_features
        )
        trained[name] = train_model(pipeline, X_train, y_train)
        logger.info(f"[{name}] Done")
    return trained


def evaluate_all_models(
    trained_models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    log_transform: bool = True,
) -> dict:
    results = {}
    for name, pipeline in trained_models.items():
        y_pred_log = pipeline.predict(X_test)
        if log_transform:
            y_pred = np.expm1(y_pred_log)
            y_true = np.expm1(y_test) if hasattr(y_test, "values") else np.expm1(y_test)
        else:
            y_pred = y_pred_log
            y_true = y_test
        results[name] = compute_metrics(y_true, y_pred)
        logger.info(
            f"[{name}] MAE={results[name]['mae']:,.0f} € | "
            f"RMSE={results[name]['rmse']:,.0f} € | "
            f"R²={results[name]['r2']:.4f}"
        )
    return results


def build_comparison_table(results: dict) -> pd.DataFrame:
    rows = [
        {
            "model": name,
            "MAE (€)": f"{m['mae']:,.0f}",
            "RMSE (€)": f"{m['rmse']:,.0f}",
            "R²": f"{m['r2']:.4f}",
            "_mae_raw": m["mae"],
            "_rmse_raw": m["rmse"],
            "_r2_raw": m["r2"],
        }
        for name, m in results.items()
    ]
    return pd.DataFrame(rows).set_index("model")


def select_best_model(results: dict, metric: str = "rmse") -> str:
    reverse = metric == "r2"
    best = sorted(results.items(), key=lambda x: x[1][metric], reverse=reverse)[0]
    logger.info(f"Best model: {best[0]} ({metric}={best[1][metric]:,.0f})")
    return best[0]


def run_comparison(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    numerical_features: list,
    categorical_features: list,
    config: dict,
) -> tuple[dict, pd.DataFrame, str]:
    log_transform = config["target"]["log_transform"]
    y_train_t = np.log1p(y_train) if log_transform else y_train
    y_test_t = np.log1p(y_test) if log_transform else y_test

    models_dict = get_models(config)
    trained = train_all_models(
        models_dict, X_train, y_train_t, numerical_features, categorical_features
    )
    results = evaluate_all_models(trained, X_test, y_test_t, log_transform)
    table = build_comparison_table(results)
    best_name = select_best_model(results, metric="rmse")

    return trained, table, best_name
