"""Model explanation helpers (SHAP)."""
import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.pipeline import Pipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)

# SHAP logge ses calculs internes (KernelExplainer) en INFO de façon très verbeuse —
# on le réduit à WARNING pour garder les sorties du pipeline lisibles.
logging.getLogger("shap").setLevel(logging.WARNING)


def _get_feature_names(pipeline: Pipeline) -> list:
    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return [f"feature_{i}" for i in range(100)]


def compute_shap_values(pipeline: Pipeline, X_sample: pd.DataFrame):
    model = pipeline.named_steps["model"]
    # Toutes les étapes de préparation (gap + preprocessor), tout sauf le modèle.
    X_transformed = pipeline[:-1].transform(X_sample)
    feature_names = _get_feature_names(pipeline)

    model_name = model.__class__.__name__

    if model_name in ("RandomForestRegressor", "XGBRegressor", "LGBMRegressor",
                      "GradientBoostingRegressor", "ExtraTreesRegressor"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)

    elif model_name in ("Ridge", "Lasso", "LinearRegression", "ElasticNet"):
        explainer = shap.LinearExplainer(model, X_transformed)
        shap_values = explainer.shap_values(X_transformed)

    else:
        # MLPRegressor or unknown — KernelExplainer on a background subsample
        logger.info("Using KernelExplainer (slow for large samples)")
        n_bg = min(100, X_transformed.shape[0])
        background = shap.sample(X_transformed, n_bg, random_state=42)
        explainer = shap.KernelExplainer(model.predict, background)
        n_explain = min(50, X_transformed.shape[0])
        shap_values = explainer.shap_values(X_transformed[:n_explain])
        X_transformed = X_transformed[:n_explain]

    return shap_values, X_transformed, feature_names


def plot_shap_summary(
    shap_values,
    X_transformed,
    feature_names: list,
    save_path: str = "reports/figures/shap_summary.png",
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"SHAP summary plot saved to {save_path}")


def explain_single_player(
    pipeline: Pipeline,
    player_df: pd.DataFrame,
    X_background: np.ndarray | None = None,
) -> tuple[np.ndarray, list]:
    """Return SHAP values for a single player row (used by Streamlit).

    X_background: pre-transformed training sample saved in the artifact.
    Required for KernelExplainer (MLP/unknown models) to produce non-zero values.
    """
    model = pipeline.named_steps["model"]
    X_transformed = pipeline[:-1].transform(player_df)
    feature_names = _get_feature_names(pipeline)
    model_name = model.__class__.__name__

    if model_name in ("RandomForestRegressor", "XGBRegressor", "LGBMRegressor",
                      "HistGradientBoostingRegressor", "GradientBoostingRegressor"):
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_transformed)
        sv = sv[0] if sv.ndim == 2 else sv

    elif model_name in ("Ridge", "Lasso", "LinearRegression", "ElasticNet"):
        bg = X_background if X_background is not None else X_transformed
        explainer = shap.LinearExplainer(model, bg)
        sv = explainer.shap_values(X_transformed)[0]

    else:
        # MLP or unknown — KernelExplainer with a proper background
        if X_background is not None and len(X_background) > 1:
            bg = X_background
        else:
            raise ValueError(
                "KernelExplainer requires X_background saved in the model artifact. "
                "Re-run python src/pipeline.py to rebuild the model."
            )
        explainer = shap.KernelExplainer(model.predict, bg)
        # l1_reg=0 disables LASSO zeroing so all features get a non-zero contribution
        sv = explainer.shap_values(X_transformed, l1_reg=0)
        sv = sv[0] if hasattr(sv, "__len__") and not isinstance(sv, np.ndarray) else sv
        if isinstance(sv, np.ndarray) and sv.ndim == 2:
            sv = sv[0]

    return sv, feature_names
