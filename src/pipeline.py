"""End-to-end ML pipeline — run from project root: python src/pipeline.py"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure project root is on sys.path when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.data.loader import load_raw_data
from src.data.validation import validate_schema, validate_target
from src.data.cleaning import clean_dataset
from src.features.feature_selection import select_features, split_features_target
from src.models.compare import run_comparison
from src.models.evaluate import residual_summary, plot_residuals
from src.models.explain import compute_shap_values, plot_shap_summary
from src.utils.serialization import save_model
from src.utils.logger import get_logger

logger = get_logger("pipeline")


def main(config_path: str = "config/params.yaml") -> None:
    config = load_config(config_path)
    paths = config["paths"]
    os.makedirs(paths["reports_dir"], exist_ok=True)
    os.makedirs(paths["figures_dir"], exist_ok=True)

    # ── 0. Données : fallback synthétique si le vrai CSV est absent ───────────
    # Garantit que `git clone && python src/pipeline.py` fonctionne sans téléchargement.
    raw_path = config["data"]["raw_path"]
    if not os.path.exists(raw_path):
        logger.warning(
            "Dataset introuvable (%s) → génération d'un échantillon SYNTHÉTIQUE. "
            "Les métriques n'auront pas de valeur métier. Pour des résultats réels, "
            "placez le vrai male_players.csv (cf. README) puis relancez.",
            raw_path,
        )
        from src.data.make_dataset import generate
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        generate(n_rows=5000).to_csv(raw_path, index=False)

    # ── 1. Load ──────────────────────────────────────────────────────────────
    df = load_raw_data(raw_path, config["data"]["fifa_version"])

    # ── 2. Validate ──────────────────────────────────────────────────────────
    target_col = config["target"]["column"]
    numerical = config["features"]["numerical"]
    categorical = config["features"]["categorical"]
    validate_schema(df, numerical + categorical + [target_col])
    validate_target(df, target_col)

    # ── 3. Clean ─────────────────────────────────────────────────────────────
    df = clean_dataset(df, target_col)

    # ── 4. Feature selection + engineering ───────────────────────────────────
    df = select_features(df, config)
    # Les features dérivées (potential_gap) sont créées DANS le pipeline par le
    # PotentialGapTransformer : on les conserve dans la liste passée au préprocesseur même
    # si elles ne sont pas encore présentes dans `df`.
    engineered = config["features"].get("engineered", [])
    present_numerical = [c for c in numerical if c in df.columns]
    all_numerical = present_numerical + engineered

    X, y = split_features_target(df, target_col)

    # ── 5. Train / test split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
    )
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # ── 6. Train all models + comparison ─────────────────────────────────────
    trained_pipelines, comparison_table, best_name = run_comparison(
        X_train, X_test, y_train, y_test,
        all_numerical, categorical, config,
    )

    # ── 7. Save comparison table ──────────────────────────────────────────────
    metrics_path = paths["metrics_csv"]
    display_cols = ["MAE (€)", "RMSE (€)", "R²", "CV RMSE (log)"]
    comparison_table[display_cols].to_csv(metrics_path)
    logger.info(f"Metrics saved to {metrics_path}")
    print("\n── Model Comparison ─────────────────────────────────")
    print(comparison_table[display_cols].to_string())
    print(f"\nBest model: {best_name}")

    # ── 8. Save best model ────────────────────────────────────────────────────
    best_pipeline = trained_pipelines[best_name]

    # Pre-transform 100 training rows → used as SHAP background in Streamlit.
    # On applique TOUTES les étapes de préparation (gap + preprocessor), donc
    # `best_pipeline[:-1]` (tout sauf le modèle final).
    X_bg_raw = X_train.sample(n=min(100, len(X_train)), random_state=42)
    X_background = best_pipeline[:-1].transform(X_bg_raw)

    artifact = {
        "pipeline": best_pipeline,
        "best_model_name": best_name,
        "numerical_features": all_numerical,
        "categorical_features": categorical,
        "log_transform": config["target"]["log_transform"],
        "X_background": X_background,   # pre-transformed, for KernelExplainer
    }
    save_model(artifact, paths["model_output"])
    logger.info(f"Best model saved to {paths['model_output']}")

    # ── 9. SHAP explanation ───────────────────────────────────────────────────
    logger.info("Computing SHAP values (sample of 500 test rows)…")
    n_shap = min(500, len(X_test))
    X_shap = X_test.sample(n=n_shap, random_state=42)
    try:
        shap_values, X_transformed, feature_names = compute_shap_values(
            best_pipeline, X_shap
        )
        plot_shap_summary(
            shap_values, X_transformed, feature_names,
            save_path=os.path.join(paths["figures_dir"], "shap_summary.png"),
        )
    except Exception as exc:
        logger.warning(f"SHAP computation failed: {exc}")

    # ── 10. Analyse de résidus (recul critique sur les erreurs du modèle) ─────
    try:
        y_pred_best = best_pipeline.predict(X_test)
        if config["target"]["log_transform"]:
            y_pred_best = np.expm1(y_pred_best)
            y_true_best = np.expm1(np.log1p(y_test))
        else:
            y_true_best = y_test
        summary = residual_summary(y_true_best, y_pred_best)
        logger.info(
            f"Résidus — moyenne {summary['residual_mean']:,.0f} € | "
            f"MAPE {summary['mape']:.1%}"
        )
        plot_residuals(
            y_true_best, y_pred_best,
            save_path=os.path.join(paths["figures_dir"], "residuals.png"),
        )
    except Exception as exc:
        logger.warning(f"Residual analysis failed: {exc}")

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
