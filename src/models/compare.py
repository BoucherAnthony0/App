"""Moteur de comparaison multi-modèles : baseline + CV + tuning + sélection.

POURQUOI cette refonte (compétence C4.2) ?
------------------------------------------
La version initiale n'évaluait que sur un **unique split 80/20** avec des hyperparamètres
**figés**. C'était fragile (dépendant d'un seul tirage) et ne démontrait aucune
« optimisation ». On ajoute donc :

1. Un **baseline trivial** (`DummyRegressor`, prédit la moyenne) : sans lui, un R² n'a pas
   de point de comparaison. Tout modèle utile doit le battre nettement.
2. Une **validation croisée K-fold** sur le train (robustesse : moyenne ± écart-type),
   qui ne touche jamais le test → pas de fuite.
3. Une **optimisation d'hyperparamètres** par `RandomizedSearchCV` (CV interne sur le train).
4. Une **évaluation finale sur le test** (jamais vu) en euros, dans l'échelle d'origine.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

from src.models.model_factory import get_models, build_full_pipeline
from src.models.train import train_model
from src.models.evaluate import compute_metrics
from src.utils.logger import get_logger

logger = get_logger(__name__)


def cross_validate_pipeline(
    pipeline, X_train, y_train, cv_folds: int, scoring: str
) -> tuple[float, float]:
    """RMSE de validation croisée (sur le train, échelle log). Retourne (moyenne, std)."""
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv_folds, scoring=scoring)
    # neg_root_mean_squared_error -> on repasse en positif.
    rmse = -scores
    return float(rmse.mean()), float(rmse.std())


def tune_pipeline(pipeline, X_train, y_train, space: dict, tuning_cfg: dict):
    """RandomizedSearchCV sur le train. Retourne le pipeline ré-entraîné optimisé."""
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=space,
        n_iter=tuning_cfg["n_iter"],
        cv=tuning_cfg["cv_folds"],
        scoring="neg_root_mean_squared_error",
        random_state=tuning_cfg.get("random_state", 42),
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    logger.info("    meilleurs params : %s", search.best_params_)
    return search.best_estimator_, search.best_params_


def run_comparison(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    numerical_features: list,
    categorical_features: list,
    config: dict,
) -> tuple[dict, pd.DataFrame, str]:
    """Entraîne, valide (CV), optimise et évalue tous les modèles + un baseline."""
    log_transform = config["target"]["log_transform"]
    y_train_t = np.log1p(y_train) if log_transform else y_train
    y_test_t = np.log1p(y_test) if log_transform else y_test

    eval_cfg = config.get("evaluation", {"cv_folds": 5, "scoring": "neg_root_mean_squared_error"})
    tuning_cfg = config.get("tuning", {"enabled": False})

    models_dict = get_models(config)
    # Baseline trivial : prédit toujours la moyenne du train.
    models_dict = {"baseline_mean": DummyRegressor(strategy="mean"), **models_dict}
    search_spaces = tuning_cfg.get("search_spaces", {})

    trained: dict = {}
    results: dict = {}
    best_params_log: dict = {}

    for name, model in models_dict.items():
        logger.info("[%s] entraînement…", name)
        pipeline = build_full_pipeline(clone(model), numerical_features, categorical_features)

        # 1. Optimisation d'hyperparamètres (sauf baseline) si activée + espace défini.
        if tuning_cfg.get("enabled") and name in search_spaces:
            logger.info("[%s] RandomizedSearchCV (n_iter=%d)…", name, tuning_cfg["n_iter"])
            pipeline, best_params = tune_pipeline(
                pipeline, X_train, y_train_t, search_spaces[name], tuning_cfg
            )
            best_params_log[name] = best_params
        else:
            pipeline = train_model(pipeline, X_train, y_train_t)

        # 2. Validation croisée (robustesse, échelle log) sur le train.
        try:
            cv_mean, cv_std = cross_validate_pipeline(
                clone(pipeline), X_train, y_train_t,
                eval_cfg["cv_folds"], eval_cfg["scoring"],
            )
        except Exception as exc:  # MLP/Kernel parfois coûteux : on dégrade proprement
            logger.warning("[%s] CV indisponible : %s", name, exc)
            cv_mean, cv_std = float("nan"), float("nan")

        # 3. Évaluation finale sur le TEST (jamais vu), en euros.
        y_pred_log = pipeline.predict(X_test)
        y_pred = np.expm1(y_pred_log) if log_transform else y_pred_log
        y_true = np.expm1(y_test_t) if log_transform else y_test_t
        m = compute_metrics(y_true, y_pred)
        m["cv_rmse_log_mean"] = cv_mean
        m["cv_rmse_log_std"] = cv_std

        trained[name] = pipeline
        results[name] = m
        logger.info(
            f"[{name}] TEST MAE={m['mae']:,.0f} € RMSE={m['rmse']:,.0f} € "
            f"R²={m['r2']:.4f} | CV(log) RMSE={cv_mean:.4f}±{cv_std:.4f}"
        )

    table = build_comparison_table(results)
    # On exclut le baseline de la sélection (référence, pas un candidat).
    candidates = {k: v for k, v in results.items() if k != "baseline_mean"}
    best_name = select_best_model(candidates, metric="rmse")

    return trained, table, best_name


def build_comparison_table(results: dict) -> pd.DataFrame:
    rows = [
        {
            "model": name,
            "MAE (€)": f"{m['mae']:,.0f}",
            "RMSE (€)": f"{m['rmse']:,.0f}",
            "R²": f"{m['r2']:.4f}",
            "CV RMSE (log)": (
                f"{m['cv_rmse_log_mean']:.4f} ± {m['cv_rmse_log_std']:.4f}"
                if not np.isnan(m.get("cv_rmse_log_mean", float("nan"))) else "n/a"
            ),
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
    logger.info(f"Meilleur modèle : {best[0]} ({metric}={best[1][metric]:,.0f})")
    return best[0]
