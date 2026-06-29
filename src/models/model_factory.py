"""Central model factory — instantiates all models from config."""
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from src.features.preprocessor import build_preprocessing_pipeline
from src.features.custom_transformers import PotentialGapTransformer
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from xgboost import XGBRegressor
    _XGB_AVAILABLE = True
except Exception:
    _XGB_AVAILABLE = False
    logger.warning("xgboost unavailable (try: brew install libomp)")

try:
    from lightgbm import LGBMRegressor
    _LGBM_AVAILABLE = True
except Exception:
    _LGBM_AVAILABLE = False


def _build_gb(cfg: dict):
    if _XGB_AVAILABLE:
        return XGBRegressor(
            n_estimators=cfg["n_estimators"],
            learning_rate=cfg["learning_rate"],
            max_depth=cfg["max_depth"],
            subsample=cfg["subsample"],
            colsample_bytree=cfg.get("colsample_bytree", 0.8),
            random_state=cfg["random_state"],
            n_jobs=cfg.get("n_jobs", -1),
            verbosity=0,
        )
    if _LGBM_AVAILABLE:
        return LGBMRegressor(
            n_estimators=cfg["n_estimators"],
            learning_rate=cfg["learning_rate"],
            max_depth=cfg["max_depth"],
            subsample=cfg["subsample"],
            random_state=cfg["random_state"],
            n_jobs=cfg.get("n_jobs", -1),
            verbose=-1,
        )
    # Fallback: sklearn's native gradient boosting (no OpenMP dependency)
    from sklearn.ensemble import HistGradientBoostingRegressor
    logger.warning(
        "xgboost/lightgbm unavailable — using HistGradientBoostingRegressor. "
        "Run `brew install libomp` then reinstall xgboost/lightgbm for full performance."
    )
    return HistGradientBoostingRegressor(
        max_iter=cfg["n_estimators"],
        learning_rate=cfg["learning_rate"],
        max_depth=cfg["max_depth"],
        random_state=cfg["random_state"],
    )


def get_models(config: dict) -> dict:
    mc = config["models"]
    from src.models.deep_model import build_mlp

    return {
        "ridge": Ridge(alpha=mc["ridge"]["alpha"]),
        "random_forest": RandomForestRegressor(
            n_estimators=mc["random_forest"]["n_estimators"],
            max_depth=mc["random_forest"]["max_depth"],
            min_samples_leaf=mc["random_forest"]["min_samples_leaf"],
            random_state=mc["random_forest"]["random_state"],
            n_jobs=mc["random_forest"].get("n_jobs", -1),
        ),
        "gradient_boosting": _build_gb(mc["gradient_boosting"]),
        "mlp": build_mlp(mc["mlp"]),
    }


def build_full_pipeline(
    model,
    numerical_features: list,
    categorical_features: list,
) -> Pipeline:
    """Assemble le pipeline complet : feature engineering -> préprocessing -> modèle.

    Le `PotentialGapTransformer` est la **source de vérité unique** pour `potential_gap`
    (= potential - overall). Il est calculé ici, à l'intérieur du pipeline, donc :
      - plus de duplication (auparavant recalculé dans feature_selection ET dans le service) ;
      - l'API n'a qu'à fournir les attributs bruts, la feature dérivée est produite seule ;
      - aucune fuite : transformation déterministe par ligne, indépendante du jeu.
    `numerical_features` inclut `potential_gap` : le ColumnTransformer le sélectionne après
    que le transformer l'a ajouté au DataFrame.
    """
    preprocessor = build_preprocessing_pipeline(numerical_features, categorical_features)
    return Pipeline(
        steps=[
            ("gap", PotentialGapTransformer()),
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
