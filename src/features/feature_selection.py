"""Feature selection and engineering helpers."""
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "potential" in df.columns and "overall" in df.columns:
        df["potential_gap"] = df["potential"] - df["overall"]
    return df


def select_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Restreint le DataFrame aux colonnes brutes utiles + cible.

    Les features dérivées (ex. `potential_gap`) ne sont PAS ajoutées ici : elles sont
    produites à l'intérieur du pipeline sklearn (`PotentialGapTransformer`), ce qui évite
    toute duplication et garantit une source de vérité unique. On conserve donc seulement
    les colonnes brutes (`potential`, `overall`, …) dont le transformer a besoin.
    `add_engineered_features()` reste exposé pour l'analyse exploratoire (notebook).
    """
    numerical = config["features"]["numerical"]
    categorical = config["features"]["categorical"]
    target = config["target"]["column"]

    wanted = [target] + numerical + categorical
    available = [c for c in wanted if c in df.columns]
    missing = set(wanted) - set(available)
    if missing:
        logger.warning(f"Features not found in DataFrame (will be skipped): {missing}")

    df = df[available].copy()
    logger.info(
        f"Selected {len(available) - 1} features + target "
        f"({len(df)} rows)"
    )
    return df


def split_features_target(
    df: pd.DataFrame, target_col: str = "value_eur"
) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
