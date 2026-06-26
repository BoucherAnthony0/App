"""Data cleaning functions."""
import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

_GK_PREFIXES = ("goalkeeping_",)
_OUTFIELD_PREFIXES = (
    "attacking_", "skill_", "movement_", "power_", "mentality_",
)


def drop_invalid_targets(df: pd.DataFrame, target_col: str = "value_eur") -> pd.DataFrame:
    before = len(df)
    df = df[df[target_col].notna() & (df[target_col] > 0)].copy()
    logger.info(f"Dropped {before - len(df)} rows with missing/zero target")
    return df


def handle_missing_values(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    gk_cols = [c for c in num_cols if c.startswith(_GK_PREFIXES)]

    for col in num_cols:
        if df[col].isna().sum() == 0:
            continue
        if col in gk_cols:
            # GK stats are NaN for outfield players → fill with 0
            df[col] = df[col].fillna(0)
        elif strategy == "median":
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mean())

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    logger.info(f"Missing values handled (strategy='{strategy}')")
    return df


def remove_outliers(
    df: pd.DataFrame, column: str = "value_eur", method: str = "iqr", factor: float = 3.0
) -> pd.DataFrame:
    if method == "iqr":
        q1, q3 = df[column].quantile(0.25), df[column].quantile(0.75)
        upper = q3 + factor * (q3 - q1)
        before = len(df)
        df = df[df[column] <= upper].copy()
        logger.info(
            f"Outlier removal (IQR×{factor}): dropped {before - len(df)} rows "
            f"(cap {upper:,.0f} €)"
        )
    return df


def clean_dataset(df: pd.DataFrame, target_col: str = "value_eur") -> pd.DataFrame:
    df = drop_invalid_targets(df, target_col)
    df = handle_missing_values(df)
    return df
