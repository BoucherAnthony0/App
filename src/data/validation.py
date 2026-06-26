"""Data validation helpers."""
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


def validate_schema(df: pd.DataFrame, expected_columns: list) -> None:
    missing = [c for c in expected_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    logger.info(f"Schema validation passed ({len(expected_columns)} columns checked)")


def validate_target(df: pd.DataFrame, target_col: str = "value_eur") -> None:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    nulls = df[target_col].isna().sum()
    negatives = (df[target_col] < 0).sum()
    if nulls > 0:
        logger.warning(f"{nulls} null values in '{target_col}'")
    if negatives > 0:
        logger.warning(f"{negatives} negative values in '{target_col}'")
    logger.info(
        f"Target stats — min: {df[target_col].min():,.0f} "
        f"median: {df[target_col].median():,.0f} "
        f"max: {df[target_col].max():,.0f}"
    )
