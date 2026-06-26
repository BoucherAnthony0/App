"""Data loading helpers."""
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_raw_data(path: str, fifa_version: float = 24.0) -> pd.DataFrame:
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path, low_memory=False)
    logger.info(f"Raw shape: {df.shape}")

    if "fifa_version" in df.columns:
        df = df[df["fifa_version"] == fifa_version].copy()
        latest_update = df["fifa_update"].max()
        df = df[df["fifa_update"] == latest_update]
        logger.info(
            f"Filtered to FIFA {fifa_version} update {latest_update}: {len(df)} players"
        )

    return df.reset_index(drop=True)
