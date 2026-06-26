"""Model training helpers."""
import pandas as pd
from sklearn.pipeline import Pipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_model(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    logger.info(f"Training {pipeline.named_steps['model'].__class__.__name__}…")
    pipeline.fit(X_train, y_train)
    logger.info("Training complete")
    return pipeline
