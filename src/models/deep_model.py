"""MLP (Deep Learning) model — implemented with sklearn MLPRegressor.

StandardScaler is applied upstream by the ColumnTransformer preprocessor,
satisfying the normalisation requirement for gradient-based learning.
"""
from sklearn.neural_network import MLPRegressor
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_mlp(cfg: dict) -> MLPRegressor:
    hidden_layer_sizes = tuple(cfg["hidden_layer_sizes"])
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=cfg.get("activation", "relu"),
        solver="adam",
        max_iter=cfg.get("max_iter", 500),
        early_stopping=cfg.get("early_stopping", True),
        validation_fraction=cfg.get("validation_fraction", 0.1),
        n_iter_no_change=cfg.get("n_iter_no_change", 20),
        random_state=cfg.get("random_state", 42),
        verbose=False,
    )
    logger.info(f"MLP built: hidden_layers={hidden_layer_sizes}")
    return model


def train_mlp(model: MLPRegressor, X_train, y_train, X_val=None, y_val=None):
    """Standalone training helper (early stopping handled internally by sklearn)."""
    model.fit(X_train, y_train)
    logger.info(
        f"MLP trained — {model.n_iter_} iterations, "
        f"best val loss: {model.best_validation_score_:.6f}"
    )
    return model
