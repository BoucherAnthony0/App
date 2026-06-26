"""Custom sklearn transformers."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PotentialGapTransformer(BaseEstimator, TransformerMixin):
    """Adds potential_gap = potential - overall as a scouting signal.

    A large gap flags undervalued/young profiles — core use case of the tool.
    Works in-place on a DataFrame before ColumnTransformer ingests it.
    """

    def __init__(self, potential_col: str = "potential", overall_col: str = "overall"):
        self.potential_col = potential_col
        self.overall_col = overall_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            if self.potential_col in X.columns and self.overall_col in X.columns:
                X["potential_gap"] = X[self.potential_col] - X[self.overall_col]
            return X
        # Fallback for numpy arrays — no-op
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array(["potential_gap"])
        return np.append(input_features, "potential_gap")
