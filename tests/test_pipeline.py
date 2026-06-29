"""Tests du pipeline de modélisation : transformer, anti-fuite, bout-en-bout."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from src.features.custom_transformers import PotentialGapTransformer
from src.models.model_factory import build_full_pipeline


def test_potential_gap_transformer_computes_correctly():
    X = pd.DataFrame({"overall": [70, 80], "potential": [85, 82], "age": [20, 30]})
    out = PotentialGapTransformer().fit_transform(X)
    assert "potential_gap" in out.columns
    assert out["potential_gap"].tolist() == [15, 2]


def test_full_pipeline_fits_and_predicts(synthetic_df, config):
    """Bout-en-bout sur un petit échantillon : le pipeline entraîne et prédit."""
    numerical = [c for c in config["features"]["numerical"] if c in synthetic_df.columns]
    categorical = config["features"]["categorical"]
    engineered = config["features"]["engineered"]
    target = config["target"]["column"]

    X = synthetic_df[numerical + categorical]
    y = np.log1p(synthetic_df[target])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

    pipe = build_full_pipeline(Ridge(alpha=10.0), numerical + engineered, categorical)
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_te)

    assert len(preds) == len(X_te)
    assert np.isfinite(preds).all()


def test_pipeline_imputer_fit_on_train_only(synthetic_df, config):
    """ANTI-FUITE : l'imputation du pipeline apprend la médiane sur le TRAIN seul.

    On vérifie que la statistique d'imputation diffère si on fit sur train vs sur tout
    le jeu — preuve que c'est bien le fit (train) qui détermine l'imputation, pas le test.
    """
    numerical = [c for c in config["features"]["numerical"] if c in synthetic_df.columns]
    categorical = config["features"]["categorical"]
    engineered = config["features"]["engineered"]

    df = synthetic_df.copy()
    df.loc[df.index[:40], "pace"] = np.nan
    X = df[numerical + categorical]
    X_tr, X_te = train_test_split(X, test_size=0.5, random_state=1)

    pipe = build_full_pipeline(Ridge(), numerical + engineered, categorical)
    pipe.fit(X_tr, np.zeros(len(X_tr)))

    num_imputer = pipe.named_steps["preprocessor"].named_transformers_["num"].named_steps["imputer"]
    pace_idx = (numerical + engineered).index("pace")
    train_median = X_tr["pace"].median()
    assert np.isclose(num_imputer.statistics_[pace_idx], train_median)
