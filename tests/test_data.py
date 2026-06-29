"""Tests de la couche données : génération, validation, nettoyage."""
import numpy as np
import pytest

from src.data.cleaning import clean_dataset, drop_invalid_targets
from src.data.validation import validate_schema, validate_target


def test_generated_schema_has_required_columns(synthetic_df, config):
    """Le générateur doit produire toutes les colonnes attendues par la config."""
    required = (
        config["features"]["numerical"]
        + config["features"]["categorical"]
        + [config["target"]["column"]]
    )
    missing = [c for c in required if c not in synthetic_df.columns]
    assert not missing, f"Colonnes manquantes dans le synthétique : {missing}"


def test_validate_schema_raises_on_missing_column(synthetic_df):
    with pytest.raises(ValueError):
        validate_schema(synthetic_df, ["colonne_inexistante"])


def test_validate_target_ok(synthetic_df):
    # Ne doit pas lever sur une cible valide.
    validate_target(synthetic_df, "value_eur")


def test_drop_invalid_targets_removes_null_and_zero(synthetic_df):
    df = synthetic_df.copy()
    df.loc[df.index[:5], "value_eur"] = np.nan
    df.loc[df.index[5:10], "value_eur"] = 0
    cleaned = drop_invalid_targets(df, "value_eur")
    assert cleaned["value_eur"].notna().all()
    assert (cleaned["value_eur"] > 0).all()
    assert len(cleaned) == len(df) - 10


def test_clean_dataset_does_not_impute(synthetic_df):
    """ANTI-FUITE : clean_dataset ne doit PAS imputer (pas de stat globale pré-split).

    On injecte des manquants dans une colonne numérique : ils doivent SUBSISTER après
    clean_dataset (l'imputation est déléguée au pipeline, fit sur le train uniquement).
    """
    df = synthetic_df.copy()
    df.loc[df.index[:20], "pace"] = np.nan
    cleaned = clean_dataset(df, "value_eur")
    assert cleaned["pace"].isna().sum() == 20, (
        "clean_dataset ne doit pas imputer : ce serait une fuite de données."
    )
