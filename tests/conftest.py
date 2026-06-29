"""Configuration partagée des tests : fixtures de données et de pipeline."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

# Rend `src` et `app` importables quand on lance pytest depuis la racine.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.make_dataset import generate  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    """Petit jeu synthétique déterministe (rapide)."""
    return generate(n_rows=400, seed=7)


@pytest.fixture(scope="session")
def config() -> dict:
    from src.config import load_config
    return load_config("config/params.yaml")
