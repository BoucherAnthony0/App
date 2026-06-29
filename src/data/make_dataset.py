"""Génération d'un jeu de données synthétique réaliste.

POURQUOI ce module ?
--------------------
Le vrai jeu de données (EA Sports FC 24 / sofifa, ~18 000 joueurs) ne peut pas être
versionné dans le dépôt (licence + volume + il provient de Kaggle qui exige une
authentification). Sans lui, personne — y compris le jury — ne peut exécuter le projet.

Ce script génère un **échantillon synthétique** au schéma strictement identique au vrai
dataset, de sorte que :
  - `python src/pipeline.py` tourne immédiatement après un `git clone` (reproductibilité) ;
  - la démonstration en soutenance ne dépend pas d'un téléchargement externe.

Les relations injectées (valeur ~ exp(overall), corrélations entre attributs, NaN sur les
stats de gardien des joueurs de champ) sont **plausibles mais artificielles** : les chiffres
de performance obtenus dessus n'ont pas de valeur métier. Pour des résultats réels, voir la
section « Données réelles » du README et placer le vrai `male_players.csv` dans `data/raw/`.

Usage :
    python src/data/make_dataset.py            # 5000 lignes -> data/raw/male_players.csv
    python src/data/make_dataset.py --rows 2000 --out data/raw/sample.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

# Attributs numériques attendus par config/params.yaml (hors features dérivées).
_NUMERIC_ATTRS = [
    "pace", "shooting", "passing", "dribbling", "defending", "physic",
    "attacking_crossing", "attacking_finishing", "attacking_heading_accuracy",
    "attacking_short_passing", "attacking_volleys",
    "skill_dribbling", "skill_curve", "skill_fk_accuracy", "skill_long_passing",
    "skill_ball_control",
    "movement_acceleration", "movement_sprint_speed", "movement_agility",
    "movement_reactions", "movement_balance",
    "power_shot_power", "power_jumping", "power_stamina", "power_strength",
    "power_long_shots",
    "mentality_aggression", "mentality_interceptions", "mentality_positioning",
    "mentality_vision", "mentality_penalties", "mentality_composure",
    "defending_marking_awareness", "defending_standing_tackle",
    "defending_sliding_tackle",
]
_GK_ATTRS = [
    "goalkeeping_diving", "goalkeeping_handling", "goalkeeping_kicking",
    "goalkeeping_positioning", "goalkeeping_reflexes", "goalkeeping_speed",
]
_WORK_RATES = ["High/High", "High/Medium", "High/Low", "Medium/High",
               "Medium/Medium", "Medium/Low", "Low/High", "Low/Medium", "Low/Low"]


def generate(n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Construit un DataFrame synthétique au schéma du dataset EA Sports FC."""
    rng = np.random.default_rng(seed)

    # overall : distribution réaliste (la masse des joueurs est autour de 60-70).
    overall = np.clip(rng.normal(65, 7, n_rows).round(), 47, 91).astype(int)
    # potential >= overall, écart plus grand chez les jeunes.
    age = np.clip(rng.normal(25, 4.5, n_rows).round(), 16, 41).astype(int)
    youth_bonus = np.clip((26 - age), 0, 10)
    potential = np.clip(overall + rng.poisson(youth_bonus * 0.5 + 1), 47, 95).astype(int)

    data: dict[str, np.ndarray] = {
        "overall": overall,
        "potential": potential,
        "age": age,
        "height_cm": np.clip(rng.normal(181, 6.8, n_rows).round(), 158, 205).astype(int),
        "weight_kg": np.clip(rng.normal(75, 7, n_rows).round(), 55, 100).astype(int),
        "weak_foot": rng.integers(1, 6, n_rows),
        "skill_moves": rng.integers(1, 6, n_rows),
        "international_reputation": np.clip(
            1 + rng.poisson((overall - 60).clip(0) / 12), 1, 5
        ).astype(int),
    }

    # Attributs techniques corrélés à overall + bruit (bornés 20-95).
    for attr in _NUMERIC_ATTRS:
        noise = rng.normal(0, 9, n_rows)
        data[attr] = np.clip(overall + noise, 20, 95).round().astype(int)

    # Stats de gardien : faibles pour les joueurs de champ (~85 % de l'effectif).
    is_gk = rng.random(n_rows) < 0.12
    for attr in _GK_ATTRS:
        gk_vals = np.where(
            is_gk,
            np.clip(overall + rng.normal(0, 8, n_rows), 20, 95),
            rng.integers(5, 22, n_rows),
        )
        data[attr] = gk_vals.round().astype(int)

    df = pd.DataFrame(data)
    df["preferred_foot"] = rng.choice(["Right", "Left"], n_rows, p=[0.76, 0.24])
    df["work_rate"] = rng.choice(_WORK_RATES, n_rows)

    # Métadonnées de version (utilisées par loader.load_raw_data pour le filtrage).
    df["fifa_version"] = 24.0
    df["fifa_update"] = rng.choice([1, 2, 3], n_rows, p=[0.2, 0.3, 0.5])

    # Colonnes "pièges" présentes dans le vrai dataset, volontairement exclues par
    # le projet (fuite/identifiants) — on les ajoute pour tester la robustesse du loader.
    df["short_name"] = [f"Player {i}" for i in range(n_rows)]
    df["player_url"] = "https://sofifa.com/player/0000"

    # Cible : valeur marchande ~ exponentielle de l'overall, bonus de potentiel,
    # malus d'âge, avec bruit log-normal (distribution très asymétrique -> log1p).
    base = np.exp((overall - 45) / 8.5) * 25_000
    pot_bonus = 1 + (potential - overall) * 0.04
    age_penalty = np.where(age > 30, 1 - (age - 30) * 0.06, 1.0).clip(0.3, 1.0)
    value = base * pot_bonus * age_penalty * rng.lognormal(0, 0.35, n_rows)
    df["value_eur"] = np.clip(value, 9_000, None).round(-3)

    # Injecter quelques manquants réalistes (stats GK des joueurs de champ).
    miss_mask = (~is_gk) & (rng.random(n_rows) < 0.15)
    df.loc[miss_mask, "goalkeeping_speed"] = np.nan

    logger.info(
        "Synthétique : %d lignes | value_eur médiane %d € | %d gardiens",
        len(df), int(df["value_eur"].median()), int(is_gk.sum()),
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Génère un dataset FIFA synthétique.")
    parser.add_argument("--rows", type=int, default=5000, help="Nombre de joueurs.")
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire.")
    parser.add_argument(
        "--out", type=str, default="data/raw/male_players.csv", help="Chemin de sortie CSV."
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = generate(args.rows, args.seed)
    df.to_csv(args.out, index=False)
    logger.info("Écrit -> %s", args.out)


if __name__ == "__main__":
    main()
