"""Model evaluation helpers : métriques + analyse de résidus.

L'analyse de résidus (compétence C4.2) répond à « le modèle se trompe-t-il de façon
structurée ? ». Un bon modèle a des résidus centrés sur 0, sans tendance vs la prédiction.
On l'utilise ici pour documenter honnêtement les limites (hétéroscédasticité sur les valeurs
extrêmes typique d'une cible log-normale).
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def residual_summary(y_true, y_pred) -> dict:
    """Statistiques descriptives des résidus (euros)."""
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    return {
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals)),
        "residual_median": float(np.median(residuals)),
        "mape": float(np.mean(np.abs(residuals) / np.clip(np.abs(y_true), 1, None))),
    }


def plot_residuals(
    y_true, y_pred, save_path: str = "reports/figures/residuals.png"
) -> None:
    """Trace résidus vs prédiction + distribution des résidus."""
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(y_pred, residuals, s=8, alpha=0.3, color="#2980b9")
    axes[0].axhline(0, color="red", linewidth=1)
    axes[0].set_xlabel("Valeur prédite (€)")
    axes[0].set_ylabel("Résidu = réel − prédit (€)")
    axes[0].set_title("Résidus vs prédiction")

    axes[1].hist(residuals, bins=60, color="#27ae60", alpha=0.8)
    axes[1].axvline(0, color="red", linewidth=1)
    axes[1].set_xlabel("Résidu (€)")
    axes[1].set_ylabel("Fréquence")
    axes[1].set_title("Distribution des résidus")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
