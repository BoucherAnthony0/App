# Scouting sportif — Estimation de la valeur marchande

Outil d'aide à la décision pour le scouting football : estimation de la valeur de transfert d'un joueur à partir de ses statistiques techniques et physiques (données EA Sports FC / FIFA).

## Objectif

Prédire `value_eur` (valeur marchande en euros) à partir des stats d'un joueur.  
Identifier des profils **sous-cotés** via le signal `potential_gap = potential − overall`.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Utilisation

### 1. Entraîner les modèles

```bash
python src/pipeline.py
```

Ce script :
- Filtre les joueurs FIFA 24 (~18 350 joueurs)
- Nettoie les données et sélectionne les features
- Entraîne et compare 4 modèles (Ridge, Random Forest, XGBoost, MLP)
- Sauvegarde le meilleur modèle dans `models/best_model.joblib`
- Génère `reports/metrics.csv` et `reports/figures/shap_summary.png`

### 2. Lancer le dashboard

```bash
streamlit run streamlit_app.py
```

### 3. Lancer l'API REST (optionnel)

```bash
uvicorn app.api:app --reload
```

Endpoint : `POST /predict` — corps JSON selon le schéma `PlayerInput`.

---

## Structure

```
config/params.yaml          Configuration centralisée (features, modèles, chemins)
src/
  config.py                 Chargement du YAML
  pipeline.py               Point d'entrée principal
  data/
    loader.py               Chargement CSV + filtrage version FIFA
    validation.py           Validation du schéma et de la cible
    cleaning.py             Nettoyage (cible nulle, valeurs manquantes)
  features/
    feature_selection.py    Sélection + feature engineering (potential_gap)
    preprocessor.py         ColumnTransformer (StandardScaler + OHE)
    custom_transformers.py  Transformer sklearn PotentialGapTransformer
  models/
    model_factory.py        Instanciation des 4 modèles
    deep_model.py           MLP (MLPRegressor sklearn)
    compare.py              Entraînement, évaluation, sélection du meilleur modèle
    train.py                Entraînement générique d'un pipeline
    evaluate.py             Métriques MAE / RMSE / R²
    predict.py              Prédiction depuis un dict de features
    explain.py              Explications SHAP (TreeExplainer / LinearExplainer / KernelExplainer)
  utils/
    logger.py               Configuration du logging
    serialization.py        Sauvegarde / chargement du modèle (joblib)
app/
  schemas.py                Schémas Pydantic (PlayerInput, PredictionResponse)
  service.py                Logique métier (appel predict.py)
  api.py                    API FastAPI (POST /predict)
streamlit_app.py            Dashboard interactif
```

---

## Modèles comparés

| Modèle | Description |
|---|---|
| Ridge | Régression linéaire régularisée (baseline) |
| Random Forest | Ensemble d'arbres de décision |
| Gradient Boosting (XGBoost) | Boosting par gradient |
| MLP | Réseau de neurones multi-couches (128→64, ReLU, early stopping) |

Le tableau comparatif MAE / RMSE / R² est généré dans `reports/metrics.csv`.

---

## Notes techniques

- La cible `value_eur` est transformée en `log1p` avant l'entraînement (distribution très asymétrique).
- La standardisation (`StandardScaler`) est appliquée dans le pipeline avant le MLP, conformément à l'exigence de normalisation des entrées.
- Les colonnes d'identifiant, de nom, d'URL et les cotes de position (type `ls`, `st`) sont exclues pour éviter toute fuite de données.
