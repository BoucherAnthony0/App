---
marp: true
title: "Projet 2 — Data Science : Estimation de valeur marchande"
paginate: true
theme: default
---

# ⚽ Estimation de la valeur marchande de joueurs

### Outil d'aide à la décision pour le scouting football

Projet 2 — Data Science · Bloc 2 (C3.1 → C4.3) · RNCP40875

*Binôme — soutenance M1 Data Engineering & IA*

<!-- Notes : se présenter, annoncer le plan : besoin → méthode → preuves → résultats → limites. -->

---

## 1. Besoin métier

- **Problème** : repérer les joueurs **sous-cotés** avant l'explosion de leur valeur.
- **Cible** : estimer `value_eur` à partir des stats techniques/physiques.
- **Signal clé** : `potential_gap = potential − overall` (jeunes à fort upside).
- **Utilisateurs** : recruteurs, directeurs sportifs.

> Besoin d'un modèle **fiable, explicable, déployé** (API + dashboard).

---

## 2. Données & analyse exploratoire *(C3.1, C3.2)*

- Dataset EA Sports FC 24 (~18 000 joueurs) ; **fallback synthétique** pour la reproductibilité.
- Cible **très asymétrique** → transformation **`log1p`** (skewness divisée par ~10).
- `overall` / `potential` = variables les plus corrélées (cohérence métier).
- Stats de gardien nulles pour les joueurs de champ → traitement dédié.

![h:300](reports/figures/shap_summary.png)

<!-- Montrer le notebook EDA : histogrammes brut vs log, heatmap corrélations. -->

---

## 3. Architecture / méthode *(C3.3, C3.4, C4.1)*

```
données → features (PotentialGapTransformer) → preprocessor
   → [Ridge | RandomForest | GradientBoosting | MLP] → SHAP → API/Dashboard
```

- **Pipeline sklearn** : imputation + scaling + OHE **appris sur le train seul** → anti-fuite.
- `potential_gap` calculé par un **transformer unique** (entraînement = API = dashboard).
- Code **modulaire** (pas de notebook monolithique).

---

## 4. Évaluation rigoureuse *(C4.2)*

- **Baseline** `DummyRegressor` (référence à battre).
- **Validation croisée** K-fold (robustesse, sur le train).
- **`RandomizedSearchCV`** par modèle (optimisation, pas de réglage manuel).
- **Analyse de résidus** (recul critique).

![h:280](reports/figures/residuals.png)

---

## 5. Preuves & résultats *(vraies données, FIFA 24, ~18 000 joueurs)*

| Modèle | MAE (€) | RMSE (€) | R² | CV RMSE (log) |
|---|---|---|---|---|
| baseline | 2 318 653 | 8 303 828 | −0,05 | 1,238 |
| ridge | 661 973 | 5 785 498 | 0,49 | 0,219 |
| random_forest | 169 561 | 1 899 166 | 0,945 | 0,064 |
| gradient_boosting | 116 262 | 1 139 951 | 0,980 | **0,047** |
| **mlp** *(retenu)* | 199 583 | **890 860** | **0,988** | 0,094 |

> 🔎 MLP = meilleur RMSE test ; GB = meilleure MAE + CV la plus stable → débat sélection test vs CV.

- ✅ 11/11 tests `pytest` (dont 2 anti-fuite) · API `/predict` opérationnelle · Docker.

---

## 6. Industrialisation *(C4.3)*

- **API FastAPI** : `POST /predict`, `GET /health`, validation Pydantic (bornes métier).
- **Dashboard Streamlit** : simulation de profil + SHAP individuel.
- **Tests** `pytest`, **Dockerfile**, dépendances **figées**.
- Reproductible : `git clone` → `pip install` → `python src/pipeline.py`.

---

## 7. Limites & améliorations

**Limites assumées**
- Données réelles non versionnées (Kaggle) ; espace de tuning restreint.
- Hétéroscédasticité sur les valeurs extrêmes ; pas de MLflow / CI.

**Améliorations**
- Intégration Kaggle + DVC ; Optuna / intervalles de prédiction.
- MLflow (tracking) ; CI GitHub Actions ; monitoring de drift.

---

## 8. Compétences démontrées

| | |
|---|---|
| **C3.1/3.2** | Préparation + EDA (notebook exécuté, log1p justifié) |
| **C3.3/3.4** | `potential_gap` centralisé ; pipeline anti-fuite (testé) |
| **C4.1** | 4 modèles + MLP |
| **C4.2** | baseline + CV + tuning + résidus |
| **C4.3** | API, dashboard, tests, Docker |

### Merci — questions ?
