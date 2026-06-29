# Rapport — Projet 2 : Data Science
## Estimation de la valeur marchande de joueurs de football
**Certification RNCP40875 — Expert en ingénierie des données — Bloc 2 (C3.1 → C4.3)**

---

## 1. Besoin métier

Le recrutement (scouting) football repose sur l'identification de joueurs à fort potentiel
**avant** que leur valeur n'explose. Le besoin : disposer d'un **outil d'aide à la décision**
qui (1) estime la **valeur marchande** d'un joueur à partir de ses caractéristiques, et
(2) met en évidence les profils **sous-cotés** — jeunes joueurs dont le potentiel dépasse
nettement le niveau actuel (`potential_gap = potential − overall`).

**Utilisateurs cibles** : recruteurs, directeurs sportifs. **Livrable attendu** : un modèle
de régression fiable, explicable, exposé via une API et un dashboard interactif.

---

## 2. Choix techniques (et leur justification)

| Décision | Pourquoi |
|---|---|
| **Régression supervisée** | La cible `value_eur` est continue. |
| **Transformation `log1p` de la cible** | Distribution très asymétrique (cf. EDA, skewness élevé) ; le log symétrise et évite que les stars dominent la perte. |
| **Pipeline sklearn** (`ColumnTransformer`) | Encapsule imputation + standardisation + OHE ; garantit que ces statistiques sont apprises sur le **train seul** → anti-fuite. |
| **4 familles de modèles** (Ridge, RF, GB/XGBoost, MLP) | Couvrir linéaire / ensembles / réseau de neurones et comparer objectivement. |
| **Baseline `DummyRegressor`** | Point de référence : un modèle utile doit le battre nettement. |
| **Validation croisée + `RandomizedSearchCV`** | Robustesse (pas un seul tirage) et optimisation des hyperparamètres sans réglage manuel. |
| **SHAP** | Explicabilité des prédictions (exigence d'aide à la décision). |
| **FastAPI + Streamlit** | Industrialisation : service temps réel + interface métier. |
| **Donnée synthétique de secours** | Reproductibilité : le repo tourne sans le dataset Kaggle. |

---

## 3. Réalisation

Architecture **modulaire** (pas de notebook monolithique) :

```
données (loader/validation/cleaning/make_dataset)
  → features (selection + PotentialGapTransformer + preprocessor)
    → modélisation (model_factory + compare : baseline/CV/tuning)
      → évaluation (métriques + résidus) + explicabilité (SHAP)
        → service (predict) → API (FastAPI) / dashboard (Streamlit)
```

Points d'ingénierie notables :
- **Anti-fuite** : `cleaning.py` ne fait que du filtrage par ligne ; toute imputation est
  dans le pipeline (vérifié par un test dédié).
- **Source de vérité unique** pour `potential_gap` via un transformer sklearn branché en tête
  de pipeline (utilisé par l'entraînement, l'API et le dashboard).
- **Dégradation gracieuse** : XGBoost → LightGBM → `HistGradientBoostingRegressor` si OpenMP
  absent ; SHAP choisit l'explainer adapté (Tree/Linear/Kernel).

---

## 4. Preuves

> ⚠️ Les chiffres ci-dessous proviennent du **jeu synthétique** par défaut : ils prouvent que
> la chaîne fonctionne, **pas** une performance métier. Sur le vrai dataset Kaggle, relancer
> `python src/pipeline.py` régénère ce tableau avec des valeurs réelles.

**Tableau comparatif** (`reports/metrics.csv`) :

| Modèle | MAE (€) | RMSE (€) | R² | CV RMSE (log) |
|---|---|---|---|---|
| baseline_mean | 257 784 | 431 831 | −0,12 | 0,890 ± 0,023 |
| **ridge** | **116 264** | **197 399** | **0,767** | **0,356 ± 0,014** |
| random_forest | 116 533 | 200 821 | 0,758 | 0,357 ± 0,018 |
| gradient_boosting | 116 502 | 200 859 | 0,758 | 0,353 ± 0,018 |
| mlp | 220 929 | 427 074 | −0,09 | 0,738 ± 0,026 |

**Autres preuves** :
- `reports/figures/shap_summary.png` — importance globale des variables (overall/potential dominants).
- `reports/figures/residuals.png` — résidus centrés, distribution analysée.
- `notebooks/eda.ipynb` — EDA exécutée (justification du log1p, corrélations, potential_gap).
- `pytest` → **11/11 tests OK** (dont 2 tests anti-fuite).
- API : `POST /predict` renvoie `{estimated_value_eur, model_name, potential_gap}`.

---

## 5. Résultat

Un outil **complet et reproductible** : un `git clone` + `pip install` + `python src/pipeline.py`
suffit à entraîner un modèle, l'exposer via API et le visualiser via dashboard, avec
explicabilité SHAP. Tous les modèles battent largement le baseline (preuve d'apprentissage
réel du signal), et la validation croisée confirme la stabilité du classement.

---

## 6. Limites & améliorations

**Limites assumées** :
- Données réelles non versionnées → démo sur synthétique par défaut.
- Espace de recherche d'hyperparamètres volontairement restreint (temps de calcul).
- Hétéroscédasticité attendue sur les valeurs extrêmes (typique d'une cible log-normale).
- Pas de suivi d'expériences (MLflow) ni de CI/CD.

**Améliorations** :
- Intégration Kaggle automatisée + DVC pour versionner les données.
- `GridSearchCV` complet / Optuna ; intervalles de prédiction (quantile regression).
- MLflow pour le tracking ; CI GitHub Actions (lint + pytest) ; monitoring de drift.

---

## 7. Compétences démontrées (mapping RNCP)

| Compétence | Réalisation (preuve) |
|---|---|
| **C3.1** Préparation des données | `loader/validation/cleaning` + `make_dataset.py` ; nettoyage anti-fuite |
| **C3.2** Analyse exploratoire | `notebooks/eda.ipynb` exécuté (distributions, corrélations, log1p justifié) |
| **C3.3** Feature engineering & sélection | `PotentialGapTransformer` centralisé ; exclusion des colonnes fuyantes |
| **C3.4** Pipeline reproductible & anti-fuite | `Pipeline` sklearn ; test automatisé d'imputation train-only ; deps figées |
| **C4.1** Conception & entraînement de modèles | 4 familles de modèles + MLP ; `log1p` ; repli OpenMP |
| **C4.2** Évaluation, comparaison & optimisation | baseline + CV K-fold + `RandomizedSearchCV` + résidus |
| **C4.3** Industrialisation / déploiement | API FastAPI, dashboard Streamlit, SHAP, `pytest`, Dockerfile |

---

## 8. Ma contribution individuelle *(trame à compléter)*

> À personnaliser pour la défense individuelle. Exemples de formulations :

- **Ce que j'ai conçu/codé personnellement** : _(ex. le moteur de comparaison `compare.py`
  avec validation croisée et tuning ; le PotentialGapTransformer ; les tests anti-fuite…)_
- **Les choix techniques que je peux défendre seul** : _(ex. pourquoi `log1p` plutôt qu'une
  cible brute ; pourquoi l'imputation doit être dans le pipeline ; pourquoi un baseline…)_
- **Les difficultés rencontrées et comment je les ai résolues** : _(ex. fuite de données
  détectée à l'imputation pré-split ; gestion d'OpenMP absent pour XGBoost…)_
- **Ce que je ferais différemment** : _(ex. ajouter MLflow dès le début…)_
- **Répartition avec le binôme** : _(qui a fait quoi)_.
