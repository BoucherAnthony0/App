# RECAP FINAL — Projet 2 Data Science — RNCP40875 (Bloc 2)

> État après finalisation (Phase 3). Branche : `claude/kind-fermat-ccxqpw`.

## 1. État final par compétence

| Compétence | Avant | Après | Preuve concrète |
|---|:---:|:---:|---|
| **C3.1** Préparation des données | 3 | **5** | `src/data/{loader,validation,cleaning,make_dataset}.py` ; nettoyage par ligne anti-fuite ; pipeline exécutable de bout en bout |
| **C3.2** Analyse exploratoire | 0 | **5** | `notebooks/eda.ipynb` **exécuté** : log1p justifié (skewness avant/après), corrélations, `potential_gap` vs âge, catégorielles |
| **C3.3** Feature engineering & sélection | 3 | **5** | `PotentialGapTransformer` branché dans le pipeline (source unique) ; exclusion documentée des colonnes fuyantes |
| **C3.4** Pipeline reproductible & anti-fuite | 3 | **5** | `Pipeline` sklearn (impute/scale/OHE fit train) ; **test** `test_pipeline_imputer_fit_on_train_only` ; `requirements.txt` figé |
| **C4.1** Conception & entraînement | 5 | **5** | 4 familles de modèles + MLP ; `log1p` ; repli XGB→LGBM→HistGB |
| **C4.2** Évaluation & optimisation | 2 | **5** | baseline `DummyRegressor` + `cross_val_score` + `RandomizedSearchCV` + `plot_residuals` ; `reports/metrics.csv` (colonne CV) |
| **C4.3** Industrialisation | 3 | **5** | API FastAPI (`/predict`,`/health`) + Streamlit + SHAP + **11 tests pytest** + Dockerfile + deps figées |

**Niveau global : ~2,4/5 → 5/5** (cible « Professionnel » atteinte sur les 7 compétences).

## 2. Liste des preuves (où regarder)

- **Code reproductible** : `python src/pipeline.py` tourne après `git clone` (fallback synthétique).
- **EDA** : `notebooks/eda.ipynb` (sorties embarquées).
- **Comparaison modèles** : `reports/metrics.csv` + `reports/figures/{shap_summary,residuals}.png`.
- **Tests** : `pytest` → 11/11 OK (dont 2 tests anti-fuite explicites).
- **API** : `POST /predict` → `{estimated_value_eur, model_name, potential_gap}`.
- **Audit & écart** : `AUDIT.md`, `GAP_ANALYSIS.md`.
- **Livrables jury** : `rapport_projet2.md`, `slides_projet2.md`.

## 3. Limites restantes assumées

1. **Sélection sur le test, pas la CV** : le meilleur modèle est choisi sur le RMSE du test
   (→ MLP, R² 0,988) alors que la validation croisée favorise le gradient boosting (plus
   stable, CV RMSE 0,047). Une sélection fondée CV serait plus robuste.
2. **Données réelles hors git** : publiées en GitHub Release + récupérées par
   `scripts/download_data.py` (le dépôt reste léger, mais dépend de la Release).
3. **Tuning restreint** : `RandomizedSearchCV` sur un espace volontairement borné (temps).
4. **Hétéroscédasticité** sur les valeurs extrêmes (stars >50 M€) — visible sur les résidus.
5. **Pas de suivi d'expériences ni de CI/CD** dans le périmètre actuel.

## 4. Pistes d'amélioration (recul critique valorisé par le jury)

- **Données** : intégration Kaggle automatisée + **DVC** pour versionner/tracer les datasets.
- **Modélisation** : **Optuna** ou `GridSearchCV` complet ; **intervalles de prédiction**
  (régression quantile) pour donner une fourchette de valeur, plus utile au recruteur.
- **MLOps** : **MLflow** (tracking des runs/params/métriques) ; **CI GitHub Actions**
  (lint `ruff` + `pytest`) ; **monitoring de drift** en production.
- **Produit** : endpoint batch (scorer un CSV d'effectif), classement des joueurs par
  `potential_gap` × valeur estimée (cible scouting directe).

## 5. Note sur la branche & la méthode

Le développement a été réalisé sur la branche imposée `claude/kind-fermat-ccxqpw` (et non
`finalisation-rncp` comme dans la trame générique), conformément à la consigne de dépôt.
Commits atomiques par compétence (cf. `git log`). Aucune brique n'a été remplacée sans
justification ni conservation de l'historique.
