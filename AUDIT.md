# AUDIT — Projet 2 (Data Science / Bloc 2) — RNCP40875

> Audit réalisé en lecture seule (Phase 1). Aucune logique métier modifiée à ce stade.
> Date : 2026-06-29 · Branche : `claude/kind-fermat-ccxqpw`

## 0. Périmètre constaté

Le dépôt ne contient **qu'un seul projet** : un outil ML d'**estimation de la valeur marchande
de joueurs de football** (données EA Sports FC / FIFA 24). Cela correspond au **Projet 2 —
Data Science (Bloc 2, C3.1 → C4.3)**. Les Projets 1 (Architecture) et 3 (IA générative)
évoqués dans la consigne **ne sont pas présents** dans ce dépôt.

Historique git : **1 seul commit** (`Initial commit: FIFA player ML project scaffold`),
deux branches identiques (`claude/kind-fermat-ccxqpw`, `youcef`). Aucune chronologie de
développement exploitable.

## 1. Ce qui existe (inventaire)

Architecture **modulaire et propre**, conforme à la séparation attendue par le jury :

| Couche | Fichiers | État |
|---|---|---|
| Config | `config/params.yaml`, `src/config.py` | ✅ Centralisée, lisible |
| Données | `src/data/{loader,validation,cleaning}.py` | ✅ Présent |
| Features | `src/features/{feature_selection,preprocessor,custom_transformers}.py` | ⚠️ voir §3 |
| Modèles | `src/models/{model_factory,deep_model,train,compare,evaluate,predict,explain}.py` | ✅ Présent |
| Utils | `src/utils/{logger,serialization}.py` | ✅ Présent |
| API | `app/{api,service,schemas}.py` (FastAPI + Pydantic) | ✅ Fonctionnel |
| Dashboard | `streamlit_app.py` (336 lignes, FR, SHAP) | ✅ Fonctionnel |
| EDA | `notebooks/eda.ipynb` | ❌ **VIDE (0 octet)** |
| Pipeline | `src/pipeline.py` | ✅ Orchestration complète |

**Points forts réels :**
- Pipeline sklearn `ColumnTransformer` (imputation + `StandardScaler` + OHE) → **pas de fuite
  de données dans l'étape de modélisation** (scaler fit sur train uniquement).
- Cible `value_eur` transformée en `log1p` (distribution asymétrique) — choix justifié.
- 4 familles de modèles comparées : Ridge (baseline) / Random Forest / Gradient Boosting
  (XGBoost, avec repli LightGBM puis `HistGradientBoostingRegressor` si OpenMP absent) / MLP.
- Interprétabilité **SHAP** avec sélection automatique de l'explainer (Tree / Linear / Kernel
  + background pré-calculé sauvegardé dans l'artefact pour le MLP).
- API `/predict` + `/health`, validation Pydantic avec bornes métier (`ge`/`le`).
- Exclusion documentée des colonnes fuyantes (id, nom, url, cotes de position `ls`/`st`).

## 2. Ce qui fonctionne (vérifié par exécution)

Environnement isolé `.venv`, `pip install -r requirements.txt` → **OK** (Python 3.11).
Versions effectivement installées (non figées dans le repo) : numpy 2.4.6, pandas 3.0.4,
scikit-learn 1.9.0, shap 0.51.0, xgboost 3.2.0, lightgbm 4.6.0, fastapi 0.138.1,
pydantic 2.13.4, streamlit 1.58.0.

- `python src/pipeline.py` sur le **vrai** dataset → ❌ échoue (cf. §3.1).
- `python src/pipeline.py` sur un **dataset synthétique** que j'ai généré pour l'audit
  (3 000 lignes, même schéma) → ✅ **tourne de bout en bout** : entraîne les 4 modèles,
  écrit `reports/metrics.csv`, sauvegarde `models/best_model.joblib`, génère
  `reports/figures/shap_summary.png`. **Le code est donc fonctionnel.**
- Couche service (`estimate_player_value`) → ✅ renvoie une prédiction valide.

> ⚠️ Les métriques obtenues lors de ce test (R² ≈ 0,68, MLP négatif) **n'ont aucune valeur** :
> elles proviennent de données aléatoires. Elles prouvent seulement que la tuyauterie marche.

## 3. Ce qui est cassé / manquant (sans complaisance)

### 3.1 BLOQUANT — Aucune donnée, aucun moyen d'en obtenir
`config/params.yaml` pointe vers `data/raw/male_players.csv`, **absent du dépôt** (et
gitignoré). Il n'existe **ni script de téléchargement, ni échantillon, ni instructions
Kaggle**. Conséquence : **le projet n'est pas reproductible** — un jury qui clone le repo ne
peut rien lancer. C'est le défaut le plus grave au regard de la grille (« reproductibilité »).

### 3.2 BLOQUANT — Notebook EDA vide
`notebooks/eda.ipynb` fait **0 octet**. L'analyse exploratoire (distributions, corrélations,
valeurs manquantes, justification du `log1p`, du choix des features) **n'existe pas**. C'est
une compétence centrale du Bloc 2 (C3.2) non démontrée.

### 3.3 IMPORTANT — Fuite de données dans le nettoyage
`src/data/cleaning.handle_missing_values()` impute les manquants avec la **médiane/mode
calculés sur l'ensemble du jeu** (`clean_dataset` est appelé **avant** le `train_test_split`
dans `pipeline.py`). C'est une **fuite de données** (le test influence l'imputation du train).
De plus c'est **redondant** : le `ColumnTransformer` ré-impute déjà proprement (fit sur train).

### 3.4 IMPORTANT — Pas de validation croisée ni d'optimisation d'hyperparamètres
Évaluation sur un **unique split 80/20**, hyperparamètres **figés en dur** dans le YAML.
Aucune `cross_val_score` / `GridSearchCV` / `RandomizedSearchCV`. La robustesse de la
comparaison de modèles et l'« optimisation » attendue (C4.2) ne sont pas démontrées.

### 3.5 MOYEN — Aucun test
Pas de `tests/`, pas de pytest. Aucune garantie de non-régression ; argument faible pour
l'industrialisation.

### 3.6 MOYEN — Dépendances non figées
`requirements.txt` ne pin aucune version → résultats non reproductibles dans le temps
(pandas 3.x / sklearn 1.9 introduisent des ruptures).

### 3.7 FAIBLE — Code mort / incohérence
`PotentialGapTransformer` (`custom_transformers.py`) **n'est jamais utilisé** : `potential_gap`
est recalculé à la main dans `feature_selection.py` ET dans `app/service.py`. Trois sources de
vérité pour la même feature → risque d'incohérence.

### 3.8 FAIBLE — Analyses post-modèle absentes
Pas d'analyse de résidus, pas de courbe d'apprentissage, pas d'intervalle d'erreur, pas de
comparaison à un baseline trivial (moyenne). Le « recul critique » sur le modèle est mince.

### 3.9 FAIBLE — Hygiène repo
Pas de `.env.example` (le `.gitignore` ignore `.env` mais aucun secret n'est utilisé — impact
réel nul, mais attendu par la consigne). Pas de `Dockerfile` (optionnel pour le Bloc 2).
README mentionne `reports/` mais aucun livrable de sortie n'est versionné pour preuve.

## 4. Dette technique — synthèse priorisée

| Prio | Sujet | Risque |
|---|---|---|
| P0 | Données absentes + non reproductible (§3.1) | Le jury ne peut rien exécuter |
| P0 | Notebook EDA vide (§3.2) | Compétence C3.2 non démontrée |
| P1 | Fuite de données à l'imputation (§3.3) | Erreur méthodo sanctionnée |
| P1 | Pas de CV / tuning (§3.4) | C4.2 au niveau « débutant » |
| P2 | Pas de tests (§3.5), deps non figées (§3.6) | Industrialisation / repro |
| P3 | Code mort (§3.7), analyses manquantes (§3.8), hygiène (§3.9) | Finition |

## 5. Conclusion de l'audit

Le projet a une **excellente ossature logicielle** (modularité, pipeline, API, dashboard, SHAP)
mais souffre de **deux blocages de reproductibilité** (données + EDA absentes) et d'une **erreur
méthodologique** (fuite à l'imputation) qui, en l'état, plafonneraient plusieurs compétences
sous le niveau « Professionnel ». La suite (analyse d'écart détaillée) est dans
`GAP_ANALYSIS.md`.
