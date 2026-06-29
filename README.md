# Scouting sportif — Estimation de la valeur marchande

Outil d'aide à la décision pour le scouting football : estimation de la valeur de transfert
d'un joueur à partir de ses statistiques techniques et physiques (données EA Sports FC / FIFA).

> **Projet 2 — Data Science (Bloc 2, RNCP40875).** Logique : besoin métier → choix techniques
> → réalisation → preuve → résultat → limites. Voir `rapport_projet2.md` et `slides_projet2.md`.

## Objectif

Prédire `value_eur` (valeur marchande en euros) à partir des stats d'un joueur, et identifier
des profils **sous-cotés** via le signal `potential_gap = potential − overall`.

---

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate   # environnement isolé
pip install -r requirements.txt                      # dépendances figées
```

---

## Démarrage rapide (reproductible après un simple `git clone`)

```bash
python src/pipeline.py        # entraîne, optimise, évalue, sauvegarde le modèle
streamlit run streamlit_app.py
```

> **Données** : si `data/raw/male_players.csv` est absent, le pipeline **génère
> automatiquement un échantillon synthétique** (`src/data/make_dataset.py`) pour que tout
> tourne sans téléchargement. ⚠️ Les métriques obtenues sur ce synthétique **n'ont pas de
> valeur métier** — voir « Données réelles » ci-dessous.

### Données réelles (résultats exploitables)

Le vrai jeu de données provient de Kaggle (*EA Sports FC 24 complete player dataset* / sofifa,
~18 000 joueurs). Il n'est pas versionné (licence + volume + auth Kaggle requise).

```bash
# Avec l'API Kaggle configurée (~/.kaggle/kaggle.json) :
kaggle datasets download -d stefanoleone992/ea-sports-fc-24-complete-player-dataset
unzip -o ea-sports-fc-24-complete-player-dataset.zip -d data/raw/
# Le fichier joueurs doit être accessible en data/raw/male_players.csv (cf. config/params.yaml).
python src/pipeline.py
```

---

## Utilisation détaillée

### 1. Entraîner & évaluer les modèles
```bash
python src/pipeline.py
```
Ce script : charge/filtre les joueurs FIFA 24 → valide le schéma → nettoie (sans fuite) →
sélectionne les features → entraîne un **baseline** + 4 modèles (Ridge, Random Forest,
Gradient Boosting, MLP) avec **validation croisée** et **optimisation d'hyperparamètres**
(`RandomizedSearchCV`) → sauvegarde le meilleur dans `models/best_model.joblib` → génère
`reports/metrics.csv`, `reports/figures/shap_summary.png` et `reports/figures/residuals.png`.

> Pour une exécution rapide (démo), mettre `tuning.enabled: false` dans `config/params.yaml`.

### 2. Générer un dataset synthétique explicitement
```bash
python src/data/make_dataset.py --rows 5000 --out data/raw/male_players.csv
```

### 3. Dashboard interactif
```bash
streamlit run streamlit_app.py
```

### 4. API REST
```bash
uvicorn app.api:app --reload
# POST /predict (corps JSON = schéma PlayerInput) · GET /health
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"overall":82,"potential":90,"age":20,"height_cm":185,"weight_kg":78}'
```

### 5. Tests
```bash
pytest                # 11 tests : données, anti-fuite, pipeline, API
```

### 6. Conteneur Docker
```bash
docker build -t football-scout .
docker run -p 8000:8000 football-scout      # API sur http://localhost:8000
```

---

## Structure

```
config/params.yaml          Configuration centralisée (features, modèles, CV, tuning, chemins)
src/
  config.py                 Chargement du YAML
  pipeline.py               Point d'entrée (orchestration + fallback données synthétiques)
  data/
    make_dataset.py         Génération d'un dataset synthétique réaliste (reproductibilité)
    loader.py               Chargement CSV + filtrage version FIFA
    validation.py           Validation du schéma et de la cible
    cleaning.py             Nettoyage par ligne (SANS imputation -> anti-fuite)
  features/
    feature_selection.py    Sélection des colonnes brutes
    preprocessor.py         ColumnTransformer (impute + StandardScaler + OHE), fit sur train
    custom_transformers.py  PotentialGapTransformer (source unique de potential_gap)
  models/
    model_factory.py        Pipeline complet (gap -> preprocessor -> modèle) + 4 modèles
    deep_model.py           MLP (MLPRegressor)
    compare.py              Baseline + CV + RandomizedSearchCV + sélection du meilleur
    evaluate.py             Métriques MAE/RMSE/R² + analyse de résidus
    predict.py              Prédiction depuis un dict de features
    explain.py              Explications SHAP (Tree / Linear / Kernel + background)
  utils/                    logging, sérialisation joblib
app/                        API FastAPI (api/service/schemas)
tests/                      Suite pytest
notebooks/eda.ipynb         Analyse exploratoire (avec sorties)
Dockerfile                  Image de l'API
```

---

## Modèles comparés

| Modèle | Rôle |
|---|---|
| Baseline (moyenne) | Référence triviale — tout modèle utile doit la battre |
| Ridge | Régression linéaire régularisée (baseline « sérieux ») |
| Random Forest | Ensemble d'arbres (capture les non-linéarités) |
| Gradient Boosting (XGBoost) | Boosting par gradient (repli LightGBM puis HistGB si OpenMP absent) |
| MLP | Réseau de neurones (128→64, ReLU, early stopping) |

Tableau comparatif MAE / RMSE / R² **+ RMSE de validation croisée** dans `reports/metrics.csv`.

---

## Notes techniques (le « pourquoi »)

- **Cible `log1p`** : `value_eur` est très asymétrique ; la transformation log symétrise la
  distribution (cf. notebook EDA, skewness avant/après) et stabilise l'apprentissage.
- **Anti-fuite de données** : aucune statistique globale (médiane, moyenne) n'est calculée
  avant le `train_test_split`. Toute imputation/standardisation vit dans le `Pipeline`
  sklearn, donc apprise sur le **train uniquement** (vérifié par test automatisé).
- **`potential_gap` centralisé** : calculé par un transformer sklearn unique, à la fois pour
  l'entraînement, l'API et le dashboard — pas de duplication.
- **Robustesse** : validation croisée K-fold + `RandomizedSearchCV` (pas de réglage à la main).
- **Exclusions** : id, nom, url et cotes de position (`ls`, `st`…) sont écartés pour éviter
  toute fuite de la cible.

---

## Limites assumées

- Données réelles non versionnées (Kaggle) → démo sur synthétique par défaut.
- `RandomizedSearchCV` explore un espace restreint (compromis temps/qualité).
- Pas de suivi d'expériences (MLflow) ni de CI/CD — pistes d'amélioration documentées dans
  `RECAP_FINAL.md`.
