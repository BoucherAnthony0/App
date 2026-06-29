# GAP ANALYSIS — Projet 2 Data Science — RNCP40875 (Bloc 2)

> Échelle de la grille : **Non démontré (0) / Débutant (2) / Intermédiaire (3) / Professionnel (5)**.
> Logique jury : Besoin → Choix → Réalisation → **Preuve** → Résultat → Limites.

> ⚠️ **Intitulés des compétences à confirmer.** La consigne cite « C3.1 → C4.3 » sans fournir
> le texte exact du référentiel. Les libellés ci-dessous sont mon interprétation des
> compétences standard du Bloc 2 « concevoir et déployer une solution de data science ».
> **Merci de corriger les intitulés** si ta grille officielle diffère — le mapping
> réalisation→compétence reste valable.

## Tableau de synthèse

| Compétence | Critère (interprété) | Niveau actuel | Preuve actuelle | Manque | Action corrective |
|---|---|:---:|---|---|---|
| **C3.1** | Collecte & préparation des données (nettoyage, manquants, outliers) | **3** | `cleaning.py`, `validation.py` existent et tournent | Données introuvables ; imputation **fuyante** (avant split) ; outliers définis mais non branchés | Script d'acquisition/échantillon ; déplacer toute imputation dans le pipeline (fit train) ; brancher/justifier le traitement outliers |
| **C3.2** | Analyse exploratoire (EDA, dataviz, stats) | **0** | — (`eda.ipynb` vide) | EDA inexistante | Notebook EDA complet : distributions, `log1p` justifié, corrélations, manquants, cardinalité catégorielles, conclusions → features |
| **C3.3** | Feature engineering & sélection de variables | **3** | `potential_gap`, exclusion colonnes fuyantes documentée | Pas de mesure d'importance/sélection objective ; feature `potential_gap` dupliquée en 3 endroits | Centraliser `potential_gap` (transformer unique) ; importance permutation / corrélation ; justifier la sélection |
| **C3.4** | Pipeline de préparation reproductible & anti-fuite | **3** | `ColumnTransformer` (scaler fit train) ✅ | Imputation hors pipeline (fuite, §3.3) ; deps non figées ; pas de seed global tracé | Tout dans `Pipeline` sklearn ; figer `requirements.txt` ; documenter la garantie anti-fuite |
| **C4.1** | Conception & entraînement de modèles ML/DL adaptés | **5** | 4 modèles (Ridge/RF/GB/MLP), `log1p`, repli XGB→LGBM→Hist | MLP « deep » à argumenter (sklearn) ; sinon conforme | Documenter le « pourquoi » de chaque modèle ; OK |
| **C4.2** | Évaluation, comparaison & optimisation des modèles | **2** | MAE/RMSE/R², tableau comparatif, SHAP | Split unique, **0 validation croisée**, **0 tuning** d'hyperparamètres, pas de baseline trivial, pas d'analyse de résidus | `cross_val_score` + `GridSearch/RandomizedSearchCV` ; baseline moyenne ; résidus + courbe d'apprentissage |
| **C4.3** | Industrialisation / déploiement (API, dashboard, sérialisation, qualité) | **3** | FastAPI `/predict` `/health`, Streamlit, `joblib`, SHAP UI | Aucun **test**, deps non figées, pas de `Dockerfile`, pas de `.env.example`, pas de CI | Suite `pytest` (data/pipeline/API) ; figer deps ; Dockerfile + `.env.example` ; (option CI) |

## Niveau global actuel (estimation)

- Moyenne pondérée ≈ **2,4 / 5** — tirée vers le bas par C3.2 (0), C4.2 (2), C3.1/C3.4 (fuite).
- Le socle technique vise le 5 ; ce sont les **preuves** (EDA, CV, repro, tests) qui manquent.

## Actions correctives regroupées par lot

### Lot A — Reproductibilité (P0) — *non destructif, peut démarrer sans validation*
- A1. Script `src/data/make_dataset.py` : génère un **échantillon synthétique réaliste**
  versionnable (pour que le repo tourne partout) **+** documentation du téléchargement du
  vrai dataset Kaggle (EA Sports FC 24). README mis à jour.
- A2. Figer `requirements.txt` (versions exactes testées) + `.env.example`.

### Lot B — Méthodologie (P1) — *touche la logique, demande validation*
- B1. Supprimer l'imputation fuyante de `cleaning.py` (la garder uniquement dans le pipeline) ;
  garder l'ancienne version en historique git + justification.
- B2. Ajouter validation croisée (`cross_val_score`) et optimisation (`RandomizedSearchCV`)
  dans `compare.py` ; baseline `DummyRegressor` ; analyse de résidus.

### Lot C — Preuves & finition (P1/P2)
- C1. Notebook EDA complet (`notebooks/eda.ipynb`).
- C2. Suite `pytest` (`tests/`) : validation schéma, anti-fuite, pipeline end-to-end, API.
- C3. Centraliser `potential_gap` (brancher `PotentialGapTransformer`, supprimer duplications).
- C4. `Dockerfile`, analyses post-modèle, captures de preuves.

### Lot D — Livrables non-code
- `rapport_projet2.md`, `slides_projet2.md`, `RECAP_FINAL.md`.

## Décisions qui requièrent ta validation (Phase 3)

1. **Stratégie de données** (le point le plus structurant) : voir question posée en chat.
2. **Refonte de `compare.py`** pour intégrer CV + tuning (Lot B2) — change le flux d'entraînement.
3. **Suppression de l'imputation dans `cleaning.py`** (Lot B1) — modifie une brique existante.
