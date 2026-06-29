# Image de l'API de scoring (FastAPI). POURQUOI : industrialisation reproductible (C4.3) —
# l'environnement d'exécution est figé et identique en dev, démo et déploiement.
FROM python:3.11-slim

# libgomp1 : requis par xgboost/lightgbm (OpenMP) en environnement slim.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Couche dépendances séparée -> cache Docker réutilisé tant que requirements ne change pas.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Entraîne un modèle au build si aucun n'est présent (utilise le fallback synthétique).
# En production réelle, monter un volume avec models/best_model.joblib pré-entraîné.
RUN python src/pipeline.py || echo "Pipeline d'amorçage ignoré (modèle fourni au runtime)."

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
