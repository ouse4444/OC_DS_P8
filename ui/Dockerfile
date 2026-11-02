# Image de l'interface Gradio : un client de l'API, séparé de celle-ci.
# Elle se construit avec le dossier ui/ comme contexte : docker build -t credit-scoring-ui ui/
FROM python:3.12-slim

# Pas de fichiers .pyc, journaux affichés immédiatement, pas de cache pip.
# Gradio n'envoie pas de statistiques d'usage vers l'extérieur.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_ANALYTICS_ENABLED=False

WORKDIR /app

# Dépendances avant le code, pour profiter du cache de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Le code est copié dans un dossier ui/ pour être lancé comme module (python -m ui.app)
COPY __init__.py app.py ./ui/

# Sécurité : l'interface ne tourne pas avec les droits d'administrateur
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

# Adresse de l'API par défaut ; elle est remplacée au déploiement (variable API_URL)
ENV API_URL=http://127.0.0.1:8000

EXPOSE 7860
CMD ["python", "-m", "ui.app"]
