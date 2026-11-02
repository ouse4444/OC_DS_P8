# Image du tableau de bord Streamlit : une application séparée de l'API.
# Elle se construit avec le dossier dashboard/ comme contexte : docker build -t credit-scoring-dashboard dashboard/
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dépendances avant le code, pour profiter du cache de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Le code et le profil agrégé de la référence (aucune donnée client)
COPY __init__.py app.py metrics.py reference_profile.json ./dashboard/

# Sécurité : le tableau de bord ne tourne pas avec les droits d'administrateur
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

# Adresse de l'API par défaut ; elle est remplacée au déploiement (variable API_URL).
# DATABASE_URL est fournie au déploiement sous forme de secret, jamais dans l'image.
ENV API_URL=http://127.0.0.1:8000

EXPOSE 8501

# Mode sans navigateur, pas de statistiques d'usage envoyées à Streamlit, barre d'outils
# réduite à la consultation (le tableau de bord ne se modifie pas en ligne)
CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", \
     "--browser.gatherUsageStats=false", "--client.toolbarMode=viewer"]
