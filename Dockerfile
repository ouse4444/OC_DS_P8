# Image de base : Python 3.12, la version du Projet 6, dans sa variante « slim »
# (Debian réduit au strict nécessaire, donc une image plus légère et moins exposée)
FROM python:3.12-slim

# Pas de fichiers .pyc (inutiles dans un conteneur), et des journaux affichés
# immédiatement, sans mise en mémoire tampon, pour les suivre en direct
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# LightGBM a besoin de la bibliothèque système OpenMP (libgomp1), absente de l'image slim.
# Le cache des paquets est supprimé dans la même instruction pour alléger l'image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Dossier de travail de l'application dans le conteneur
WORKDIR /app

# Les dépendances sont copiées et installées AVANT le code : Docker garde cette étape en
# cache tant que requirements.txt ne change pas. Une modification du code ne relance donc
# pas l'installation des bibliothèques, ce qui accélère fortement les reconstructions.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du strict nécessaire : le code de l'API et le modèle
COPY api/ ./api/
COPY model/ ./model/

# Sécurité : l'API ne tourne pas avec les droits d'administrateur (root),
# mais avec un utilisateur dédié sans privilèges
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

# Port sur lequel l'API écoute (documentation pour les outils de déploiement)
EXPOSE 8000

# Vérification périodique de l'état de l'API par Docker, via la route /health
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

# Commande de démarrage. --host 0.0.0.0 : l'API accepte les connexions venant de
# l'extérieur du conteneur (avec 127.0.0.1, elle ne serait joignable que de l'intérieur)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
