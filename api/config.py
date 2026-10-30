"""Paramètres de l'API, regroupés dans un seul fichier.

Les valeurs peuvent être remplacées par des variables d'environnement :
c'est ce qui permettra plus tard de configurer l'API dans Docker ou sur Azure
sans modifier le code.
"""

import os
from pathlib import Path

# Racine du projet : le dossier qui contient api/ et model/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dossier du modèle exporté depuis le registre MLflow du Projet 6
MODEL_DIR = Path(os.getenv("MODEL_DIR", PROJECT_ROOT / "model"))

# Identité du modèle servi, renvoyée dans chaque réponse pour la traçabilité
MODEL_NAME = "credit_scoring_lightgbm_fbeta"
MODEL_VERSION = "1"

# Seuil de décision optimisé au Projet 6 (F-bêta avec bêta = racine de 10)
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.36"))

# Variables créées par l'étape de préparation du pipeline (classe ConstructeurFeatures).
# Elles sont calculées par le modèle lui-même : le client ne doit donc pas les envoyer.
ENGINEERED_FEATURES = [
    "RATIO_CREDIT_REVENU",
    "RATIO_ANNUITE_REVENU",
    "RATIO_ANNUITE_CREDIT",
    "REVENU_PAR_PERSONNE",
    "RATIO_EMPLOI_AGE",
    "AGE_ANNEES",
    "ANCIENNETE_EMPLOI_ANNEES",
]