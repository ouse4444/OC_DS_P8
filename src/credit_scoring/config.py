"""Configuration centrale du projet : chemins et constantes partages."""
from pathlib import Path

# Racine du projet (trois niveaux au-dessus : src/credit_scoring/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Repertoires de donnees
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# Repertoires de sortie
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Colonne cible et cles de jointure
TARGET = "TARGET"
KEY_CURR = "SK_ID_CURR"
KEY_BUREAU = "SK_ID_BUREAU"
KEY_PREV = "SK_ID_PREV"

# Graine aleatoire pour la reproductibilite
RANDOM_STATE = 42
