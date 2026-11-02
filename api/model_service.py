"""Chargement du modèle et calcul des prédictions.

Ce module isole tout ce qui touche au modèle : l'API (main.py) ne manipule
ni pandas ni scikit-learn, ce qui la rend plus simple à lire et à tester.

Optimisations (phase 9, mesurées puis validées sans aucun écart de prédiction) :
- le modèle est lu directement dans le fichier model.pkl exporté par MLflow (format
  cloudpickle), sans importer MLflow : démarrage plus rapide et image plus légère.
  MLflow reste l'outil de suivi des expériences et de versionnement du modèle ;
- la ligne de données est construite colonne par colonne avec le bon type, au lieu
  d'être convertie après coup : environ 7 fois plus rapide.
"""

import time
import warnings
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd

from api.config import ENGINEERED_FEATURES

# La préparation transmet au LightGBM un tableau sans noms de colonnes, alors qu'il a été
# entraîné avec des noms. L'ordre des colonnes est garanti par le pipeline : cet
# avertissement, sans conséquence, est masqué pour ne pas encombrer les journaux.
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Avertissement de pandas émis par la classe ConstructeurFeatures du Projet 6, enregistrée
# dans le modèle. Sans effet avec la version figée de pandas : il est masqué pour garder
# des journaux de production lisibles.
warnings.filterwarnings("ignore", message="Downcasting behavior", category=FutureWarning)


class CreditScoringModel:
    """Enveloppe autour du pipeline du Projet 6 (préparation des variables, puis LightGBM)."""

    def __init__(self, model_dir: Path):
        # Chargement unique : cet objet est créé une seule fois, au démarrage de l'API.
        # Le fichier model.pkl est celui du registre MLflow (format cloudpickle, indiqué dans
        # le fichier MLmodel) : il est lu directement, sans charger toute la bibliothèque MLflow
        with open(Path(model_dir) / "model.pkl", "rb") as model_file:
            self.pipeline = cloudpickle.load(model_file)

        # Les variables attendues sont lues dans le modèle lui-même, et non recopiées
        # à la main : une seule source de vérité, aucun risque de divergence.
        column_transformer = self.pipeline.named_steps["preparation"].named_steps["preparation"]
        self.input_features = [
            col for col in column_transformer.feature_names_in_
            if col not in ENGINEERED_FEATURES
        ]
        # Les variables catégorielles sont celles de la branche « cat » du préprocesseur
        self.categorical_features = next(
            cols for name, _, cols in column_transformer.transformers_ if name == "cat"
        )
        self.numeric_features = [
            col for col in self.input_features if col not in self.categorical_features
        ]

    def check_input(self, data: dict) -> list[str]:
        """Contrôle les variables reçues : nom connu et type correct.

        Retourne la liste des erreurs détectées (liste vide si les données sont valides).
        """
        errors = []

        # Une variable inconnue du modèle signale une erreur côté client (faute de frappe,
        # mauvaise version du contrat) : elle est refusée plutôt qu'ignorée en silence
        unknown = sorted(set(data) - set(self.input_features))
        if unknown:
            errors.append(f"Unknown features: {unknown}")

        for name, value in data.items():
            # Une valeur absente est acceptée : le pipeline l'imputera
            if value is None or name in unknown:
                continue
            # bool est un sous-type de int en Python : il est exclu explicitement
            if name in self.numeric_features and (
                isinstance(value, bool) or not isinstance(value, (int, float))
            ):
                errors.append(f"Feature '{name}' must be a number, got {value!r}")
            if name in self.categorical_features and not isinstance(value, str):
                errors.append(f"Feature '{name}' must be a string, got {value!r}")

        return errors

    def to_dataframe(self, data: dict) -> pd.DataFrame:
        """Construit une ligne au format exact attendu par le pipeline.

        Les variables absentes deviennent des valeurs manquantes : le pipeline les
        impute comme à l'entraînement (zéro pour une absence d'historique, médiane
        sinon, modalité « Non renseigné » pour les catégorielles).

        Chaque colonne est créée directement avec son type final (décimal pour les
        numériques, objet pour les catégorielles). Convertir 188 colonnes après coup,
        comme dans la première version, coûtait environ 17 ms par requête, soit plus
        que le modèle lui-même ; la construction directe en coûte environ 2.
        """
        # Une valeur absente (None) devient NaN, la valeur manquante attendue par le pipeline
        values = {
            col: [np.nan if data.get(col) is None else float(data[col])]
            for col in self.numeric_features
        }
        values.update({
            col: np.array([np.nan if data.get(col) is None else data[col]], dtype=object)
            for col in self.categorical_features
        })
        # Toutes les colonnes, dans l'ordre exact de l'entraînement
        return pd.DataFrame(values, columns=self.input_features)

    def predict_proba(self, data: dict) -> tuple[float, float]:
        """Retourne la probabilité de défaut et le temps d'inférence en millisecondes."""
        X = self.to_dataframe(data)
        # Seul le calcul du modèle est chronométré : c'est le temps d'inférence
        start = time.perf_counter()
        proba = float(self.pipeline.predict_proba(X)[0, 1])
        inference_ms = (time.perf_counter() - start) * 1000
        return proba, inference_ms