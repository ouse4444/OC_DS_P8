"""Chargement du modèle et calcul des prédictions.

Ce module isole tout ce qui touche au modèle : l'API (main.py) ne manipule
ni pandas ni scikit-learn, ce qui la rend plus simple à lire et à tester.
"""

import time
import warnings
from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd

from api.config import ENGINEERED_FEATURES

# La préparation transmet au LightGBM un tableau sans noms de colonnes, alors qu'il a été
# entraîné avec des noms. L'ordre des colonnes est garanti par le pipeline : cet
# avertissement, sans conséquence, est masqué pour ne pas encombrer les journaux.
warnings.filterwarnings("ignore", message="X does not have valid feature names")


class CreditScoringModel:
    """Enveloppe autour du pipeline du Projet 6 (préparation des variables, puis LightGBM)."""

    def __init__(self, model_dir: Path):
        # Chargement unique : cet objet est créé une seule fois, au démarrage de l'API
        self.pipeline = mlflow.sklearn.load_model(str(model_dir))

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
        """
        # Toutes les colonnes, dans l'ordre exact de l'entraînement
        row = {col: data.get(col) for col in self.input_features}
        df = pd.DataFrame([row], columns=self.input_features)

        # Les variables numériques sont converties en nombres décimaux (None devient NaN)
        df[self.numeric_features] = df[self.numeric_features].astype(float)

        # Les catégorielles manquantes doivent valoir np.nan, et non None, pour être imputées
        cat = df[self.categorical_features].astype(object)
        df[self.categorical_features] = cat.where(cat.notna(), np.nan)
        return df

    def predict_proba(self, data: dict) -> tuple[float, float]:
        """Retourne la probabilité de défaut et le temps d'inférence en millisecondes."""
        X = self.to_dataframe(data)
        # Seul le calcul du modèle est chronométré : c'est le temps d'inférence
        start = time.perf_counter()
        proba = float(self.pipeline.predict_proba(X)[0, 1])
        inference_ms = (time.perf_counter() - start) * 1000
        return proba, inference_ms