"""Construction du profil de référence utilisé par le tableau de bord.

Le tableau de bord surveille la dérive des données sans avoir accès aux données
d'entraînement : il compare la production à un PROFIL de la référence, c'est-à-dire
des statistiques agrégées, et non des lignes de clients :
- pour une variable numérique : les bornes des déciles et la part de clients dans chaque
  tranche, plus la part de valeurs manquantes ;
- pour une variable catégorielle : la part de chaque catégorie, plus la part manquante.

Ce fichier ne contient aucune donnée individuelle : il peut être versionné et embarqué
dans l'image du tableau de bord.

Les variables retenues sont les 12 variables envoyées à l'API les plus importantes pour le
modèle (gain LightGBM), plus 4 variables métier (revenu, crédit, annuité, type de contrat). Usage, depuis la racine du projet :
    python scripts/build_reference_profile.py
"""

import json
import sys
import warnings
from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from credit_scoring.config import RANDOM_STATE, TARGET  # noqa: E402

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message="Downcasting behavior", category=FutureWarning)

REFERENCE_DATA = PROJECT_ROOT / "data" / "processed" / "application_train_enriched.parquet"
OUTPUT = PROJECT_ROOT / "dashboard" / "reference_profile.json"
N_FEATURES = 12
N_BINS = 10

# Variables construites par le modèle (classe ConstructeurFeatures du Projet 6) : elles ne
# sont pas envoyées à l'API, le tableau de bord ne peut donc pas les suivre directement
ENGINEERED = {"RATIO_CREDIT_REVENU", "RATIO_ANNUITE_REVENU", "RATIO_ANNUITE_CREDIT",
              "REVENU_PAR_PERSONNE", "RATIO_EMPLOI_AGE", "AGE_ANNEES", "ANCIENNETE_EMPLOI_ANNEES"}

# Variables métier suivies en plus des plus importantes : ce sont celles que le métier
# regarde en premier (montants et type de crédit), même si leur poids dans le modèle est faible
BUSINESS_FEATURES = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "NAME_CONTRACT_TYPE"]


def feature_importance(model) -> pd.Series:
    """Part du gain LightGBM de chaque variable d'origine (colonnes encodées regroupées)."""
    column_transformer = model.named_steps["preparation"].named_steps["preparation"]
    categorical = list(next(cols for name, _, cols in column_transformer.transformers_ if name == "cat"))
    gains = model.named_steps["modele"].booster_.feature_importance(importance_type="gain")

    def original_name(encoded: str) -> str:
        branch, name = encoded.split("__", 1)
        if branch == "cat":
            return max((c for c in categorical if name.startswith(c + "_")), key=len)
        return name

    names = [original_name(n) for n in column_transformer.get_feature_names_out()]
    importance = pd.Series(gains, index=names).groupby(level=0).sum()
    return (importance / importance.sum()).sort_values(ascending=False), categorical


def numeric_profile(values: pd.Series) -> dict:
    """Bornes des déciles et part de clients dans chaque tranche."""
    present = values.dropna()
    # Bornes uniques : une variable avec beaucoup de valeurs identiques a moins de tranches
    edges = np.unique(np.quantile(present, np.linspace(0, 1, N_BINS + 1)))
    inner = edges[1:-1].tolist()
    counts = np.histogram(present, bins=[-np.inf, *inner, np.inf])[0]
    return {
        "type": "numeric",
        "edges": inner,
        "proportions": (counts / len(present)).tolist(),
        "missing_share": float(values.isna().mean()),
    }


def categorical_profile(values: pd.Series) -> dict:
    """Part de chaque catégorie."""
    return {
        "type": "categorical",
        "proportions": values.dropna().value_counts(normalize=True).to_dict(),
        "missing_share": float(values.isna().mean()),
    }


def main():
    model = mlflow.sklearn.load_model(str(PROJECT_ROOT / "model"))
    importance, categorical = feature_importance(model)
    # Seules les variables reçues par l'API sont suivies : les plus importantes pour le
    # modèle, puis les variables métier qui n'en font pas déjà partie
    inputs = [f for f in importance.index if f not in ENGINEERED]
    selected = inputs[:N_FEATURES]
    selected += [f for f in BUSINESS_FEATURES if f not in selected]

    df = pd.read_parquet(REFERENCE_DATA)
    train, _ = train_test_split(df, test_size=0.20, stratify=df[TARGET], random_state=RANDOM_STATE)

    profile = {
        "description": "Aggregated profile of the training data (Project 6), no individual data",
        "n_reference": int(len(train)),
        "features": {},
    }
    for feature in selected:
        stats = categorical_profile(train[feature]) if feature in categorical else numeric_profile(train[feature])
        stats["importance"] = round(float(importance[feature]), 4)
        profile["features"][feature] = stats

    OUTPUT.write_text(json.dumps(profile, indent=2, ensure_ascii=False))
    print(f"Profile of {len(selected)} features saved to {OUTPUT.relative_to(PROJECT_ROOT)}")
    for feature, stats in profile["features"].items():
        print(f"  {feature:<40} {stats['type']:<12} importance {stats['importance']:.3f}")


if __name__ == "__main__":
    main()
