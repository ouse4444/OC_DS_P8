"""Extraction de clients de test pour l'API.

Les clients sont tirés du jeu de test du Projet 6 (même découpage 80/20) : ce sont
des clients que le modèle n'a jamais vus, comme de nouvelles demandes de crédit.
Pour chacun, la probabilité calculée directement par le modèle est enregistrée :
elle servira à vérifier que l'API renvoie exactement le même résultat.
"""

import argparse
import json
import sys
from pathlib import Path

import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split

# Le script est lancé depuis la racine du projet : on rend importables api/ et src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from api.config import MODEL_DIR  # noqa: E402
from credit_scoring.config import RANDOM_STATE, TARGET  # noqa: E402

REFERENCE_DATA = PROJECT_ROOT / "data" / "processed" / "application_train_enriched.parquet"
OUTPUT = PROJECT_ROOT / "data" / "samples" / "clients_sample.json"


def main():
    parser = argparse.ArgumentParser(description="Export sample clients for API tests.")
    parser.add_argument("--n", type=int, default=10, help="Number of clients to export")
    args = parser.parse_args()

    df = pd.read_parquet(REFERENCE_DATA)
    # Même découpage qu'au Projet 6 : seuls les clients du jeu de test sont utilisés
    _, test = train_test_split(df, test_size=0.20, stratify=df[TARGET], random_state=RANDOM_STATE)
    sample = test.sample(n=args.n, random_state=RANDOM_STATE)

    X = sample.drop(columns=[TARGET, "SK_ID_CURR"])
    model = mlflow.sklearn.load_model(str(MODEL_DIR))
    expected = model.predict_proba(X)[:, 1]

    # to_json convertit les types NumPy en types JSON et les valeurs manquantes en null
    features = json.loads(X.to_json(orient="records"))
    clients = [
        {
            "client_id": int(cid),
            "target": int(t),
            "expected_probability": round(float(p), 4),
            "features": f,
        }
        for cid, t, p, f in zip(sample["SK_ID_CURR"], sample[TARGET], expected, features)
    ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(clients, indent=2, ensure_ascii=False))
    print(f"{len(clients)} clients exported to {OUTPUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()