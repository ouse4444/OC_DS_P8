"""Mesure des performances de l'inférence, avant et après optimisation (phase 9).

Le script compare deux versions du traitement d'une requête :
- AVANT : modèle chargé avec MLflow, ligne de données construite puis convertie colonne
  par colonne (première version de l'API) ;
- APRÈS : modèle lu directement avec cloudpickle, ligne construite avec les bons types
  (version actuelle de api/model_service.py).

Il mesure :
1. le temps de démarrage (import des bibliothèques et chargement du modèle), dans un
   processus Python neuf, comme au démarrage du conteneur ;
2. le temps de chaque étape d'une requête, client par client (médiane et 95e centile) ;
3. l'absence de régression : prédictions identiques client par client, et AUC identique
   sur l'ensemble du jeu de test du Projet 6.

Usage, depuis la racine du projet (environnement de développement, avec MLflow) :
    python scripts/benchmark_inference.py
"""

import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from api.model_service import CreditScoringModel  # noqa: E402
from credit_scoring.config import RANDOM_STATE, TARGET  # noqa: E402

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message="Downcasting behavior", category=FutureWarning)

MODEL_DIR = PROJECT_ROOT / "model"
REFERENCE_DATA = PROJECT_ROOT / "data" / "processed" / "application_train_enriched.parquet"
OUTPUT = PROJECT_ROOT / "monitoring" / "reports" / "benchmark_inference.json"
N_CLIENTS = 500
N_STARTUPS = 5


def legacy_to_dataframe(service: CreditScoringModel, data: dict) -> pd.DataFrame:
    """Première version de la construction de la ligne, conservée ici pour la comparaison."""
    row = {col: data.get(col) for col in service.input_features}
    df = pd.DataFrame([row], columns=service.input_features)
    df[service.numeric_features] = df[service.numeric_features].astype(float)
    cat = df[service.categorical_features].astype(object)
    df[service.categorical_features] = cat.where(cat.notna(), np.nan)
    return df


def startup_time(code: str) -> float:
    """Durée d'exécution d'un code dans un processus Python neuf (secondes, médiane)."""
    times = []
    for _ in range(N_STARTUPS):
        result = subprocess.run(
            [sys.executable, "-W", "ignore", "-c",
             f"import time; t = time.perf_counter(); {code}; print(time.perf_counter() - t)"],
            capture_output=True, text=True, cwd=PROJECT_ROOT, check=True,
        )
        times.append(float(result.stdout.strip().splitlines()[-1]))
    return float(np.median(times))


def timings(func, items) -> dict:
    """Médiane et 95e centile du temps d'exécution, en millisecondes."""
    values = []
    for item in items:
        start = time.perf_counter()
        func(item)
        values.append((time.perf_counter() - start) * 1000)
    return {"p50_ms": round(float(np.median(values)), 2), "p95_ms": round(float(np.percentile(values, 95)), 2)}


def main():
    print("1. Startup time (fresh Python process)...")
    startup = {
        "before_mlflow_s": round(startup_time("import mlflow.sklearn; mlflow.sklearn.load_model('model')"), 2),
        "after_cloudpickle_s": round(startup_time(
            "import cloudpickle; cloudpickle.load(open('model/model.pkl', 'rb'))"), 2),
    }

    print("2. Per-request timings...")
    service = CreditScoringModel(MODEL_DIR)
    legacy_model = mlflow.sklearn.load_model(str(MODEL_DIR))
    df = pd.read_parquet(REFERENCE_DATA)
    _, test = train_test_split(df, test_size=0.20, stratify=df[TARGET], random_state=RANDOM_STATE)
    sample = test.sample(n=min(N_CLIENTS, len(test)), random_state=RANDOM_STATE)
    clients = json.loads(sample[service.input_features].to_json(orient="records"))

    steps = {
        "build_row_before": timings(lambda c: legacy_to_dataframe(service, c), clients),
        "build_row_after": timings(service.to_dataframe, clients),
        # Lignes construites à l'avance : seul le calcul du pipeline du modèle est mesuré
        "model_pipeline": timings(service.pipeline.predict_proba, [service.to_dataframe(c) for c in clients]),
        "full_request_before": timings(
            lambda c: legacy_model.predict_proba(legacy_to_dataframe(service, c)), clients),
        "full_request_after": timings(lambda c: service.predict_proba(c), clients),
    }

    print("3. Regression checks...")
    before = np.array([legacy_model.predict_proba(legacy_to_dataframe(service, c))[0, 1] for c in clients])
    after = np.array([service.predict_proba(c)[0] for c in clients])
    X_test = test[service.input_features]
    auc_before = roc_auc_score(test[TARGET], legacy_model.predict_proba(X_test)[:, 1])
    auc_after = roc_auc_score(test[TARGET], service.pipeline.predict_proba(X_test)[:, 1])
    regression = {
        "clients_compared": len(clients),
        "max_probability_gap": float(np.abs(before - after).max()),
        "auc_before": round(float(auc_before), 4),
        "auc_after": round(float(auc_after), 4),
    }

    results = {"startup": startup, "per_request": steps, "regression": regression}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2))

    print("\nStartup (import + model loading, median of 5 fresh processes):")
    print(f"  before (MLflow)      : {startup['before_mlflow_s']:.2f} s")
    print(f"  after (cloudpickle)  : {startup['after_cloudpickle_s']:.2f} s")
    print(f"\nPer request, {len(clients)} test clients (median / 95th percentile):")
    for name, value in steps.items():
        print(f"  {name:<22}: {value['p50_ms']:6.2f} ms / {value['p95_ms']:6.2f} ms")
    gain = 1 - steps["full_request_after"]["p50_ms"] / steps["full_request_before"]["p50_ms"]
    print(f"  median gain on the full request: {gain:.0%}")
    print("\nRegression checks:")
    print(f"  max probability gap over {len(clients)} clients: {regression['max_probability_gap']}")
    print(f"  AUC on the full test set: before {regression['auc_before']}, after {regression['auc_after']}")
    print(f"\nResults saved to {OUTPUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
