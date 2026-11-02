"""Simulation de trafic de production sur l'API de scoring.

Le script envoie à l'API de vraies demandes de crédit, tirées du jeu de test du Projet 6
(des clients que le modèle n'a jamais vus), en trois lots :

1. « normal » : des clients tels quels. Leur distribution est celle de l'entraînement :
   aucune dérive n'est attendue ;
2. « drift » : des clients volontairement modifiés pour simuler un changement de
   population (scénario décrit dans apply_drift). Une dérive doit être détectée ;
3. « invalid » : quelques requêtes erronées, pour alimenter le suivi des erreurs.

Chaque appel est enregistré par l'API dans la base PostgreSQL de production. Le script
garde en local (dossier data/, non versionné) la correspondance entre l'identifiant de
requête, le lot et la vraie cible du client : elle servira à l'analyse de dérive et à
mesurer la performance réelle du modèle sur les données de production.

Usage, depuis la racine du projet :
    python scripts/simulate_traffic.py --api-url https://...azurecontainerapps.io
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from credit_scoring.config import RANDOM_STATE, TARGET  # noqa: E402

REFERENCE_DATA = PROJECT_ROOT / "data" / "processed" / "application_train_enriched.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "simulation"

# Bornes de l'API : âge entre 18 et 100 ans, en jours comptés à rebours
MIN_DAYS_BIRTH, MAX_DAYS_BIRTH = -36500, -6570


def load_test_clients(data_path: Path) -> pd.DataFrame:
    """Charge les clients du jeu de test du Projet 6 (même découpage 80/20)."""
    df = pd.read_parquet(data_path)
    _, test = train_test_split(df, test_size=0.20, stratify=df[TARGET], random_state=RANDOM_STATE)
    return test


def apply_drift(clients: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Simule un changement de population, comme après une campagne commerciale.

    Scénario : Prêt à Dépenser lance une offre en ligne qui attire des clients plus
    jeunes, aux revenus plus modestes, qui demandent des montants plus élevés, avec
    davantage de crédits renouvelables et des scores externes plus faibles.
    Les modifications portent sur des variables importantes du modèle : l'analyse de
    dérive doit les détecter, et le score moyen doit augmenter.
    """
    drifted = clients.copy()
    # Clients plus jeunes : l'âge est réduit d'un tiers, sans descendre sous 18 ans
    drifted["DAYS_BIRTH"] = (drifted["DAYS_BIRTH"] * 0.67).round().clip(MIN_DAYS_BIRTH, MAX_DAYS_BIRTH)
    # Revenus plus modestes et montants demandés plus élevés
    drifted["AMT_INCOME_TOTAL"] = drifted["AMT_INCOME_TOTAL"] * 0.75
    drifted["AMT_CREDIT"] = drifted["AMT_CREDIT"] * 1.5
    drifted["AMT_ANNUITY"] = drifted["AMT_ANNUITY"] * 1.5
    # Scores externes plus faibles, maintenus dans l'intervalle [0, 1]
    for col in ("EXT_SOURCE_2", "EXT_SOURCE_3"):
        drifted[col] = (drifted[col] - 0.15).clip(0, 1)
    # Un client sur deux passe en crédit renouvelable
    switch = rng.random(len(drifted)) < 0.5
    drifted.loc[switch, "NAME_CONTRACT_TYPE"] = "Revolving loans"
    return drifted


def invalid_payloads(base: dict) -> list[dict]:
    """Requêtes volontairement erronées : elles doivent être rejetées (422)."""
    return [
        {**base, "AMT_INCOME_TOTAL": -5},                             # valeur hors limites
        {k: v for k, v in base.items() if k != "AMT_CREDIT"},         # champ obligatoire absent
        {**base, "CODE_GENDER": "Z"},                                 # catégorie inconnue
        {**base, "FAVOURITE_COLOR": "blue"},                          # variable inconnue
        {**base, "DAYS_BIRTH": -1825},                                # âge de 5 ans
    ]


def to_payloads(clients: pd.DataFrame) -> list[dict]:
    """Convertit des lignes en requêtes JSON (les valeurs manquantes deviennent null)."""
    features = clients.drop(columns=[TARGET, "SK_ID_CURR"])
    return json.loads(features.to_json(orient="records"))


def send(api_url: str, payload: dict) -> dict:
    """Envoie une requête et mesure le temps de réponse vu par le client."""
    start = time.perf_counter()
    try:
        response = requests.post(f"{api_url}/predict", json=payload, timeout=60)
        elapsed_ms = (time.perf_counter() - start) * 1000
        body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
        return {
            "status_code": response.status_code,
            "request_id": body.get("request_id"),
            "probability_default": body.get("probability_default"),
            "decision": body.get("decision"),
            "client_latency_ms": round(elapsed_ms, 1),
        }
    except requests.RequestException as exc:
        return {"status_code": None, "error": type(exc).__name__,
                "client_latency_ms": round((time.perf_counter() - start) * 1000, 1)}


def run_batch(api_url: str, batch: str, payloads: list[dict], meta: list[dict], workers: int) -> list[dict]:
    """Envoie un lot de requêtes en parallèle (plusieurs clients simultanés).

    Retourne une ligne par requête : le lot, la réponse de l'API, et les informations
    gardées en local (identifiant client et vraie cible, absents pour les requêtes invalides).
    """
    print(f"Sending batch '{batch}': {len(payloads)} requests...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda p: send(api_url, p), payloads))
    return [{"batch": batch, **result, **m} for result, m in zip(results, meta)]


def main():
    parser = argparse.ArgumentParser(description="Simulate production traffic on the scoring API.")
    parser.add_argument("--api-url", required=True, help="Base URL of the API")
    parser.add_argument("--data", type=Path, default=REFERENCE_DATA, help="Reference data (parquet)")
    parser.add_argument("--n-normal", type=int, default=500, help="Number of unchanged clients")
    parser.add_argument("--n-drift", type=int, default=300, help="Number of drifted clients")
    parser.add_argument("--workers", type=int, default=4, help="Number of simultaneous clients")
    args = parser.parse_args()
    api_url = args.api_url.rstrip("/")

    rng = np.random.default_rng(RANDOM_STATE)
    test = load_test_clients(args.data)
    # Deux échantillons distincts : un même client n'est pas envoyé deux fois
    sample = test.sample(n=args.n_normal + args.n_drift, random_state=RANDOM_STATE)
    normal, drift = sample.iloc[:args.n_normal], apply_drift(sample.iloc[args.n_normal:], rng)

    # Réveil de l'API (mise en veille sur Azure) avant de mesurer quoi que ce soit
    print("Waking up the API...")
    print(requests.get(f"{api_url}/health", timeout=120).json())

    def meta(clients):
        # Données gardées en local uniquement : identifiant client et vraie cible
        return [{"client_id": int(c), "target": int(t)} for c, t in zip(clients["SK_ID_CURR"], clients[TARGET])]

    invalid = invalid_payloads(to_payloads(normal.iloc[:1])[0])
    # Un seul tableau construit à partir de toutes les lignes : les colonnes absentes
    # d'un lot (score des requêtes invalides, par exemple) deviennent des valeurs manquantes
    records = (
        run_batch(api_url, "normal", to_payloads(normal), meta(normal), args.workers)
        + run_batch(api_url, "drift", to_payloads(drift), meta(drift), args.workers)
        + run_batch(api_url, "invalid", invalid, [{} for _ in invalid], args.workers)
    )
    results = pd.DataFrame.from_records(records)
    # Entiers « nullables » : l'identifiant et la cible restent des entiers malgré les vides
    results[["client_id", "target"]] = results[["client_id", "target"]].astype("Int64")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / "simulation_results.csv"
    results.to_csv(output, index=False)

    # Résumé : codes de réponse, taux de refus et latence vue par le client, par lot
    print("\nSummary by batch:")
    ok = results[results["status_code"] == 200]
    summary = results.groupby("batch").agg(
        requests=("batch", "size"),
        success=("status_code", lambda s: int((s == 200).sum())),
        rejected_422=("status_code", lambda s: int((s == 422).sum())),
    )
    summary["mean_score"] = ok.groupby("batch")["probability_default"].mean().round(3)
    summary["refusal_rate"] = ok.groupby("batch")["decision"].apply(lambda d: round((d == "refused").mean(), 3))
    summary["p95_latency_ms"] = results.groupby("batch")["client_latency_ms"].quantile(0.95).round(0)
    print(summary.to_string())
    print(f"\nResults saved to {output.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
