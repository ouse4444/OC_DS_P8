"""Test manuel de l'API lancée en local.

Prérequis : l'API tourne dans un autre terminal (uvicorn api.main:app --reload)
et les clients de test ont été exportés (scripts/export_sample_clients.py).

Le script vérifie trois choses :
1. l'API est en ligne et le modèle est chargé ;
2. l'API renvoie exactement la probabilité calculée directement par le modèle ;
3. les entrées invalides sont rejetées avec un code 422.
"""

import json
from pathlib import Path

import requests

API_URL = "http://127.0.0.1:8000"
SAMPLE_FILE = Path(__file__).resolve().parent.parent / "data" / "samples" / "clients_sample.json"


def main():
    # 1. État de l'API
    health = requests.get(f"{API_URL}/health", timeout=10).json()
    print(f"Health: {health}\n")

    # 2. Prédictions sur des clients réels, comparées au calcul direct du modèle
    clients = json.loads(SAMPLE_FILE.read_text())
    print(f"{'Client':>8} {'True':>5} {'Expected':>9} {'API':>7} {'Decision':>9} {'Time (ms)':>10} {'Match':>6}")
    for c in clients:
        r = requests.post(f"{API_URL}/predict", json=c["features"], timeout=10)
        r.raise_for_status()
        res = r.json()
        # Tolérance d'un dix-millième : l'API arrondit la probabilité à 4 décimales
        match = abs(res["probability_default"] - c["expected_probability"]) < 1e-4
        print(f"{c['client_id']:>8} {c['target']:>5} {c['expected_probability']:>9.4f} "
              f"{res['probability_default']:>7.4f} {res['decision']:>9} "
              f"{res['inference_time_ms']:>10.2f} {str(match):>6}")

    # 3. Cas d'erreur : chaque requête invalide doit être rejetée (code 422)
    base = clients[0]["features"]
    invalid_cases = {
        "Missing required field (AMT_CREDIT)": {k: v for k, v in base.items() if k != "AMT_CREDIT"},
        "Out of range value (income = -5)": {**base, "AMT_INCOME_TOTAL": -5},
        "Wrong type (text as income)": {**base, "AMT_INCOME_TOTAL": "high"},
        "Unknown category (CODE_GENDER = 'Z')": {**base, "CODE_GENDER": "Z"},
        "Unknown feature name": {**base, "FAVOURITE_COLOR": "blue"},
        "Wrong type on optional feature (text)": {**base, "BUREAU_COUNT": "many"},
    }
    print("\nInvalid inputs:")
    for label, payload in invalid_cases.items():
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        status = "rejected" if r.status_code == 422 else "NOT REJECTED"
        print(f"  {label:<42} -> HTTP {r.status_code} ({status})")


if __name__ == "__main__":
    main()