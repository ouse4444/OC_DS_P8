"""Tests d'intégration de l'API : les requêtes passent par toute la chaîne
(validation, service du modèle, réponse), comme pour un vrai client."""

import pytest

from api.config import DECISION_THRESHOLD
from api.main import ml_models


# Routes générales

def test_root_returns_welcome_message(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "docs" in response.json()


def test_health_reports_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_features_lists_199_inputs(client):
    body = client.get("/features").json()
    assert body["n_features"] == 199
    assert len(body["categorical"]) + len(body["numeric"]) == 199


# Prédiction : cas valides

def test_predict_valid_request_returns_probability(client, valid_payload):
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["probability_default"] <= 1
    assert body["decision"] in ("accepted", "refused")
    assert body["inference_time_ms"] >= 0


def test_predict_decision_follows_threshold(client, valid_payload):
    body = client.post("/predict", json=valid_payload).json()
    expected = "refused" if body["probability_default"] >= DECISION_THRESHOLD else "accepted"
    assert body["decision"] == expected


def test_predict_matches_direct_model(client, valid_payload):
    """L'API doit renvoyer la probabilité calculée directement par le modèle."""
    api_proba = client.post("/predict", json=valid_payload).json()["probability_default"]
    model = ml_models["credit_scoring"]
    direct_proba, _ = model.predict_proba(valid_payload)
    assert api_proba == pytest.approx(direct_proba, abs=1e-4)


def test_predict_accepts_missing_optional_fields(client, valid_payload):
    """Seuls les champs obligatoires suffisent : les autres sont imputés par le pipeline."""
    required = ("NAME_CONTRACT_TYPE", "CODE_GENDER", "AMT_INCOME_TOTAL", "AMT_CREDIT", "DAYS_BIRTH")
    minimal = {key: valid_payload[key] for key in required}
    assert client.post("/predict", json=minimal).status_code == 200


# Prédiction : cas invalides, tous rejetés avec le code 422

@pytest.mark.parametrize(
    "change",
    [
        {"AMT_CREDIT": None},           # champ obligatoire vide
        {"AMT_INCOME_TOTAL": -5},       # valeur hors limites
        {"AMT_INCOME_TOTAL": 0},        # revenu nul
        {"AMT_INCOME_TOTAL": "high"},   # mauvais type
        {"DAYS_BIRTH": -1825},          # âge de 5 ans
        {"EXT_SOURCE_2": 1.5},          # score hors de [0, 1]
        {"CODE_GENDER": "Z"},           # catégorie inconnue
        {"FAVOURITE_COLOR": "blue"},    # variable inconnue du modèle
        {"BUREAU_COUNT": "many"},       # texte sur une variable numérique complémentaire
        {"FLAG_OWN_CAR": 1},            # nombre sur une variable catégorielle complémentaire
    ],
)
def test_predict_invalid_input_is_rejected(client, valid_payload, change):
    response = client.post("/predict", json={**valid_payload, **change})
    assert response.status_code == 422


def test_predict_missing_required_field_is_rejected(client, valid_payload):
    payload = {key: value for key, value in valid_payload.items() if key != "AMT_CREDIT"}
    assert client.post("/predict", json=payload).status_code == 422


# Indisponibilité du modèle

def test_predict_returns_503_when_model_unavailable(client, valid_payload, monkeypatch):
    """Si le modèle n'est pas chargé, l'API répond 503 au lieu de planter."""
    monkeypatch.delitem(ml_models, "credit_scoring")
    assert client.post("/predict", json=valid_payload).status_code == 503
    assert client.get("/health").json()["status"] == "unavailable"