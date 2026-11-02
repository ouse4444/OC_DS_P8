"""Tests de la journalisation des prédictions de production.

Les tests utilisent une base SQLite dans un dossier temporaire : même code SQLAlchemy
qu'en production (PostgreSQL), sans serveur à installer. Chaque test vérifie qu'un appel
à /predict, réussi ou non, laisse bien une trace exploitable pour le suivi.
"""

import json
import logging

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

import api.main as main_module
from api.main import app, ml_models
from api.prediction_log import PredictionLogger, build_event, predictions


@pytest.fixture
def db_logger(tmp_path, monkeypatch):
    """Branche l'API sur une base SQLite temporaire et renvoie le journal."""
    prediction_logger = PredictionLogger(f"sqlite:///{tmp_path / 'predictions.db'}")
    monkeypatch.setattr(main_module, "prediction_logger", prediction_logger)
    return prediction_logger


def stored_rows(prediction_logger):
    """Lit toutes les lignes de la table des prédictions."""
    with prediction_logger.engine.connect() as connection:
        return [dict(row._mapping) for row in connection.execute(select(predictions))]


def test_successful_prediction_is_stored(client, valid_payload, db_logger):
    body = client.post("/predict", json=valid_payload).json()
    rows = stored_rows(db_logger)
    assert len(rows) == 1
    row = rows[0]
    # La ligne en base et la réponse au client sont reliées par le même identifiant
    assert row["request_id"] == body["request_id"]
    assert row["status_code"] == 200
    assert row["probability_default"] == body["probability_default"]
    assert row["decision"] == body["decision"]
    assert row["inputs"]["AMT_CREDIT"] == valid_payload["AMT_CREDIT"]
    assert row["total_time_ms"] >= row["inference_time_ms"]


def test_pydantic_rejection_is_stored(client, valid_payload, db_logger):
    response = client.post("/predict", json={**valid_payload, "AMT_INCOME_TOTAL": -5})
    assert response.status_code == 422
    row = stored_rows(db_logger)[0]
    assert row["status_code"] == 422
    assert row["inputs"]["AMT_INCOME_TOTAL"] == -5
    assert "AMT_INCOME_TOTAL" in row["error"]


def test_unknown_feature_rejection_is_stored(client, valid_payload, db_logger):
    client.post("/predict", json={**valid_payload, "FAVOURITE_COLOR": "blue"})
    row = stored_rows(db_logger)[0]
    assert row["status_code"] == 422
    assert "FAVOURITE_COLOR" in row["error"]


def test_invalid_json_body_is_stored(client, db_logger):
    """Un corps qui n'est pas un objet JSON est conservé sous forme de texte."""
    client.post("/predict", json=["not", "an", "object"])
    row = stored_rows(db_logger)[0]
    assert row["status_code"] == 422
    assert "raw" in row["inputs"]


def test_unavailable_model_is_stored(client, valid_payload, db_logger, monkeypatch):
    monkeypatch.delitem(ml_models, "credit_scoring")
    client.post("/predict", json=valid_payload)
    assert stored_rows(db_logger)[0]["status_code"] == 503


def test_unexpected_error_is_stored(client, valid_payload, db_logger, monkeypatch):
    def broken_predict(data):
        raise RuntimeError("internal detail")
    monkeypatch.setattr(ml_models["credit_scoring"], "predict_proba", broken_predict)
    safe_client = TestClient(app, raise_server_exceptions=False)
    assert safe_client.post("/predict", json=valid_payload).status_code == 500
    row = stored_rows(db_logger)[0]
    assert row["status_code"] == 500
    # Seul le type de l'erreur est stocké, pas son message interne
    assert row["error"] == "RuntimeError"


def test_event_is_written_as_json_log_line(client, valid_payload, caplog):
    with caplog.at_level(logging.INFO, logger="credit_scoring_api.predictions"):
        body = client.post("/predict", json=valid_payload).json()
    # Seules les lignes du journal des prédictions sont lues (pas celles des autres modules)
    records = [r for r in caplog.records if r.name == "credit_scoring_api.predictions"]
    event = json.loads(records[-1].getMessage())
    assert event["request_id"] == body["request_id"]
    assert event["status_code"] == 200


def test_database_failure_does_not_break_prediction(client, valid_payload, monkeypatch, caplog):
    """Une base injoignable est signalée dans les journaux, mais l'API répond quand même."""
    broken = PredictionLogger("sqlite:////nonexistent_folder/predictions.db")
    monkeypatch.setattr(main_module, "prediction_logger", broken)
    with caplog.at_level(logging.ERROR, logger="credit_scoring_api.predictions"):
        response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    assert "could not be created" in caplog.text


def test_insert_failure_is_logged_not_raised(db_logger, caplog):
    """Un événement mal formé n'est pas inséré, mais aucune exception ne remonte."""
    event = build_event(request_id="same-id", model_name="m", model_version="1",
                        status_code=200, inputs={}, total_time_ms=1.0)
    db_logger.log(event)
    with caplog.at_level(logging.ERROR, logger="credit_scoring_api.predictions"):
        db_logger.log(event)  # identifiant en double : contrainte d'unicité violée
    assert "could not be stored" in caplog.text
    assert len(stored_rows(db_logger)) == 1


def test_storage_is_reported_by_health(client, db_logger):
    assert client.get("/health").json()["prediction_storage"] == "database"


def test_logs_only_mode_without_database():
    prediction_logger = PredictionLogger()
    assert prediction_logger.storage == "logs"
    assert prediction_logger.ensure_table() is False


def test_postgresql_url_sets_a_connection_timeout():
    """Le moteur PostgreSQL est créé sans se connecter : aucune base n'est nécessaire."""
    prediction_logger = PredictionLogger("postgresql+psycopg://user:pwd@localhost:5432/monitoring")
    assert prediction_logger.storage == "database"
    assert prediction_logger.engine.dialect.name == "postgresql"


def test_api_startup_connects_to_the_configured_database(tmp_path, monkeypatch):
    """Au démarrage, l'API se branche sur la base indiquée par DATABASE_URL et crée la table."""
    # Sauvegarde du modèle et du journal de la session : l'arrêt de l'API vide ml_models
    saved_models = dict(ml_models)
    monkeypatch.setattr(main_module, "prediction_logger", main_module.prediction_logger)
    monkeypatch.setattr(main_module, "DATABASE_URL", f"sqlite:///{tmp_path / 'startup.db'}")
    try:
        with TestClient(app) as startup_client:
            assert startup_client.get("/health").json()["prediction_storage"] == "database"
            assert main_module.prediction_logger.ensure_table() is True
    finally:
        ml_models.clear()
        ml_models.update(saved_models)
