"""Tests de robustesse : comportement de l'API face aux pannes internes."""

from fastapi.testclient import TestClient

import api.main as main_module
from api.main import app, ml_models


def test_api_starts_even_if_model_loading_fails(monkeypatch, tmp_path):
    """Un modèle introuvable ne doit pas empêcher l'API de démarrer : /health le signale."""
    # Sauvegarde du modèle chargé pour la session, car l'arrêt de l'API vide le conteneur
    saved_models = dict(ml_models)
    # Le conteneur est vidé pour simuler un démarrage à neuf, avec un modèle introuvable
    ml_models.clear()
    monkeypatch.setattr(main_module, "MODEL_DIR", tmp_path / "missing_model")
    try:
        with TestClient(app) as failing_client:
            body = failing_client.get("/health").json()
            assert body["status"] == "unavailable"
            assert body["model_loaded"] is False
    finally:
        # Rétablissement du modèle pour les tests suivants
        ml_models.clear()
        ml_models.update(saved_models)


def test_unexpected_error_returns_generic_500(client, valid_payload, monkeypatch):
    """Une erreur imprévue renvoie un code 500 et un message générique, sans détail interne."""
    def broken_predict(data):
        raise RuntimeError("internal detail that must not leak")

    monkeypatch.setattr(ml_models["credit_scoring"], "predict_proba", broken_predict)
    # raise_server_exceptions=False : le client renvoie la réponse 500 au lieu de relancer
    # l'exception, comme le ferait un vrai serveur. Sans bloc « with », le lifespan ne se
    # relance pas : le modèle déjà chargé (et modifié ci-dessus) reste utilisé.
    safe_client = TestClient(app, raise_server_exceptions=False)
    response = safe_client.post("/predict", json=valid_payload)
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert "internal detail" not in response.text