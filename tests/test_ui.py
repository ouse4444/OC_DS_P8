"""Tests de l'interface Gradio, cliente de l'API.

L'interface est un service séparé qui appelle l'API par HTTP. Les tests remplacent
l'appel HTTP (monkeypatch) :
- soit par de fausses réponses, pour vérifier l'affichage de chaque cas
  (succès, donnée invalide, erreur de l'API, API injoignable) ;
- soit par un appel à la vraie API, exécutée en mémoire (TestClient), pour vérifier
  que l'interface et l'API se comprennent réellement.
"""

import gradio as gr
import pytest
import requests

from ui import app as ui_app
from ui.app import build_demo, format_errors, form_to_payload, score_form

# Valeurs par défaut du formulaire : une demande valide
VALID_FORM = ["Cash loans", "F", 135000, 568800, 20560.5, 41, 7, 0.52, 0.48]


class FakeResponse:
    """Réponse HTTP simulée, avec le même usage qu'une réponse de requests."""

    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body
        self.text = str(body)

    def json(self):
        return self._body


def fake_post(status_code, body, calls=None):
    """Fabrique une fausse fonction requests.post qui renvoie toujours la même réponse."""
    def _post(url, json, timeout):
        if calls is not None:
            calls.append({"url": url, "json": json, "timeout": timeout})
        return FakeResponse(status_code, body)
    return _post


# Conversion du formulaire

def test_form_to_payload_converts_years_to_negative_days():
    payload = form_to_payload(*VALID_FORM)
    assert payload["DAYS_BIRTH"] == -round(41 * 365.25)
    assert payload["DAYS_EMPLOYED"] == -round(7 * 365.25)


def test_form_to_payload_omits_empty_optional_fields():
    """Une ancienneté vide signifie « sans emploi » : le champ n'est pas transmis."""
    form = VALID_FORM.copy()
    form[6] = None
    assert "DAYS_EMPLOYED" not in form_to_payload(*form)


# Affichage des réponses de l'API (réponses simulées)

def test_ui_calls_the_predict_route_of_the_api(monkeypatch):
    calls = []
    monkeypatch.setattr(ui_app, "API_URL", "https://api.example.org")
    monkeypatch.setattr(ui_app.requests, "post", fake_post(200, {
        "probability_default": 0.12, "decision": "accepted", "threshold": 0.36}, calls))
    score_form(*VALID_FORM)
    assert calls[0]["url"] == "https://api.example.org/predict"
    assert calls[0]["json"]["AMT_CREDIT"] == 568800


def test_ui_shows_refused_decision(monkeypatch):
    body = {"probability_default": 0.48, "decision": "refused", "threshold": 0.36}
    monkeypatch.setattr(ui_app.requests, "post", fake_post(200, body))
    summary, details = score_form(*VALID_FORM)
    assert summary.startswith("### Credit REFUSED")
    assert "48.0%" in summary
    assert details == body


def test_ui_shows_validation_errors(monkeypatch):
    detail = [{"loc": ["body", "AMT_INCOME_TOTAL"], "msg": "Input should be greater than 0"}]
    monkeypatch.setattr(ui_app.requests, "post", fake_post(422, {"detail": detail}))
    summary, details = score_form(*VALID_FORM)
    assert summary.startswith("### Invalid input")
    assert "AMT_INCOME_TOTAL" in summary
    assert details == {}


def test_ui_shows_api_error(monkeypatch):
    monkeypatch.setattr(ui_app.requests, "post", fake_post(503, {"detail": "Model not available"}))
    summary, _ = score_form(*VALID_FORM)
    assert summary.startswith("### Error 503")


def test_ui_handles_unreachable_api(monkeypatch):
    """Si l'API ne répond pas, l'interface affiche un message au lieu de planter."""
    def broken_post(url, json, timeout):
        raise requests.ConnectionError("connection refused")
    monkeypatch.setattr(ui_app.requests, "post", broken_post)
    summary, details = score_form(*VALID_FORM)
    assert summary.startswith("### API unreachable")
    assert details == {}


def test_format_errors_accepts_plain_messages():
    assert format_errors(["Unknown feature: FAVOURITE_COLOR"]) == "- Unknown feature: FAVOURITE_COLOR"
    assert format_errors("Model not available") == "Model not available"


def test_build_demo_returns_gradio_app():
    assert isinstance(build_demo(), gr.Blocks)


# Interface et vraie API ensemble (API exécutée en mémoire)

@pytest.fixture
def ui_wired_to_api(client, monkeypatch):
    """Redirige les appels HTTP de l'interface vers la vraie API, via le client de test."""
    monkeypatch.setattr(ui_app.requests, "post",
                        lambda url, json, timeout: client.post("/predict", json=json))


def test_ui_and_api_agree_on_valid_form(ui_wired_to_api):
    summary, details = score_form(*VALID_FORM)
    assert summary.startswith("### Credit")
    assert 0 <= details["probability_default"] <= 1


@pytest.mark.parametrize("index, value", [(2, -5), (5, 5)])  # revenu négatif, âge de 5 ans
def test_ui_and_api_agree_on_invalid_form(ui_wired_to_api, index, value):
    form = VALID_FORM.copy()
    form[index] = value
    summary, details = score_form(*form)
    assert summary.startswith("### Invalid input")
    assert details == {}


def test_ui_handles_non_json_error(monkeypatch):
    """Une erreur sans corps JSON (par exemple une page d'erreur du serveur) est affichée telle quelle."""
    class HtmlResponse:
        status_code = 502
        text = "Bad Gateway"

        def json(self):
            raise ValueError("not JSON")

    monkeypatch.setattr(ui_app.requests, "post", lambda url, json, timeout: HtmlResponse())
    summary, _ = score_form(*VALID_FORM)
    assert summary == "### Error 502\nBad Gateway"
