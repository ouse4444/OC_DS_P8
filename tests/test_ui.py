"""Tests de l'interface Gradio : conversion du formulaire et affichage du résultat."""

import pytest

from api.main import ml_models, run_prediction
from api.ui import form_to_payload, score_form

# Valeurs par défaut du formulaire : une demande valide
VALID_FORM = ["Cash loans", "F", 135000, 568800, 20560.5, 41, 7, 0.52, 0.48]


def test_form_to_payload_converts_years_to_negative_days():
    payload = form_to_payload(*VALID_FORM)
    assert payload["DAYS_BIRTH"] == -round(41 * 365.25)
    assert payload["DAYS_EMPLOYED"] == -round(7 * 365.25)


def test_form_to_payload_omits_empty_optional_fields():
    """Une ancienneté vide signifie « sans emploi » : le champ n'est pas transmis."""
    form = VALID_FORM.copy()
    form[6] = None
    assert "DAYS_EMPLOYED" not in form_to_payload(*form)


def test_ui_returns_decision_for_valid_form(client):
    summary, details = score_form(run_prediction, *VALID_FORM)
    assert summary.startswith("### Credit")
    assert 0 <= details["probability_default"] <= 1


@pytest.mark.parametrize("index, value", [(2, -5), (5, 5)])  # revenu négatif, âge de 5 ans
def test_ui_reports_invalid_input(client, index, value):
    form = VALID_FORM.copy()
    form[index] = value
    summary, details = score_form(run_prediction, *form)
    assert summary.startswith("### Invalid input")
    assert details == {}


def test_ui_reports_unavailable_model(client, monkeypatch):
    monkeypatch.delitem(ml_models, "credit_scoring")
    summary, _ = score_form(run_prediction, *VALID_FORM)
    assert summary.startswith("### Error 503")


def test_ui_page_is_served(client):
    assert client.get("/ui/").status_code == 200