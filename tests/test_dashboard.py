"""Tests du tableau de bord de suivi.

Les événements sont écrits avec le journal de l'API (PredictionLogger) dans une base
SQLite temporaire, puis relus par le tableau de bord : on vérifie ainsi que les deux
applications, bien que séparées, partagent le même format de données.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from api.prediction_log import PredictionLogger, build_event
from dashboard.metrics import (
    compute_kpis, drift_level, drift_table, feature_psi, load_events, load_profile, psi,
    status_category, timeline, weighted_drift,
)

# Chemin absolu : l'outil de test de Streamlit résout les chemins relatifs depuis le dossier tests/
APP_PATH = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"

PROFILE = {
    "features": {
        "EXT_SOURCE_2": {"type": "numeric", "edges": [0.25, 0.5, 0.75],
                         "proportions": [0.25, 0.25, 0.25, 0.25], "missing_share": 0.0, "importance": 0.6},
        "CODE_GENDER": {"type": "categorical", "proportions": {"F": 0.6, "M": 0.4},
                        "missing_share": 0.0, "importance": 0.4},
    }
}


def make_events(scores, ext_source_2, genders, status=200):
    """Construit un tableau d'événements au format renvoyé par load_events."""
    now = datetime.now(timezone.utc)
    return pd.DataFrame({
        "created_at": [now - timedelta(minutes=i) for i in range(len(scores))],
        "status_code": status,
        "probability_default": scores,
        "decision": ["refused" if s >= 0.36 else "accepted" for s in scores],
        "threshold": 0.36,
        "inference_time_ms": 12.0,
        "total_time_ms": 35.0,
        "inputs": [{"EXT_SOURCE_2": e, "CODE_GENDER": g} for e, g in zip(ext_source_2, genders)],
    })


@pytest.fixture
def database(tmp_path):
    """Base SQLite remplie par le journal de l'API : 3 succès et 1 rejet."""
    url = f"sqlite:///{tmp_path / 'monitoring.db'}"
    logger = PredictionLogger(url)
    for i, (score, status) in enumerate([(0.2, 200), (0.5, 200), (0.7, 200), (None, 422)]):
        logger.log(build_event(
            request_id=f"id-{i}", model_name="m", model_version="1", status_code=status,
            inputs={"EXT_SOURCE_2": 0.5, "CODE_GENDER": "F"}, total_time_ms=30.0,
            probability_default=score, decision=None if score is None else ("refused" if score >= 0.36 else "accepted"),
            threshold=0.36, inference_time_ms=10.0 if score else None,
        ))
    return url, logger


def test_dashboard_reads_events_written_by_the_api(database):
    url, logger = database
    events = load_events(logger.engine)
    assert len(events) == 4
    # Les entrées sont toujours des dictionnaires, quelle que soit la base
    assert events["inputs"].iloc[0]["CODE_GENDER"] == "F"
    # Filtre par date : aucun événement dans le futur
    assert load_events(logger.engine, since=datetime.now(timezone.utc) + timedelta(hours=1)).empty


def test_kpis():
    events = make_events([0.2, 0.5, 0.7], [0.1, 0.4, 0.9], ["F", "M", "F"])
    kpis = compute_kpis(events)
    assert kpis["requests"] == 3
    assert kpis["error_rate"] == 0
    assert kpis["refusal_rate"] == pytest.approx(2 / 3)
    assert kpis["mean_score"] == pytest.approx(0.4667, abs=1e-3)


def test_kpis_without_successful_prediction():
    events = make_events([np.nan], [0.1], ["F"], status=422)
    kpis = compute_kpis(events)
    assert kpis["error_rate"] == 1
    assert kpis["mean_score"] is None


@pytest.mark.parametrize("code, category", [(200, "Succès (200)"), (422, "Rejet client (4xx)"), (500, "Erreur serveur (5xx)")])
def test_status_category(code, category):
    assert status_category(code) == category


def test_timeline_counts_requests_by_category():
    events = make_events([0.2, 0.5], [0.1, 0.4], ["F", "M"])
    assert timeline(events, "1D").to_numpy().sum() == 2
    assert timeline(events.iloc[0:0]).empty


def test_psi_is_zero_for_identical_distributions():
    assert psi([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0)


def test_psi_detects_a_shift():
    assert psi([0.5, 0.5], [0.9, 0.1]) > 0.25


def test_feature_psi_numeric_and_categorical():
    stats = PROFILE["features"]
    same = pd.Series([0.1, 0.3, 0.6, 0.9] * 25)
    assert feature_psi(stats["EXT_SOURCE_2"], same) == pytest.approx(0, abs=1e-6)
    shifted = pd.Series([0.1] * 100)
    assert feature_psi(stats["EXT_SOURCE_2"], shifted) > 0.25
    # Une catégorie inconnue de la référence est comptée dans une tranche « autre »
    assert feature_psi(stats["CODE_GENDER"], pd.Series(["XNA"] * 100)) > 0.25
    # Une hausse des valeurs manquantes est aussi une dérive
    assert feature_psi(stats["CODE_GENDER"], pd.Series([None] * 100)) > 0.25


@pytest.mark.parametrize("value, level", [(0.05, "Stable"), (0.15, "Dérive modérée"), (0.4, "Dérive forte")])
def test_drift_level(value, level):
    assert drift_level(value) == level


def test_drift_table_and_weighted_drift():
    # EXT_SOURCE_2 dérive (toutes les valeurs dans la première tranche), le genre est stable
    events = make_events([0.5] * 100, [0.1] * 100, ["F"] * 60 + ["M"] * 40)
    table = drift_table(PROFILE, events)
    assert table.loc["EXT_SOURCE_2", "niveau"] == "Dérive forte"
    assert table.loc["CODE_GENDER", "niveau"] == "Stable"
    assert weighted_drift(table) == pytest.approx(0.6)


def test_drift_table_without_data():
    table = drift_table(PROFILE, make_events([], [], []))
    assert (table["niveau"] == "Pas de données").all()


def test_reference_profile_contains_aggregates_only():
    """Le profil embarqué ne contient que des statistiques agrégées, aucune donnée client."""
    profile = load_profile()
    assert len(profile["features"]) == 16
    for stats in profile["features"].values():
        assert set(stats) <= {"type", "edges", "proportions", "missing_share", "importance"}


def run_app(monkeypatch, database_url=None, api_url="http://127.0.0.1:9"):
    """Exécute le tableau de bord sans navigateur (outil de test de Streamlit)."""
    monkeypatch.setenv("API_URL", api_url)
    if database_url:
        monkeypatch.setenv("DATABASE_URL", database_url)
    else:
        monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: None)
    # Les caches de Streamlit sont partagés dans le processus de test : ils sont vidés pour
    # que chaque test parte d'une application neuve (connexion et données)
    st.cache_data.clear()
    st.cache_resource.clear()
    return AppTest.from_file(str(APP_PATH), default_timeout=60).run()


class FakeHealth:
    """Réponse simulée de la route /health de l'API."""

    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body


def test_app_displays_indicators_even_if_the_api_is_unreachable(database, monkeypatch):
    """Tableau de bord et API sont indépendants : une API injoignable n'empêche pas l'affichage."""
    url, _ = database
    app = run_app(monkeypatch, url)
    assert not app.exception
    assert any("injoignable" in w.value for w in app.warning)
    assert {m.label: m.value for m in app.metric}["Requêtes"] == "4"


def test_app_without_database_configuration(monkeypatch):
    app = run_app(monkeypatch)
    assert not app.exception
    assert any("non configurée" in e.value for e in app.error)


def test_app_with_unreachable_database(monkeypatch):
    app = run_app(monkeypatch, "sqlite:////nonexistent_folder/monitoring.db")
    assert not app.exception
    assert any("injoignable" in e.value for e in app.error)


@pytest.mark.parametrize("body, kind, text", [
    ({"status": "ok", "model_name": "m", "model_version": "1", "prediction_storage": "database"},
     "success", "API en ligne"),
    ({"status": "unavailable"}, "error", "modèle n'est pas chargé"),
])
def test_app_shows_the_api_status(database, monkeypatch, body, kind, text):
    url, _ = database
    monkeypatch.setattr("requests.get", lambda *args, **kwargs: FakeHealth(body))
    app = run_app(monkeypatch, url)
    assert any(text in element.value for element in getattr(app, kind))


def test_app_with_an_empty_period(database, monkeypatch):
    """Une période sans requête affiche un message, sans erreur."""
    url, _ = database
    app = run_app(monkeypatch, url)
    app.selectbox[0].select("Dernière heure").run()
    assert not app.exception
