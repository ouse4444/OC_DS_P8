"""Éléments partagés par tous les tests (fixtures Pytest).

Les tests n'utilisent ni les données de référence ni les clients extraits, qui ne sont
pas versionnés : ils doivent pouvoir s'exécuter dans le pipeline CI/CD, à partir du
seul contenu du dépôt (code et modèle).
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="session")
def client():
    """Client de test de l'API.

    Le bloc « with » déclenche le lifespan : le modèle est chargé une seule fois
    pour toute la session de tests, comme au démarrage réel de l'API.
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_payload():
    """Demande de crédit valide et fictive (les champs obligatoires et quelques champs bornés)."""
    return {
        "NAME_CONTRACT_TYPE": "Cash loans",
        "CODE_GENDER": "F",
        "AMT_INCOME_TOTAL": 135000.0,
        "AMT_CREDIT": 568800.0,
        "AMT_ANNUITY": 20560.5,
        "CNT_FAM_MEMBERS": 2.0,
        "DAYS_BIRTH": -15000,
        "DAYS_EMPLOYED": -2500.0,
        "EXT_SOURCE_2": 0.52,
    }