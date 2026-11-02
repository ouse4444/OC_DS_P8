"""Calculs du tableau de bord de suivi.

Ce module ne contient que des fonctions de calcul, sans affichage : elles sont testées
avec Pytest, indépendamment de Streamlit.

Le tableau de bord lit la table des prédictions de production (PostgreSQL) et compare
les variables reçues au profil agrégé de la référence (reference_profile.json).
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

PROFILE_PATH = Path(__file__).resolve().parent / "reference_profile.json"

# Seuils usuels du PSI (Population Stability Index), utilisés notamment dans le secteur
# bancaire pour surveiller les modèles de score
PSI_MODERATE, PSI_HIGH = 0.10, 0.25

# Taille minimale d'un échantillon pour interpréter le PSI : en dessous, les fluctuations
# naturelles produisent trop de fausses alertes
MIN_SAMPLE = 100

# Petite proportion utilisée à la place de zéro, pour éviter la division par zéro et le
# logarithme de zéro dans le calcul du PSI
EPSILON = 1e-4


def load_profile(path: Path = PROFILE_PATH) -> dict:
    """Charge le profil agrégé de la référence."""
    return json.loads(Path(path).read_text())


def load_events(engine: Engine, since: Optional[datetime] = None) -> pd.DataFrame:
    """Lit les événements de prédiction, éventuellement à partir d'une date."""
    query = """
        SELECT created_at, status_code, probability_default, decision, threshold,
               inference_time_ms, total_time_ms, inputs
        FROM predictions
    """
    params = {}
    if since is not None:
        query += " WHERE created_at >= :since"
        params["since"] = since
    query += " ORDER BY created_at"
    with engine.connect() as connection:
        events = pd.read_sql(text(query), connection, params=params)
    events["created_at"] = pd.to_datetime(events["created_at"], utc=True)
    # PostgreSQL renvoie la colonne JSONB sous forme de dictionnaires ; d'autres bases
    # (SQLite, utilisée dans les tests) la renvoient sous forme de texte JSON
    events["inputs"] = events["inputs"].map(lambda v: json.loads(v) if isinstance(v, str) else v)
    return events


def compute_kpis(events: pd.DataFrame) -> dict:
    """Indicateurs principaux : volume, erreurs, scores, décisions et temps de traitement."""
    ok = events[events["status_code"] == 200]
    has_ok = len(ok) > 0
    return {
        "requests": int(len(events)),
        "predictions": int(len(ok)),
        "error_rate": float((events["status_code"] != 200).mean()) if len(events) else 0.0,
        "server_errors": int((events["status_code"] >= 500).sum()),
        "mean_score": float(ok["probability_default"].mean()) if has_ok else None,
        "refusal_rate": float((ok["decision"] == "refused").mean()) if has_ok else None,
        "inference_p50_ms": float(ok["inference_time_ms"].median()) if has_ok else None,
        "total_p95_ms": float(ok["total_time_ms"].quantile(0.95)) if has_ok else None,
    }


def status_category(code: int) -> str:
    """Regroupe les codes HTTP en trois familles lisibles."""
    if code == 200:
        return "Succès (200)"
    if code < 500:
        return "Rejet client (4xx)"
    return "Erreur serveur (5xx)"


def timeline(events: pd.DataFrame, freq: str = "h") -> pd.DataFrame:
    """Nombre de requêtes par période et par famille de code de réponse."""
    if events.empty:
        return pd.DataFrame()
    df = events.assign(category=events["status_code"].map(status_category))
    return (
        df.groupby([pd.Grouper(key="created_at", freq=freq), "category"])
        .size()
        .unstack(fill_value=0)
    )


def psi(reference_shares: np.ndarray, current_shares: np.ndarray) -> float:
    """Population Stability Index entre deux répartitions (mêmes tranches, même ordre)."""
    ref = np.clip(np.asarray(reference_shares, dtype=float), EPSILON, None)
    cur = np.clip(np.asarray(current_shares, dtype=float), EPSILON, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def feature_psi(stats: dict, values: pd.Series) -> float:
    """PSI d'une variable, en comparant la production au profil de la référence.

    Les valeurs manquantes forment une tranche à part : une hausse soudaine des valeurs
    manquantes est elle aussi une dérive (problème de collecte, par exemple).
    """
    n = len(values)
    missing = values.isna()
    present = values[~missing]
    if stats["type"] == "numeric":
        bins = [-np.inf, *stats["edges"], np.inf]
        counts = np.histogram(pd.to_numeric(present, errors="coerce").dropna(), bins=bins)[0]
        current = counts / n
        reference = np.array(stats["proportions"]) * (1 - stats["missing_share"])
    else:
        categories = list(stats["proportions"])
        shares = present.value_counts().reindex(categories, fill_value=0) / n
        # Les catégories absentes de la référence sont regroupées dans une tranche « autre »
        other = (~present.isin(categories)).sum() / n
        current = np.append(shares.to_numpy(), other)
        reference = np.append(np.array(list(stats["proportions"].values())) * (1 - stats["missing_share"]), 0.0)
    current = np.append(current, missing.mean())
    reference = np.append(reference, stats["missing_share"])
    return psi(reference, current)


def drift_level(value: float) -> str:
    """Traduit un PSI en niveau d'alerte."""
    if value >= PSI_HIGH:
        return "Dérive forte"
    if value >= PSI_MODERATE:
        return "Dérive modérée"
    return "Stable"


def drift_table(profile: dict, events: pd.DataFrame) -> pd.DataFrame:
    """PSI de chaque variable suivie, sur les prédictions réussies de la période."""
    ok = events[events["status_code"] == 200]
    inputs = pd.DataFrame(list(ok["inputs"])) if len(ok) else pd.DataFrame()
    rows = []
    for feature, stats in profile["features"].items():
        values = inputs[feature] if feature in inputs else pd.Series([np.nan] * len(inputs))
        value = feature_psi(stats, values) if len(inputs) else np.nan
        rows.append({
            "variable": feature,
            "importance": stats["importance"],
            "psi": value,
            "niveau": drift_level(value) if len(inputs) else "Pas de données",
        })
    table = pd.DataFrame(rows).set_index("variable")
    return table.sort_values("importance", ascending=False)


def weighted_drift(table: pd.DataFrame) -> float:
    """Part de l'importance (des variables suivies) portée par les variables en dérive."""
    drifted = table["psi"] >= PSI_MODERATE
    total = table["importance"].sum()
    return float(table.loc[drifted, "importance"].sum() / total) if total else 0.0
