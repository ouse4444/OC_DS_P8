"""Tableau de bord de suivi de l'API de scoring crédit (Streamlit).

Application SÉPARÉE de l'API, conformément au principe d'architecture retenu : si le
tableau de bord tombe en panne, l'API continue de servir ses clients. Elle s'appuie sur
deux sources :
- la base PostgreSQL de production (variable DATABASE_URL), où l'API enregistre chaque appel ;
- la route /health de l'API (variable API_URL), interrogée par HTTP pour afficher son état.

Lancement en local, depuis la racine du projet :
    streamlit run dashboard/app.py
"""

import os
import sys
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
import altair as alt
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Streamlit exécute ce fichier comme un script : le dossier parent (racine du projet en
# local, /app dans le conteneur) est ajouté au chemin pour importer le module dashboard
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard.metrics import (  # noqa: E402
    MIN_SAMPLE, PSI_HIGH, PSI_MODERATE, compute_kpis, drift_table, load_events,
    load_profile, timeline, weighted_drift,
)

# En local, les adresses sont lues dans le fichier .env (non versionné) ;
# sur Azure, elles sont fournies par l'environnement (DATABASE_URL est un secret)
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="Suivi du modèle de scoring", layout="wide")
st.title("Suivi de production du modèle de scoring crédit")
st.caption("Prêt à Dépenser : API de scoring (modèle LightGBM du Projet 6), données enregistrées dans PostgreSQL")

# Période analysée
PERIODS = {
    "Dernière heure": (timedelta(hours=1), "1min"),
    "Dernières 24 heures": (timedelta(days=1), "15min"),
    "7 derniers jours": (timedelta(days=7), "1h"),
    "Tout l'historique": (None, "1h"),
}
period = st.sidebar.selectbox("Période analysée", list(PERIODS), index=3)
window, freq = PERIODS[period]
if st.sidebar.button("Actualiser"):
    st.cache_data.clear()


@st.cache_data(ttl=30)
def api_health() -> dict:
    """État de l'API. Une API injoignable n'empêche pas le tableau de bord de fonctionner."""
    try:
        return requests.get(f"{API_URL}/health", timeout=60).json()
    except (requests.RequestException, ValueError):
        return {"status": "unreachable"}


@st.cache_resource
def get_engine():
    """Connexion à la base, créée une seule fois pour toute l'application."""
    return create_engine(DATABASE_URL, pool_pre_ping=True)


@st.cache_data(ttl=60)
def get_events(since):
    """Événements de la période, gardés en cache une minute pour limiter les requêtes."""
    return load_events(get_engine(), since)


# 1. État de l'API
health = api_health()
if health.get("status") == "ok":
    st.success(
        f"API en ligne : modèle {health['model_name']} version {health['model_version']} chargé, "
        f"prédictions stockées en base : {'oui' if health.get('prediction_storage') == 'database' else 'non'}"
    )
elif health.get("status") == "unavailable":
    st.error("API en ligne, mais le modèle n'est pas chargé (les prédictions renvoient une erreur 503)")
else:
    st.warning(f"API injoignable ({API_URL}) : les indicateurs ci-dessous restent consultables")

if not DATABASE_URL:
    st.error("Base de données non configurée (variable DATABASE_URL absente)")
    st.stop()

since = datetime.now(timezone.utc) - window if window else None
try:
    events = get_events(since)
except Exception as exc:
    # Base injoignable (pare-feu, mise en veille, coupure) : message clair au lieu d'un plantage.
    # Le détail complet part dans les journaux (terminal en local, Log Analytics sur Azure) ;
    # seul le type d'erreur est affiché à l'écran, sans information sensible
    logging.getLogger("dashboard").exception("Production events could not be loaded")
    st.error(f"Base de données de production injoignable ({type(exc).__name__}) : réessayer dans quelques instants")
    st.stop()

# 2. Indicateurs principaux
kpis = compute_kpis(events)
cols = st.columns(6)
cols[0].metric("Requêtes", f"{kpis['requests']:,}".replace(",", " "))
cols[1].metric("Taux d'erreur", f"{kpis['error_rate']:.1%}", help="Part des requêtes rejetées (4xx) ou en erreur (5xx)")
cols[2].metric("Score moyen", f"{kpis['mean_score']:.3f}" if kpis["mean_score"] is not None else "-")
cols[3].metric("Taux de refus", f"{kpis['refusal_rate']:.1%}" if kpis["refusal_rate"] is not None else "-")
cols[4].metric("Inférence (médiane)", f"{kpis['inference_p50_ms']:.1f} ms" if kpis["inference_p50_ms"] is not None else "-")
cols[5].metric("Traitement total (p95)", f"{kpis['total_p95_ms']:.0f} ms" if kpis["total_p95_ms"] is not None else "-")
if kpis["server_errors"]:
    st.error(f"{kpis['server_errors']} erreur(s) serveur (5xx) sur la période : à analyser dans les journaux")

ok = events[events["status_code"] == 200]
tab_activity, tab_scores, tab_latency, tab_drift = st.tabs(
    ["Activité", "Scores et décisions", "Temps de réponse", "Dérive des données"]
)

# 3. Activité : volume de requêtes par période et par famille de code
with tab_activity:
    counts = timeline(events, freq).reset_index().melt(id_vars="created_at", var_name="Réponse", value_name="Requêtes")
    chart = alt.Chart(counts).mark_bar().encode(
        x=alt.X("created_at:T", title="Date et heure (UTC)"),
        y=alt.Y("Requêtes:Q", stack=True),
        color=alt.Color("Réponse:N", scale=alt.Scale(
            domain=["Succès (200)", "Rejet client (4xx)", "Erreur serveur (5xx)"],
            range=["#4c78a8", "#f58518", "#e45756"])),
        tooltip=["created_at:T", "Réponse:N", "Requêtes:Q"],
    ).properties(height=320, title="Requêtes reçues par l'API")
    st.altair_chart(chart, width="stretch")

# 4. Scores et décisions
with tab_scores:
    left, right = st.columns(2)
    threshold = float(ok["threshold"].dropna().iloc[0]) if ok["threshold"].notna().any() else 0.36
    hist = alt.Chart(ok).mark_bar(opacity=0.8).encode(
        x=alt.X("probability_default:Q", bin=alt.Bin(step=0.025), title="Probabilité de défaut"),
        y=alt.Y("count():Q", title="Nombre de clients"),
    )
    rule = alt.Chart(pd.DataFrame({"seuil": [threshold]})).mark_rule(color="black", strokeDash=[6, 4]).encode(x="seuil:Q")
    left.altair_chart((hist + rule).properties(height=320, title=f"Distribution des scores (seuil {threshold})"),
                      width="stretch")
    refusals = (
        ok.set_index("created_at")["decision"].eq("refused").resample(freq).mean().dropna().rename("Taux de refus").reset_index()
    )
    line = alt.Chart(refusals).mark_line(point=True).encode(
        x=alt.X("created_at:T", title="Date et heure (UTC)"),
        y=alt.Y("Taux de refus:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        tooltip=["created_at:T", alt.Tooltip("Taux de refus:Q", format=".1%")],
    ).properties(height=320, title="Taux de refus dans le temps")
    right.altair_chart(line, width="stretch")

# 5. Temps de réponse
with tab_latency:
    latency = (
        ok.set_index("created_at")[["inference_time_ms", "total_time_ms"]]
        .resample(freq).quantile(0.95).dropna()
        .rename(columns={"inference_time_ms": "Inférence (p95)", "total_time_ms": "Traitement total (p95)"})
        .reset_index().melt(id_vars="created_at", var_name="Mesure", value_name="Millisecondes")
    )
    chart = alt.Chart(latency).mark_line(point=True).encode(
        x=alt.X("created_at:T", title="Date et heure (UTC)"),
        y=alt.Y("Millisecondes:Q"),
        color="Mesure:N",
        tooltip=["created_at:T", "Mesure:N", alt.Tooltip("Millisecondes:Q", format=".1f")],
    ).properties(height=320, title="Temps de traitement dans l'API (95e centile)")
    st.altair_chart(chart, width="stretch")
    st.caption("Inférence : calcul du modèle seul. Traitement total : de la réception de la requête validée "
               "à la réponse (préparation des données, contrôles et modèle).")

# 6. Dérive des données
with tab_drift:
    profile = load_profile()
    table = drift_table(profile, events)
    share = weighted_drift(table)
    c1, c2, c3 = st.columns(3)
    c1.metric("Variables en dérive", f"{(table['psi'] >= PSI_MODERATE).sum()} / {len(table)}")
    c2.metric("Importance portée par ces variables", f"{share:.0%}",
              help="Part de l'importance des variables suivies portée par celles dont le PSI dépasse 0,10")
    c3.metric("Prédictions analysées", f"{len(ok):,}".replace(",", " "))
    if len(ok) < MIN_SAMPLE:
        st.warning(f"Moins de {MIN_SAMPLE} prédictions sur la période : le PSI est peu fiable (fluctuations d'échantillonnage)")
    elif share >= 0.20:
        st.error("Alerte : des variables importantes pour le modèle dérivent. Vérifier la source des données, "
                 "puis envisager un réentraînement.")
    else:
        st.success("Pas de dérive significative sur les variables importantes du modèle")

    bars = alt.Chart(table.reset_index()).mark_bar().encode(
        x=alt.X("psi:Q", title="PSI"),
        # Noms de variables longs : affichés en entier, une ligne par variable
        y=alt.Y("variable:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320, labelOverlap=False)),
        color=alt.Color("niveau:N", title="Niveau", scale=alt.Scale(
            domain=["Stable", "Dérive modérée", "Dérive forte"], range=["#54a24b", "#f58518", "#e45756"])),
        tooltip=["variable:N", alt.Tooltip("psi:Q", format=".3f"), alt.Tooltip("importance:Q", format=".1%"), "niveau:N"],
    )
    rules = alt.Chart(pd.DataFrame({"x": [PSI_MODERATE, PSI_HIGH]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q")
    st.altair_chart((bars + rules).properties(height=28 * len(table), title="Dérive des variables suivies (PSI)"),
                    width="stretch")
    st.dataframe(
        table.style.format({"importance": "{:.1%}", "psi": "{:.3f}"}),
        width="stretch",
    )
    st.caption(
        f"PSI (Population Stability Index) : écart entre la répartition des valeurs en production et "
        f"dans les données d'entraînement. Moins de {PSI_MODERATE:.2f} : stable ; de {PSI_MODERATE:.2f} à "
        f"{PSI_HIGH:.2f} : dérive modérée ; au-delà : dérive forte. Variables suivies : les 12 plus importantes "
        f"pour le modèle et 4 variables métier. L'analyse détaillée est faite avec Evidently "
        f"(notebook 02_analyse_derive)."
    )
