"""Journalisation des prédictions de production.

Chaque appel à /predict produit un événement : date, identifiant de requête, version du
modèle, code de réponse, entrées reçues, probabilité, décision, temps d'inférence, temps
total de traitement et message d'erreur éventuel. Cet événement est :

1. toujours écrit dans les journaux, sous forme d'une ligne JSON : sur Azure, ces lignes
   sont collectées automatiquement dans l'espace de journaux (Log Analytics) ;
2. enregistré dans une base de données si la variable d'environnement DATABASE_URL est
   définie (PostgreSQL en production). Cette table est la source de l'analyse de dérive
   et du tableau de bord de suivi.

Principe essentiel : la journalisation ne doit jamais faire échouer la prédiction. Une base
indisponible est signalée dans les journaux, mais l'API continue de répondre.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    JSON, BigInteger, Column, DateTime, Float, Integer, MetaData, String, Table, Text,
    create_engine,
)
from sqlalchemy.dialects.postgresql import JSONB

logger = logging.getLogger("credit_scoring_api.predictions")

metadata = MetaData()

# Une ligne par appel à /predict, réussi ou non : les erreurs font partie du suivi
predictions = Table(
    "predictions",
    metadata,
    # BigInteger en PostgreSQL ; Integer en SQLite, seul type auto-incrémenté par SQLite
    Column("id", BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), nullable=False, index=True),
    Column("request_id", String(36), nullable=False, unique=True),
    Column("model_name", String(100), nullable=False),
    Column("model_version", String(20), nullable=False),
    Column("status_code", Integer, nullable=False, index=True),
    Column("probability_default", Float),
    Column("decision", String(10)),
    Column("threshold", Float),
    Column("inference_time_ms", Float),
    Column("total_time_ms", Float),
    # Entrées reçues, conservées pour l'analyse de dérive. JSONB en PostgreSQL :
    # format binaire indexable, interrogeable en SQL (inputs->>'AMT_CREDIT')
    Column("inputs", JSON().with_variant(JSONB(), "postgresql")),
    Column("error", Text),
)


class PredictionLogger:
    """Écrit les événements de prédiction dans les journaux et, si configurée, dans la base."""

    def __init__(self, database_url: Optional[str] = None):
        self.engine = None
        self._table_ready = False
        if database_url:
            connect_args = {}
            if database_url.startswith("postgresql"):
                # Délai de connexion court : une base injoignable ne doit pas bloquer l'API
                connect_args["connect_timeout"] = 5
            # pool_pre_ping : la connexion est vérifiée avant chaque usage, ce qui évite les
            # erreurs après une coupure réseau ou une mise en veille de la base
            self.engine = create_engine(database_url, pool_pre_ping=True, connect_args=connect_args)

    @property
    def storage(self) -> str:
        """Destination des événements, affichée par la route /health."""
        return "database" if self.engine is not None else "logs"

    def ensure_table(self) -> bool:
        """Crée la table si elle n'existe pas encore. Retourne True si la base est prête."""
        if self.engine is None:
            return False
        if not self._table_ready:
            try:
                metadata.create_all(self.engine)
                self._table_ready = True
            except Exception:
                logger.exception("Prediction table could not be created")
        return self._table_ready

    def log(self, event: dict[str, Any]) -> None:
        """Enregistre un événement. Ne lève jamais d'exception."""
        # 1. Une ligne JSON dans les journaux, lisible par un humain comme par une machine
        logger.info(json.dumps(event, default=str, ensure_ascii=False))

        # 2. Une ligne dans la base, si elle est configurée
        if not self.ensure_table():
            return
        try:
            with self.engine.begin() as connection:
                connection.execute(predictions.insert().values(**event))
        except Exception:
            # La prédiction a déjà été servie : l'échec est seulement signalé
            logger.exception("Prediction event could not be stored in the database")


def build_event(
    *,
    request_id: str,
    model_name: str,
    model_version: str,
    status_code: int,
    inputs: Any,
    total_time_ms: float,
    probability_default: Optional[float] = None,
    decision: Optional[str] = None,
    threshold: Optional[float] = None,
    inference_time_ms: Optional[float] = None,
    error: Optional[str] = None,
) -> dict[str, Any]:
    """Construit un événement au format de la table predictions."""
    return {
        "created_at": datetime.now(timezone.utc),
        "request_id": request_id,
        "model_name": model_name,
        "model_version": model_version,
        "status_code": status_code,
        "probability_default": probability_default,
        "decision": decision,
        "threshold": threshold,
        "inference_time_ms": inference_time_ms,
        "total_time_ms": round(total_time_ms, 2),
        # Les entrées sont toujours stockées comme un objet JSON
        "inputs": inputs if isinstance(inputs, dict) else {"raw": str(inputs)},
        "error": error,
    }
