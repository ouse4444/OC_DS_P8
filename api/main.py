"""API FastAPI de scoring crédit.

Lancement en local, depuis la racine du projet :
    uvicorn api.main:app --reload
Documentation interactive : http://127.0.0.1:8000/docs

L'API est un service autonome : ses clients (interface Gradio, tableau de bord,
applications métier) sont des applications séparées qui l'appellent par HTTP.

Chaque appel à /predict, réussi ou non, est journalisé (module prediction_log) :
une ligne JSON dans les journaux et, si DATABASE_URL est définie, une ligne en base.
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from api.config import DATABASE_URL, DECISION_THRESHOLD, MODEL_DIR, MODEL_NAME, MODEL_VERSION
from api.model_service import CreditScoringModel
from api.prediction_log import PredictionLogger, build_event
from api.schemas import ClientData, HealthResponse, PredictionResponse

logger = logging.getLogger("credit_scoring_api")
logging.basicConfig(level=logging.INFO)

# Conteneur partagé : le modèle y est déposé au démarrage, puis réutilisé par chaque requête
ml_models: dict = {}

# Journal des prédictions. Par défaut, journaux seuls ; la base est branchée au démarrage
prediction_logger = PredictionLogger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle et prépare la journalisation, une seule fois, au démarrage.

    Charger le modèle à chaque requête ralentirait fortement les réponses et
    saturerait la mémoire sous charge.
    """
    global prediction_logger
    try:
        ml_models["credit_scoring"] = CreditScoringModel(MODEL_DIR)
        logger.info("Model loaded from %s", MODEL_DIR)
    except Exception:
        # L'API démarre quand même : /health signalera que le modèle est indisponible
        logger.exception("Model loading failed")

    # Une base injoignable n'empêche pas l'API de démarrer : les événements restent
    # écrits dans les journaux, et la table sera créée dès que la base répondra
    prediction_logger = PredictionLogger(DATABASE_URL)
    if prediction_logger.ensure_table():
        logger.info("Prediction storage: database")
    else:
        logger.info("Prediction storage: logs only")
    yield
    # Code exécuté à l'arrêt de l'API : libération du modèle
    ml_models.clear()


app = FastAPI(
    title="Credit Scoring API - Prêt à Dépenser",
    description=(
        "Returns the probability of default of a credit applicant and the credit "
        f"decision, using the LightGBM model of Project 6 (threshold {DECISION_THRESHOLD})."
    ),
    version="1.1.0",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    """Toute erreur imprévue renvoie un message générique, sans exposer de détail interne."""
    logger.exception("Unexpected error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Journalise les requêtes rejetées par Pydantic, puis renvoie la réponse 422 standard.

    Ces rejets ont lieu avant la route /predict : sans ce gestionnaire, ils
    n'apparaîtraient pas dans le suivi de production.
    """
    if request.url.path == "/predict":
        event = build_event(
            request_id=str(uuid.uuid4()),
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            status_code=422,
            inputs=exc.body,
            total_time_ms=0.0,
            error=str(exc.errors()),
        )
        # L'écriture en base est bloquante : elle est exécutée hors de la boucle asynchrone
        await run_in_threadpool(prediction_logger.log, event)
    return await request_validation_exception_handler(request, exc)


def get_model() -> CreditScoringModel:
    """Retourne le modèle chargé, ou une erreur 503 s'il n'est pas disponible."""
    model = ml_models.get("credit_scoring")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    return model


@app.get("/", tags=["General"])
def root():
    """Message d'accueil et liens utiles."""
    return {
        "message": "Credit Scoring API - Prêt à Dépenser",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health():
    """Indique si l'API fonctionne, si le modèle est chargé et où sont stockées les prédictions."""
    loaded = "credit_scoring" in ml_models
    return HealthResponse(
        status="ok" if loaded else "unavailable",
        model_loaded=loaded,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        prediction_storage=prediction_logger.storage,
    )


@app.get("/features", tags=["Model"])
def features():
    """Liste des variables acceptées par le modèle, utile aux applications clientes."""
    model = get_model()
    return {
        "n_features": len(model.input_features),
        "categorical": model.categorical_features,
        "numeric": model.numeric_features,
    }


def run_prediction(client: ClientData, request_id: str) -> PredictionResponse:
    """Logique de prédiction de la route /predict."""
    model = get_model()

    # exclude_unset : seules les variables réellement envoyées sont transmises,
    # les variables absentes seront imputées par le pipeline
    data = client.model_dump(exclude_unset=True)

    # Contrôle des variables complémentaires (noms et types)
    errors = model.check_input(data)
    if errors:
        raise HTTPException(status_code=422, detail=errors)

    proba, inference_ms = model.predict_proba(data)
    # Au-delà du seuil, le risque de défaut est jugé trop élevé : le crédit est refusé
    decision = "refused" if proba >= DECISION_THRESHOLD else "accepted"

    return PredictionResponse(
        request_id=request_id,
        probability_default=round(proba, 4),
        decision=decision,
        threshold=DECISION_THRESHOLD,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        inference_time_ms=round(inference_ms, 2),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Model"])
def predict(client: ClientData, background_tasks: BackgroundTasks):
    """Calcule la probabilité de défaut d'un client et la décision de crédit."""
    # Identifiant unique : il relie la réponse renvoyée au client et la ligne en base
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    inputs = client.model_dump(exclude_unset=True)

    def event(status_code: int, **fields):
        return build_event(
            request_id=request_id,
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            status_code=status_code,
            inputs=inputs,
            total_time_ms=(time.perf_counter() - start) * 1000,
            **fields,
        )

    try:
        result = run_prediction(client, request_id)
    except HTTPException as exc:
        # Erreur prévue (422 ou 503) : journalisée, puis renvoyée au client
        prediction_logger.log(event(exc.status_code, error=str(exc.detail)))
        raise
    except Exception as exc:
        # Erreur imprévue : seul le type est journalisé ici, la trace complète part
        # dans les journaux techniques via le gestionnaire d'erreurs global
        prediction_logger.log(event(500, error=type(exc).__name__))
        raise

    # Succès : l'écriture est faite APRÈS l'envoi de la réponse (tâche de fond),
    # pour ne pas ajouter le temps d'accès à la base au temps de réponse
    background_tasks.add_task(prediction_logger.log, event(
        200,
        probability_default=result.probability_default,
        decision=result.decision,
        threshold=result.threshold,
        inference_time_ms=result.inference_time_ms,
    ))
    return result
