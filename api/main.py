"""API FastAPI de scoring crédit.

Lancement en local, depuis la racine du projet :
    uvicorn api.main:app --reload
Documentation interactive : http://127.0.0.1:8000/docs
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from api.config import DECISION_THRESHOLD, MODEL_DIR, MODEL_NAME, MODEL_VERSION
from api.model_service import CreditScoringModel
from api.schemas import ClientData, HealthResponse, PredictionResponse

logger = logging.getLogger("credit_scoring_api")
logging.basicConfig(level=logging.INFO)

# Conteneur partagé : le modèle y est déposé au démarrage, puis réutilisé par chaque requête
ml_models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle une seule fois, au démarrage de l'API.

    Charger le modèle à chaque requête ralentirait fortement les réponses et
    saturerait la mémoire sous charge.
    """
    try:
        ml_models["credit_scoring"] = CreditScoringModel(MODEL_DIR)
        logger.info("Model loaded from %s", MODEL_DIR)
    except Exception:
        # L'API démarre quand même : /health signalera que le modèle est indisponible
        logger.exception("Model loading failed")
    yield
    # Code exécuté à l'arrêt de l'API : libération du modèle
    ml_models.clear()


app = FastAPI(
    title="Credit Scoring API - Prêt à Dépenser",
    description=(
        "Returns the probability of default of a credit applicant and the credit "
        f"decision, using the LightGBM model of Project 6 (threshold {DECISION_THRESHOLD})."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    """Toute erreur imprévue renvoie un message générique, sans exposer de détail interne."""
    logger.exception("Unexpected error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


def get_model() -> CreditScoringModel:
    """Retourne le modèle chargé, ou une erreur 503 s'il n'est pas disponible."""
    model = ml_models.get("credit_scoring")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    return model


@app.get("/", tags=["General"])
def root():
    """Message d'accueil et liens utiles."""
    return {"message": "Credit Scoring API - Prêt à Dépenser", "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health():
    """Indique si l'API fonctionne et si le modèle est chargé."""
    loaded = "credit_scoring" in ml_models
    return HealthResponse(
        status="ok" if loaded else "unavailable",
        model_loaded=loaded,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
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


@app.post("/predict", response_model=PredictionResponse, tags=["Model"])
def predict(client: ClientData):
    """Calcule la probabilité de défaut d'un client et la décision de crédit."""
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
        probability_default=round(proba, 4),
        decision=decision,
        threshold=DECISION_THRESHOLD,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        inference_time_ms=round(inference_ms, 2),
    )