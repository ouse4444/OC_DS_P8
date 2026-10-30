"""Schémas de validation des entrées et des sorties de l'API (Pydantic).

Le modèle attend 199 variables. Les plus importantes pour le métier sont déclarées
explicitement, avec leurs contraintes : elles sont obligatoires ou bornées, et
documentées dans Swagger. Les autres variables sont acceptées en complément,
puis contrôlées par le service du modèle (nom connu, type correct).
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ClientData(BaseModel):
    """Données d'une demande de crédit."""

    # extra="allow" : les autres variables du modèle peuvent être envoyées en complément
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": {
                "NAME_CONTRACT_TYPE": "Cash loans",
                "CODE_GENDER": "F",
                "AMT_INCOME_TOTAL": 135000.0,
                "AMT_CREDIT": 568800.0,
                "AMT_ANNUITY": 20560.5,
                "CNT_FAM_MEMBERS": 2.0,
                "DAYS_BIRTH": -15000,
                "DAYS_EMPLOYED": -2500.0,
                "EXT_SOURCE_2": 0.52,
                "EXT_SOURCE_3": 0.48,
            }
        },
    )

    # Champs obligatoires : sans eux, une demande de crédit n'a pas de sens
    NAME_CONTRACT_TYPE: Literal["Cash loans", "Revolving loans"] = Field(
        ..., description="Type of loan"
    )
    CODE_GENDER: Literal["M", "F", "XNA"] = Field(..., description="Gender of the client")
    AMT_INCOME_TOTAL: float = Field(..., gt=0, description="Annual income (must be > 0)")
    AMT_CREDIT: float = Field(..., gt=0, description="Credit amount (must be > 0)")
    DAYS_BIRTH: int = Field(
        ..., ge=-36500, le=-6570,
        description="Age in days, counted backwards (between 18 and 100 years)",
    )

    # Champs facultatifs mais bornés : une valeur absente est imputée par le pipeline
    AMT_ANNUITY: Optional[float] = Field(None, gt=0, description="Loan annuity")
    CNT_FAM_MEMBERS: Optional[float] = Field(None, ge=1, le=20, description="Family size")
    DAYS_EMPLOYED: Optional[float] = Field(
        None, le=0,
        description="Employment length in days, counted backwards (empty if unemployed)",
    )
    EXT_SOURCE_1: Optional[float] = Field(None, ge=0, le=1, description="External score 1")
    EXT_SOURCE_2: Optional[float] = Field(None, ge=0, le=1, description="External score 2")
    EXT_SOURCE_3: Optional[float] = Field(None, ge=0, le=1, description="External score 3")


class PredictionResponse(BaseModel):
    """Réponse de l'API : le schéma valide aussi les sorties avant leur envoi."""

    probability_default: float = Field(..., ge=0, le=1, description="Probability of default")
    decision: Literal["accepted", "refused"] = Field(..., description="Credit decision")
    threshold: float = Field(..., description="Decision threshold used")
    model_name: str
    model_version: str
    inference_time_ms: float = Field(..., ge=0, description="Model inference time (ms)")


class HealthResponse(BaseModel):
    """État de l'API."""

    status: Literal["ok", "unavailable"]
    model_loaded: bool
    model_name: str
    model_version: str