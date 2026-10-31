"""Interface Gradio de démonstration de l'API.

L'interface est montée dans la même application que l'API (adresse /ui) : un seul
conteneur à déployer. Elle appelle exactement la même fonction de prédiction que la
route /predict, avec la même validation : ce qui est refusé par l'API l'est aussi ici.
"""

from typing import Callable, Optional

import gradio as gr
from fastapi import HTTPException
from pydantic import ValidationError

from api.schemas import ClientData, PredictionResponse

DAYS_PER_YEAR = 365.25


def form_to_payload(
    contract_type: str,
    gender: str,
    income: float,
    credit: float,
    annuity: Optional[float],
    age_years: float,
    years_employed: Optional[float],
    ext_source_2: Optional[float],
    ext_source_3: Optional[float],
) -> dict:
    """Convertit les champs du formulaire, lisibles par un humain, au format du modèle.

    Le modèle attend l'âge et l'ancienneté en jours comptés à rebours (valeurs négatives),
    comme dans les données Home Credit. Une ancienneté vide signifie « sans emploi ».
    """
    payload = {
        "NAME_CONTRACT_TYPE": contract_type,
        "CODE_GENDER": gender,
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "DAYS_BIRTH": -round(age_years * DAYS_PER_YEAR) if age_years is not None else None,
    }
    optional = {
        "AMT_ANNUITY": annuity,
        "DAYS_EMPLOYED": -round(years_employed * DAYS_PER_YEAR) if years_employed is not None else None,
        "EXT_SOURCE_2": ext_source_2,
        "EXT_SOURCE_3": ext_source_3,
    }
    # Seuls les champs facultatifs renseignés sont transmis : les autres seront imputés
    payload.update({k: v for k, v in optional.items() if v is not None})
    return payload


def score_form(predict_fn: Callable[[ClientData], PredictionResponse], *fields) -> tuple[str, dict]:
    """Valide le formulaire, appelle la prédiction et met en forme le résultat."""
    try:
        client = ClientData(**form_to_payload(*fields))
        result = predict_fn(client)
    except ValidationError as exc:
        # Mêmes règles que l'API : chaque erreur est affichée avec le champ concerné
        messages = [f"- {'.'.join(map(str, e['loc']))} : {e['msg']}" for e in exc.errors()]
        return "### Invalid input\n" + "\n".join(messages), {}
    except HTTPException as exc:
        return f"### Error {exc.status_code}\n{exc.detail}", {}

    verdict = "REFUSED" if result.decision == "refused" else "ACCEPTED"
    summary = (
        f"### Credit {verdict}\n"
        f"Probability of default: **{result.probability_default:.1%}** "
        f"(threshold {result.threshold:.0%})"
    )
    return summary, result.model_dump()


def build_demo(predict_fn: Callable[[ClientData], PredictionResponse]) -> gr.Blocks:
    """Construit l'interface. La fonction de prédiction est passée en paramètre pour
    éviter une dépendance circulaire avec main.py."""
    with gr.Blocks(title="Credit Scoring - Prêt à Dépenser") as demo:
        gr.Markdown(
            "# Credit Scoring - Prêt à Dépenser\n"
            "Fill in the main characteristics of the application. "
            "Fields left empty are imputed by the model."
        )
        with gr.Row():
            with gr.Column():
                contract_type = gr.Dropdown(["Cash loans", "Revolving loans"], value="Cash loans", label="Contract type")
                gender = gr.Dropdown(["F", "M"], value="F", label="Gender")
                income = gr.Number(value=135000, label="Annual income")
                credit = gr.Number(value=568800, label="Credit amount")
                annuity = gr.Number(value=20560.5, label="Annuity (optional)")
            with gr.Column():
                age_years = gr.Number(value=41, label="Age (years)")
                years_employed = gr.Number(value=7, label="Years employed (empty if unemployed)")
                ext_source_2 = gr.Slider(0, 1, value=0.52, step=0.01, label="External score 2")
                ext_source_3 = gr.Slider(0, 1, value=0.48, step=0.01, label="External score 3")
        button = gr.Button("Score the application", variant="primary")
        summary = gr.Markdown()
        details = gr.JSON(label="API response")

        inputs = [contract_type, gender, income, credit, annuity,
                  age_years, years_employed, ext_source_2, ext_source_3]
        button.click(lambda *fields: score_form(predict_fn, *fields), inputs, [summary, details])
    return demo