"""Interface Gradio de test de l'API de scoring crédit.

Cette interface est un CLIENT de l'API, au même titre qu'un tableau de bord ou qu'une
application métier : elle tourne dans son propre conteneur et appelle l'API par HTTP.
Si l'interface tombe en panne, l'API continue de répondre à ses autres clients.
L'interface ne contient ni le modèle, ni les règles de validation : c'est l'API qui
valide les données et calcule la prédiction.

Lancement en local, depuis la racine du projet (l'API doit tourner par ailleurs) :
    API_URL=http://127.0.0.1:8000 python -m ui.app
puis ouvrir http://127.0.0.1:7860
"""

import os
from typing import Optional

# Pas d'envoi de statistiques d'usage ni de vérification de version vers les serveurs
# de Gradio : l'interface ne communique qu'avec l'API. La variable doit être définie
# AVANT l'import de Gradio, qui la lit à ce moment-là.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr  # noqa: E402
import requests  # noqa: E402

# Adresse de l'API, fournie par une variable d'environnement : la même interface peut
# appeler l'API locale ou l'API de production, sans modifier le code
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

# Délai d'attente maximal : l'API sur Azure peut mettre jusqu'à une minute à se réveiller
TIMEOUT_SECONDS = 90

DAYS_PER_YEAR = 365.25


def form_to_payload(
    contract_type: str,
    gender: str,
    income: Optional[float],
    credit: Optional[float],
    annuity: Optional[float],
    age_years: Optional[float],
    years_employed: Optional[float],
    ext_source_2: Optional[float],
    ext_source_3: Optional[float],
) -> dict:
    """Convertit les champs du formulaire, lisibles par un humain, au format de l'API.

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


def format_errors(detail) -> str:
    """Met en forme le détail d'une erreur 422 renvoyée par l'API.

    Deux formats sont possibles : une liste de dictionnaires (validation Pydantic des
    champs déclarés) ou une liste de messages (contrôle des variables complémentaires).
    """
    if not isinstance(detail, list):
        return str(detail)
    lines = []
    for error in detail:
        if isinstance(error, dict):
            # « body » désigne le corps de la requête : inutile pour l'utilisateur
            field = ".".join(str(part) for part in error.get("loc", []) if part != "body")
            lines.append(f"- {field} : {error.get('msg', '')}")
        else:
            lines.append(f"- {error}")
    return "\n".join(lines)


def score_form(*fields) -> tuple[str, dict]:
    """Envoie le formulaire à l'API et met en forme la réponse, quelle qu'elle soit."""
    payload = form_to_payload(*fields)
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=TIMEOUT_SECONDS)
    except requests.RequestException:
        # L'interface gère l'indisponibilité de l'API sans planter
        return "### API unreachable\nThe scoring API did not respond. Please try again later.", {}

    if response.status_code == 200:
        result = response.json()
        verdict = "REFUSED" if result["decision"] == "refused" else "ACCEPTED"
        summary = (
            f"### Credit {verdict}\n"
            f"Probability of default: **{result['probability_default']:.1%}** "
            f"(threshold {result['threshold']:.0%})"
        )
        return summary, result

    # Erreur renvoyée par l'API : le message est affiché tel quel, sans plantage
    try:
        detail = response.json().get("detail", "")
    except ValueError:
        detail = response.text
    if response.status_code == 422:
        return "### Invalid input\n" + format_errors(detail), {}
    return f"### Error {response.status_code}\n{detail}", {}


def build_demo() -> gr.Blocks:
    """Construit le formulaire de test."""
    with gr.Blocks(title="Credit Scoring - Prêt à Dépenser") as demo:
        gr.Markdown(
            "# Credit Scoring - Prêt à Dépenser\n"
            "Fill in the main characteristics of the application. "
            "Fields left empty are imputed by the model.\n\n"
            f"Scoring API: `{API_URL}`"
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
        button.click(score_form, inputs, [summary, details])
    return demo


if __name__ == "__main__":  # pragma: no cover
    # 0.0.0.0 : l'interface accepte les connexions venant de l'extérieur du conteneur
    build_demo().launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
