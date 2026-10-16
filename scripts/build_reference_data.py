"""Reconstruction du jeu de données enrichi du Projet 6.

Ce script reproduit à l'identique les cellules du notebook P6 qui ont produit
les données d'entraînement du modèle (cellules 164, 181 à 186, 195 à 219, 224 et 230).
Le résultat sert de données de référence pour l'analyse de dérive et de source
de clients de test pour l'API.
"""

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# Tables du zip Home Credit réellement utilisées par le modèle.
# Le dictionnaire des colonnes et les fichiers de test Kaggle sont ignorés.
NEEDED_TABLES = {
    "application_train",
    "bureau",
    "bureau_balance",
    "previous_application",
    "POS_CASH_balance",
    "installments_payments",
    "credit_card_balance",
}

DEFAULT_ZIP = Path("data/raw/home-credit-default-risk.zip")
DEFAULT_OUTPUT = Path("data/processed/application_train_enriched.parquet")


def load_tables(zip_path: Path) -> dict:
    """Charge les tables utiles directement depuis l'archive, sans la décompresser."""
    tables = {}
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            key = Path(name).stem
            # Le nom du fichier sans extension sert de clé, quel que soit le dossier interne
            if name.endswith(".csv") and key in NEEDED_TABLES:
                with archive.open(name) as f:
                    tables[key] = pd.read_csv(f)
                print(f"Loaded {key}: {tables[key].shape}")
    return tables


def build_application(app: pd.DataFrame) -> pd.DataFrame:
    """Table centrale réduite (cellules 163, 164 et 224 du notebook P6)."""
    # Versions _AVG et _MODE du logement : redondantes avec les versions _MEDI
    housing_avg_mode = [c for c in app.columns if c.endswith(("_AVG", "_MODE"))]
    # Redondances identifiées par l'analyse des corrélations
    redundant = ["FLAG_EMP_PHONE", "OBS_60_CNT_SOCIAL_CIRCLE",
                 "AMT_GOODS_PRICE", "REGION_RATING_CLIENT_W_CITY"]
    # Variables liées au moment de la demande, jugées peu pertinentes pour le risque
    low_relevance = ["WEEKDAY_APPR_PROCESS_START", "HOUR_APPR_PROCESS_START"]
    # Flags documents : remplacés par une variable de synthèse
    doc_flags = [c for c in app.columns if c.startswith("FLAG_DOCUMENT")]

    to_drop = set(housing_avg_mode + redundant + low_relevance + doc_flags)
    # L'ordre d'origine des colonnes est conservé : le pipeline y est sensible
    kept = [c for c in app.columns if c not in to_drop]

    app_reduced = app[kept].copy()
    app_reduced["NB_DOCUMENTS_FOURNIS"] = app[doc_flags].sum(axis=1)

    # La valeur 365243 (environ 1000 ans) signale une absence d'emploi.
    # Un drapeau conserve l'information, puis la valeur est remplacée par un manquant.
    app_reduced["DAYS_EMPLOYED_ANOM"] = (app_reduced["DAYS_EMPLOYED"] == 365243).astype(int)
    app_reduced["DAYS_EMPLOYED"] = app_reduced["DAYS_EMPLOYED"].replace(365243, np.nan)
    return app_reduced


def aggregate_bureau(bureau: pd.DataFrame, bureau_balance: pd.DataFrame) -> pd.DataFrame:
    """Agrégation en cascade bureau_balance puis bureau (cellules 181, 183 et 186)."""
    # Niveau 1 : bureau_balance résumé au niveau du crédit externe
    bb = pd.get_dummies(bureau_balance, columns=["STATUS"], dummy_na=False, dtype=int)
    bb_num = [c for c in bb.select_dtypes(include="number").columns if c != "SK_ID_BUREAU"]
    bb_agg = bb.groupby("SK_ID_BUREAU")[bb_num].agg(["mean", "max", "min", "sum"])
    bb_agg.columns = [f"BB_{col}_{stat}".upper() for col, stat in bb_agg.columns]
    bb_agg = bb_agg.reset_index()

    # Niveau 2 : jointure à bureau, puis résumé au niveau du client
    enriched = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")
    enriched = pd.get_dummies(enriched, columns=["CREDIT_ACTIVE"], dummy_na=False, dtype=int)

    agg_dict = {
        "DAYS_CREDIT": ["min", "max", "mean"],
        "DAYS_CREDIT_ENDDATE": ["min", "max", "mean"],
        "DAYS_ENDDATE_FACT": ["mean"],
        "DAYS_CREDIT_UPDATE": ["mean"],
        "CREDIT_DAY_OVERDUE": ["max", "mean"],
        "AMT_CREDIT_MAX_OVERDUE": ["max", "mean"],
        "AMT_CREDIT_SUM_OVERDUE": ["max", "sum"],
        "AMT_CREDIT_SUM": ["sum", "mean", "max"],
        "AMT_CREDIT_SUM_DEBT": ["sum", "mean", "max"],
        "AMT_CREDIT_SUM_LIMIT": ["sum", "mean"],
        "AMT_ANNUITY": ["mean", "sum"],
        "CNT_CREDIT_PROLONG": ["sum", "max"],
    }
    # Colonnes binaires ou proportionnelles : leur moyenne donne une proportion
    for c in enriched.columns:
        if c.startswith(("BB_", "CREDIT_ACTIVE_")):
            agg_dict[c] = ["mean"]

    agg = enriched.groupby("SK_ID_CURR").agg(agg_dict)
    agg.columns = [f"BUREAU_{col}_{stat}".upper() for col, stat in agg.columns]
    agg["BUREAU_COUNT"] = enriched.groupby("SK_ID_CURR").size()
    agg = agg.reset_index()

    # Redondances retirées au P6 (corrélations supérieures à 0,9)
    redundant = [
        "BUREAU_CNT_CREDIT_PROLONG_MAX", "BUREAU_AMT_CREDIT_SUM_OVERDUE_MAX",
        "BUREAU_AMT_CREDIT_SUM_DEBT_MAX", "BUREAU_AMT_CREDIT_MAX_OVERDUE_MEAN",
        "BUREAU_BB_STATUS_5_SUM_MEAN", "BUREAU_CREDIT_ACTIVE_CLOSED_MEAN",
        "BUREAU_BB_MONTHS_BALANCE_MIN_MEAN", "BUREAU_BB_MONTHS_BALANCE_SUM_MEAN",
        "BUREAU_BB_MONTHS_BALANCE_MEAN_MEAN",
    ]
    return agg.drop(columns=[c for c in redundant if c in agg.columns])


def aggregate_previous(prev: pd.DataFrame) -> pd.DataFrame:
    """Agrégation de previous_application (cellules 195 et 198)."""
    prev = prev.copy()
    # Écart entre montant accordé et montant demandé : un écart négatif est un signal
    prev["ECART_CREDIT_DEMANDE"] = prev["AMT_CREDIT"] - prev["AMT_APPLICATION"]
    cat_cols = ["NAME_CONTRACT_STATUS", "NAME_CONTRACT_TYPE", "NAME_YIELD_GROUP"]
    prev = pd.get_dummies(prev, columns=cat_cols, dummy_na=False, dtype=int)

    agg_dict = {
        "AMT_APPLICATION": ["mean", "max", "sum"],
        "AMT_CREDIT": ["mean", "max", "sum"],
        "AMT_ANNUITY": ["mean", "max"],
        "AMT_DOWN_PAYMENT": ["mean", "max"],
        "AMT_GOODS_PRICE": ["mean", "max"],
        "ECART_CREDIT_DEMANDE": ["mean", "min"],
        "RATE_DOWN_PAYMENT": ["mean"],
        "DAYS_DECISION": ["min", "max", "mean"],
        "CNT_PAYMENT": ["mean", "max"],
        "NFLAG_INSURED_ON_APPROVAL": ["mean"],
    }
    for c in prev.columns:
        if c.startswith(("NAME_CONTRACT_STATUS_", "NAME_CONTRACT_TYPE_", "NAME_YIELD_GROUP_")):
            agg_dict[c] = ["mean"]

    agg = prev.groupby("SK_ID_CURR").agg(agg_dict)
    agg.columns = [f"PREV_{col}_{stat}".upper() for col, stat in agg.columns]
    agg["PREV_COUNT"] = prev.groupby("SK_ID_CURR").size()
    agg = agg.reset_index()

    # Montants quasi identiques à AMT_CREDIT (corrélations supérieures à 0,98)
    redundant = ["PREV_AMT_APPLICATION_MAX", "PREV_AMT_APPLICATION_SUM",
                 "PREV_AMT_APPLICATION_MEAN", "PREV_AMT_GOODS_PRICE_MAX",
                 "PREV_AMT_GOODS_PRICE_MEAN"]
    return agg.drop(columns=[c for c in redundant if c in agg.columns])


def aggregate_pos(pos: pd.DataFrame) -> pd.DataFrame:
    """Agrégation de POS_CASH_balance (cellules 205 et 207)."""
    pos = pd.get_dummies(pos, columns=["NAME_CONTRACT_STATUS"], dummy_na=False, dtype=int)
    agg_dict = {
        "MONTHS_BALANCE": ["min", "max"],
        "CNT_INSTALMENT": ["mean", "max"],
        "CNT_INSTALMENT_FUTURE": ["mean", "min"],
        "SK_DPD": ["max", "mean", "sum"],
        "SK_DPD_DEF": ["max", "mean"],
    }
    for c in pos.columns:
        if c.startswith("NAME_CONTRACT_STATUS_"):
            agg_dict[c] = ["mean"]

    agg = pos.groupby("SK_ID_CURR").agg(agg_dict)
    agg.columns = [f"POS_{col}_{stat}".upper() for col, stat in agg.columns]
    agg["POS_COUNT"] = pos.groupby("SK_ID_CURR").size()
    agg = agg.reset_index()

    # Le maximum des retards est conservé : c'est l'événement le plus parlant pour le risque
    redundant = ["POS_SK_DPD_MEAN", "POS_SK_DPD_SUM",
                 "POS_SK_DPD_DEF_MEAN", "POS_CNT_INSTALMENT_FUTURE_MEAN"]
    return agg.drop(columns=[c for c in redundant if c in agg.columns])


def aggregate_installments(instal: pd.DataFrame) -> pd.DataFrame:
    """Agrégation de installments_payments (cellules 210 et 212)."""
    instal = instal.copy()
    # Écarts entre prévu et réalisé : le signal de risque est dans ces écarts
    instal["RETARD_PAIEMENT"] = instal["DAYS_ENTRY_PAYMENT"] - instal["DAYS_INSTALMENT"]
    instal["ECART_PAIEMENT"] = instal["AMT_PAYMENT"] - instal["AMT_INSTALMENT"]
    instal["RATIO_PAIEMENT"] = instal["AMT_PAYMENT"] / instal["AMT_INSTALMENT"]
    instal["RATIO_PAIEMENT"] = instal["RATIO_PAIEMENT"].replace([np.inf, -np.inf], np.nan)

    agg_dict = {
        "RETARD_PAIEMENT": ["max", "mean", "sum"],
        "ECART_PAIEMENT": ["min", "mean"],
        "RATIO_PAIEMENT": ["mean", "min"],
        "AMT_INSTALMENT": ["mean", "sum"],
        "AMT_PAYMENT": ["mean", "sum"],
        "NUM_INSTALMENT_VERSION": ["mean", "max"],
    }
    agg = instal.groupby("SK_ID_CURR").agg(agg_dict)
    agg.columns = [f"INSTAL_{col}_{stat}".upper() for col, stat in agg.columns]
    agg["INSTAL_COUNT"] = instal.groupby("SK_ID_CURR").size()
    agg = agg.reset_index()

    # Montant payé quasi identique au montant dû (corrélation 0,98)
    redundant = ["INSTAL_AMT_PAYMENT_SUM", "INSTAL_AMT_PAYMENT_MEAN"]
    return agg.drop(columns=[c for c in redundant if c in agg.columns])


def aggregate_credit_card(cc: pd.DataFrame) -> pd.DataFrame:
    """Agrégation de credit_card_balance (cellules 217 et 219)."""
    cc = cc.copy()
    # Taux d'utilisation : une carte utilisée au maximum traduit une tension financière
    cc["TAUX_UTILISATION"] = cc["AMT_BALANCE"] / cc["AMT_CREDIT_LIMIT_ACTUAL"]
    cc["TAUX_UTILISATION"] = cc["TAUX_UTILISATION"].replace([np.inf, -np.inf], np.nan)
    cc = pd.get_dummies(cc, columns=["NAME_CONTRACT_STATUS"], dummy_na=False, dtype=int)

    agg_dict = {
        "AMT_BALANCE": ["mean", "max"],
        "AMT_CREDIT_LIMIT_ACTUAL": ["mean", "max"],
        "TAUX_UTILISATION": ["mean", "max"],
        "SK_DPD": ["max", "mean", "sum"],
        "SK_DPD_DEF": ["max", "mean"],
        "AMT_DRAWINGS_CURRENT": ["mean", "sum"],
        "CNT_DRAWINGS_CURRENT": ["mean", "sum"],
        "AMT_INST_MIN_REGULARITY": ["mean"],
        "CNT_INSTALMENT_MATURE_CUM": ["max"],
        "MONTHS_BALANCE": ["min"],
    }
    for c in cc.columns:
        if c.startswith("NAME_CONTRACT_STATUS_"):
            agg_dict[c] = ["mean"]

    agg = cc.groupby("SK_ID_CURR").agg(agg_dict)
    agg.columns = [f"CC_{col}_{stat}".upper() for col, stat in agg.columns]
    agg["CC_COUNT"] = cc.groupby("SK_ID_CURR").size()
    agg = agg.reset_index()

    redundant = [
        "CC_MONTHS_BALANCE_MIN", "CC_SK_DPD_MEAN", "CC_SK_DPD_SUM",
        "CC_SK_DPD_DEF_MEAN", "CC_AMT_INST_MIN_REGULARITY_MEAN",
        "CC_NAME_CONTRACT_STATUS_COMPLETED_MEAN", "CC_AMT_CREDIT_LIMIT_ACTUAL_MAX",
    ]
    return agg.drop(columns=[c for c in redundant if c in agg.columns])


def main():
    parser = argparse.ArgumentParser(description="Rebuild the P6 enriched dataset.")
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    tables = load_tables(args.zip_path)

    app = build_application(tables["application_train"])
    aggregates = [
        aggregate_bureau(tables["bureau"], tables["bureau_balance"]),
        aggregate_previous(tables["previous_application"]),
        aggregate_pos(tables["POS_CASH_balance"]),
        aggregate_installments(tables["installments_payments"]),
        aggregate_credit_card(tables["credit_card_balance"]),
    ]

    # Jointure à gauche : chaque client de la table centrale garde exactement une ligne.
    # L'ordre de fusion est celui du P6, ce qui garantit le même ordre de colonnes.
    df = app
    for agg in aggregates:
        df = df.merge(agg, on="SK_ID_CURR", how="left")

    print(f"Enriched dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Default rate: {df['TARGET'].mean() * 100:.2f} %")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()