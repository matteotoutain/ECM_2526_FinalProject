"""
tgvmax_backend.py
Backend pour la prévision d'ouverture TGVmax à partir des snapshots journaliers.

À placer à la racine de ton projet (ou dans un dossier backend/).
"""

from __future__ import annotations

import os
import glob
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd


# =====================
# Configuration globale
# =====================

# Colonnes attendues dans les CSV tgvmax_YYYY-MM-DD.csv
COL_DATE = "date"               # date de circulation du train (YYYY-MM-DD)
COL_TRAIN_NO = "train_no"
COL_ENTITY = "entity"
COL_ORIGIN = "origine"
COL_DEST = "destination"
COL_OD_HAPPY = "od_happy_card"  # "OUI" => dispo TGVmax, "NON" sinon

DEFAULT_PATTERN = "tgvmax_*.csv"


@dataclass
class TgvMaxModel:
    """
    Objet léger qui contient :
      - le DataFrame brut filtré (trains)
      - la table proba globale en fonction de delta_days
    """
    trains: pd.DataFrame
    proba_by_delta: pd.Series


# =====================
# Utilitaires
# =====================

def parse_snapshot_date_from_filename(path: str) -> Optional[date]:
    """
    Extrait la date du snapshot à partir du nom de fichier.
    On attend un pattern du type tgvmax_YYYY-MM-DD.csv ou tgvmax_YYYY-MM-DD_blabla.csv
    """
    filename = os.path.basename(path)
    base = filename
    if base.startswith("tgvmax_"):
        base = base[len("tgvmax_") :]
    if base.endswith(".csv"):
        base = base[: -4]
    # Si on a un suffixe après "_", on garde la première partie
    base = base.split("_")[0]

    try:
        return datetime.strptime(base, "%Y-%m-%d").date()
    except ValueError:
        return None


def load_all_snapshots(data_dir: str, pattern: str = DEFAULT_PATTERN) -> pd.DataFrame:
    """
    Charge tous les fichiers tgvmax_*.csv d'un dossier, ajoute snapshot_date,
    concatène dans un seul DataFrame.
    """
    full_pattern = os.path.join(data_dir, pattern)
    paths = glob.glob(full_pattern)
    if not paths:
        raise FileNotFoundError(f"Aucun fichier trouvé avec le pattern : {full_pattern!r}")

    df_list: list[pd.DataFrame] = []
    for path in paths:
        snap_date = parse_snapshot_date_from_filename(path)
        if snap_date is None:
            print(f"[WARN] Impossible de parser une date depuis {path}, on ignore ce fichier.")
            continue

        tmp = pd.read_csv(path, sep=";", dtype=str)
        tmp["snapshot_date"] = pd.to_datetime(snap_date)
        df_list.append(tmp)

    if not df_list:
        raise RuntimeError("Tous les fichiers ont été ignorés, vérifier le format de nommage.")

    trains_raw = pd.concat(df_list, ignore_index=True)
    return trains_raw


def filter_trains_for_model(trains_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Applique les filtres de base :
      - garde uniquement les entités Nord/Sud (JCNORDSUD, JCSUDNORD)
      - supprime les lignes avec delta_days négatif
      - ajoute les colonnes delta_days et tgvmax_available
    """
    df = trains_raw.copy()

    # Filtre sur entity (comme dans ton notebook)
    mask_entity = (
        df[COL_ENTITY].str.contains("JCNORDSUD", case=False, na=False)
        | df[COL_ENTITY].str.contains("JCSUDNORD", case=False, na=False)
    )
    df = df[mask_entity].copy()

    # Date de départ (à partir de la colonne "date" du CSV)
    df["departure_date"] = pd.to_datetime(df[COL_DATE]).dt.date
    df["snapshot_date_only"] = pd.to_datetime(df["snapshot_date"]).dt.date

    # delta_days = (date départ) - (date snapshot)
    df["delta_days"] = (df["departure_date"] - df["snapshot_date_only"]).apply(lambda d: d.days)

    # On garde seulement delta_days >= 0 (pas de snapshot après le départ)
    df = df[df["delta_days"] >= 0].copy()

    # Variable binaire de dispo TGVmax
    df["tgvmax_available"] = df[COL_OD_HAPPY].str.upper().eq("OUI")

    return df


def compute_global_proba_by_delta(trains: pd.DataFrame) -> pd.Series:
    """
    Calcule la probabilité empirique de disponibilité TGVmax en fonction de delta_days,
    toutes lignes confondues.
    """
    proba_by_delta = (
        trains.groupby("delta_days")["tgvmax_available"]
        .mean()
        .sort_index()
    )
    return proba_by_delta


# =====================
# Construction du modèle
# =====================

def build_model(data_dir: str) -> TgvMaxModel:
    """
    Pipeline complet :
      1. chargement de tous les snapshots tgvmax_*.csv
      2. filtrage + features (delta_days, tgvmax_available)
      3. calcul de la proba globale par delta_days
    """
    trains_raw = load_all_snapshots(data_dir=data_dir, pattern=DEFAULT_PATTERN)
    trains = filter_trains_for_model(trains_raw)
    proba_by_delta = compute_global_proba_by_delta(trains)

    return TgvMaxModel(trains=trains, proba_by_delta=proba_by_delta)


# =====================
# Fonction de prévision
# =====================

def forecast_opening_curve(
    model: TgvMaxModel,
    departure_date: date,
    today: Optional[date] = None,
) -> pd.DataFrame:
    """
    Construit la courbe de probabilité d'ouverture entre (today) et (departure_date - 1).

    Hypothèse : on utilise une proba globale P(ouvert) en fonction de delta_days,
    basée sur les observations historiques.

    Retourne un DataFrame avec colonnes :
      - date             : date calendaire
      - prob_open        : proba approx. que la résa "s'ouvre" ce jour-là
      - prob_open_cum    : proba qu'elle soit déjà ouverte à cette date
    """
    if today is None:
        today = date.today()

    if departure_date <= today:
        raise ValueError("La date de départ doit être dans le futur.")

    proba_by_delta = model.proba_by_delta

    max_delta_known = int(proba_by_delta.index.max())
    # Si le départ est très lointain, on commence la courbe à departure_date - max_delta_known.
    start_date = max(today, departure_date - timedelta(days=max_delta_known))

    nb_days = (departure_date - start_date).days
    dates = [start_date + timedelta(days=i) for i in range(nb_days)]

    # Proba "ouverte ce jour-là" (daily_p) = P(TGVmax disponible pour ce delta_days)
    daily_p = []
    for d in dates:
        delta = (departure_date - d).days
        p = float(proba_by_delta.get(delta, 0.0))
        p = max(0.0, min(1.0, p))
        daily_p.append(p)

    # On passe à une interprétation "événement d'ouverture" + cumul.
    prob_open = []
    prob_open_cum = []
    prob_not_open_yet = 1.0

    for p in daily_p:
        # Approximation : si la résa n'est pas déjà ouverte,
        # probabilité qu'elle s'ouvre ce jour = p * prob_not_open_yet
        p_new = p * prob_not_open_yet
        prob_open.append(p_new)
        prob_not_open_yet *= (1 - p)
        prob_open_cum.append(1 - prob_not_open_yet)

    forecast_df = pd.DataFrame({
        "date": dates,
        "prob_open": prob_open,
        "prob_open_cum": prob_open_cum,
    })

    return forecast_df


def get_most_likely_opening_date(forecast_df: pd.DataFrame) -> tuple[date, float]:
    """
    Renvoie (date_ouverte_max, probabilité_ce_jour_là) à partir d'une courbe de forecast.
    """
    idx = forecast_df["prob_open"].idxmax()
    row = forecast_df.loc[idx]
    return row["date"], float(row["prob_open"])
