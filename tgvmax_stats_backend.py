"""
tgvmax_stats_backend.py

Backend léger qui lit uniquement les stats pré-calculées dans ./precomputed
et, si disponible, un snapshot du jour pour connaître l'état réel du trajet aujourd'hui.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PRECOMPUTED_DIR = Path("precomputed")
COL_ORIGIN = "origine"
COL_DEST = "destination"
COL_DATE = "date"
COL_OD_HAPPY = "od_happy_card"


@dataclass
class TgvMaxStats:
    proba_global: pd.DataFrame   # delta_days, proba_open
    proba_od: pd.DataFrame       # origine, destination, delta_days, proba_open
    stations: list[str]
    mean_open: float             # moyenne globale d'ouverture
    snapshot_today: Optional[pd.DataFrame] = None  # snapshot du jour (optionnel)


def _load_snapshot_today() -> Optional[pd.DataFrame]:
    """
    Charge un snapshot du jour si disponible dans precomputed/snapshot_today.csv.

    Format attendu (aligné sur les CSV d'origine) :
      - date              : date de circulation du train (YYYY-MM-DD)
      - origine           : nom/label de la gare d'origine
      - destination       : nom/label de la gare de destination
      - od_happy_card     : "OUI" / "NON" (TGVmax dispo ou non)
      - (optionnel) snapshot_date : date du snapshot ; si absent, on suppose today()

    Si le fichier n'existe pas, retourne None.
    """
    snapshot_path = PRECOMPUTED_DIR / "snapshot_today.csv"
    if not snapshot_path.exists():
        return None

    df = pd.read_csv(snapshot_path, sep=";", dtype=str)

    # Normalisation minimale
    # - departure_date : type date
    df["departure_date"] = pd.to_datetime(df[COL_DATE]).dt.date

    # - snapshot_date_only : si pas présent, on met la date du jour
    if "snapshot_date" in df.columns:
        df["snapshot_date_only"] = pd.to_datetime(df["snapshot_date"]).dt.date
    else:
        df["snapshot_date_only"] = date.today()

    # - tgvmax_available : booléen dérivé de od_happy_card
    df["tgvmax_available"] = df[COL_OD_HAPPY].str.upper().eq("OUI")

    return df


def load_stats() -> TgvMaxStats:
    proba_global = pd.read_csv(PRECOMPUTED_DIR / "proba_global.csv")
    proba_od = pd.read_parquet(PRECOMPUTED_DIR / "proba_od.parquet")

    with open(PRECOMPUTED_DIR / "stations.json", "r", encoding="utf-8") as f:
        stations = json.load(f)

    mean_open = float(proba_global["proba_open"].mean())

    snapshot_today = _load_snapshot_today()

    return TgvMaxStats(
        proba_global=proba_global,
        proba_od=proba_od,
        stations=stations,
        mean_open=mean_open,
        snapshot_today=snapshot_today,
    )


def _get_daily_probability(
    stats: TgvMaxStats,
    delta: int,
    origin: Optional[str],
    destination: Optional[str],
) -> float:
    p = None

    # Proba spécifique OD si disponible
    if origin is not None and destination is not None:
        subset = stats.proba_od[
            (stats.proba_od[COL_ORIGIN] == origin)
            & (stats.proba_od[COL_DEST] == destination)
            & (stats.proba_od["delta_days"] == delta)
        ]
        if not subset.empty:
            p = float(subset["proba_open"].iloc[0])

    # Sinon proba globale par delta
    if p is None:
        subset_g = stats.proba_global[stats.proba_global["delta_days"] == delta]
        if not subset_g.empty:
            p = float(subset_g["proba_open"].iloc[0])

    # Sinon moyenne globale
    if p is None:
        p = stats.mean_open

    return max(0.0, min(1.0, p))


def _get_today_availability_status(
    stats: TgvMaxStats,
    departure_date: date,
    today: date,
    origin: Optional[str],
    destination: Optional[str],
) -> tuple[str, Optional[bool]]:
    """
    Utilise snapshot_today (si fourni) pour savoir si le trajet est déjà ouvert aujourd'hui.

    Retourne :
      - status_today : "open_today", "closed_today", "no_data_today", "unknown_od"
      - is_open_today : True / False / None
    """
    if origin is None or destination is None:
        return "unknown_od", None

    if stats.snapshot_today is None:
        return "no_data_today", None

    df = stats.snapshot_today

    mask = (
        (df["departure_date"] == departure_date)
        & (df["snapshot_date_only"] == today)
        & (df[COL_ORIGIN] == origin)
        & (df[COL_DEST] == destination)
    )

    subset = df[mask]

    if subset.empty:
        return "no_data_today", None

    is_open = bool(subset["tgvmax_available"].any())
    if is_open:
        return "open_today", True
    else:
        return "closed_today", False


def forecast_opening_curve(
    stats: TgvMaxStats,
    departure_date: date,
    today: Optional[date] = None,
    origin: Optional[str] = None,
    destination: Optional[str] = None,
) -> pd.DataFrame:
    """
    Construit la courbe de probabilité d'ouverture entre (today) et (departure_date - 1).

    - Utilise les stats pré-calculées (proba_global / proba_od) pour les probabilités.
    - Si un fichier snapshot_today.csv est présent dans ./precomputed, il est utilisé
      pour déterminer si le trajet est déjà ouvert aujourd'hui.

    Colonnes renvoyées :
      - date             : date calendaire
      - prob_open        : proba approx. que la résa "s'ouvre" ce jour-là
      - prob_open_cum    : proba qu'elle soit déjà ouverte à cette date
      - status_today     : "open_today", "closed_today", "no_data_today", "unknown_od"
      - open_today       : True / False / None
    """
    if today is None:
        today = date.today()
    if departure_date <= today:
        raise ValueError("La date de départ doit être dans le futur.")

    # 1) État actuel dans le snapshot du jour (si existant)
    status_today, is_open_today = _get_today_availability_status(
        stats=stats,
        departure_date=departure_date,
        today=today,
        origin=origin,
        destination=destination,
    )

    # Si déjà ouvert aujourd'hui, on renvoie une "courbe" dégénérée :
    # un seul point, aujourd'hui, avec proba=1.
    if is_open_today is True:
        return pd.DataFrame(
            {
                "date": [today],
                "prob_open": [1.0],
                "prob_open_cum": [1.0],
                "status_today": [status_today],
                "open_today": [True],
            }
        )

    # 2) Sinon : calcul standard de la courbe avec les stats pré-calculées
    max_delta_known = int(stats.proba_global["delta_days"].max())
    start_date = max(today, departure_date - timedelta(days=max_delta_known))

    nb_days = (departure_date - start_date).days
    dates = [start_date + timedelta(days=i) for i in range(nb_days)]

    daily_p = []
    for d in dates:
        delta = (departure_date - d).days
        daily_p.append(_get_daily_probability(stats, delta, origin, destination))

    prob_open = []
    prob_open_cum = []
    # Ici on suppose "fermée" au début, cohérent avec l'interprétation historique
    prob_not_open_yet = 1.0

    for p in daily_p:
        p_new = p * prob_not_open_yet
        prob_open.append(p_new)
        prob_not_open_yet *= (1 - p)
        prob_open_cum.append(1 - prob_not_open_yet)

    forecast_df = pd.DataFrame({
        "date": dates,
        "prob_open": prob_open,
        "prob_open_cum": prob_open_cum,
    })

    # On ajoute les infos d'état "aujourd'hui" sur toutes les lignes pour que le front
    # puisse les lire facilement (on dupliquera la même valeur sur chaque ligne).
    forecast_df["status_today"] = status_today
    forecast_df["open_today"] = is_open_today

    return forecast_df


def get_most_likely_opening_date(forecast_df: pd.DataFrame) -> tuple[date, float]:
    idx = forecast_df["prob_open"].idxmax()
    row = forecast_df.loc[idx]
    return row["date"], float(row["prob_open"])
