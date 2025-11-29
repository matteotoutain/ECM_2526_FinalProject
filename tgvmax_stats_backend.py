"""
tgvmax_stats_backend.py

Backend léger qui lit :
- les stats pré-calculées dans ./precomputed
- le dernier snapshot brut dans ./snapshots (fichiers tgvmax_YYYY-MM-DD*.csv)
pour connaître l'état réel du trajet aujourd'hui.
"""

from __future__ import annotations

import json
import glob
import os
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Dossier des stats pré-calculées
PRECOMPUTED_DIR = Path("precomputed")

# Dossier des snapshots bruts
SNAPSHOT_DIR = Path("snapshots")

# Colonnes standard
COL_ORIGIN = "origine"
COL_DEST = "destination"
COL_DATE = "date"
COL_OD_HAPPY = "od_happy_card"


@dataclass
class TgvMaxStats:
    proba_global: pd.DataFrame   # colonnes : delta_days, proba_open
    proba_od: pd.DataFrame       # colonnes : origine, destination, delta_days, proba_open
    stations: list[str]
    mean_open: float             # moyenne globale d'ouverture
    snapshot_today: Optional[pd.DataFrame] = None  # dernier snapshot brut, si dispo


# =====================
# Helpers snapshots
# =====================

def _extract_snapshot_date_from_path(path: str) -> date:
    """
    Extrait une date (date du snapshot) à partir d'un chemin de fichier de type :
      - tgvmax_YYYY-MM-DD.csv
      - tgvmax_YYYY-MM-DD_blabla.csv

    Si parsing impossible, retourne date.min (pour que ce fichier soit trié en premier).
    """
    name = os.path.basename(path)
    base = name
    if base.startswith("tgvmax_"):
        base = base[len("tgvmax_"):]
    if base.endswith(".csv"):
        base = base[:-4]
    base = base.split("_")[0]

    try:
        return pd.to_datetime(base).date()
    except Exception:
        return date.min


def _find_latest_snapshot() -> Optional[Path]:
    """
    Cherche le dernier fichier tgvmax_YYYY-MM-DD*.csv dans SNAPSHOT_DIR.
    Retourne le Path correspondant ou None si aucun snapshot trouvé.
    """
    pattern = str(SNAPSHOT_DIR / "tgvmax_*.csv")
    paths = glob.glob(pattern)
    if not paths:
        return None

    paths_sorted = sorted(paths, key=_extract_snapshot_date_from_path)
    latest = paths_sorted[-1]
    return Path(latest)


def _load_snapshot_today() -> Optional[pd.DataFrame]:
    """
    Charge automatiquement le **dernier snapshot tgvmax_YYYY-MM-DD*.csv**
    trouvé dans SNAPSHOT_DIR.

    Si aucun snapshot n'est trouvé, retourne None.

    Colonnes attendues dans les CSV :
      - date              : date de circulation du train (YYYY-MM-DD)
      - origine           : gare d'origine
      - destination       : gare d'arrivée
      - od_happy_card     : "OUI" / "NON" (TGVmax dispo ou non)
      - (éventuellement d'autres colonnes ignorées ici)
    """
    latest_path = _find_latest_snapshot()
    if latest_path is None:
        return None

    df = pd.read_csv(latest_path, sep=";", dtype=str)

    # departure_date : date de circulation du train
    df["departure_date"] = pd.to_datetime(df[COL_DATE]).dt.date

    # snapshot_date_only : date du snapshot dérivée du nom du fichier
    snap_date = _extract_snapshot_date_from_path(str(latest_path))
    df["snapshot_date_only"] = snap_date

    # tgvmax_available : booléen sur la dispo TGVmax
    df["tgvmax_available"] = df[COL_OD_HAPPY].str.upper().eq("OUI")

    return df


# =====================
# Chargement des stats
# =====================

def load_stats() -> TgvMaxStats:
    """
    Charge les stats pré-calculées et, si possible, le dernier snapshot brut.
    """
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


# =====================
# Helpers OD / probabilités
# =====================

def _od_exists_in_data(
    stats: TgvMaxStats,
    origin: Optional[str],
    destination: Optional[str],
) -> bool:
    """
    Indique si le couple (origine, destination) existe dans les données :
      - soit dans les stats pré-calculées (proba_od)
      - soit dans le dernier snapshot brut (snapshot_today)

    Retourne True si au moins un train a été vu pour cet OD, False sinon.
    """
    if origin is None or destination is None:
        return False

    # 1) présence dans les stats pré-calculées
    subset_stats = stats.proba_od[
        (stats.proba_od[COL_ORIGIN] == origin)
        & (stats.proba_od[COL_DEST] == destination)
    ]
    if not subset_stats.empty:
        return True

    # 2) présence dans le snapshot brut le plus récent
    if stats.snapshot_today is not None:
        df = stats.snapshot_today
        subset_snap = df[
            (df[COL_ORIGIN] == origin)
            & (df[COL_DEST] == destination)
        ]
        if not subset_snap.empty:
            return True

    return False


def _get_daily_probability(
    stats: TgvMaxStats,
    delta: int,
    origin: Optional[str],
    destination: Optional[str],
) -> float:
    """
    Récupère la proba d'ouverture pour un delta_days donné :
      - d'abord spécifique au trajet (origine, destination, delta_days)
      - sinon stat globale proba_global[delta_days]
      - sinon moyenne globale
    """
    p = None

    # Proba spécifique à l'OD si dispo
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


# =====================
# État "aujourd'hui" via snapshot brut
# =====================

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
      - status_today : "open_today", "closed_today", "no_data_today",
                       "invalid_od", "unknown_od"
      - is_open_today : True / False / None
    """
    if origin is None or destination is None:
        return "unknown_od", None

    # OD jamais vu dans les données
    if not _od_exists_in_data(stats, origin, destination):
        return "invalid_od", None

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


# =====================
# Courbe de prévision
# =====================

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
    - Utilise, si présent, le dernier snapshot brut pour savoir si le trajet est déjà
      ouvert aujourd'hui.

    Colonnes renvoyées :
      - date             : date calendaire
      - prob_open        : proba approx. que la résa "s'ouvre" ce jour-là
      - prob_open_cum    : proba qu'elle soit déjà ouverte à cette date
      - status_today     : "open_today", "closed_today", "no_data_today",
                           "invalid_od", "unknown_od"
      - open_today       : True / False / None
      - od_exists        : True / False
    """
    if today is None:
        today = date.today()
    if departure_date <= today:
        raise ValueError("La date de départ doit être dans le futur.")

    # OD existe-t-il dans les données ?
    od_exists = _od_exists_in_data(stats, origin, destination)

    # 1) État actuel dans le snapshot du jour (si existant)
    status_today, is_open_today = _get_today_availability_status(
        stats=stats,
        departure_date=departure_date,
        today=today,
        origin=origin,
        destination=destination,
    )

    # Si déjà ouvert aujourd'hui, on renvoie une "courbe" dégénérée :
    # un seul point, aujourd'hui, avec proba = 1.
    if is_open_today is True:
        return pd.DataFrame(
            {
                "date": [today],
                "prob_open": [1.0],
                "prob_open_cum": [1.0],
                "status_today": [status_today],
                "open_today": [True],
                "od_exists": [od_exists],
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
    # Au début on suppose "non ouvert", cohérent avec l’interprétation historique
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

    # On ajoute les infos d'état "aujourd'hui" et d'existence de l'OD
    forecast_df["status_today"] = status_today
    forecast_df["open_today"] = is_open_today
    forecast_df["od_exists"] = od_exists

    return forecast_df


# =====================
# Date la plus probable
# =====================

def get_most_likely_opening_date(forecast_df: pd.DataFrame) -> tuple[date, float]:
    """
    Renvoie (date_ouverte_max, probabilité_ce_jour_là) à partir d'une courbe de forecast.
    """
    idx = forecast_df["prob_open"].idxmax()
    row = forecast_df.loc[idx]
    return row["date"], float(row["prob_open"])
