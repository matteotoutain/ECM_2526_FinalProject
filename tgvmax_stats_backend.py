"""
tgvmax_stats_backend.py

Backend léger qui lit uniquement les stats pré-calculées dans ./precomputed.
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


@dataclass
class TgvMaxStats:
    proba_global: pd.DataFrame   # delta_days, proba_open
    proba_od: pd.DataFrame       # origine, destination, delta_days, proba_open
    stations: list[str]
    mean_open: float             # moyenne globale d'ouverture


def load_stats() -> TgvMaxStats:
    proba_global = pd.read_csv(PRECOMPUTED_DIR / "proba_global.csv")
    proba_od = pd.read_parquet(PRECOMPUTED_DIR / "proba_od.parquet")

    with open(PRECOMPUTED_DIR / "stations.json", "r", encoding="utf-8") as f:
        stations = json.load(f)

    mean_open = float(proba_global["proba_open"].mean())

    return TgvMaxStats(
        proba_global=proba_global,
        proba_od=proba_od,
        stations=stations,
        mean_open=mean_open,
    )


def _get_daily_probability(
    stats: TgvMaxStats,
    delta: int,
    origin: Optional[str],
    destination: Optional[str],
) -> float:
    p = None

    if origin is not None and destination is not None:
        subset = stats.proba_od[
            (stats.proba_od[COL_ORIGIN] == origin)
            & (stats.proba_od[COL_DEST] == destination)
            & (stats.proba_od["delta_days"] == delta)
        ]
        if not subset.empty:
            p = float(subset["proba_open"].iloc[0])

    if p is None:
        subset_g = stats.proba_global[stats.proba_global["delta_days"] == delta]
        if not subset_g.empty:
            p = float(subset_g["proba_open"].iloc[0])

    if p is None:
        p = stats.mean_open

    return max(0.0, min(1.0, p))


def forecast_opening_curve(
    stats: TgvMaxStats,
    departure_date: date,
    today: Optional[date] = None,
    origin: Optional[str] = None,
    destination: Optional[str] = None,
) -> pd.DataFrame:
    if today is None:
        today = date.today()
    if departure_date <= today:
        raise ValueError("La date de départ doit être dans le futur.")

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
    prob_not_open_yet = 1.0

    for p in daily_p:
        p_new = p * prob_not_open_yet
        prob_open.append(p_new)
        prob_not_open_yet *= (1 - p)
        prob_open_cum.append(1 - prob_not_open_yet)

    return pd.DataFrame({
        "date": dates,
        "prob_open": prob_open,
        "prob_open_cum": prob_open_cum,
    })


def get_most_likely_opening_date(forecast_df: pd.DataFrame) -> tuple[date, float]:
    idx = forecast_df["prob_open"].idxmax()
    row = forecast_df.loc[idx]
    return row["date"], float(row["prob_open"])
