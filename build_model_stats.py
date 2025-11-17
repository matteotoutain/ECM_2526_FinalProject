"""
build_model_stats.py

Script OFFLINE pour construire les stats TGVmax à partir des snapshots :
- Probabilité globale par delta_days
- Probabilité par (origine, destination, delta_days)
- Liste des gares

Sortie dans ./precomputed :
- proba_global.csv
- proba_od.parquet
- stations.json
- metadata.json
"""

from __future__ import annotations

import os
import glob
import json
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd

# Colonnes attendues
COL_DATE = "date"
COL_ENTITY = "entity"
COL_ORIGIN = "origine"
COL_DEST = "destination"
COL_OD_HAPPY = "od_happy_card"

SNAPSHOTS_DIR = Path("snapshots")
PRECOMPUTED_DIR = Path("precomputed")


def parse_snapshot_date_from_filename(path: str) -> date | None:
    base = os.path.basename(path)
    if base.startswith("tgvmax_"):
        base = base[len("tgvmax_") :]
    if base.endswith(".csv"):
        base = base[:-4]
    base = base.split("_")[0]
    try:
        return datetime.strptime(base, "%Y-%m-%d").date()
    except ValueError:
        return None


def load_all_snapshots() -> pd.DataFrame:
    pattern = str(SNAPSHOTS_DIR / "tgvmax_*.csv")
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"Aucun fichier trouvé dans {SNAPSHOTS_DIR}/ avec pattern tgvmax_*.csv")

    dfs = []
    for path in paths:
        snap_date = parse_snapshot_date_from_filename(path)
        if snap_date is None:
            print(f"[WARN] Impossible de parser la date depuis {path}, ignoré.")
            continue
        tmp = pd.read_csv(path, sep=";", dtype=str)
        tmp["snapshot_date"] = pd.to_datetime(snap_date)
        dfs.append(tmp)

    if not dfs:
        raise RuntimeError("Tous les fichiers ont été ignorés, vérifier nom et format.")
    return pd.concat(dfs, ignore_index=True)


def build_enriched_df(trains_raw: pd.DataFrame) -> pd.DataFrame:
    df = trains_raw.copy()

    # Filtre entités Nord/Sud
    mask_entity = (
        df[COL_ENTITY].str.contains("JCNORDSUD", case=False, na=False)
        | df[COL_ENTITY].str.contains("JCSUDNORD", case=False, na=False)
        | df[COL_ENTITY].str.contains("PAPROVENCE", case=False, na=False)
    )
    df = df[mask_entity].copy()

    df["departure_date"] = pd.to_datetime(df[COL_DATE]).dt.date
    df["snapshot_date_only"] = pd.to_datetime(df["snapshot_date"]).dt.date
    df["delta_days"] = (df["departure_date"] - df["snapshot_date_only"]).apply(lambda d: d.days)

    df = df[df["delta_days"] >= 0].copy()
    df["tgvmax_available"] = df[COL_OD_HAPPY].str.upper().eq("OUI")
    return df


def compute_probas(df: pd.DataFrame):
    proba_global = (
        df.groupby("delta_days")["tgvmax_available"]
        .mean()
        .reset_index()
        .rename(columns={"tgvmax_available": "proba_open"})
        .sort_values("delta_days")
    )

    proba_od = (
        df.groupby([COL_ORIGIN, COL_DEST, "delta_days"])["tgvmax_available"]
        .mean()
        .reset_index()
        .rename(columns={"tgvmax_available": "proba_open"})
        .sort_values([COL_ORIGIN, COL_DEST, "delta_days"])
    )

    return proba_global, proba_od


def extract_stations(df: pd.DataFrame) -> list[str]:
    s = pd.concat([df[COL_ORIGIN], df[COL_DEST]], axis=0)
    stations = (
        s.dropna()
        .astype(str)
        .str.strip()
        .replace({"": None})
        .dropna()
        .unique()
        .tolist()
    )
    return sorted(stations)


def main():
    PRECOMPUTED_DIR.mkdir(exist_ok=True)

    print("Chargement des snapshots...")
    raw = load_all_snapshots()
    print(f"{len(raw):,} lignes brutes")

    print("Construction du DataFrame enrichi...")
    df = build_enriched_df(raw)
    print(f"{len(df):,} lignes après filtrage / enrichissement")

    print("Calcul des probabilités...")
    proba_global, proba_od = compute_probas(df)

    print("Extraction de la liste des gares...")
    stations = extract_stations(df)

    print("Sauvegarde dans ./precomputed ...")
    proba_global.to_csv(PRECOMPUTED_DIR / "proba_global.csv", index=False)
    proba_od.to_parquet(PRECOMPUTED_DIR / "proba_od.parquet", index=False)

    with open(PRECOMPUTED_DIR / "stations.json", "w", encoding="utf-8") as f:
        json.dump(stations, f, ensure_ascii=False, indent=2)

    metadata = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "n_rows_raw": int(len(raw)),
        "n_rows_enriched": int(len(df)),
        "n_stations": int(len(stations)),
        "n_rows_proba_global": int(len(proba_global)),
        "n_rows_proba_od": int(len(proba_od)),
    }
    with open(PRECOMPUTED_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("OK ✅")


if __name__ == "__main__":
    main()
