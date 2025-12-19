"""
build_model_stats.py

Script OFFLINE pour construire les stats TGVmax à partir des snapshots :
- Probabilité globale par delta_days
- Probabilité par (origine, destination, delta_days)
- Liste des gares
- Snapshot "du jour" (dernier snapshot) agrégé par (date de départ, OD) pour afficher "dispo/pas dispo"

Sortie dans ./precomputed :
- proba_global.csv
- proba_od.parquet
- proba_od.csv
- snapshot_today_od.csv
- stations.json
- metadata.json

✅ Modif (compat ML / cohérence backend) :
- On réutilise les mêmes fonctions de préparation/features que tgvmax_backend.py
  (filter_trains_for_model + extract_stations) pour garantir que les stats offline
  sont alignées avec le nouveau backend ML.
- AUCUNE sortie existante n’est changée (mêmes fichiers, mêmes colonnes).
"""

from __future__ import annotations

import os
import glob
import json
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd

# ---- IMPORTANT: on s'aligne sur le backend (mêmes filtres / features)
# Le fichier tgvmax_backend.py doit être présent dans le repo (même dossier ou importable).
from tgvmax_backend import (
    filter_trains_for_model,
    extract_stations as backend_extract_stations,
)

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


def list_snapshot_paths() -> list[str]:
    pattern = str(SNAPSHOTS_DIR / "tgvmax_*.csv")
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(
            f"Aucun fichier trouvé dans {SNAPSHOTS_DIR}/ avec pattern tgvmax_*.csv"
        )
    return paths


def get_latest_snapshot_path(paths: list[str]) -> tuple[str, date]:
    dated = []
    for p in paths:
        d = parse_snapshot_date_from_filename(p)
        if d is not None:
            dated.append((p, d))
    if not dated:
        raise RuntimeError("Aucune date de snapshot parsable (noms de fichiers invalides).")
    dated.sort(key=lambda x: x[1])
    return dated[-1][0], dated[-1][1]


def load_snapshot(path: str) -> pd.DataFrame:
    snap_date = parse_snapshot_date_from_filename(path)
    if snap_date is None:
        raise ValueError(f"Impossible de parser la date depuis {path}")
    tmp = pd.read_csv(path, sep=";", dtype=str)
    tmp["snapshot_date"] = pd.to_datetime(snap_date)
    return tmp


def load_all_snapshots() -> pd.DataFrame:
    paths = list_snapshot_paths()

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
    """
    IMPORTANT : on délègue au backend (filter_trains_for_model)
    pour être strictement cohérent avec la version ML du projet.
    """
    return filter_trains_for_model(trains_raw)


def compute_probas(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    """
    Aligné backend (strip, drop blanks, etc.)
    """
    return backend_extract_stations(df)


def build_snapshot_today_od(latest_snapshot_df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Construit un tableau pour le bandeau "dispo / pas dispo" sur le dernier snapshot disponible.

    IMPORTANT : on agrège par (departure_date, origin, destination) car "dispo"
    dépend de la date de départ.

    Sortie colonnes :
    - departure_date (YYYY-MM-DD)
    - origin
    - destination
    - is_open_today (0/1)
    """
    df_today = build_enriched_df(latest_snapshot_df_raw)

    snap_od = (
        df_today.groupby(["departure_date", COL_ORIGIN, COL_DEST])["tgvmax_available"]
        .max()  # dispo si au moins une ligne OD est "OUI"
        .reset_index()
        .rename(columns={"tgvmax_available": "is_open_today"})
        .sort_values(["departure_date", COL_ORIGIN, COL_DEST])
    )

    # Export-friendly (inchangé)
    snap_od["departure_date"] = snap_od["departure_date"].astype(str)
    snap_od["is_open_today"] = snap_od["is_open_today"].astype(int)
    snap_od = snap_od.rename(columns={COL_ORIGIN: "origin", COL_DEST: "destination"})
    return snap_od


def main():
    PRECOMPUTED_DIR.mkdir(exist_ok=True)

    # 1) Détermine le dernier snapshot dispo (pour le bandeau "dispo")
    paths = list_snapshot_paths()
    latest_path, latest_date = get_latest_snapshot_path(paths)

    print("Chargement de TOUS les snapshots...")
    raw = load_all_snapshots()
    print(f"{len(raw):,} lignes brutes")

    print("Construction du DataFrame enrichi (aligné tgvmax_backend ML)...")
    df = build_enriched_df(raw)
    print(f"{len(df):,} lignes après filtrage / enrichissement")

    print("Calcul des probabilités (stats offline, inchangées)...")
    proba_global, proba_od = compute_probas(df)

    print("Extraction de la liste des gares...")
    stations = extract_stations(df)

    # 2) Snapshot du jour (dernier snapshot)
    print(f"Chargement du dernier snapshot : {os.path.basename(latest_path)}")
    raw_latest = load_snapshot(latest_path)
    snapshot_today_od = build_snapshot_today_od(raw_latest)
    print(f"{len(snapshot_today_od):,} lignes snapshot_today_od")

    print("Sauvegarde dans ./precomputed ...")

    # Proba globale
    proba_global.to_csv(PRECOMPUTED_DIR / "proba_global.csv", index=False)

    # Proba OD : parquet (comme avant)
    proba_od.to_parquet(PRECOMPUTED_DIR / "proba_od.parquet", index=False)

    # Proba OD en CSV (pour GitHub Pages)
    proba_od_csv = proba_od.rename(columns={COL_ORIGIN: "origin", COL_DEST: "destination"})
    proba_od_csv.to_csv(PRECOMPUTED_DIR / "proba_od.csv", index=False)

    # Snapshot du jour agrégé OD + departure_date
    snapshot_today_od.to_csv(PRECOMPUTED_DIR / "snapshot_today_od.csv", index=False)

    # Stations
    with open(PRECOMPUTED_DIR / "stations.json", "w", encoding="utf-8") as f:
        json.dump(stations, f, ensure_ascii=False, indent=2)

    # Metadata (inchangé)
    metadata = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "latest_snapshot_date": str(latest_date),
        "latest_snapshot_file": os.path.basename(latest_path),
        "n_rows_raw": int(len(raw)),
        "n_rows_enriched": int(len(df)),
        "n_stations": int(len(stations)),
        "n_rows_proba_global": int(len(proba_global)),
        "n_rows_proba_od": int(len(proba_od)),
        "n_rows_snapshot_today_od": int(len(snapshot_today_od)),
    }
    with open(PRECOMPUTED_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("OK ✅")


if __name__ == "__main__":
    main()
