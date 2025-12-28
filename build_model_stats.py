"""
build_model_stats.py

Script OFFLINE pour construire les stats TGVmax à partir des snapshots.

Sortie dans ./precomputed :
- proba_global.csv
- proba_od.parquet
- proba_od.csv
- snapshot_today_od.csv
- stations.json
- metadata.json

✅ Version ML :
- entraînement d'un classifieur (XGBoost si dispo, sinon RandomForest)
- on calcule proba_global et proba_od via les PROBAS PRÉDITES (pas les fréquences)
- on garde strictement les mêmes fichiers et schémas de sortie

✅ Correction métier :
- 1 train ouvert => journée OD ouverte (cible = day_open)
"""

from __future__ import annotations

import os
import glob
import json
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


# Colonnes attendues
COL_DATE = "date"
COL_TRAIN_NO = "train_no"
COL_ENTITY = "entity"
COL_ORIGIN = "origine"
COL_DEST = "destination"
COL_DEP_TIME = "heure_depart"
COL_ARR_TIME = "heure_arrivee"
COL_OD_HAPPY = "od_happy_card"

SNAPSHOTS_DIR = Path("snapshots")
PRECOMPUTED_DIR = Path("precomputed")


# ---------------------
# Helpers snapshots
# ---------------------

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


# ---------------------
# Feature engineering (niveau train) (comme notebook)
# ---------------------

def build_enriched_df(trains_raw: pd.DataFrame, allowed_entities: list[str] | None = None) -> pd.DataFrame:
    """
    Construit un DF enrichi aligné notebook (niveau train).
    Si allowed_entities est fourni, on filtre strictement dessus.
    """
    df = trains_raw.copy()

    # Filtre entity (batch-friendly)
    if allowed_entities is not None:
        allowed = {e.strip() for e in allowed_entities if e and e.strip()}
        df[COL_ENTITY] = df[COL_ENTITY].astype(str)
        df = df[df[COL_ENTITY].isin(allowed)].copy()

    # Parse dates
    df[COL_DATE] = df[COL_DATE].astype(str)
    df[COL_DEP_TIME] = df.get(COL_DEP_TIME, "").astype(str)

    df["departure_date"] = pd.to_datetime(df[COL_DATE], errors="coerce").dt.date
    df["snapshot_date_only"] = pd.to_datetime(df["snapshot_date"], errors="coerce").dt.date

    # departure_datetime (robuste)
    dep_dt_str = df[COL_DATE].astype(str) + " " + df[COL_DEP_TIME].astype(str)
    df["departure_datetime"] = pd.to_datetime(dep_dt_str, errors="coerce", format=None)

    # delta_days
    df["delta_days"] = (df["departure_date"] - df["snapshot_date_only"]).apply(
        lambda d: d.days if pd.notna(d) else np.nan
    )
    df = df[df["delta_days"].notna()].copy()
    df["delta_days"] = df["delta_days"].astype(int)

    # filtre delta >=0
    df = df[df["delta_days"] >= 0].copy()

    # cible train-level
    df["tgvmax_available"] = df[COL_OD_HAPPY].astype(str).str.upper().eq("OUI").astype(int)

    # filtre fenêtre delta 0..60
    MIN_D, MAX_D = 0, 60
    df = df[(df["delta_days"] >= MIN_D) & (df["delta_days"] <= MAX_D)].copy()

    # filtre "has_ever_max" (train_no + departure_date)
    if COL_TRAIN_NO in df.columns:
        group_cols = [COL_TRAIN_NO, "departure_date"]
        has_ever_max = df.groupby(group_cols)["tgvmax_available"].transform("max")
        df = df[has_ever_max == 1].copy()

    # Features temporelles train-level (on les gardera uniquement pour filtrer la validité des dates)
    df["dep_hour"] = pd.to_datetime(df["departure_datetime"], errors="coerce").dt.hour
    df["dep_weekday"] = pd.to_datetime(df["departure_datetime"], errors="coerce").dt.weekday
    df["dep_month"] = pd.to_datetime(df["departure_datetime"], errors="coerce").dt.month
    df["is_weekend"] = df["dep_weekday"].isin([5, 6]).astype(int)

    # Nettoyage OD
    df[COL_ORIGIN] = df[COL_ORIGIN].astype(str).str.strip()
    df[COL_DEST] = df[COL_DEST].astype(str).str.strip()

    # On vire les lignes où on ne peut pas faire les features time
    df = df[df["dep_weekday"].notna() & df["dep_month"].notna()].copy()

    # Cast num
    df["dep_hour"] = pd.to_numeric(df["dep_hour"], errors="coerce")
    df["dep_weekday"] = df["dep_weekday"].astype(int)
    df["dep_month"] = df["dep_month"].astype(int)

    return df


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


# ---------------------
# ✅ Agrégation jour-OD : 1 train ouvert => jour ouvert
# ---------------------

def build_day_level_df(df_train_enriched: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège au niveau (snapshot_date, departure_date, entity, origine, destination, delta_days)
    avec day_open = max(tgvmax_available).

    On reconstruit des features "jour" pertinentes (pas dep_hour, etc.)
    """
    group_keys = ["snapshot_date", "departure_date", COL_ENTITY, COL_ORIGIN, COL_DEST, "delta_days"]

    day = (
        df_train_enriched
        .groupby(group_keys, as_index=False)["tgvmax_available"]
        .max()
        .rename(columns={"tgvmax_available": "day_open"})
    )

    # Features jour
    dep_dt = pd.to_datetime(day["departure_date"], errors="coerce")
    day["dep_weekday"] = dep_dt.dt.weekday
    day["dep_month"] = dep_dt.dt.month
    day["is_weekend"] = day["dep_weekday"].isin([5, 6]).astype(int)

    # casts
    day["dep_weekday"] = day["dep_weekday"].astype(int)
    day["dep_month"] = day["dep_month"].astype(int)
    day["is_weekend"] = day["is_weekend"].astype(int)
    day["day_open"] = day["day_open"].astype(int)

    return day


# ---------------------
# ML training (sur day_open)
# ---------------------

def train_classifier(df_day: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame]:
    """
    Entraîne sur la cible day_open (jour-OD).
    """
    target_col = "day_open"

    data_ml = df_day.copy()

    # on garde snapshot_date en colonne (utile si tu fais un split ailleurs),
    # mais on ne l'utilise pas dans le modèle.
    if "snapshot_date" not in data_ml.columns:
        raise ValueError("snapshot_date absent : attendu pour cohérence dataset.")

    X = data_ml.drop(columns=[target_col])
    y = data_ml[target_col].astype(int).values

    numeric_features = []
    categorical_features = []
    for col in X.columns:
        if col == "snapshot_date":
            continue
        if pd.api.types.is_numeric_dtype(X[col]):
            numeric_features.append(col)
        else:
            categorical_features.append(col)

    if "snapshot_date" in numeric_features:
        numeric_features.remove("snapshot_date")
    if "snapshot_date" in categorical_features:
        categorical_features.remove("snapshot_date")

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    model = None
    try:
        from xgboost import XGBClassifier  # type: ignore
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
        )
    except Exception:
        model = RandomForestClassifier(
            n_estimators=60,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        )

    clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    X_fit = X.drop(columns=["snapshot_date"]) if "snapshot_date" in X.columns else X
    clf.fit(X_fit, y)

    return clf, X_fit


def compute_probas_from_ml(df_day: pd.DataFrame, clf: Pipeline) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calcule proba_global et proba_od à partir des probabilités PRÉDITES (day-level).
    Sorties identiques (colonnes) à l'ancien système.
    """
    target_col = "day_open"

    X = df_day.drop(columns=[target_col])
    if "snapshot_date" in X.columns:
        X = X.drop(columns=["snapshot_date"])

    y_proba = clf.predict_proba(X)[:, 1]

    dfp = df_day.copy()
    dfp["proba_pred"] = y_proba

    proba_global = (
        dfp.groupby("delta_days")["proba_pred"]
        .mean()
        .reset_index()
        .rename(columns={"proba_pred": "proba_open"})
        .sort_values("delta_days")
    )

    proba_od = (
        dfp.groupby([COL_ORIGIN, COL_DEST, "delta_days"])["proba_pred"]
        .mean()
        .reset_index()
        .rename(columns={"proba_pred": "proba_open"})
        .sort_values([COL_ORIGIN, COL_DEST, "delta_days"])
    )

    return proba_global, proba_od


# ---------------------
# Snapshot today OD (inchangé)
# ---------------------

def build_snapshot_today_od(latest_snapshot_df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Tableau bandeau "dispo/pas dispo" sur le dernier snapshot.

    Sortie colonnes :
    - departure_date (YYYY-MM-DD)
    - origine
    - destination
    - is_open_today (0/1)
    """
    df_today = build_enriched_df(latest_snapshot_df_raw)

    snap_od = (
        df_today.groupby(["departure_date", COL_ORIGIN, COL_DEST])["tgvmax_available"]
        .max()
        .reset_index()
        .rename(columns={"tgvmax_available": "is_open_today"})
        .sort_values(["departure_date", COL_ORIGIN, COL_DEST])
    )

    snap_od["departure_date"] = snap_od["departure_date"].astype(str)
    snap_od["is_open_today"] = snap_od["is_open_today"].astype(int)
    snap_od = snap_od.rename(columns={COL_ORIGIN: "origine", COL_DEST: "destination"})
    return snap_od


def main():
    PRECOMPUTED_DIR.mkdir(exist_ok=True)

    # 1) Dernier snapshot dispo (bandeau)
    paths = list_snapshot_paths()
    latest_path, latest_date = get_latest_snapshot_path(paths)

    print("Chargement de TOUS les snapshots...")
    raw = load_all_snapshots()
    print(f"{len(raw):,} lignes brutes")

    print("Construction du DataFrame enrichi (train-level)...")
    df_train = build_enriched_df(raw)
    print(f"{len(df_train):,} lignes après filtrage / enrichissement (train-level)")

    print("Agrégation au niveau journée OD (day-level : 1 train => jour ouvert)...")
    df_day = build_day_level_df(df_train)
    print(f"{len(df_day):,} lignes (day-level)")

    print("Entraînement du modèle ML (day-level)...")
    clf, _ = train_classifier(df_day)

    print("Calcul des probabilités (probas PRÉDITES par ML, day-level)...")
    proba_global, proba_od = compute_probas_from_ml(df_day, clf)

    print("Extraction de la liste des gares...")
    stations = extract_stations(df_train)

    # Snapshot du jour
    print(f"Chargement du dernier snapshot : {os.path.basename(latest_path)}")
    raw_latest = load_snapshot(latest_path)
    snapshot_today_od = build_snapshot_today_od(raw_latest)
    print(f"{len(snapshot_today_od):,} lignes snapshot_today_od")

    print("Sauvegarde dans ./precomputed ...")

    # Proba globale
    proba_global.to_csv(PRECOMPUTED_DIR / "proba_global.csv", index=False)

    # Proba OD
    proba_od.to_parquet(PRECOMPUTED_DIR / "proba_od.parquet", index=False)

    proba_od_csv = proba_od.rename(columns={COL_ORIGIN: "origine", COL_DEST: "destination"})
    proba_od_csv.to_csv(PRECOMPUTED_DIR / "proba_od.csv", index=False)

    # Snapshot today OD
    snapshot_today_od.to_csv(PRECOMPUTED_DIR / "snapshot_today_od.csv", index=False)

    # Stations
    with open(PRECOMPUTED_DIR / "stations.json", "w", encoding="utf-8") as f:
        json.dump(stations, f, ensure_ascii=False, indent=2)

    # Metadata (mêmes clés)
    metadata = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "latest_snapshot_date": str(latest_date),
        "latest_snapshot_file": os.path.basename(latest_path),
        "n_rows_raw": int(len(raw)),
        "n_rows_enriched": int(len(df_train)),
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
