"""
tgvmax_backend.py
Backend pour la prévision d'ouverture TGVmax à partir des snapshots journaliers.

✅ Version ML (alignée notebook + alignée build_model_stats.py):
- build_model entraîne un modèle (XGBoost si dispo, sinon RandomForest)
- puis construit proba_by_delta et proba_by_od via PROBAS PRÉDITES (pas fréquences)
- le reste (get_today_availability_status / forecast_opening_curve) ne change pas
"""

from __future__ import annotations

import os
import glob
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional, List

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


# =====================
# Configuration globale
# =====================

COL_DATE = "date"
COL_TRAIN_NO = "train_no"
COL_ENTITY = "entity"
COL_ORIGIN = "origine"
COL_DEST = "destination"
COL_DEP_TIME = "heure_depart"
COL_ARR_TIME = "heure_arrivee"
COL_OD_HAPPY = "od_happy_card"

DEFAULT_PATTERN = "tgvmax_*.csv"


@dataclass
class TgvMaxModel:
    trains: pd.DataFrame
    proba_by_delta: pd.Series
    proba_by_od: pd.Series
    stations: List[str]


# =====================
# Utilitaires fichiers
# =====================

def parse_snapshot_date_from_filename(path: str) -> Optional[date]:
    filename = os.path.basename(path)
    base = filename
    if base.startswith("tgvmax_"):
        base = base[len("tgvmax_") :]
    if base.endswith(".csv"):
        base = base[: -4]
    base = base.split("_")[0]
    try:
        return datetime.strptime(base, "%Y-%m-%d").date()
    except ValueError:
        return None


def load_all_snapshots(data_dir: str, pattern: str = DEFAULT_PATTERN) -> pd.DataFrame:
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

    return pd.concat(df_list, ignore_index=True)


# =====================
# Enrichissement (aligné notebook)
# =====================

def filter_trains_for_model(trains_raw: pd.DataFrame) -> pd.DataFrame:
    df = trains_raw.copy()

    # Filtre entity Nord/Sud
    mask_entity = (
        df[COL_ENTITY].str.contains("JCNORDSUD", case=False, na=False)
        | df[COL_ENTITY].str.contains("JCSUDNORD", case=False, na=False)
        | df[COL_ENTITY].str.contains("PAPROVENCE", case=False, na=False)
    )
    df = df[mask_entity].copy()

    # Parse dates/heures
    df[COL_DATE] = df[COL_DATE].astype(str)
    df[COL_DEP_TIME] = df.get(COL_DEP_TIME, "").astype(str)

    df["departure_date"] = pd.to_datetime(df[COL_DATE], errors="coerce").dt.date
    df["snapshot_date_only"] = pd.to_datetime(df["snapshot_date"], errors="coerce").dt.date

    dep_dt_str = df[COL_DATE].astype(str) + " " + df[COL_DEP_TIME].astype(str)
    df["departure_datetime"] = pd.to_datetime(dep_dt_str, errors="coerce", format=None)

    # delta
    df["delta_days"] = (df["departure_date"] - df["snapshot_date_only"]).apply(
        lambda d: d.days if pd.notna(d) else np.nan
    )
    df = df[df["delta_days"].notna()].copy()
    df["delta_days"] = df["delta_days"].astype(int)
    df = df[df["delta_days"] >= 0].copy()

    # cible binaire
    df["tgvmax_available"] = df[COL_OD_HAPPY].astype(str).str.upper().eq("OUI").astype(int)

    # fenêtre delta 0..60
    df = df[(df["delta_days"] >= 0) & (df["delta_days"] <= 60)].copy()

    # filtre has_ever_max
    if COL_TRAIN_NO in df.columns:
        group_cols = [COL_TRAIN_NO, "departure_date"]
        has_ever_max = df.groupby(group_cols)["tgvmax_available"].transform("max")
        df = df[has_ever_max == 1].copy()

    # features time
    df["dep_hour"] = pd.to_datetime(df["departure_datetime"], errors="coerce").dt.hour
    df["dep_weekday"] = pd.to_datetime(df["departure_datetime"], errors="coerce").dt.weekday
    df["dep_month"] = pd.to_datetime(df["departure_datetime"], errors="coerce").dt.month
    df["is_weekend"] = df["dep_weekday"].isin([5, 6]).astype(int)

    df[COL_ORIGIN] = df[COL_ORIGIN].astype(str).str.strip()
    df[COL_DEST] = df[COL_DEST].astype(str).str.strip()

    df = df[df["dep_hour"].notna() & df["dep_weekday"].notna() & df["dep_month"].notna()].copy()
    df["dep_hour"] = df["dep_hour"].astype(int)
    df["dep_weekday"] = df["dep_weekday"].astype(int)
    df["dep_month"] = df["dep_month"].astype(int)

    return df


def extract_stations(trains: pd.DataFrame) -> List[str]:
    series_stations = pd.concat([trains[COL_ORIGIN], trains[COL_DEST]], axis=0)
    stations = (
        series_stations.dropna()
        .astype(str)
        .str.strip()
        .replace({"": None})
        .dropna()
        .unique()
        .tolist()
    )
    return sorted(stations)


# =====================
# ML : training + proba tables (aligné offline)
# =====================

def _train_classifier(trains: pd.DataFrame) -> Pipeline:
    target_col = "tgvmax_available"

    drop_cols = [
        "departure_date",
        "departure_datetime",
        "arrival_datetime",
        COL_DEP_TIME,
        COL_ARR_TIME,
        COL_OD_HAPPY,
    ]

    data_ml = trains.copy()
    for c in drop_cols:
        if c in data_ml.columns:
            data_ml = data_ml.drop(columns=c)

    if "snapshot_date" not in data_ml.columns:
        raise ValueError("snapshot_date absent : attendu.")

    X = data_ml.drop(columns=[target_col])
    y = data_ml[target_col].astype(int).values

    # retirer snapshot_date du modèle
    if "snapshot_date" in X.columns:
        X = X.drop(columns=["snapshot_date"])

    numeric_features = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_features = [c for c in X.columns if c not in numeric_features]

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

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    clf.fit(X, y)
    return clf


def _compute_proba_tables_from_ml(trains: pd.DataFrame, clf: Pipeline) -> tuple[pd.Series, pd.Series]:
    """
    Retourne :
    - proba_by_delta : Series index=delta_days
    - proba_by_od    : Series MultiIndex (origine, destination, delta_days)
    """
    target_col = "tgvmax_available"

    drop_cols = [
        "departure_date",
        "departure_datetime",
        "arrival_datetime",
        COL_DEP_TIME,
        COL_ARR_TIME,
        COL_OD_HAPPY,
    ]

    data_ml = trains.copy()
    for c in drop_cols:
        if c in data_ml.columns:
            data_ml = data_ml.drop(columns=c)

    X = data_ml.drop(columns=[target_col])
    if "snapshot_date" in X.columns:
        X = X.drop(columns=["snapshot_date"])

    proba = clf.predict_proba(X)[:, 1]
    dfp = trains.copy()
    dfp["_proba_pred"] = proba

    proba_by_delta = (
        dfp.groupby("delta_days")["_proba_pred"]
        .mean()
        .sort_index()
    )

    proba_by_od = (
        dfp.groupby([COL_ORIGIN, COL_DEST, "delta_days"])["_proba_pred"]
        .mean()
    )
    proba_by_od.sort_index(inplace=True)

    return proba_by_delta, proba_by_od


# =====================
# Construction du modèle
# =====================

def build_model(data_dir: str) -> TgvMaxModel:
    trains_raw = load_all_snapshots(data_dir=data_dir, pattern=DEFAULT_PATTERN)
    trains = filter_trains_for_model(trains_raw)

    clf = _train_classifier(trains)
    proba_by_delta, proba_by_od = _compute_proba_tables_from_ml(trains, clf)
    stations = extract_stations(trains)

    return TgvMaxModel(
        trains=trains,
        proba_by_delta=proba_by_delta,
        proba_by_od=proba_by_od,
        stations=stations,
    )


# =====================
# État "aujourd'hui" (inchangé)
# =====================

def get_today_availability_status(
    model: TgvMaxModel,
    departure_date: date,
    today: date,
    origin: Optional[str],
    destination: Optional[str],
) -> tuple[str, Optional[bool]]:
    if origin is None or destination is None:
        return "unknown_od", None

    df = model.trains

    mask = (
        (df["departure_date"] == departure_date)
        & (df["snapshot_date_only"] == today)
        & (df[COL_ORIGIN] == origin)
        & (df[COL_DEST] == destination)
    )

    subset = df[mask]

    if subset.empty:
        return "no_data_today", None

    is_open = bool((subset["tgvmax_available"] == 1).any())
    return ("open_today", True) if is_open else ("closed_today", False)


# =====================
# Forecast (inchangé)
# =====================

def _get_daily_probability(
    model: TgvMaxModel,
    delta: int,
    origin: Optional[str],
    destination: Optional[str],
) -> float:
    p = None

    if origin is not None and destination is not None:
        try:
            p = float(model.proba_by_od.loc[(origin, destination, delta)])
        except KeyError:
            p = None

    if p is None:
        p = float(model.proba_by_delta.get(delta, 0.0))

    return max(0.0, min(1.0, p))


def forecast_opening_curve(
    model: TgvMaxModel,
    departure_date: date,
    today: Optional[date] = None,
    origin: Optional[str] = None,
    destination: Optional[str] = None,
) -> pd.DataFrame:
    if today is None:
        today = date.today()

    if departure_date <= today:
        raise ValueError("La date de départ doit être dans le futur.")

    status_today, is_open_today = get_today_availability_status(
        model=model,
        departure_date=departure_date,
        today=today,
        origin=origin,
        destination=destination,
    )

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

    max_delta_known = int(model.proba_by_delta.index.max()) if len(model.proba_by_delta.index) else 0
    start_date = max(today, departure_date - timedelta(days=max_delta_known))

    nb_days = (departure_date - start_date).days
    dates = [start_date + timedelta(days=i) for i in range(nb_days)]

    daily_p = []
    for d in dates:
        delta = (departure_date - d).days
        daily_p.append(_get_daily_probability(model, delta, origin, destination))

    prob_open = []
    prob_open_cum = []
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
    forecast_df["status_today"] = status_today
    forecast_df["open_today"] = is_open_today

    return forecast_df


def get_most_likely_opening_date(forecast_df: pd.DataFrame) -> tuple[date, float]:
    idx = forecast_df["prob_open"].idxmax()
    row = forecast_df.loc[idx]
    return row["date"], float(row["prob_open"])
