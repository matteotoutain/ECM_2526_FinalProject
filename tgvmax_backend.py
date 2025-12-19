"""
tgvmax_backend.py
Backend pour la prévision d'ouverture TGVmax à partir des snapshots journaliers.

Version ML (drop-in, mêmes entrées/sorties) :
- On conserve EXACTEMENT les mêmes fonctions publiques + mêmes retours
- On garde proba_by_delta / proba_by_od (compatibilité)
- On ajoute un vrai modèle supervisé (RandomForest) stocké dans model.trains.attrs
  -> utilisé en priorité pour estimer P(dispo | origine, destination, delta_days)
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
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


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

# Clés attrs pandas (pour stocker le modèle ML sans casser l'API)
_ATTR_ML_PIPELINE = "ml_pipeline"
_ATTR_ML_FEATURES_NUM = "ml_features_num"
_ATTR_ML_FEATURES_CAT = "ml_features_cat"


@dataclass
class TgvMaxModel:
    """
    Objet qui contient :
      - trains          : le DataFrame filtré
      - proba_by_delta  : proba globale vs delta_days
      - proba_by_od     : proba par (origine, destination, delta_days)
      - stations        : liste dédupliquée de gares (origine + destination)
    """
    trains: pd.DataFrame
    proba_by_delta: pd.Series
    proba_by_od: pd.Series
    stations: List[str]


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
      - garde uniquement les entités Nord/Sud (JCNORDSUD, JCSUDNORD, PAPROVENCE)
      - supprime les lignes avec delta_days négatif
      - ajoute les colonnes delta_days et tgvmax_available
    """
    df = trains_raw.copy()

    # Filtre sur entity (comme dans ton notebook)
    mask_entity = (
        df[COL_ENTITY].str.contains("JCNORDSUD", case=False, na=False)
        | df[COL_ENTITY].str.contains("JCSUDNORD", case=False, na=False)
        | df[COL_ENTITY].str.contains("PAPROVENCE", case=False, na=False)
    )
    df = df[mask_entity].copy()

    # Date de départ (à partir de la colonne "date" du CSV)
    df["departure_date"] = pd.to_datetime(df[COL_DATE], errors="coerce").dt.date
    df["snapshot_date_only"] = pd.to_datetime(df["snapshot_date"], errors="coerce").dt.date

    # delta_days = (date départ) - (date snapshot)
    df["delta_days"] = (df["departure_date"] - df["snapshot_date_only"]).apply(
        lambda d: d.days if pd.notna(d) else np.nan
    )

    # On garde seulement delta_days >= 0 (pas de snapshot après le départ)
    df = df[df["delta_days"].notna()].copy()
    df["delta_days"] = df["delta_days"].astype(int)
    df = df[df["delta_days"] >= 0].copy()

    # Variable binaire de dispo TGVmax
    df["tgvmax_available"] = df[COL_OD_HAPPY].astype(str).str.upper().eq("OUI")

    # Nettoyage minimal OD (évite les faux mismatchs)
    df[COL_ORIGIN] = df[COL_ORIGIN].astype(str).str.strip()
    df[COL_DEST] = df[COL_DEST].astype(str).str.strip()

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


def compute_proba_by_od(trains: pd.DataFrame) -> pd.Series:
    """
    Calcule la probabilité empirique de disponibilité TGVmax
    par (origine, destination, delta_days).
    Retourne une Series avec un MultiIndex (origine, destination, delta_days).
    """
    grouped = (
        trains
        .groupby([COL_ORIGIN, COL_DEST, "delta_days"])["tgvmax_available"]
        .mean()
    )
    grouped.sort_index(inplace=True)
    return grouped


def extract_stations(trains: pd.DataFrame) -> List[str]:
    """
    Extrait la liste dédupliquée des gares (Origine + Destination).
    """
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
    stations = sorted(stations)
    return stations


# =====================
# ML (supervisé) : entraînement + stockage dans attrs
# =====================

def _train_ml_pipeline(trains: pd.DataFrame) -> Optional[Pipeline]:
    """
    Entraîne un vrai modèle ML supervisé pour approximer :
        P(tgvmax_available | origine, destination, delta_days)

    Important :
    - On NE change pas les sorties publiques : on stocke le modèle dans trains.attrs.
    - On reste léger : RandomForest (inspiré notebook), one-hot sur OD.
    """
    needed_cols = {COL_ORIGIN, COL_DEST, "delta_days", "tgvmax_available"}
    if not needed_cols.issubset(trains.columns):
        return None

    df = trains[[COL_ORIGIN, COL_DEST, "delta_days", "tgvmax_available"]].dropna().copy()
    if df.empty:
        return None

    # On évite les modèles débiles si une seule classe
    y = df["tgvmax_available"].astype(int).values
    if len(np.unique(y)) < 2:
        return None

    X = df[[COL_ORIGIN, COL_DEST, "delta_days"]].copy()

    numeric_features = ["delta_days"]
    categorical_features = [COL_ORIGIN, COL_DEST]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    model = RandomForestClassifier(
        n_estimators=30,
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


# =====================
# Construction du modèle
# =====================

def build_model(data_dir: str) -> TgvMaxModel:
    """
    Pipeline complet :
      1. chargement de tous les snapshots tgvmax_*.csv
      2. filtrage + features (delta_days, tgvmax_available)
      3. calcul de la proba globale par delta_days
      4. calcul de la proba par (origine, destination, delta_days)
      5. extraction de la liste de gares
      6. entraînement d'un modèle ML (stocké dans trains.attrs) utilisé ensuite en priorité
    """
    trains_raw = load_all_snapshots(data_dir=data_dir, pattern=DEFAULT_PATTERN)
    trains = filter_trains_for_model(trains_raw)

    proba_by_delta = compute_global_proba_by_delta(trains)
    proba_by_od = compute_proba_by_od(trains)
    stations = extract_stations(trains)

    # Entraînement ML (drop-in)
    ml = _train_ml_pipeline(trains)
    if ml is not None:
        trains.attrs[_ATTR_ML_PIPELINE] = ml
        trains.attrs[_ATTR_ML_FEATURES_NUM] = ["delta_days"]
        trains.attrs[_ATTR_ML_FEATURES_CAT] = [COL_ORIGIN, COL_DEST]
    else:
        trains.attrs[_ATTR_ML_PIPELINE] = None
        trains.attrs[_ATTR_ML_FEATURES_NUM] = ["delta_days"]
        trains.attrs[_ATTR_ML_FEATURES_CAT] = [COL_ORIGIN, COL_DEST]

    return TgvMaxModel(
        trains=trains,
        proba_by_delta=proba_by_delta,
        proba_by_od=proba_by_od,
        stations=stations,
    )


# =====================
# Prise en compte de l'état "aujourd'hui"
# =====================

def get_today_availability_status(
    model: TgvMaxModel,
    departure_date: date,
    today: date,
    origin: Optional[str],
    destination: Optional[str],
) -> tuple[str, Optional[bool]]:
    """
    Regarde dans les snapshots si, pour ce trajet (origine, destination, departure_date),
    le snapshot du jour (today) indique une dispo TGVmax.

    Retourne :
      - status_today : "open_today", "closed_today", "no_data_today", "unknown_od"
      - is_open_today : True / False / None
    """
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

    is_open = bool(subset["tgvmax_available"].any())
    if is_open:
        return "open_today", True
    else:
        return "closed_today", False


# =====================
# Fonction de prévision
# =====================

def _predict_proba_ml(
    model: TgvMaxModel,
    delta: int,
    origin: Optional[str],
    destination: Optional[str],
) -> Optional[float]:
    """
    Renvoie P(dispo) via le modèle ML si dispo, sinon None.
    """
    pipe: Optional[Pipeline] = model.trains.attrs.get(_ATTR_ML_PIPELINE, None)
    if pipe is None:
        return None
    if origin is None or destination is None:
        return None

    X = pd.DataFrame([{
        COL_ORIGIN: str(origin).strip(),
        COL_DEST: str(destination).strip(),
        "delta_days": int(delta),
    }])

    try:
        proba = float(pipe.predict_proba(X)[:, 1][0])
        return max(0.0, min(1.0, proba))
    except Exception:
        return None


def _get_daily_probability(
    model: TgvMaxModel,
    delta: int,
    origin: Optional[str],
    destination: Optional[str],
) -> float:
    """
    Récupère la proba pour un (delta_days) donné :
      0) d'abord via ML supervisé (si entraîné)
      1) sinon spécifique au trajet (origine, destination, delta) empirique
      2) sinon fallback sur la proba globale proba_by_delta
      3) sinon 0.0
    """
    # 0) ML
    p_ml = _predict_proba_ml(model=model, delta=delta, origin=origin, destination=destination)
    if p_ml is not None:
        return p_ml

    p = None

    # 1) tentative par trajet (stats)
    if origin is not None and destination is not None:
        try:
            p = float(model.proba_by_od.loc[(origin, destination, delta)])
        except KeyError:
            p = None

    # 2) fallback global (stats)
    if p is None:
        p = float(model.proba_by_delta.get(delta, 0.0))

    # 3) clamp
    p = max(0.0, min(1.0, p))
    return p


def forecast_opening_curve(
    model: TgvMaxModel,
    departure_date: date,
    today: Optional[date] = None,
    origin: Optional[str] = None,
    destination: Optional[str] = None,
) -> pd.DataFrame:
    """
    Construit la courbe de probabilité d'ouverture entre (today) et (departure_date - 1).

    Étapes :
      1. Si possible, on regarde l'état actuel du trajet dans le snapshot d'aujourd'hui :
         - s'il est déjà ouvert => on renvoie une "courbe" dégénérée avec proba=1 aujourd'hui.
         - s'il est fermé => on continue avec les stats historiques / ML.
         - s'il n'y a pas de données => on continue aussi avec les stats historiques / ML.
      2. Utilise si possible le ML (OD + delta_days),
         sinon la proba par trajet, sinon la proba globale.

    Retourne un DataFrame avec colonnes :
      - date             : date calendaire
      - prob_open        : proba approx. que la résa "s'ouvre" ce jour-là
      - prob_open_cum    : proba qu'elle soit déjà ouverte à cette date
      - status_today     : état détecté ("open_today", "closed_today", "no_data_today", "unknown_od")
      - open_today       : True / False / None
    """
    if today is None:
        today = date.today()

    if departure_date <= today:
        raise ValueError("La date de départ doit être dans le futur.")

    # 1) État actuel dans le snapshot du jour (si OD fourni)
    status_today, is_open_today = get_today_availability_status(
        model=model,
        departure_date=departure_date,
        today=today,
        origin=origin,
        destination=destination,
    )

    # Si déjà ouvert aujourd'hui, on ne fait pas de prévision : proba=1, cum=1
    if is_open_today is True:
        forecast_df = pd.DataFrame(
            {
                "date": [today],
                "prob_open": [1.0],
                "prob_open_cum": [1.0],
                "status_today": [status_today],
                "open_today": [True],
            }
        )
        return forecast_df

    # 2) Sinon, calcul standard de la courbe à partir de today
    max_delta_known = int(model.proba_by_delta.index.max()) if len(model.proba_by_delta.index) else 0
    # Si le départ est très lointain, on commence la courbe à departure_date - max_delta_known.
    start_date = max(today, departure_date - timedelta(days=max_delta_known))

    nb_days = (departure_date - start_date).days
    dates = [start_date + timedelta(days=i) for i in range(nb_days)]

    # Proba "ouverte ce jour-là" (daily_p) en fonction de delta_days
    daily_p = []
    for d in dates:
        delta = (departure_date - d).days
        p = _get_daily_probability(
            model=model,
            delta=delta,
            origin=origin,
            destination=destination,
        )
        daily_p.append(p)

    # Interprétation "événement d'ouverture" + cumul
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

    # Tag état détecté aujourd'hui
    forecast_df["status_today"] = status_today
    forecast_df["open_today"] = is_open_today

    return forecast_df


def get_most_likely_opening_date(forecast_df: pd.DataFrame) -> tuple[date, float]:
    """
    Renvoie (date_ouverte_max, probabilité_ce_jour_là) à partir d'une courbe de forecast.
    Si le trajet est déjà ouvert aujourd'hui et que la courbe ne contient qu'un point,
    la date retournée sera la date du jour avec probabilité 1.0.
    """
    idx = forecast_df["prob_open"].idxmax()
    row = forecast_df.loc[idx]
    return row["date"], float(row["prob_open"])
