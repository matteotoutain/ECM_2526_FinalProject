from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# Ajoute la racine du repo au PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import build_model_stats as bms  # pour parse_snapshot_date_from_filename + noms de colonnes

SNAPSHOTS_DIR = Path("snapshots")
PRECOMPUTED_DIR = Path("precomputed")
BY_ENTITY_DIR = PRECOMPUTED_DIR / "first_signal_od" / "by_entity"
BATCH_FILE = PRECOMPUTED_DIR / "batch_entities.txt"

CHUNKSIZE = 400_000

NEEDED_COLS = [
    bms.COL_DATE,
    bms.COL_TRAIN_NO,
    bms.COL_ENTITY,
    bms.COL_ORIGIN,
    bms.COL_DEST,
    bms.COL_DEP_TIME,
    bms.COL_ARR_TIME,
    bms.COL_OD_HAPPY,
]


def _read_batch_entities() -> List[str]:
    if not BATCH_FILE.exists():
        return []
    return [x.strip() for x in BATCH_FILE.read_text(encoding="utf-8").splitlines() if x.strip()]


def _iter_snapshots() -> List[Path]:
    paths = sorted(SNAPSHOTS_DIR.glob("tgvmax_*.csv"))
    if not paths:
        raise FileNotFoundError(f"Aucun snapshot trouvé dans {SNAPSHOTS_DIR}/")
    return paths


def _load_all_snapshots_filtered(allowed_entities: List[str]) -> pd.DataFrame:
    allowed = set(allowed_entities)
    frames: List[pd.DataFrame] = []

    for p in _iter_snapshots():
        snap_date = bms.parse_snapshot_date_from_filename(str(p))
        if snap_date is None:
            continue

        for chunk in pd.read_csv(
            p, sep=";", dtype=str, usecols=NEEDED_COLS, chunksize=CHUNKSIZE
        ):
            chunk[bms.COL_ENTITY] = chunk[bms.COL_ENTITY].astype(str)
            chunk = chunk[chunk[bms.COL_ENTITY].isin(allowed)]
            if chunk.empty:
                continue

            chunk["snapshot_date"] = pd.to_datetime(snap_date)
            frames.append(chunk)

    if not frames:
        return pd.DataFrame(columns=NEEDED_COLS + ["snapshot_date"])
    return pd.concat(frames, ignore_index=True)


def _compute_first_signal_od_from_enriched(df_enriched: pd.DataFrame) -> pd.DataFrame:
    """
    df_enriched attendu au niveau train (avec delta_days, departure_date, snapshot_date, tgvmax_available)
    Retour:
      origine, destination,
      first_open_delta_median, first_open_delta_p25, first_open_delta_p75,
      n_departure_dates
    """
    if df_enriched.empty:
        return pd.DataFrame(columns=[
            "origine", "destination",
            "first_open_delta_median", "first_open_delta_p25", "first_open_delta_p75",
            "n_departure_dates"
        ])

    # 1 train ouvert => journée OD ouverte (au niveau snapshot_date + departure_date + OD + delta)
    group_keys = ["snapshot_date", "departure_date", bms.COL_ORIGIN, bms.COL_DEST, "delta_days"]
    day = (
        df_enriched
        .groupby(group_keys, as_index=False)["tgvmax_available"]
        .max()
        .rename(columns={"tgvmax_available": "day_open"})
    )

    opened = day[day["day_open"] == 1].copy()
    if opened.empty:
        return pd.DataFrame(columns=[
            "origine", "destination",
            "first_open_delta_median", "first_open_delta_p25", "first_open_delta_p75",
            "n_departure_dates"
        ])

    # 1er signal pour chaque (departure_date, OD) = dispo la plus "tôt" => delta_days max
    first_per_departure = (
        opened
        .groupby(["departure_date", bms.COL_ORIGIN, bms.COL_DEST], as_index=False)["delta_days"]
        .max()
        .rename(columns={"delta_days": "first_open_delta"})
    )

    def q25(x): return float(np.quantile(x, 0.25))
    def q75(x): return float(np.quantile(x, 0.75))

    out = (
        first_per_departure
        .groupby([bms.COL_ORIGIN, bms.COL_DEST])["first_open_delta"]
        .agg(
            first_open_delta_median="median",
            first_open_delta_p25=q25,
            first_open_delta_p75=q75,
            n_departure_dates="count",
        )
        .reset_index()
        .rename(columns={bms.COL_ORIGIN: "origine", bms.COL_DEST: "destination"})
    )

    out["first_open_delta_median"] = out["first_open_delta_median"].round(0).astype(int)
    out["first_open_delta_p25"] = out["first_open_delta_p25"].round(0).astype(int)
    out["first_open_delta_p75"] = out["first_open_delta_p75"].round(0).astype(int)
    out["n_departure_dates"] = out["n_departure_dates"].astype(int)

    return out.sort_values(["origine", "destination"]).reset_index(drop=True)


def main() -> None:
    entities = _read_batch_entities()
    if not entities:
        print("No entities selected today (batch_entities.txt empty).")
        return

    BY_ENTITY_DIR.mkdir(parents=True, exist_ok=True)

    raw = _load_all_snapshots_filtered(entities)
    if raw.empty:
        print("No data found for selected entities in snapshots.")
        return

    for ent in entities:
        df_ent = raw[raw[bms.COL_ENTITY] == ent].copy()
        if df_ent.empty:
            continue

        df_enriched = bms.build_enriched_df(df_ent, allowed_entities=[ent])
        if df_enriched.empty:
            print(f"[{ent}] enriched empty -> skip")
            continue

        out = _compute_first_signal_od_from_enriched(df_enriched)

        out_path = BY_ENTITY_DIR / f"{ent}.csv"
        out.to_csv(out_path, index=False)
        print(f"[{ent}] wrote {len(out)} rows -> {out_path.as_posix()}")


if __name__ == "__main__":
    main()
