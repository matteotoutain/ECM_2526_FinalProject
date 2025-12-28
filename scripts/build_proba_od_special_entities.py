from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pandas as pd

# Ajoute la racine du repo au PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import build_model_stats as bms

SNAPSHOTS_DIR = Path("snapshots")
OUT_DIR = Path("precomputed") / "proba_od" / "special_entities"

# ⚠️ entities ciblées
SPECIAL_ENTITIES = ["PAPROVENCE", "PROVENCEPA"]

# perf
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

        for chunk in pd.read_csv(p, sep=";", dtype=str, usecols=NEEDED_COLS, chunksize=CHUNKSIZE):
            chunk[bms.COL_ENTITY] = chunk[bms.COL_ENTITY].astype(str)
            chunk = chunk[chunk[bms.COL_ENTITY].isin(allowed)]
            if chunk.empty:
                continue
            chunk["snapshot_date"] = pd.to_datetime(snap_date)
            frames.append(chunk)

    if not frames:
        return pd.DataFrame(columns=NEEDED_COLS + ["snapshot_date"])
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = _load_all_snapshots_filtered(SPECIAL_ENTITIES)
    if raw.empty:
        print("No data found for special entities in snapshots.")
        return

    for ent in SPECIAL_ENTITIES:
        df_ent = raw[raw[bms.COL_ENTITY] == ent].copy()
        if df_ent.empty:
            print(f"[{ent}] no rows -> skip")
            continue

        df_enriched = bms.build_enriched_df(df_ent, allowed_entities=[ent])
        if df_enriched.empty:
            print(f"[{ent}] enriched empty -> skip")
            continue

        clf, _ = bms.train_classifier(df_enriched)
        _, proba_od = bms.compute_probas_from_ml(df_enriched, clf)

        out_path = OUT_DIR / f"{ent}.csv"
        proba_od.to_csv(out_path, index=False)
        print(f"[{ent}] wrote {len(proba_od)} rows -> {out_path.as_posix()}")


if __name__ == "__main__":
    main()
