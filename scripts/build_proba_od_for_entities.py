# scripts/build_proba_od_for_entities.py
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

import build_model_stats as bms  # ton script existant

SNAPSHOTS_DIR = Path("snapshots")
PRECOMPUTED_DIR = Path("precomputed")
BY_ENTITY_DIR = PRECOMPUTED_DIR / "proba_od" / "by_entity"
BATCH_FILE = PRECOMPUTED_DIR / "batch_entities.txt"

# perf: adapte si besoin
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

        for chunk in pd.read_csv(p, sep=";", dtype=str, usecols=NEEDED_COLS, chunksize=CHUNKSIZE):
            # filtre entity au plus tôt
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
    entities = _read_batch_entities()
    if not entities:
        print("No entities selected today (batch_entities.txt empty).")
        return

    BY_ENTITY_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Charger uniquement les lignes des entities du batch
    raw = _load_all_snapshots_filtered(entities)
    if raw.empty:
        print("No data found for selected entities in snapshots.")
        return

    # 2) Pour chaque entity: enrich + train + proba_od + write
    for ent in entities:
        df_ent = raw[raw[bms.COL_ENTITY] == ent].copy()
        if df_ent.empty:
            continue

        df_enriched = bms.build_enriched_df(df_ent, allowed_entities=[ent])
        if df_enriched.empty:
            print(f"[{ent}] enriched empty -> skip")
            continue

        clf = bms.train_classifier(df_enriched)
        _proba_global, proba_od = bms.compute_probas_from_ml(df_enriched, clf)

        out_path = BY_ENTITY_DIR / f"{ent}.csv"
        proba_od.to_csv(out_path, index=False)
        print(f"[{ent}] wrote {len(proba_od)} rows -> {out_path.as_posix()}")


if __name__ == "__main__":
    main()
