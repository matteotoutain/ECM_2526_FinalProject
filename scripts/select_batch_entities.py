# scripts/select_batch_entities.py
from __future__ import annotations

import hashlib
from pathlib import Path
from datetime import date
from typing import List, Set

import pandas as pd

SNAPSHOTS_DIR = Path("snapshots")
PRECOMPUTED_DIR = Path("precomputed")
ENTITIES_FILE = PRECOMPUTED_DIR / "entities.txt"
BATCH_FILE = PRECOMPUTED_DIR / "batch_entities.txt"

# Paramètres (tu peux aussi les passer en env si tu veux)
NB_BUCKETS = 20   # plus grand = moins d'entities/jour
MAX_PER_DAY = 4   # 3-4 comme tu veux


def _latest_snapshot_path() -> Path:
    paths = sorted(SNAPSHOTS_DIR.glob("tgvmax_*.csv"))
    if not paths:
        raise FileNotFoundError(f"Aucun snapshot trouvé dans {SNAPSHOTS_DIR}/ (tgvmax_*.csv)")
    return paths[-1]


def _parse_date_from_filename(p: Path) -> date:
    # tgvmax_YYYY-MM-DD.csv
    stem = p.stem
    s = stem.replace("tgvmax_", "").split("_")[0]
    return date.fromisoformat(s)


def _stable_hash_mod(s: str, mod: int) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h, 16) % mod


def _load_known_entities(latest_snapshot: Path) -> List[str]:
    known: Set[str] = set()

    if ENTITIES_FILE.exists():
        known |= {x.strip() for x in ENTITIES_FILE.read_text(encoding="utf-8").splitlines() if x.strip()}

    # On enrichit avec les entities du dernier snapshot (léger)
    cols = ["entity"]
    df = pd.read_csv(latest_snapshot, sep=";", usecols=cols, dtype=str)
    known |= set(df["entity"].dropna().astype(str).str.strip().unique().tolist())

    out = sorted(e for e in known if e)
    PRECOMPUTED_DIR.mkdir(parents=True, exist_ok=True)
    ENTITIES_FILE.write_text("\n".join(out) + "\n", encoding="utf-8")
    return out


def main() -> None:
    latest = _latest_snapshot_path()
    d = _parse_date_from_filename(latest)
    bucket_today = d.toordinal() % NB_BUCKETS

    entities = _load_known_entities(latest)
    selected = [e for e in entities if _stable_hash_mod(e, NB_BUCKETS) == bucket_today]
    selected = sorted(selected)[:MAX_PER_DAY]

    PRECOMPUTED_DIR.mkdir(parents=True, exist_ok=True)
    BATCH_FILE.write_text("\n".join(selected) + "\n", encoding="utf-8")

    print(f"latest_snapshot={latest.name} date={d.isoformat()} bucket={bucket_today}/{NB_BUCKETS}")
    print(f"selected_entities={len(selected)}")
    for e in selected:
        print(e)


if __name__ == "__main__":
    main()
