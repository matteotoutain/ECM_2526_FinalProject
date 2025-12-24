# scripts/init_legacy_by_entity.py
from __future__ import annotations

from pathlib import Path
import shutil

PRECOMPUTED_DIR = Path("precomputed")
BY_ENTITY_DIR = PRECOMPUTED_DIR / "proba_od" / "by_entity"

def main() -> None:
    BY_ENTITY_DIR.mkdir(parents=True, exist_ok=True)
    src = PRECOMPUTED_DIR / "proba_od.csv"
    dst = BY_ENTITY_DIR / "__LEGACY__.csv"

    if dst.exists():
        print("LEGACY already exists, skip.")
        return

    if not src.exists():
        print("No precomputed/proba_od.csv found, skip legacy init.")
        return

    shutil.copyfile(src, dst)
    print("Initialized LEGACY:", dst.as_posix())

if __name__ == "__main__":
    main()
