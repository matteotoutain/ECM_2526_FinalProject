from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PRECOMPUTED_DIR = Path("precomputed")
BY_ENTITY_DIR = PRECOMPUTED_DIR / "first_signal_od" / "by_entity"
META_PATH = PRECOMPUTED_DIR / "metadata.json"


def main() -> None:
    BY_ENTITY_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(BY_ENTITY_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError("No by_entity CSV files found to aggregate for first_signal_od.")

    # même logique que proba_od : LEGACY d'abord si un jour tu en ajoutes
    files_sorted = sorted(files, key=lambda p: (p.name != "__LEGACY__.csv", p.name))

    frames = []
    for p in files_sorted:
        df = pd.read_csv(p)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)

    # sécurité doublons OD -> garder le dernier
    out = out.drop_duplicates(["origine", "destination"], keep="last")
    out = out.sort_values(["origine", "destination"])

    csv_path = PRECOMPUTED_DIR / "first_signal_od.csv"
    out.to_csv(csv_path, index=False)

    # metadata update
    meta = {}
    if META_PATH.exists():
        try:
            meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

    meta.update(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "n_rows_first_signal_od": int(len(out)),
        }
    )
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"OK: aggregated {len(out)} rows -> {csv_path.name}")


if __name__ == "__main__":
    main()
