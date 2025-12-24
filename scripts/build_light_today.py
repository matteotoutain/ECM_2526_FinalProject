# scripts/build_light_today.py
from __future__ import annotations

import json
from datetime import datetime, timezone, date
from pathlib import Path

import pandas as pd

SNAPSHOTS_DIR = Path("snapshots")
PRECOMPUTED_DIR = Path("precomputed")

COL_DATE = "date"
COL_ORIGIN = "origin"
COL_DEST = "destination"


def latest_snapshot_path() -> Path:
    paths = sorted(SNAPSHOTS_DIR.glob("tgvmax_*.csv"))
    if not paths:
        raise FileNotFoundError(f"Aucun snapshot trouvé dans {SNAPSHOTS_DIR}/")
    return paths[-1]


def parse_date_from_filename(p: Path) -> date:
    s = p.stem.replace("tgvmax_", "").split("_")[0]
    return date.fromisoformat(s)


def build_snapshot_today_od(snapshot_path: Path) -> pd.DataFrame:
    # On agrège "ouvert aujourd'hui" par (departure_date, origin, destination)
    usecols = [COL_DATE, COL_ORIGIN, COL_DEST, "od_happy_card"]
    df = pd.read_csv(snapshot_path, sep=";", usecols=usecols, dtype=str)

    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce").dt.date
    df[COL_ORIGIN] = df[COL_ORIGIN].astype(str).str.strip()
    df[COL_DEST] = df[COL_DEST].astype(str).str.strip()

    df["is_open_today"] = df["od_happy_card"].astype(str).str.upper().eq("OUI").astype(int)

    out = (
        df.groupby([COL_DATE, COL_ORIGIN, COL_DEST], as_index=False)["is_open_today"]
        .max()
        .rename(columns={COL_DATE: "departure_date", COL_ORIGIN: "origin", COL_DEST: "destination"})
        .sort_values(["departure_date", "origin", "destination"])
    )
    return out


def extract_stations(snapshot_path: Path) -> list[str]:
    usecols = [COL_ORIGIN, COL_DEST]
    df = pd.read_csv(snapshot_path, sep=";", usecols=usecols, dtype=str)
    s = pd.concat([df[COL_ORIGIN], df[COL_DEST]], ignore_index=True)
    stations = sorted(s.dropna().astype(str).str.strip().unique().tolist())
    return [x for x in stations if x]


def main() -> None:
    PRECOMPUTED_DIR.mkdir(parents=True, exist_ok=True)

    snap = latest_snapshot_path()
    snap_date = parse_date_from_filename(snap)

    snapshot_today_od = build_snapshot_today_od(snap)
    snapshot_today_od.to_csv(PRECOMPUTED_DIR / "snapshot_today_od.csv", index=False)

    stations = extract_stations(snap)
    (PRECOMPUTED_DIR / "stations.json").write_text(
        json.dumps(stations, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # metadata léger (le reste sera complété par l'agrégateur proba)
    meta = {}
    meta_path = PRECOMPUTED_DIR / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

    meta.update(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "latest_snapshot_date": snap_date.isoformat(),
            "latest_snapshot_file": snap.name,
            "n_stations": len(stations),
            "n_rows_snapshot_today_od": int(len(snapshot_today_od)),
        }
    )
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"OK: snapshot_today_od={len(snapshot_today_od)} stations={len(stations)} from {snap.name}")


if __name__ == "__main__":
    main()
