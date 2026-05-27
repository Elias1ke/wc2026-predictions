"""data_fetcher.py
=================
Utilities to download international results (martj42) and ClubElo histories.

Usage:
    python data_fetcher.py

Outputs:
    data/raw/martj42_international_results.csv
    data/raw/clubeelo/<team>.csv
"""

from __future__ import annotations

import os
import sys
import logging
import requests
import pandas as pd
import time
from urllib.parse import quote_plus


class UnicodeSafeStreamHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            super().emit(record)
        except UnicodeEncodeError:
            msg = self.format(record)
            encoding = self.stream.encoding or "utf-8"
            safe_msg = msg.encode(encoding, errors="replace").decode(encoding)
            try:
                self.stream.write(safe_msg + self.terminator)
                self.flush()
            except Exception:
                pass


log = logging.getLogger(__name__)
handler = UnicodeSafeStreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[handler])

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR = os.path.join(ROOT, "data", "raw")
CLUBELO_DIR = os.path.join(RAW_DIR, "clubeelo")
MARTJ42_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

for d in (RAW_DIR, CLUBELO_DIR):
    os.makedirs(d, exist_ok=True)


def download_martj42(out_path: str | None = None, force: bool = False) -> str:
    """Download the martj42 international results CSV and save locally.

    Returns the path to the saved file.
    """
    out_path = out_path or os.path.join(RAW_DIR, "martj42_international_results.csv")
    if os.path.exists(out_path) and not force:
        log.info("Cached martj42 CSV found — skipping download.")
        return out_path

    log.info(f"Downloading martj42 international results from {MARTJ42_URL}")
    resp = requests.get(MARTJ42_URL, timeout=30)
    resp.raise_for_status()

    with open(out_path, "wb") as f:
        f.write(resp.content)

    log.info(f"Saved martj42 CSV -> {out_path}")
    return out_path


def fetch_clubeelo_api(team: str, out_dir: str | None = None, delay: float = 0.5) -> str | None:
    """Fetch ClubElo history via the public API and save as CSV.

    API: http://api.clubelo.com/TEAMNAME  (returns CSV)
    Team names should be URL-encoded (spaces -> + or %20).
    Returns the saved path or None on failure.
    """
    out_dir = out_dir or CLUBELO_DIR
    safe_name = quote_plus(team)
    url = f"http://api.clubelo.com/{safe_name}"
    out_path = os.path.join(out_dir, f"{safe_name}.csv")

    if os.path.exists(out_path):
        log.info(f"Cached ClubElo for '{team}' found — skipping API fetch.")
        return out_path

    try:
        log.info(f"Fetching ClubElo (API) for: {team}")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(resp.content)
        time.sleep(delay)
        log.info(f"Saved ClubElo CSV -> {out_path}")
        return out_path
    except Exception as e:
        log.warning(f"ClubElo API fetch failed for '{team}': {e}")
        return None


def fetch_clubeelo_soccerdata(team: str, out_dir: str | None = None) -> str | None:
    """Attempt to fetch ClubElo history using the `soccerdata` wrapper as a fallback.

    Returns saved CSV path or None.
    """
    try:
        import soccerdata as sd
    except ImportError:
        log.debug("soccerdata not installed; skipping soccerdata ClubElo fetch")
        return None

    out_dir = out_dir or CLUBELO_DIR
    safe_name = quote_plus(team)
    out_path = os.path.join(out_dir, f"{safe_name}_sd.csv")

    if os.path.exists(out_path):
        log.info(f"Cached soccerdata ClubElo for '{team}' found — skipping.")
        return out_path

    try:
        log.info(f"Fetching ClubElo (soccerdata) for: {team}")
        ce = sd.ClubElo(teams=[team])
        # soccerdata APIs vary; try a few common reader names and save first successful
        df = None
        for reader in ("read", "read_team_history", "read_elo"):
            try:
                fn = getattr(ce, reader, None)
                if fn is None:
                    continue
                df = fn()
                if df is not None and not df.empty:
                    break
            except Exception:
                continue

        if df is None or df.empty:
            log.warning(f"soccerdata returned no ClubElo data for '{team}'")
            return None

        df.to_csv(out_path, index=False)
        log.info(f"Saved soccerdata ClubElo -> {out_path}")
        return out_path
    except Exception as e:
        log.debug(f"soccerdata ClubElo fetch error for '{team}': {e}")
        return None


def teams_from_squads(squads_path: str) -> set[str]:
    """Read `data/raw/squads/all_squads.csv` and return unique club names.
    Falls back to an empty set if file isn't found.
    """
    try:
        df = pd.read_csv(squads_path)
        clubs = set(df["club"].dropna().astype(str).unique())
        return clubs
    except Exception:
        return set()


def main():
    # 1) Download martj42 CSV
    download_martj42()

    # 2) If squads exist, fetch ClubElo for each club
    squads_path = os.path.join(RAW_DIR, "squads", "all_squads.csv")
    clubs = teams_from_squads(squads_path)

    if not clubs:
        log.info("No squads file found or no clubs discovered; skipping ClubElo batch fetch.")
        return

    log.info(f"Found {len(clubs)} unique clubs in squads; fetching ClubElo histories (API first, soccerdata fallback).")
    for club in sorted(clubs):
        ok = fetch_clubeelo_api(club)
        if not ok:
            fetch_clubeelo_soccerdata(club)


if __name__ == "__main__":
    main()
