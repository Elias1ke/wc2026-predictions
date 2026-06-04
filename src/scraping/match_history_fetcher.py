"""
match_history_fetcher.py
========================
Fetches and assembles ALL historical data needed to train the WC2026
prediction models.

Sources
-------
  1. martj42/international_results (GitHub)
       49,000+ international results from 1872-2026
       No API key. Direct raw CSV from GitHub.

  2. ELO ratings computed from scratch using martj42 dataset
       Custom Elo system tuned for international football.
       No external dependency.

  3. StatsBomb Open Data (GitHub) — corners + cards
       Fully free, no API key, no rate limiting.
       Has WC2018 + WC2022 with exact corner kicks and cards per match.
       Extracted from event-level data (pass type=Corner, foul card type).

Outputs
-------
  data/raw/match_history/international_results_raw.csv
  data/raw/match_history/elo_all_teams_current.csv
  data/processed/match_history_with_elo.csv       <- main training set
  data/processed/corners_cards_history.csv         <- corners + cards

Requirements
------------
  pip install pandas requests tqdm
"""

import os
import time
import logging
from io import StringIO

import requests
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ─── Directories ──────────────────────────────────────────────────────────────
RAW_DIR       = "data/raw/match_history"
PROCESSED_DIR = "data/processed"

for d in (RAW_DIR, PROCESSED_DIR):
    os.makedirs(d, exist_ok=True)

# ─── Constants ────────────────────────────────────────────────────────────────
MARTJ42_URL = (
    "https://raw.githubusercontent.com/martj42/"
    "international_results/master/results.csv"
)

STATSBOMB_BASE = (
    "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
)

# StatsBomb competition IDs for World Cups
# comp_id=43 is FIFA World Cup; season_id identifies the year
STATSBOMB_COMPETITIONS = [
    {"comp_id": 43, "season_id": 3,   "name": "FIFA World Cup 2018"},
    {"comp_id": 43, "season_id": 106, "name": "FIFA World Cup 2022"},
]

TRAINING_START_YEAR = 2010

WC2026_TEAMS = {
    "Argentina", "Australia", "Belgium", "Bolivia", "Brazil",
    "Cameroon", "Canada", "Chile", "China PR", "Colombia",
    "Costa Rica", "Croatia", "Czech Republic", "Denmark", "Ecuador",
    "Egypt", "England", "France", "Germany", "Ghana",
    "Honduras", "Hungary", "Indonesia", "Iran", "Iraq",
    "Italy", "Ivory Coast", "Jamaica", "Japan", "Kenya",
    "Mali", "Mexico", "Morocco", "Netherlands", "New Zealand",
    "Nigeria", "Norway", "Panama", "Paraguay", "Peru",
    "Poland", "Portugal", "Qatar", "Romania", "Saudi Arabia",
    "Senegal", "Serbia", "Slovakia", "Slovenia", "South Korea",
    "South Africa", "Spain", "Switzerland", "Trinidad and Tobago",
    "Tunisia", "Turkey", "Ukraine", "United States", "Uruguay",
    "Uzbekistan", "Venezuela",
}


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 1 — martj42 international results
# ══════════════════════════════════════════════════════════════════════════════

def fetch_international_results() -> pd.DataFrame:
    raw_path = os.path.join(RAW_DIR, "international_results_raw.csv")

    if os.path.exists(raw_path):
        log.info("  Cached martj42 results — loading from disk.")
        df = pd.read_csv(raw_path, parse_dates=["date"])
    else:
        log.info("  Downloading from GitHub ...")
        resp = requests.get(
            MARTJ42_URL,
            headers={"User-Agent": "Mozilla/5.0 (WC2026-prediction)"},
            timeout=60,
        )
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text), parse_dates=["date"])
        df.to_csv(raw_path, index=False)
        log.info(f"  Saved {len(df):,} total rows -> {raw_path}")

    df = df[df["date"].dt.year >= TRAINING_START_YEAR].copy()
    df = df.dropna(subset=["home_score", "away_score"])
    mask = df["home_team"].isin(WC2026_TEAMS) | df["away_team"].isin(WC2026_TEAMS)
    df = df[mask].reset_index(drop=True)

    df["result"] = df.apply(
        lambda r: "H" if r["home_score"] > r["away_score"]
                  else ("A" if r["home_score"] < r["away_score"] else "D"),
        axis=1,
    )
    df["goal_diff"]   = (df["home_score"] - df["away_score"]).astype(int)
    df["total_goals"] = (df["home_score"] + df["away_score"]).astype(int)
    df["is_neutral"]  = df["neutral"].astype(bool)

    def _weight(t):
        t = str(t)
        if "World Cup" in t and "qualification" not in t.lower():
            return 2.0
        if any(k in t for k in ["qualification", "Nations League", "Euro",
                                  "Copa", "African Cup", "Asian Cup"]):
            return 1.5
        return 1.0

    df["tournament_weight"] = df["tournament"].apply(_weight)
    log.info(f"  Filtered: {len(df):,} matches | "
             f"{df['date'].dt.year.min()}-{df['date'].dt.year.max()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 2 — ELO ratings computed from scratch
# ══════════════════════════════════════════════════════════════════════════════

ELO_DEFAULT    = 1500
ELO_K_WC       = 60
ELO_K_MAJOR    = 50
ELO_K_NATIONS  = 40
ELO_K_FRIENDLY = 30
HOME_ADVANTAGE = 100


def _k_factor(tournament: str) -> int:
    t = str(tournament)
    if "World Cup" in t and "qualification" not in t.lower():
        return ELO_K_WC
    if any(k in t for k in ["Euro", "Copa America", "African Cup",
                              "Asian Cup", "Gold Cup"]):
        return ELO_K_MAJOR
    if "qualification" in t.lower() or "Nations League" in t:
        return ELO_K_NATIONS
    return ELO_K_FRIENDLY


def _expected_score(r_home: float, r_away: float, neutral: bool) -> float:
    adv = 0 if neutral else HOME_ADVANTAGE
    return 1 / (1 + 10 ** ((r_away - r_home - adv) / 400))


def compute_elo_ratings(full_df: pd.DataFrame) -> dict:
    log.info("  Computing ELO from full match history ...")
    raw_path = os.path.join(RAW_DIR, "international_results_raw.csv")
    df_full = pd.read_csv(raw_path, parse_dates=["date"])
    df_full = df_full.dropna(subset=["home_score", "away_score"])
    df_full = df_full.sort_values("date").reset_index(drop=True)

    elo = {}

    for _, row in df_full.iterrows():
        h, a    = row["home_team"], row["away_team"]
        r_h     = elo.get(h, ELO_DEFAULT)
        r_a     = elo.get(a, ELO_DEFAULT)
        neutral = bool(row["neutral"])

        E_h  = _expected_score(r_h, r_a, neutral)
        S_h  = 1.0 if row["home_score"] > row["away_score"] else \
               (0.5 if row["home_score"] == row["away_score"] else 0.0)
        K    = _k_factor(row["tournament"])
        delta = K * (S_h - E_h)
        elo[h] = r_h + delta
        elo[a] = r_a - delta

    current_df = pd.DataFrame([
        {"team": t, "elo": round(elo.get(t, ELO_DEFAULT), 1)}
        for t in sorted(WC2026_TEAMS)
    ]).sort_values("elo", ascending=False)

    current_df.to_csv(
        os.path.join(RAW_DIR, "elo_all_teams_current.csv"), index=False
    )
    log.info(f"  ELO computed for {len(elo)} teams.")
    log.info("  Top 5: " + ", ".join(
        f"{r['team']} ({r['elo']:.0f})"
        for _, r in current_df.head(5).iterrows()
    ))
    return elo


def attach_elo_to_matches(
    matches_df: pd.DataFrame,
    full_df_raw: pd.DataFrame,
) -> pd.DataFrame:
    log.info("  Attaching pre-match ELO to historical matches ...")

    df_full = full_df_raw.dropna(subset=["home_score", "away_score"])
    df_full = df_full.sort_values("date").reset_index(drop=True)

    elo = {}
    target_keys = set(zip(
        matches_df["date"].dt.normalize(),
        matches_df["home_team"],
        matches_df["away_team"],
    ))
    pre_match_elo = {}

    for _, row in df_full.iterrows():
        h, a       = row["home_team"], row["away_team"]
        r_h        = elo.get(h, ELO_DEFAULT)
        r_a        = elo.get(a, ELO_DEFAULT)
        date_norm  = pd.Timestamp(row["date"]).normalize()
        key        = (date_norm, h, a)

        if key in target_keys:
            pre_match_elo[key] = (r_h, r_a)

        neutral = bool(row["neutral"])
        E_h     = _expected_score(r_h, r_a, neutral)
        S_h     = 1.0 if row["home_score"] > row["away_score"] else \
                  (0.5 if row["home_score"] == row["away_score"] else 0.0)
        K       = _k_factor(row["tournament"])
        delta   = K * (S_h - E_h)
        elo[h]  = r_h + delta
        elo[a]  = r_a - delta

    home_elos, away_elos = [], []
    for _, row in matches_df.iterrows():
        key  = (pd.Timestamp(row["date"]).normalize(), row["home_team"], row["away_team"])
        pair = pre_match_elo.get(key)
        home_elos.append(pair[0] if pair else elo.get(row["home_team"], ELO_DEFAULT))
        away_elos.append(pair[1] if pair else elo.get(row["away_team"], ELO_DEFAULT))

    matches_df = matches_df.copy()
    matches_df["home_elo"] = [round(e, 1) for e in home_elos]
    matches_df["away_elo"] = [round(e, 1) for e in away_elos]
    matches_df["elo_diff"] = (matches_df["home_elo"] - matches_df["away_elo"]).round(1)

    log.info(f"  ELO attached. Coverage: {(matches_df['home_elo'] != ELO_DEFAULT).mean():.1%}")
    return matches_df


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 3 — StatsBomb Open Data (corners + cards)
# No API key. No rate limiting. Fully free static GitHub files.
# Coverage: WC2018 (64 matches) + WC2022 (64 matches) = 128 matches
# ══════════════════════════════════════════════════════════════════════════════

def _extract_corners_cards(events: list) -> dict:
    """
    Extract corner kicks and cards from a StatsBomb match event list.

    Corners: Pass events where pass.type.name == 'Corner'
    Yellow cards: Foul Committed events where foul_committed.card.name == 'Yellow Card'
    Red cards:    Foul Committed events where foul_committed.card.name contains 'Red'
                  (covers 'Red Card' and 'Second Yellow')
    """
    corners = sum(
        1 for e in events
        if e.get("type", {}).get("name") == "Pass"
        and e.get("pass", {}).get("type", {}).get("name") == "Corner"
    )
    yellows = sum(
        1 for e in events
        if e.get("type", {}).get("name") == "Foul Committed"
        and e.get("foul_committed", {}).get("card", {}).get("name") == "Yellow Card"
    )
    reds = sum(
        1 for e in events
        if e.get("type", {}).get("name") == "Foul Committed"
        and "Red" in str(e.get("foul_committed", {}).get("card", {}).get("name", ""))
    )
    return {"total_corners": corners, "yellow_cards": yellows, "red_cards": reds}


def _fetch_statsbomb_competition(comp_id: int, season_id: int, name: str) -> pd.DataFrame:
    """Fetch all matches + corners/cards for one StatsBomb competition/season."""
    # Get match list
    matches_url = f"{STATSBOMB_BASE}/matches/{comp_id}/{season_id}.json"
    resp = requests.get(matches_url, timeout=30)
    resp.raise_for_status()
    matches = resp.json()
    log.info(f"    {name}: {len(matches)} matches found")

    rows = []
    for m in tqdm(matches, desc=f"  {name}", leave=False):
        match_id   = m["match_id"]
        events_url = f"{STATSBOMB_BASE}/events/{match_id}.json"

        try:
            resp2  = requests.get(events_url, timeout=30)
            resp2.raise_for_status()
            events = resp2.json()

            stats = _extract_corners_cards(events)
            rows.append({
                "date"          : m["match_date"],
                "home_team"     : m["home_team"]["home_team_name"],
                "away_team"     : m["away_team"]["away_team_name"],
                "home_score"    : m["home_score"],
                "away_score"    : m["away_score"],
                "competition"   : name,
                "match_week"    : m.get("match_week"),
                "stage"         : m.get("competition_stage", {}).get("name"),
                **stats,
            })
            time.sleep(0.2)  # gentle on GitHub CDN

        except Exception as e:
            log.warning(f"      Failed match {match_id}: {e}")

    return pd.DataFrame(rows)


def fetch_statsbomb_corners_cards() -> pd.DataFrame:
    """
    Fetch corners + cards for WC2018 and WC2022 from StatsBomb Open Data.

    Why StatsBomb instead of FBref/Sofascore:
      - Fully free, no API key, no rate limiting, no 403 errors
      - Hosted as static JSON on GitHub — always accessible
      - Event-level granularity: exact corner kicks and cards extracted
        from raw match events, not scraped from summary tables
      - Covers both WC2018 (64 matches) and WC2022 (64 matches)

    Output columns:
        date, home_team, away_team, home_score, away_score,
        competition, total_corners, yellow_cards, red_cards
    """
    out_path = os.path.join(PROCESSED_DIR, "corners_cards_history.csv")

    if os.path.exists(out_path):
        log.info("  Cached corners/cards data — loading from disk.")
        return pd.read_csv(out_path, parse_dates=["date"])

    log.info("  Fetching corners + cards from StatsBomb Open Data ...")
    log.info("  Source: github.com/statsbomb/open-data (free, no key needed)")

    frames = []
    for comp in STATSBOMB_COMPETITIONS:
        try:
            df = _fetch_statsbomb_competition(
                comp["comp_id"],
                comp["season_id"],
                comp["name"],
            )
            if not df.empty:
                frames.append(df)
                log.info(
                    f"    Loaded {len(df)} matches | "
                    f"avg corners: {df['total_corners'].mean():.1f} | "
                    f"avg yellows: {df['yellow_cards'].mean():.1f}"
                )
        except Exception as e:
            log.warning(f"  Failed to fetch {comp['name']}: {e}")

    if not frames:
        log.warning("  No StatsBomb data retrieved.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined.to_csv(out_path, index=False)
    log.info(f"  Saved {len(combined)} matches -> {out_path}")
    return combined


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Build master training dataset
# ══════════════════════════════════════════════════════════════════════════════

def build_training_dataset(
    matches_elo: pd.DataFrame,
    corners_cards: pd.DataFrame,
) -> pd.DataFrame:
    out_path = os.path.join(PROCESSED_DIR, "match_history_with_elo.csv")
    log.info("  Building master training dataset ...")
    master = matches_elo.copy()

    if not corners_cards.empty and "home_team" in corners_cards.columns:
        cc = corners_cards[[
            "date", "home_team", "away_team",
            "total_corners", "yellow_cards", "red_cards"
        ]].copy()
        cc["date"]     = pd.to_datetime(cc["date"]).dt.normalize()
        master["date"] = pd.to_datetime(master["date"]).dt.normalize()
        master = master.merge(cc, on=["date", "home_team", "away_team"], how="left")

    master["is_wc"]  = (
        master["tournament"].str.contains("World Cup", na=False) &
        ~master["tournament"].str.contains("qualification", case=False, na=False)
    )
    master["is_wcq"] = master["tournament"].str.contains("qualification", case=False, na=False)
    master["year"]   = master["date"].dt.year
    master["month"]  = master["date"].dt.month

    most_recent = master["date"].max()
    master["days_ago"]     = (most_recent - master["date"]).dt.days
    master["decay_weight"] = master["tournament_weight"] * (0.995 ** (master["days_ago"] / 30))

    master = master.sort_values("date").reset_index(drop=True)
    master.to_csv(out_path, index=False)
    log.info(f"  Master training set: {len(master):,} rows -> {out_path}")
    return master


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> pd.DataFrame:
    log.info("=" * 65)
    log.info("WC2026 Match History Fetcher")
    log.info("=" * 65)

    log.info("\n[1/4] International match results ...")
    results_df = fetch_international_results()

    raw_full = pd.read_csv(
        os.path.join(RAW_DIR, "international_results_raw.csv"),
        parse_dates=["date"],
    )

    log.info("\n[2/4] Computing ELO ratings ...")
    compute_elo_ratings(raw_full)

    log.info("\n[3/4] Attaching pre-match ELO ...")
    results_elo = attach_elo_to_matches(results_df, raw_full)

    log.info("\n[4/4] Fetching corners + cards from StatsBomb Open Data ...")
    corners_cards = fetch_statsbomb_corners_cards()

    log.info("\nBuilding master training dataset ...")
    master = build_training_dataset(results_elo, corners_cards)

    print("\n" + "=" * 65)
    print("  Pipeline complete")
    print(f"  Matches in training set : {len(master):,}")
    print(f"  Date range              : {master['date'].min().date()} -> {master['date'].max().date()}")
    print(f"  ELO coverage            : {(master['home_elo'] != ELO_DEFAULT).mean():.1%}")
    if "total_corners" in master.columns:
        print(f"  Corners coverage        : {master['total_corners'].notna().mean():.1%}")
    if "yellow_cards" in master.columns:
        print(f"  Yellow cards coverage   : {master['yellow_cards'].notna().mean():.1%}")
    print(f"\n  Outputs:")
    print(f"    data/processed/match_history_with_elo.csv")
    print(f"    data/processed/corners_cards_history.csv")
    print(f"    data/raw/match_history/elo_all_teams_current.csv")
    print("=" * 65)
    return master


if __name__ == "__main__":
    main()