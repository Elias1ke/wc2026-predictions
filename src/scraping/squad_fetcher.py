"""
squad_fetcher.py
================
WC2026 squad list + player stats pipeline using:
  - Wikipedia  -> official 48-team squad lists (no API key needed)
  - FBref via soccerdata -> player-level stats (standard, shooting, passing,
    defense, possession) for the 2025-26 club season

Strategy
--------
1. Scrape the 2026 FIFA World Cup squads page on Wikipedia to get the
   authoritative list of all called-up players, their positions, clubs,
   caps, and date of birth.
2. Pull Big-5 league + World Cup qualifier player stats from FBref using
   the `soccerdata` library.
3. Merge on player name (fuzzy if needed) to attach stats to each squad player.
4. Save per-team CSVs and a combined master file.

Requirements
------------
    pip install soccerdata pandas requests beautifulsoup4 rapidfuzz tqdm lxml

Usage
-----
    python squad_fetcher.py

Output
------
    data/raw/squads/all_squads.csv          <- Wikipedia squad data for all 48 teams
    data/raw/player_stats/standard_2526.csv <- FBref standard stats (Big-5 + WCQ)
    data/raw/player_stats/shooting_2526.csv
    data/raw/player_stats/passing_2526.csv
    data/raw/player_stats/defense_2526.csv
    data/raw/player_stats/possession_2526.csv
    data/processed/squad_with_stats.csv     <- merged master file
"""

import os
import sys
import time
import re
import warnings
import logging

import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

import random


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


warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)
handler = UnicodeSafeStreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[handler])


# --- User-Agent rotation to avoid blocking ---
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
]

def _random_ua() -> str:
    """Return a random User-Agent string."""
    return random.choice(USER_AGENTS)


def _fbref_headers() -> dict[str, str]:
    """Return a conservative headers set for FBref requests."""
    return {
        "User-Agent": _random_ua(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://fbref.com/",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

# --- Paths ---
SQUADS_DIR       = "data/raw/squads"
STATS_DIR        = "data/raw/player_stats"
PROCESSED_DIR    = "data/processed"
WIKI_SQUADS_URL  = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads"
FBREF_WC_URL     = "https://fbref.com/en/comps/1/history/World-Cup-Seasons"  # WC2026 player stats page

for d in (SQUADS_DIR, STATS_DIR, PROCESSED_DIR):
    os.makedirs(d, exist_ok=True)


# --- Mapping: Wikipedia country name -> FIFA three-letter code ---
# Used to tag each player row with a consistent team identifier.
TEAM_TLA = {
    "Algeria": "ALG", "Argentina": "ARG", "Australia": "AUS",
    "Belgium": "BEL", "Bolivia": "BOL", "Brazil": "BRA",
    "Cameroon": "CMR", "Canada": "CAN", "Chile": "CHI",
    "China": "CHN", "Colombia": "COL", "Costa Rica": "CRC",
    "Croatia": "CRO", "Czech Republic": "CZE", "Denmark": "DEN",
    "Ecuador": "ECU", "Egypt": "EGY", "England": "ENG",
    "France": "FRA", "Germany": "GER", "Ghana": "GHA",
    "Honduras": "HON", "Hungary": "HUN", "Indonesia": "IDN",
    "Iran": "IRN", "Iraq": "IRQ", "Italy": "ITA",
    "Ivory Coast": "CIV", "Jamaica": "JAM", "Japan": "JPN",
    "Kenya": "KEN", "Mali": "MLI", "Mexico": "MEX",
    "Morocco": "MAR", "Netherlands": "NED", "New Zealand": "NZL",
    "Nigeria": "NGA", "Norway": "NOR", "Panama": "PAN",
    "Paraguay": "PAR", "Peru": "PER", "Poland": "POL",
    "Portugal": "POR", "Qatar": "QAT", "Romania": "ROU",
    "Saudi Arabia": "KSA", "Senegal": "SEN", "Serbia": "SRB",
    "Slovakia": "SVK", "Slovenia": "SVN", "South Korea": "KOR",
    "South Africa": "RSA", "Spain": "ESP", "Switzerland": "SUI",
    "Trinidad and Tobago": "TTO", "Tunisia": "TUN", "Turkey": "TUR",
    "Ukraine": "UKR", "United States": "USA", "Uruguay": "URU",
    "Uzbekistan": "UZB", "Venezuela": "VEN",
    # aliases that appear on Wikipedia
    "Republic of Ireland": "IRL", "Korea Republic": "KOR",
    "United States of America": "USA", "Côte d'Ivoire": "CIV",
}


# ========================================
# STEP 1 -- Scrape Wikipedia for official WC2026 squad lists
# ========================================

def _clean_player_name(raw: str) -> str:
    """Strip Wikipedia citation markers and whitespace from a player name."""
    return re.sub(r"\[.*?\]", "", raw).strip()


def scrape_wikipedia_squads() -> pd.DataFrame:
    """
    Parse the Wikipedia '2026 FIFA World Cup squads' page.

    Returns a DataFrame with columns:
        team, tla, coach, number, position, player, date_of_birth,
        age, caps, club
    """
    out_path = os.path.join(SQUADS_DIR, "all_squads.csv")
    if os.path.exists(out_path):
        log.info("Cached Wikipedia squads found — loading from disk.")
        return pd.read_csv(out_path)

    log.info("Fetching Wikipedia squads page ...")
    headers = {"User-Agent": _random_ua()}
    resp = requests.get(WIKI_SQUADS_URL, headers=headers, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")
    rows = []
    current_team  = None
    current_coach = None

    # Wikipedia structures each national team as:
    #   <h3>  Team name heading
    #   Coach: <a> ... </a>
    #   <table class="wikitable"> with player rows

    for element in soup.find_all(["h3", "p", "table"]):
        # ── Team heading ──────────────────────────────────────────────────────
        if element.name == "h3":
            headline = element.get_text(strip=True)
            # Remove edit-section brackets: "Argentina[edit]" -> "Argentina"
            headline = re.sub(r"\[.*?\]", "", headline).strip()
            if headline:
                current_team  = headline
                current_coach = None

        # ── Coach paragraph ───────────────────────────────────────────────────
        elif element.name == "p":
            txt = element.get_text(strip=True)
            if txt.startswith("Coach:"):
                current_coach = txt.replace("Coach:", "").strip()

        # ── Squad table ───────────────────────────────────────────────────────
        elif element.name == "table" and current_team:
            headers_row = [th.get_text(strip=True) for th in element.find_all("th")]
            # We want tables that have a "Player" column
            if "Player" not in headers_row:
                continue

            col_map = {h: i for i, h in enumerate(headers_row)}

            for tr in element.find_all("tr")[1:]:  # skip header row
                cells = tr.find_all(["td", "th"])
                if len(cells) < 4:
                    continue

                def cell(col_name: str, default="") -> str:
                    idx = col_map.get(col_name)
                    if idx is None or idx >= len(cells):
                        return default
                    return cells[idx].get_text(strip=True)

                player_name = _clean_player_name(cell("Player"))
                if not player_name:
                    continue

                rows.append({
                    "team"          : current_team,
                    "tla"           : TEAM_TLA.get(current_team, current_team[:3].upper()),
                    "coach"         : current_coach,
                    "number"        : cell("No."),
                    "position"      : cell("Pos."),
                    "player"        : player_name,
                    "date_of_birth" : cell("Date of birth (age)"),
                    "caps"          : cell("Caps"),
                    "goals"         : cell("Goals"),
                    "club"          : _clean_player_name(cell("Club")),
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    log.info(f"Saved {len(df)} players from {df['team'].nunique()} teams -> {out_path}")
    return df


# ========================================
# STEP 2 -- Pull player stats from FBref via soccerdata
# ========================================

# Stat types we want.  Each is a separate FBref table.
STAT_TYPES = ["standard", "shooting", "passing", "defense", "possession"]

# Leagues to pull club stats from (Big-5 + WCQ competitions in soccerdata)
BIG5_LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "GER-Bundesliga",
    "ITA-Serie A",
    "FRA-Ligue 1",
]
INTL_LEAGUES = [
    "INT-World Cup",           # WC2026 itself (will have data once tournament starts)
    "INT-World Cup Qual. (UEFA)",
    "INT-World Cup Qual. (CONMEBOL)",
    "INT-World Cup Qual. (CAF)",
    "INT-World Cup Qual. (AFC)",
    "INT-World Cup Qual. (CONCACAF)",
    "INT-World Cup Qual. (OFC)",
]
SEASON = "2526"   # 2025-26

# Delay (seconds) between FBref requests to avoid rate-limiting
REQUEST_DELAY = 5

def fetch_fbref_stats(stat_type: str) -> pd.DataFrame:
    """
    Download one stat_type from FBref for Big-5 leagues + international WCQ.
    Falls back gracefully if a league isn't available in soccerdata.

    Returns a combined DataFrame.
    """
    try:
        import soccerdata as sd
    except ImportError:
        raise ImportError("Run: pip install soccerdata")

    out_path = os.path.join(STATS_DIR, f"{stat_type}_{SEASON}.csv")
    if os.path.exists(out_path):
        log.info(f"  Cached {stat_type} stats found — loading from disk.")
        return pd.read_csv(out_path)

    # Some FBref table names have changed across site versions; try aliases
    STAT_ALIASES = {
        "passing": ["passing", "passing_types"],
    }

    frames = []

    # ── Big-5 club stats ──────────────────────────────────────────────────────
    log.info(f"  Fetching {stat_type} stats from Big-5 leagues ...")
    try:
        fbref_big5 = sd.FBref(leagues=BIG5_LEAGUES, seasons=[SEASON])

        aliases = STAT_ALIASES.get(stat_type, [stat_type])
        df = pd.DataFrame()
        for st in aliases:
            try:
                df = fbref_big5.read_player_season_stats(stat_type=st)
                if df is not None and not df.empty:
                    log.info(f"    Got data from Big-5 using stat_type='{st}'")
                    break
            except Exception as e:
                log.debug(f"    Big-5 read failed for stat_type='{st}': {e}")

        if df is None or df.empty:
            log.warning(f"    No Big-5 club data found for stat_type variants: {aliases}")
        else:
            df["data_source"] = "big5_club"
            frames.append(df)
            time.sleep(REQUEST_DELAY)   # be polite to FBref
    except Exception as e:
        log.warning(f"  Big-5 fetch failed for {stat_type}: {e}")

    # ── International / WCQ stats ─────────────────────────────────────────────
    for league in INTL_LEAGUES:
        try:
            fbref_intl = sd.FBref(leagues=[league], seasons=[SEASON])

            aliases = STAT_ALIASES.get(stat_type, [stat_type])
            intl_df = pd.DataFrame()
            for st in aliases:
                try:
                    intl_df = fbref_intl.read_player_season_stats(stat_type=st)
                    if intl_df is not None and not intl_df.empty:
                        log.info(f"    Got data from {league} using stat_type='{st}'")
                        break
                except Exception as e:
                    log.debug(f"    {league} read failed for stat_type='{st}': {e}")

            if intl_df is None or intl_df.empty:
                log.debug(f"    No international data for {league} (stat_type variants: {aliases})")
            else:
                intl_df["data_source"] = "international"
                frames.append(intl_df)
                time.sleep(REQUEST_DELAY)
        except Exception as e:
            log.debug(f"  Skipping {league} ({stat_type}): {e}")

    if not frames:
        log.warning(f"  No data retrieved for stat_type={stat_type}")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Flatten MultiIndex columns that soccerdata sometimes returns
    if isinstance(combined.columns, pd.MultiIndex):
        combined.columns = ["_".join(filter(None, map(str, c))).strip("_")
                            for c in combined.columns]

    combined.to_csv(out_path, index=False)
    log.info(f"  Saved {len(combined)} rows -> {out_path}")
    return combined


def fetch_all_fbref_stats() -> dict[str, pd.DataFrame]:
    """Fetch all stat types and return as a dict."""
    stats = {}
    for st in STAT_TYPES:
        log.info(f"Fetching stat_type: {st}")
        stats[st] = fetch_fbref_stats(st)
    return stats


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2b — Direct FBref HTML scrape (fallback / supplement)
#           Scrapes the WC2026 player stats page directly with requests +
#           pandas.read_html, bypassing soccerdata entirely.
#           Useful once the tournament has started (live stats appear there).
# ══════════════════════════════════════════════════════════════════════════════

def scrape_fbref_wc_page(url: str = FBREF_WC_URL) -> pd.DataFrame:
    """
    Scrape the FBref World Cup Stats page directly.
    Returns the 'standard' stats table as a DataFrame.

    FBref serves tables as commented-out HTML that requests can still read.
    We pass the raw HTML through pandas.read_html with the correct table id.
    """
    out_path = os.path.join(STATS_DIR, "wc2026_fbref_direct.csv")
    if os.path.exists(out_path):
        log.info("Cached FBref WC direct scrape found — loading from disk.")
        return pd.read_csv(out_path)

    log.info(f"Direct FBref scrape: {url}")
    session = requests.Session()
    session.headers.update(_fbref_headers())

    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        log.warning(f"FBref direct scrape failed: {e}")
        return pd.DataFrame()

    # FBref wraps data tables in HTML comments; uncomment them first
    html = resp.text.replace("<!--", "").replace("-->", "")

    try:
        tables = pd.read_html(html, attrs={"id": "stats_standard"})
        if not tables:
            raise ValueError("Table 'stats_standard' not found")
        df = tables[0]
    except Exception:
        # Fallback: grab any table with a 'Player' column
        all_tables = pd.read_html(html)
        tables_with_player = [t for t in all_tables if "Player" in t.columns]
        if not tables_with_player:
            log.warning("No player tables found on FBref WC page.")
            return pd.DataFrame()
        df = tables_with_player[0]

    # Drop duplicate header rows that FBref embeds mid-table
    if "Player" in df.columns:
        df = df[df["Player"] != "Player"].copy()

    # Flatten MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(filter(None, map(str, c))).strip("_")
                      for c in df.columns]

    df["data_source"] = "fbref_direct_wc2026"
    df.to_csv(out_path, index=False)
    log.info(f"Saved {len(df)} rows -> {out_path}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Merge squads + stats
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_name(name: str) -> str:
    """Lowercase, strip accents, remove punctuation for matching."""
    import unicodedata
    name = str(name).lower().strip()
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = re.sub(r"[^a-z ]", "", name)
    return name.strip()


def merge_squads_with_stats(
    squads_df: pd.DataFrame,
    stats: dict[str, pd.DataFrame],
    fuzzy: bool = True,
) -> pd.DataFrame:
    """
    Left-join squad list with FBref stats on player name.
    Uses fuzzy matching (rapidfuzz) when exact match fails.
    """
    # Identify the player-name column in each stats frame
    def player_col(df: pd.DataFrame) -> str | None:
        for c in df.columns:
            if c.lower() in ("player", "player_player", "name"):
                return c
        return None

    # Build a combined stats frame (standard is the anchor)
    std = stats.get("standard", pd.DataFrame())
    if std.empty:
        log.warning("Standard stats empty — returning squads only.")
        return squads_df

    pc = player_col(std)
    if pc is None:
        log.warning("Could not find player column in standard stats.")
        return squads_df

    # Keep the most useful columns from standard stats
    keep_cols = [pc]
    for col in ["Nation", "Pos", "Squad", "Age", "Born",
                "MP", "Starts", "Min", "90s",
                "Gls", "Ast", "G+A", "G-PK", "PK",
                "xG", "xAG", "npxG", "PrgC", "PrgP", "PrgR"]:
        # column names may have suffixes after MultiIndex flattening
        matches = [c for c in std.columns if c == col or c.startswith(col + "_")]
        keep_cols.extend(matches)

    keep_cols = list(dict.fromkeys(keep_cols))  # deduplicate, preserve order
    std_slim  = std[[c for c in keep_cols if c in std.columns]].copy()
    std_slim.rename(columns={pc: "player_fbref"}, inplace=True)
    std_slim["_norm"] = std_slim["player_fbref"].apply(_normalize_name)

    # Normalise squad names
    squads_df = squads_df.copy()
    squads_df["_norm"] = squads_df["player"].apply(_normalize_name)

    # ── Exact match ───────────────────────────────────────────────────────────
    merged = squads_df.merge(std_slim, on="_norm", how="left")

    unmatched_mask = merged["player_fbref"].isna()
    unmatched_count = unmatched_mask.sum()
    log.info(f"Exact name matches: {len(merged) - unmatched_count} / {len(merged)}")

    # ── Fuzzy match for unmatched rows ────────────────────────────────────────
    if fuzzy and unmatched_count > 0:
        try:
            from rapidfuzz import process, fuzz
            log.info(f"Fuzzy matching {unmatched_count} unmatched players ...")
            fbref_names = std_slim["_norm"].tolist()
            fbref_lookup = dict(zip(std_slim["_norm"], std_slim.to_dict("records")))

            for idx in tqdm(merged[unmatched_mask].index, desc="Fuzzy matching"):
                query = merged.at[idx, "_norm"]
                result = process.extractOne(
                    query, fbref_names,
                    scorer=fuzz.token_sort_ratio,
                    score_cutoff=82,
                )
                if result:
                    best_norm = result[0]
                    for col, val in fbref_lookup[best_norm].items():
                        if col in merged.columns and col != "_norm":
                            merged.at[idx, col] = val
        except ImportError:
            log.warning("rapidfuzz not installed — skipping fuzzy match. "
                        "Run: pip install rapidfuzz")

    merged.drop(columns=["_norm"], inplace=True, errors="ignore")
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("WC2026 Squad + Player Stats Pipeline")
    log.info("=" * 60)

    # ── Step 1: Get squads ────────────────────────────────────────────────────
    log.info("\n[1/3] Scraping WC2026 squads from Wikipedia ...")
    squads_df = scrape_wikipedia_squads()
    print(f"\n  Teams found   : {squads_df['team'].nunique()}")
    print(f"  Players found : {len(squads_df)}")

    # ── Step 2: Get FBref stats ───────────────────────────────────────────────
    log.info("\n[2/3] Fetching player stats from FBref ...")

    # Try soccerdata first; fall back to direct scrape for the WC page
    stats = fetch_all_fbref_stats()

    # Always also scrape the FBref WC2026 page directly (tournament stats)
    wc_direct = scrape_fbref_wc_page()
    if not wc_direct.empty:
        stats["wc_direct"] = wc_direct

    # ── Step 3: Merge ─────────────────────────────────────────────────────────
    log.info("\n[3/3] Merging squad list with player stats ...")
    master = merge_squads_with_stats(squads_df, stats, fuzzy=True)

    out_path = os.path.join(PROCESSED_DIR, "squad_with_stats.csv")
    master.to_csv(out_path, index=False)
    log.info(f"\nMaster file saved -> {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    matched = master["player_fbref"].notna().sum() if "player_fbref" in master.columns else 0
    print("\n" + "=" * 60)
    print("  Pipeline complete")
    print(f"  Total players    : {len(master)}")
    print(f"  FBref stat match : {matched} ({matched/len(master)*100:.1f}%)")
    print(f"  Output           : {out_path}")
    print("=" * 60)

    return master


if __name__ == "__main__":
    main()