"""Snapshots current NFL spread odds from The Odds API (free tier) into
`odds_snapshots`. Run this periodically (e.g. via cron) between now and
kickoff for a slate of games -- repeated snapshots over time are what let
us later compute opening-vs-closing line movement, since the free tier has
no historical-odds endpoint.
"""

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    ODDS_API_BASE_URL, ODDS_API_KEY, ODDS_API_MARKETS, ODDS_API_REGIONS,
    ODDS_API_SPORT, ODDS_SNAPSHOT_CSV,
)
from data.db import get_connection, init_db

CSV_HEADER = [
    "fetched_at", "commence_time", "home_team", "away_team", "sportsbook",
    "home_spread", "away_spread", "home_spread_price", "away_spread_price",
]

# The Odds API returns full team names ("Atlanta Falcons"); nflverse (and
# our games/game_features tables) use abbreviations ("ATL"). Normalize here
# so line-movement joins against games actually match.
TEAM_NAME_TO_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}


def fetch_current_odds() -> list[dict]:
    if not ODDS_API_KEY:
        raise RuntimeError(
            "ODDS_API_KEY is not set. Get a free key at https://the-odds-api.com "
            "and put it in a .env file (see .env.example)."
        )
    url = f"{ODDS_API_BASE_URL}/sports/{ODDS_API_SPORT}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": ODDS_API_REGIONS,
        "markets": ODDS_API_MARKETS,
        "oddsFormat": "american",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    remaining = resp.headers.get("x-requests-remaining")
    if remaining is not None:
        print(f"Odds API requests remaining this period: {remaining}")
    return resp.json()


def parse_events(events: list[dict], fetched_at: str) -> list[tuple]:
    rows = []
    for event in events:
        raw_home, raw_away = event.get("home_team"), event.get("away_team")
        home_team = TEAM_NAME_TO_ABBR.get(raw_home, raw_home)
        away_team = TEAM_NAME_TO_ABBR.get(raw_away, raw_away)
        commence_time = event.get("commence_time")
        for bookmaker in event.get("bookmakers", []):
            sportsbook = bookmaker.get("key")
            for market in bookmaker.get("markets", []):
                if market.get("key") != "spreads":
                    continue
                outcomes = {o["name"]: o for o in market.get("outcomes", [])}
                home_outcome = outcomes.get(raw_home, {})
                away_outcome = outcomes.get(raw_away, {})
                rows.append((
                    fetched_at, commence_time, home_team, away_team, sportsbook,
                    home_outcome.get("point"), away_outcome.get("point"),
                    home_outcome.get("price"), away_outcome.get("price"),
                ))
    return rows


def store_snapshots(rows: list[tuple]) -> None:
    with get_connection() as conn:
        conn.executemany(
            """
            INSERT INTO odds_snapshots
                (fetched_at, commence_time, home_team, away_team, sportsbook,
                 home_spread, away_spread, home_spread_price, away_spread_price)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    write_header = not ODDS_SNAPSHOT_CSV.exists()
    with open(ODDS_SNAPSHOT_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(CSV_HEADER)
        writer.writerows(rows)

    print(f"Stored {len(rows)} odds snapshot rows (db + {ODDS_SNAPSHOT_CSV}).")


def main():
    init_db()
    fetched_at = datetime.now(timezone.utc).isoformat()
    events = fetch_current_odds()
    rows = parse_events(events, fetched_at)
    if not rows:
        print("No current NFL odds returned (likely off-season or no games in window).")
        return
    store_snapshots(rows)


if __name__ == "__main__":
    main()
