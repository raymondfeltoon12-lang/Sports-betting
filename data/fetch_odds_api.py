"""Snapshots current NFL spread odds from The Odds API (free tier) into
`odds_snapshots`. Run this periodically (e.g. via cron) between now and
kickoff for a slate of games -- repeated snapshots over time are what let
us later compute opening-vs-closing line movement, since the free tier has
no historical-odds endpoint.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import ODDS_API_BASE_URL, ODDS_API_KEY, ODDS_API_MARKETS, ODDS_API_REGIONS, ODDS_API_SPORT
from data.db import get_connection, init_db


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
        home_team = event.get("home_team")
        away_team = event.get("away_team")
        commence_time = event.get("commence_time")
        for bookmaker in event.get("bookmakers", []):
            sportsbook = bookmaker.get("key")
            for market in bookmaker.get("markets", []):
                if market.get("key") != "spreads":
                    continue
                outcomes = {o["name"]: o for o in market.get("outcomes", [])}
                home_outcome = outcomes.get(home_team, {})
                away_outcome = outcomes.get(away_team, {})
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
    print(f"Stored {len(rows)} odds snapshot rows.")


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
