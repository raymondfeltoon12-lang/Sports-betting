"""Opening-vs-closing spread movement, built from repeated Odds API
snapshots (see data/fetch_odds_api.py). Historical seasons have no opening
line on record -- nfl_data_py only gives us the closing line -- so this will
be all-NaN until enough live snapshots accumulate for upcoming games.

Reads from the tracked CSV (data/processed/odds_snapshots.csv), not the
local SQLite db: snapshots capture a moment in time and can't be
regenerated, so they're committed to git and shared across environments
(e.g. the scheduled GitHub Actions job), unlike the db file itself.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import ODDS_SNAPSHOT_CSV


def load_line_movement() -> pd.DataFrame:
    empty = pd.DataFrame(columns=["home_team", "away_team", "commence_time", "opening_spread", "closing_spread", "line_movement"])
    if not ODDS_SNAPSHOT_CSV.exists():
        return empty

    snapshots = pd.read_csv(ODDS_SNAPSHOT_CSV)
    if snapshots.empty:
        return empty

    # Collapse across sportsbooks to a per-round consensus line first, so
    # "opening" and "closing" aren't accidentally comparing two different
    # books' numbers.
    consensus = (
        snapshots.groupby(["home_team", "away_team", "commence_time", "fetched_at"])["home_spread"]
        .median()
        .reset_index()
        .sort_values("fetched_at")
    )
    grouped = consensus.groupby(["home_team", "away_team", "commence_time"])["home_spread"]
    movement = grouped.agg(opening_spread="first", closing_spread="last").reset_index()
    movement["line_movement"] = movement["closing_spread"] - movement["opening_spread"]
    return movement


def attach_line_movement(games: pd.DataFrame) -> pd.DataFrame:
    movement = load_line_movement()
    if movement.empty:
        games = games.copy()
        games["opening_spread"] = pd.NA
        games["closing_spread"] = pd.NA
        games["line_movement"] = pd.NA
        return games

    # Match on team names + game date (commence_time is a full timestamp,
    # gameday is a date -- compare on date only).
    movement = movement.copy()
    movement["gameday"] = pd.to_datetime(movement["commence_time"]).dt.strftime("%Y-%m-%d")
    merged = games.merge(
        movement[["home_team", "away_team", "gameday", "opening_spread", "closing_spread", "line_movement"]],
        on=["home_team", "away_team", "gameday"],
        how="left",
    )
    return merged
