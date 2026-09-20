"""Opening-vs-closing spread movement, built from repeated Odds API
snapshots (see data/fetch_odds_api.py). Historical seasons have no opening
line on record -- nfl_data_py only gives us the closing line -- so this will
be all-NaN until enough live snapshots accumulate for upcoming games.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from data.db import get_connection


def load_line_movement() -> pd.DataFrame:
    with get_connection() as conn:
        snapshots = pd.read_sql("SELECT * FROM odds_snapshots", conn)

    if snapshots.empty:
        return pd.DataFrame(columns=["home_team", "away_team", "commence_time", "opening_spread", "closing_spread", "line_movement"])

    snapshots = snapshots.sort_values("fetched_at")
    grouped = snapshots.groupby(["home_team", "away_team", "commence_time"])["home_spread"]
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
