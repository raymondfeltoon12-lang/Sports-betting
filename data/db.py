import sqlite3
from contextlib import contextmanager

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    game_type TEXT,
    gameday TEXT,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    home_score REAL,
    away_score REAL,
    home_rest INTEGER,
    away_rest INTEGER,
    div_game INTEGER,
    roof TEXT,
    surface TEXT,
    temp REAL,
    wind REAL,
    stadium TEXT,
    -- betting lines as recorded by nflverse (home-team perspective:
    -- negative spread_line = home team favored)
    spread_line REAL,
    away_spread_odds REAL,
    home_spread_odds REAL,
    total_line REAL,
    over_odds REAL,
    under_odds REAL,
    away_moneyline REAL,
    home_moneyline REAL
);

CREATE TABLE IF NOT EXISTS team_game_stats (
    game_id TEXT NOT NULL,
    team TEXT NOT NULL,
    opponent TEXT NOT NULL,
    is_home INTEGER NOT NULL,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    points_scored REAL,
    points_allowed REAL,
    offensive_plays INTEGER,
    total_yards REAL,
    yards_per_play REAL,
    pass_yards REAL,
    pass_plays INTEGER,
    rush_yards REAL,
    rush_plays INTEGER,
    turnovers INTEGER,
    epa_per_play REAL,
    success_rate REAL,
    PRIMARY KEY (game_id, team),
    FOREIGN KEY (game_id) REFERENCES games (game_id)
);

CREATE TABLE IF NOT EXISTS odds_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fetched_at TEXT NOT NULL,
    commence_time TEXT,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    sportsbook TEXT NOT NULL,
    home_spread REAL,
    away_spread REAL,
    home_spread_price REAL,
    away_spread_price REAL,
    matched_game_id TEXT,
    FOREIGN KEY (matched_game_id) REFERENCES games (game_id)
);

CREATE INDEX IF NOT EXISTS idx_team_game_team ON team_game_stats (team, season, week);
CREATE INDEX IF NOT EXISTS idx_games_season_week ON games (season, week);
CREATE INDEX IF NOT EXISTS idx_odds_snapshots_teams ON odds_snapshots (home_team, away_team, commence_time);
"""


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_connection() as conn:
        conn.executescript(SCHEMA)


if __name__ == "__main__":
    init_db()
    print(f"Initialized database at {DB_PATH}")
