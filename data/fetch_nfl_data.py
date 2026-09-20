"""Pulls historical NFL schedules (incl. betting lines) and play-by-play data
via nfl_data_py, aggregates pbp to team-game level, and loads both into the
local SQLite database.
"""

import sys
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import RAW_DIR, SEASONS
from data.db import get_connection, init_db

GAME_COLUMNS = [
    "game_id", "season", "week", "game_type", "gameday",
    "home_team", "away_team", "home_score", "away_score",
    "home_rest", "away_rest", "div_game", "roof", "surface", "temp", "wind",
    "stadium", "spread_line", "away_spread_odds", "home_spread_odds",
    "total_line", "over_odds", "under_odds", "away_moneyline", "home_moneyline",
]

SCRIMMAGE_PLAYS = ("pass", "run")


def fetch_schedules() -> pd.DataFrame:
    df = nfl.import_schedules(SEASONS)
    for col in GAME_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df["div_game"] = df["div_game"].fillna(0).astype(int)
    return df[GAME_COLUMNS]


def load_games(df: pd.DataFrame) -> None:
    with get_connection() as conn:
        df.to_sql("games_staging", conn, if_exists="replace", index=False)
        conn.execute(
            "DELETE FROM team_game_stats WHERE game_id IN (SELECT game_id FROM games_staging)"
        )
        conn.execute("DELETE FROM games WHERE game_id IN (SELECT game_id FROM games_staging)")
        conn.execute(f"""
            INSERT INTO games ({", ".join(GAME_COLUMNS)})
            SELECT {", ".join(GAME_COLUMNS)} FROM games_staging
        """)
        conn.execute("DROP TABLE games_staging")
    print(f"Loaded {len(df)} games into `games`.")


def fetch_and_aggregate_pbp(season: int) -> pd.DataFrame:
    cache_path = RAW_DIR / f"pbp_{season}.parquet"
    if cache_path.exists():
        pbp = pd.read_parquet(cache_path)
    else:
        pbp = nfl.import_pbp_data([season], downcast=True, cache=False)
        pbp.to_parquet(cache_path)

    plays = pbp[pbp["play_type"].isin(SCRIMMAGE_PLAYS)].copy()
    plays["is_pass"] = (plays["play_type"] == "pass").astype(int)
    plays["is_rush"] = (plays["play_type"] == "run").astype(int)
    plays["turnover"] = plays["interception"].fillna(0) + plays["fumble_lost"].fillna(0)

    grouped = plays.groupby(["game_id", "posteam"]).agg(
        offensive_plays=("play_id", "count"),
        total_yards=("yards_gained", "sum"),
        pass_yards=("passing_yards", "sum"),
        pass_plays=("is_pass", "sum"),
        rush_yards=("rushing_yards", "sum"),
        rush_plays=("is_rush", "sum"),
        turnovers=("turnover", "sum"),
        epa_per_play=("epa", "mean"),
        success_rate=("success", "mean"),
        season=("season", "first"),
        week=("week", "first"),
        defteam=("defteam", "first"),
    ).reset_index()

    grouped["yards_per_play"] = grouped["total_yards"] / grouped["offensive_plays"]
    grouped = grouped.rename(columns={"posteam": "team", "defteam": "opponent"})
    return grouped


def load_team_game_stats(games: pd.DataFrame) -> None:
    home_scores = games.set_index("game_id")[["home_team", "away_team", "home_score", "away_score"]]

    all_rows = []
    for season in SEASONS:
        try:
            agg = fetch_and_aggregate_pbp(season)
        except Exception as exc:
            print(f"Skipping pbp for {season}: {exc}")
            continue

        agg = agg.join(home_scores, on="game_id", how="inner")
        agg["is_home"] = (agg["team"] == agg["home_team"]).astype(int)
        agg["points_scored"] = agg.apply(
            lambda r: r["home_score"] if r["is_home"] else r["away_score"], axis=1
        )
        agg["points_allowed"] = agg.apply(
            lambda r: r["away_score"] if r["is_home"] else r["home_score"], axis=1
        )

        cols = [
            "game_id", "team", "opponent", "is_home", "season", "week",
            "points_scored", "points_allowed", "offensive_plays", "total_yards",
            "yards_per_play", "pass_yards", "pass_plays", "rush_yards", "rush_plays",
            "turnovers", "epa_per_play", "success_rate",
        ]
        all_rows.append(agg[cols])
        print(f"Aggregated {season}: {len(agg)} team-game rows.")

    full = pd.concat(all_rows, ignore_index=True)
    with get_connection() as conn:
        full.to_sql("team_game_stats_staging", conn, if_exists="replace", index=False)
        conn.execute(
            "DELETE FROM team_game_stats WHERE game_id IN (SELECT game_id FROM team_game_stats_staging)"
        )
        conn.execute(f"""
            INSERT INTO team_game_stats ({", ".join(cols)})
            SELECT {", ".join(cols)} FROM team_game_stats_staging
        """)
        conn.execute("DROP TABLE team_game_stats_staging")
    print(f"Loaded {len(full)} team-game stat rows into `team_game_stats`.")


def main():
    init_db()
    games = fetch_schedules()
    load_games(games)
    load_team_game_stats(games)


if __name__ == "__main__":
    main()
