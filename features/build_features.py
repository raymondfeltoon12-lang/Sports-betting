"""Builds the game-level feature table used for modeling: one row per game,
combining home/away rolling form, venue splits, rest days, SRS ratings, and
line movement. Writes to the `game_features` table and a parquet snapshot.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DIR
from data.db import get_connection, init_db
from features.line_movement import attach_line_movement
from features.rolling_stats import build_team_features
from features.srs import compute_srs_asof

# Only pre-game-known, engineered columns get joined into the feature table.
# team_game_stats also carries the *actual result* of each game (raw
# points_scored, yards_per_play, etc.) which exists purely so the rolling/
# venue-avg columns below could be derived from it -- including those raw
# columns as "features" would leak the outcome being predicted.
PREGAME_FEATURE_SUFFIXES = ("_avg5", "_avg10", "_venue_avg")

INDOOR_ROOFS = ("dome", "closed")


def _clean_weather(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["roof"] = df["roof"].replace({"None": pd.NA}).fillna("unknown").str.strip()
    df["surface"] = df["surface"].replace({"None": pd.NA}).fillna("unknown").str.strip()
    # Indoor games have no weather effect -- filling with the outdoor-game
    # median would bias the "temp"/"wind" columns for dome games, since a
    # dome always plays like ~70F and no wind, not an average of outdoor
    # conditions it never sees.
    indoor = df["roof"].isin(INDOOR_ROOFS)
    df.loc[indoor & df["temp"].isna(), "temp"] = 70.0
    df.loc[indoor & df["wind"].isna(), "wind"] = 0.0
    return df


def _side_frame(team_features: pd.DataFrame, is_home: int, prefix: str) -> pd.DataFrame:
    side = team_features[team_features["is_home"] == is_home].copy()
    keep = [c for c in side.columns if c.endswith(PREGAME_FEATURE_SUFFIXES)]
    rename = {c: f"{prefix}_{c}" for c in keep}
    side = side.rename(columns=rename)
    return side[["game_id"] + list(rename.values())]


def build_game_features() -> pd.DataFrame:
    with get_connection() as conn:
        games = pd.read_sql("SELECT * FROM games", conn)
        team_stats = pd.read_sql("SELECT * FROM team_game_stats", conn)

    games = _clean_weather(games)
    team_features = build_team_features(team_stats)

    home_side = _side_frame(team_features, is_home=1, prefix="home")
    away_side = _side_frame(team_features, is_home=0, prefix="away")

    merged = games.merge(home_side, on="game_id", how="left")
    merged = merged.merge(away_side, on="game_id", how="left")

    srs = compute_srs_asof(games)
    merged = merged.merge(srs, on="game_id", how="left")
    merged["srs_diff"] = merged["home_srs"] - merged["away_srs"]

    merged = attach_line_movement(merged)

    merged["home_margin"] = merged["home_score"] - merged["away_score"]
    has_result = merged["home_score"].notna() & merged["spread_line"].notna()
    covered = pd.Series(pd.NA, index=merged.index, dtype="Int64")
    covered[has_result] = (merged.loc[has_result, "home_margin"] > merged.loc[has_result, "spread_line"]).astype(int)
    merged["home_covered_spread"] = covered

    return merged


def save_game_features(df: pd.DataFrame) -> None:
    with get_connection() as conn:
        df.to_sql("game_features", conn, if_exists="replace", index=False)
    df.to_parquet(PROCESSED_DIR / "game_features.parquet", index=False)
    print(f"Saved {len(df)} rows x {len(df.columns)} cols to `game_features` and data/processed/game_features.parquet")


def main():
    init_db()
    df = build_game_features()
    save_game_features(df)


if __name__ == "__main__":
    main()
