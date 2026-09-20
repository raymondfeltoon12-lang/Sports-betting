"""Rolling-form and home/away-split features computed from team_game_stats.

Everything here is shifted by one game before any rolling/expanding window is
applied, so a team's feature value for game N only ever uses games strictly
before N -- no look-ahead leakage.
"""

import pandas as pd

ROLLING_METRICS = [
    "points_scored", "points_allowed", "yards_per_play",
    "epa_per_play", "success_rate",
]
ROLLING_WINDOWS = (5, 10)

SPLIT_METRICS = ["points_scored", "points_allowed"]


def compute_rolling_team_features(team_stats: pd.DataFrame) -> pd.DataFrame:
    df = team_stats.sort_values(["team", "season", "week"]).copy()
    grouped = df.groupby("team")
    for window in ROLLING_WINDOWS:
        for metric in ROLLING_METRICS:
            df[f"{metric}_avg{window}"] = grouped[metric].transform(
                lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
            )
    return df


def compute_home_away_splits(team_stats: pd.DataFrame) -> pd.DataFrame:
    """For each team-game row, the team's average performance in its prior
    games at that same venue (home or away), i.e. a home team's row gets its
    historical home-game average, an away team's row gets its away-game
    average.
    """
    df = team_stats.sort_values(["team", "is_home", "season", "week"]).copy()
    grouped = df.groupby(["team", "is_home"])
    for metric in SPLIT_METRICS:
        df[f"{metric}_venue_avg"] = grouped[metric].transform(
            lambda s: s.shift(1).expanding().mean()
        )
    return df.sort_values(["team", "season", "week"])


def build_team_features(team_stats: pd.DataFrame) -> pd.DataFrame:
    df = compute_rolling_team_features(team_stats)
    df = compute_home_away_splits(df)
    return df
