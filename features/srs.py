"""Simple Rating System (SRS): rating_team = avg_margin + avg(opponent_rating),
solved iteratively. Computed as-of each week using only games completed
strictly before that week, so it's safe to use as a pre-game feature.

Cold start: a new season carries over the previous season's final ratings,
regressed 50% toward 0 (typical "returning teams aren't as good/bad as last
year's endpoint" shrinkage), rather than resetting blind to 0 in week 1.
"""

import pandas as pd

CARRYOVER_DECAY = 0.5
ITERATIONS = 20


def _iterative_srs(games: pd.DataFrame, base_ratings: dict) -> dict:
    teams = set(games["home_team"]).union(games["away_team"]) | set(base_ratings.keys())
    ratings = {t: base_ratings.get(t, 0.0) for t in teams}
    if games.empty:
        return ratings

    matchups = list(zip(games["home_team"], games["away_team"], games["home_score"] - games["away_score"]))

    for _ in range(ITERATIONS):
        sums = {t: 0.0 for t in ratings}
        counts = {t: 0 for t in ratings}
        for home, away, margin in matchups:
            sums[home] += margin + ratings[away]
            counts[home] += 1
            sums[away] += -margin + ratings[home]
            counts[away] += 1

        new_ratings = {
            t: (sums[t] / counts[t] if counts[t] else ratings[t])
            for t in ratings
        }
        mean_r = sum(new_ratings.values()) / len(new_ratings)
        ratings = {t: r - mean_r for t, r in new_ratings.items()}

    return ratings


def compute_srs_asof(games: pd.DataFrame) -> pd.DataFrame:
    completed = games[games["home_score"].notna()].sort_values(["season", "week"])
    rows = []
    prev_season_final: dict = {}

    for season, season_games in games.sort_values(["season", "week"]).groupby("season"):
        completed_season = completed[completed["season"] == season]
        carryover = {t: r * CARRYOVER_DECAY for t, r in prev_season_final.items()}

        for week in sorted(season_games["week"].unique()):
            prior_games = completed_season[completed_season["week"] < week]
            ratings = _iterative_srs(prior_games, carryover)
            for _, g in season_games[season_games["week"] == week].iterrows():
                rows.append({
                    "game_id": g["game_id"],
                    "home_srs": ratings.get(g["home_team"], 0.0),
                    "away_srs": ratings.get(g["away_team"], 0.0),
                })

        prev_season_final = _iterative_srs(completed_season, carryover)

    return pd.DataFrame(rows)
