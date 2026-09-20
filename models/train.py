"""Trains a logistic regression baseline and an XGBoost classifier to predict
P(home team covers the closing spread), split by season (never randomly, to
avoid leaking future seasons into training), and evaluates both on log loss
plus closing line value (CLV) where line-movement data is available.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DIR, ROOT_DIR, TEST_START_SEASON
from data.db import get_connection

ARTIFACTS_DIR = ROOT_DIR / "models" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "home_covered_spread"

FEATURE_COLUMNS = (
    [f"home_{m}" for m in [
        "points_scored_avg5", "points_allowed_avg5", "yards_per_play_avg5",
        "epa_per_play_avg5", "success_rate_avg5",
        "points_scored_avg10", "points_allowed_avg10", "yards_per_play_avg10",
        "epa_per_play_avg10", "success_rate_avg10",
        "points_scored_venue_avg", "points_allowed_venue_avg",
    ]]
    + [f"away_{m}" for m in [
        "points_scored_avg5", "points_allowed_avg5", "yards_per_play_avg5",
        "epa_per_play_avg5", "success_rate_avg5",
        "points_scored_avg10", "points_allowed_avg10", "yards_per_play_avg10",
        "epa_per_play_avg10", "success_rate_avg10",
        "points_scored_venue_avg", "points_allowed_venue_avg",
    ]]
    + ["home_rest", "away_rest", "div_game", "home_srs", "away_srs", "srs_diff",
       "spread_line", "total_line"]
)


def load_dataset() -> pd.DataFrame:
    with get_connection() as conn:
        df = pd.read_sql("SELECT * FROM game_features", conn)
    df = df.dropna(subset=[TARGET_COL]).copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    return df


def season_split(df: pd.DataFrame):
    train = df[df["season"] < TEST_START_SEASON]
    test = df[df["season"] >= TEST_START_SEASON]
    return train, test


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def compute_clv(test_df: pd.DataFrame, model_probs: np.ndarray) -> dict:
    """CLV in points: for the side the model favors, how much the line moved
    in that bettor's favor between the recorded line and closing. Requires
    opening_spread/closing_spread, which only populate once live Odds API
    snapshots have been collected for a game (see data/fetch_odds_api.py) --
    historical seasons only ever have a single (closing) line on record, so
    this will be N/A for them.
    """
    has_movement = test_df["opening_spread"].notna() & test_df["closing_spread"].notna()
    if not has_movement.any():
        return {"avg_clv_points": None, "n_games_with_line_data": 0, "note": "no historical opening-line data available"}

    movement = test_df.loc[has_movement, "closing_spread"] - test_df.loc[has_movement, "opening_spread"]
    bet_home = model_probs[has_movement.values] > 0.5
    clv_points = np.where(bet_home, movement, -movement)
    return {"avg_clv_points": float(np.mean(clv_points)), "n_games_with_line_data": int(has_movement.sum())}


def evaluate_model(name: str, model, X_test: pd.DataFrame, y_test: pd.Series, test_df: pd.DataFrame) -> dict:
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs > 0.5).astype(int)
    metrics = {
        "model": name,
        "log_loss": log_loss(y_test, probs),
        "accuracy": accuracy_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
    }
    metrics.update(compute_clv(test_df, probs))
    return metrics


def main():
    df = load_dataset()
    train_df, test_df = season_split(df)
    print(f"Train: {len(train_df)} games (seasons < {TEST_START_SEASON}); "
          f"Test: {len(test_df)} games (seasons >= {TEST_START_SEASON})")

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COL]

    logreg = train_logistic_regression(X_train, y_train)
    xgb = train_xgboost(X_train, y_train)

    results = [
        evaluate_model("logistic_regression", logreg, X_test, y_test, test_df),
        evaluate_model("xgboost", xgb, X_test, y_test, test_df),
    ]
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    results_df.to_csv(ARTIFACTS_DIR / "model_eval_summary.csv", index=False)
    joblib.dump(logreg, ARTIFACTS_DIR / "logistic_regression.joblib")
    joblib.dump(xgb, ARTIFACTS_DIR / "xgboost.joblib")
    joblib.dump(FEATURE_COLUMNS, ARTIFACTS_DIR / "feature_columns.joblib")
    print(f"Saved models and summary to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
