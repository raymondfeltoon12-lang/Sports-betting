"""Scores upcoming, already-lined games with the trained models and ranks
them by Kelly edge. This is model output, not betting advice: Phase 3/4
established these models have no demonstrated edge over the closing line,
so treat agreement between models as a (weak) confidence signal and
everything here as informational.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from backtest.kelly import kelly_fraction
from config import PROCESSED_DIR, ROOT_DIR
from models.train import CATEGORICAL_COLUMNS

ARTIFACTS_DIR = ROOT_DIR / "models" / "artifacts"


def load_upcoming() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / "game_features.parquet")
    df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, prefix=CATEGORICAL_COLUMNS)
    upcoming = df[df["home_score"].isna() & df["spread_line"].notna()].copy()

    feature_columns = joblib.load(ARTIFACTS_DIR / "feature_columns.joblib")
    for c in feature_columns:
        if c not in upcoming.columns:
            upcoming[c] = 0
    return upcoming, feature_columns


def choose_side(prob_home: float, home_odds: float, away_odds: float) -> tuple:
    home_f = kelly_fraction(prob_home, home_odds, 1.0, 0.2)
    away_f = kelly_fraction(1 - prob_home, away_odds, 1.0, 0.2)
    if home_f <= 0 and away_f <= 0:
        return None, 0.0
    return ("home", home_f) if home_f >= away_f else ("away", away_f)


def build_picks() -> pd.DataFrame:
    upcoming, feature_columns = load_upcoming()
    xgb = joblib.load(ARTIFACTS_DIR / "xgboost.joblib")
    logreg = joblib.load(ARTIFACTS_DIR / "logistic_regression.joblib")

    upcoming["p_xgb"] = xgb.predict_proba(upcoming[feature_columns])[:, 1]
    upcoming["p_lr"] = logreg.predict_proba(upcoming[feature_columns])[:, 1]

    rows = []
    for _, r in upcoming.iterrows():
        side, edge = choose_side(r["p_xgb"], r["home_spread_odds"], r["away_spread_odds"])
        xgb_leans_home = r["p_xgb"] > 0.5
        lr_leans_home = r["p_lr"] > 0.5
        rows.append({
            "game_id": r["game_id"], "season": int(r["season"]), "week": int(r["week"]),
            "home_team": r["home_team"], "away_team": r["away_team"],
            "spread_line": r["spread_line"],
            "p_xgb": round(float(r["p_xgb"]), 3), "p_lr": round(float(r["p_lr"]), 3),
            "models_agree": bool(xgb_leans_home == lr_leans_home),
            "side": side, "edge": round(float(edge), 4),
        })

    picks = pd.DataFrame(rows).sort_values("edge", ascending=False).reset_index(drop=True)
    return picks


def main():
    picks = build_picks()
    out_path = PROCESSED_DIR / "upcoming_picks.csv"
    picks.to_csv(out_path, index=False)
    print(picks.to_string(index=False))
    print(f"Saved {len(picks)} rows to {out_path}")


if __name__ == "__main__":
    main()
