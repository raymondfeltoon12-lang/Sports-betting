"""Simulates historical spread betting using each trained model's predicted
probabilities, sized with the Kelly Criterion, walking through the held-out
test seasons in chronological order with a single running bankroll.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from backtest.kelly import american_to_decimal, kelly_fraction
from config import KELLY_MULTIPLIER, MAX_BET_FRACTION, ROOT_DIR, STARTING_BANKROLL
from models.train import FEATURE_COLUMNS, load_dataset, season_split

MODELS_DIR = ROOT_DIR / "models" / "artifacts"
RESULTS_DIR = ROOT_DIR / "backtest" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def choose_bet(prob_home: float, home_odds: float, away_odds: float, kelly_multiplier: float) -> tuple[str, float]:
    """Returns (side, kelly_fraction) for whichever side has the larger
    positive edge, or (None, 0.0) if neither side clears a positive edge.
    """
    home_f = kelly_fraction(prob_home, home_odds, kelly_multiplier, MAX_BET_FRACTION)
    away_f = kelly_fraction(1 - prob_home, away_odds, kelly_multiplier, MAX_BET_FRACTION)
    if home_f <= 0 and away_f <= 0:
        return None, 0.0
    return ("home", home_f) if home_f >= away_f else ("away", away_f)


def resolve_bet(side: str, home_margin: float, spread_line: float) -> str:
    if home_margin == spread_line:
        return "push"
    home_covers = home_margin > spread_line
    won = home_covers if side == "home" else not home_covers
    return "win" if won else "loss"


def simulate(test_df: pd.DataFrame, probs: np.ndarray, kelly_multiplier: float) -> pd.DataFrame:
    df = test_df.sort_values(["season", "week"]).copy()
    prob_series = pd.Series(probs, index=test_df.index)
    df["prob_home_cover"] = prob_series.reindex(df.index).values

    bankroll = STARTING_BANKROLL
    rows = []
    for _, g in df.iterrows():
        side, frac = choose_bet(g["prob_home_cover"], g["home_spread_odds"], g["away_spread_odds"], kelly_multiplier)
        if side is None:
            rows.append({**g.to_dict(), "bet_side": None, "stake": 0.0, "outcome": "no_bet",
                         "profit": 0.0, "bankroll_after": bankroll})
            continue

        stake = frac * bankroll
        outcome = resolve_bet(side, g["home_margin"], g["spread_line"])
        odds = g["home_spread_odds"] if side == "home" else g["away_spread_odds"]
        b = american_to_decimal(odds) - 1

        if outcome == "win":
            profit = stake * b
        elif outcome == "loss":
            profit = -stake
        else:  # push
            profit = 0.0

        bankroll += profit
        rows.append({**g.to_dict(), "bet_side": side, "stake": stake, "outcome": outcome,
                     "profit": profit, "bankroll_after": bankroll})

    return pd.DataFrame(rows)


def compute_max_drawdown(bankroll_curve: pd.Series) -> float:
    running_peak = bankroll_curve.cummax()
    drawdown = (bankroll_curve - running_peak) / running_peak
    return float(drawdown.min())  # most negative value = largest drawdown


def summarize(bet_log: pd.DataFrame, model_name: str, kelly_multiplier: float) -> dict:
    placed = bet_log[bet_log["bet_side"].notna()]
    decided = placed[placed["outcome"].isin(["win", "loss"])]
    wins = (decided["outcome"] == "win").sum()
    losses = (decided["outcome"] == "loss").sum()
    pushes = (placed["outcome"] == "push").sum()

    total_staked = placed["stake"].sum()
    total_profit = placed["profit"].sum()
    final_bankroll = bet_log["bankroll_after"].iloc[-1] if len(bet_log) else STARTING_BANKROLL

    bankroll_curve = pd.concat([pd.Series([STARTING_BANKROLL]), bet_log["bankroll_after"]], ignore_index=True)

    return {
        "model": model_name,
        "kelly_multiplier": kelly_multiplier,
        "n_games": len(bet_log),
        "n_bets": len(placed),
        "wins": int(wins),
        "losses": int(losses),
        "pushes": int(pushes),
        "win_rate": wins / (wins + losses) if (wins + losses) else None,
        "total_staked": float(total_staked),
        "total_profit": float(total_profit),
        "roi_on_turnover": float(total_profit / total_staked) if total_staked else None,
        "starting_bankroll": STARTING_BANKROLL,
        "final_bankroll": float(final_bankroll),
        "total_return_pct": float(final_bankroll / STARTING_BANKROLL - 1),
        "max_drawdown_pct": compute_max_drawdown(bankroll_curve),
    }


def run_backtest_for_model(model_name: str, test_df: pd.DataFrame, kelly_multiplier: float) -> dict:
    model = joblib.load(MODELS_DIR / f"{model_name}.joblib")
    probs = model.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]
    bet_log = simulate(test_df, probs, kelly_multiplier)
    tag = f"{model_name}_k{kelly_multiplier}"
    bet_log.to_csv(RESULTS_DIR / f"bet_log_{tag}.csv", index=False)
    return summarize(bet_log, model_name, kelly_multiplier)


def main():
    df = load_dataset()
    _, test_df = season_split(df)
    test_df = test_df.dropna(subset=["home_spread_odds", "away_spread_odds"]).reset_index(drop=True)

    # Run at both full Kelly (as classically defined) and a conservative
    # quarter-Kelly. Full Kelly is extremely sensitive to probability-
    # estimate error -- with a model that has little real edge over the
    # closing line (see Phase 3), its near-0.5 predictions are mostly noise,
    # and full Kelly stakes aggressively on that noise. Quarter-Kelly is the
    # realistic way anyone would actually size these bets.
    summaries = []
    for kelly_multiplier in (KELLY_MULTIPLIER, 0.25):
        summaries.append(run_backtest_for_model("logistic_regression", test_df, kelly_multiplier))
        summaries.append(run_backtest_for_model("xgboost", test_df, kelly_multiplier))

    summary_df = pd.DataFrame(summaries)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(RESULTS_DIR / "backtest_summary.csv", index=False)
    print(f"Saved backtest summary and per-bet logs to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
