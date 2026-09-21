"""Runs the full pipeline: data ingestion -> feature engineering -> model
training/evaluation -> Kelly-criterion backtest.
"""

import argparse

from backtest import simulate as backtest_simulate
from data import fetch_nfl_data, fetch_odds_api
from features import build_features
from models import predict_upcoming, prop_rates, train as train_models


def run_pipeline(with_live_odds: bool = False):
    fetch_nfl_data.main()
    if with_live_odds:
        fetch_odds_api.main()
    build_features.main()
    train_models.main()
    backtest_simulate.main()
    predict_upcoming.main()
    prop_rates.main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NFL spread prediction pipeline")
    parser.add_argument(
        "--with-live-odds", action="store_true",
        help="Also snapshot current lines from The Odds API (requires ODDS_API_KEY)",
    )
    args = parser.parse_args()
    run_pipeline(with_live_odds=args.with_live_odds)
