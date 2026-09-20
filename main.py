"""Pipeline entrypoint. Currently wires up Phase 1 (data ingestion) only;
feature engineering, modeling, and backtesting phases will be added here as
they're built.
"""

import argparse

from data import fetch_nfl_data, fetch_odds_api


def run_data_pipeline(with_live_odds: bool = False):
    fetch_nfl_data.main()
    if with_live_odds:
        fetch_odds_api.main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NFL spread prediction pipeline")
    parser.add_argument(
        "--with-live-odds", action="store_true",
        help="Also snapshot current lines from The Odds API (requires ODDS_API_KEY)",
    )
    args = parser.parse_args()
    run_data_pipeline(with_live_odds=args.with_live_odds)
